% ----------------------------------------------------------------------- %
% Differential Evolution Strategy (DES)
% Journal-version code of the CEC 2017 5th-place entry (not the contest build)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4*D, mu = floor(lambda/2)  % Offspring and selected points, log weights
%   F = 1                       % Scale of the difference vectors (its adaptation is off)
%   c_c = mu/(mu + 2)           % Weight of history differences against the path term
%   c_p = 1/sqrt(D)             % Decay of the evolution path
%   H = ceil(6 + ceil(3*sqrt(D)))  % Generations kept in the history window
%   tol = 1e-12                 % Scale of the vanishing Gaussian noise term
%
% Algorithm Concept:
%   - The mean moves to the weighted mean of the mu best; the selected points of the
%     last H generations are archived, scaled by 1/sqrt(2)
%   - Offspring = mean + difference of two archived points of a random past
%     generation + N(0,1) times that generation's mean shift + N(0,1) times a path
%   - The mean shift is the selected mean minus the population mean; the path
%     accumulates mean-to-mean steps as in CMA-ES cumulation
%   - Offspring outside the box are not evaluated; they rank on a death penalty
%   - The exponentially smoothed midpoint of the means is evaluated every generation
%     and can become the best-so-far solution
%
% Reference:
% Dariusz Jagodzinski, Jaroslaw Arabas,
% A differential evolution strategy,
% 2017 IEEE Congress on Evolutionary Computation (CEC), 2017, pp. 1872-1876.
% https://doi.org/10.1109/CEC.2017.7969529
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' R release (DES.R, github.com/Jagorius/DES), which its README
% ties to the journal version (Arabas, Jagodzinski, IEEE TEVC 24(1) 84-98, 2020). It is
% not the CEC 2017 build: at D = 10 its CEC 2017 F10 error exceeds 50 in 7/13 runs (10/22
% in a Python transcription of DES.R) against 1/51 submitted, and the submission stalls
% near 1e-7 on F1 and F6 where the release reaches 0. Kept as released: F adaptation and
% the restart trigger are commented out (re-enabled, the transcription gets F10 2/20);
% out-of-box offspring cost no evaluation; the path restarts at each ring-buffer wrap; the
% population mean weights members in generation order. Harness: the start mean is the box
% centre (README: 0 on [-100,100]); the first population fills the inner 80 % of the box
% around it instead of [0.8*lb, 0.8*ub].
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = des(problem)

    N     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;

    lambda     = 4 * N;
    mu         = floor(lambda / 2);
    weights    = log(mu + 1) - log(1:mu)';
    weights    = weights / sum(weights);
    weightsPop = log(lambda + 1) - log(1:lambda)';
    weightsPop = weightsPop / sum(weightsPop);
    cc         = mu / (mu + 2);
    cp         = 1 / sqrt(N);
    histSize   = ceil(6 + ceil(3 * sqrt(N)));
    Ft         = 1;
    tol        = 1e-12;
    chiN       = sqrt(2) * exp(gammaln((N + 1) / 2) - gammaln(N / 2));
    histNorm   = 1 / sqrt(2);
    BIG        = realmax;              % R's .Machine$double.xmax

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = ((lb + ub) / 2)';
    ctxX = zeros(0, N);                % evaluated members of the current generation
    ctxF = zeros(0, 1);

    centre     = (lb + ub) / 2;
    population = centre + 0.8 * ((lb + rand(N, lambda) .* (ub - lb)) - centre);
    fitness    = fn_l(population);
    newMean    = centre;
    worst_fit  = max(fitness);
    popMean    = population * weightsPop;
    cumMean    = centre;

    history  = zeros(N, mu * histSize);   % slot h holds columns (h-1)*mu+1 .. h*mu
    dMean    = zeros(N, histSize);
    pc       = zeros(N, histSize);
    histHead = 0;
    iter     = 0;

    while FE < maxFE
        iter     = iter + 1;
        histHead = mod(histHead, histSize) + 1;

        [~, order]     = sort(fitness);
        selectedPoints = population(:, order(1:mu));
        history(:, (histHead - 1) * mu + (1:mu)) = selectedPoints * histNorm / Ft;

        oldMean = newMean;
        newMean = selectedPoints * weights;
        dMean(:, histHead) = (newMean - popMean) / Ft;
        step = (newMean - oldMean) / Ft;

        % At a wrap of the ring index the path is rebuilt from zero, as in the release
        if histHead == 1
            pc(:, 1) = sqrt(mu * cp * (2 - cp)) * step;
        else
            pc(:, histHead) = (1 - cp) * pc(:, histHead - 1) + sqrt(mu * cp * (2 - cp)) * step;
        end

        if iter < histSize
            limit = histHead;
        else
            limit = histSize;
        end
        hs1 = randi(limit, 1, lambda);
        hs2 = randi(limit, 1, lambda);
        j1  = randi(mu, 1, lambda);
        j2  = randi(mu, 1, lambda);
        x1  = history(:, (hs1 - 1) * mu + j1);
        x2  = history(:, (hs1 - 1) * mu + j2);
        diffs = sqrt(cc) * ((x1 - x2) + randn(1, lambda) .* dMean(:, hs1)) + ...
                sqrt(1 - cc) * randn(1, lambda) .* pc(:, hs2);

        population = newMean + Ft * diffs + tol * (1 - 2 / N ^ 2) ^ (iter / 2) * randn(N, lambda) / chiN;
        population(~isfinite(population)) = BIG;
        populationRepaired = bounce_back(population, lb, ub);

        popMean = population * weightsPop;
        ctxX = zeros(0, N);
        ctxF = zeros(0, 1);
        fitness = fn_l(population);

        % Non-Lamarckian ranking: only a point violated in EVERY coordinate gets worst + distance
        fitnessNL = fitness;
        repaired  = all(population ~= populationRepaired, 1);
        vecDist   = sum((population - populationRepaired) .^ 2, 1);
        fitnessNL(repaired) = worst_fit + vecDist(repaired);
        fitnessNL(~isfinite(fitnessNL)) = BIG;

        fmax = max(fitness);
        if fmax > worst_fit
            worst_fit = fmax;
        end
        fitness = fitnessNL;

        cumMean = 0.8 * cumMean + 0.2 * newMean;
        if FE < maxFE
            evaluate(bounce_back(cumMean, lb, ub));
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;

    % R's fn_l: in-box columns are evaluated (within the budget), the rest score BIG at no cost
    function f = fn_l(P)
        n = size(P, 2);
        f = BIG * ones(1, n);
        feasible = all(P >= lb & P <= ub, 1);
        if FE + n <= maxFE
            cand = 1:n;
        else
            cand = 1:max(0, maxFE - FE);
        end
        idx = cand(feasible(cand));
        if ~isempty(idx)
            f(idx) = evaluate(P(:, idx));
        end
    end

    % Evaluates the columns of P, tracking the best and recording the evaluated prefix
    function f = evaluate(P)
        n = size(P, 2);
        [fv, FE_new] = calculate_fitness(P, problem, FE);
        f = fv(:)';
        Xr = P';
        for q = 1:n
            if f(q) < bsf
                bsf  = f(q);
                bsfx = Xr(q, :);
            end
            ec = FE + q;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                ec, [ctxX; Xr(1:q, :)], [ctxF; f(1:q)'], population_history, ...
                fitness_history, history_index, maxFE);
        end
        FE = FE_new;
        ctxX = [ctxX; Xr];
        ctxF = [ctxF; f'];
    end
end

function x = bounce_back(x, lb, ub)
% bounceBackBoundary2: modular reflection back into the box, lower violations first
    range = ub - lb;
    below = x < lb;
    if any(below(:))
        LB = repmat(lb, 1, size(x, 2));
        R  = repmat(range, 1, size(x, 2));
        x(below) = LB(below) + mod(LB(below) - x(below), R(below));
    end
    above = x > ub;
    if any(above(:))
        UB = repmat(ub, 1, size(x, 2));
        R  = repmat(range, 1, size(x, 2));
        x(above) = UB(above) - mod(x(above) - UB(above), R(above));
    end
    x = min(max(x, lb), ub);           % the release recurses until inside; this catches rounding
end

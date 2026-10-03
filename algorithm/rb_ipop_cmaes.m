% ----------------------------------------------------------------------- %
% IPOP-CMA-ES with Midpoint (RB-IPOP-CMA-ES)
% CEC 2017 competition -- 8th place (the ranking slide misprints it as 7th)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 2^run * 4*N, mu = lambda/2   % IPOP doubling of the release's 4*N
%   sigma0 = 0.035*(ub - lb)              % sigma = 7 on the release's [-100, 100] box
%   max_dx = (ub - lb)/5                  % Sigma repair threshold; 5 repairs -> restart
%   LastIts = 10 + ceil(30*N/(4*N)) = 18  % Midpoint check period in iterations
%   midpoint tolerance = 1e-8             % Midpoint fitness change that restarts
%   flat-fitness boost: sigma = 2*sigma*exp(0.2 + cs/ds)
%
% Algorithm Concept:
%   - CMA-ES with cumulative step-size adaptation and the rank-one/rank-mu update
%   - IPOP: restart from a uniform point with the population doubled when any
%     trigger fires: tolX, condition, no-effect axis/coordinate, indefinite C
%   - Midpoint trigger: every LastIts iterations the mean m is evaluated; a
%     change of its fitness below 1e-8 since the last check restarts the run
%   - Sigma repair: a coordinate std above (ub-lb)/5 resets sigma to a quarter
%     of the largest admissible value; more than five repairs restart the run
%   - Out-of-box samples are evaluated at their projection, fitness multiplied
%     by 1 + squared distance to the box
%
% Reference:
% Rafal Biedrzycki,
% A version of IPOP-CMA-ES algorithm with midpoint for CEC 2017 single
% objective bound constrained problems,
% 2017 IEEE Congress on Evolutionary Computation (CEC), 2017, pp. 1489-1494.
% https://doi.org/10.1109/CEC.2017.7969479
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author-linked R release, github.com/ewarchul/cec2021
% R/rb-ipop-cma-esr-csa.R (cmaesr-based, rewritten by E. Warchulski), with
% last_its_type = "mean". Where it differs from the paper the release is kept:
% lambda = 4*N (paper 4*(4+floor(3 ln N))), a uniform start (paper: best of 100),
% restart once min eig <= sqrt(eps)*max eig (paper: floor at 1e-15), no extra
% midpoints at report times. Quirks kept: tolX reads an undefined p.c and tests
% sqrt(eig(C)) < 1e-12 alone; iter and h_sigma's count span restarts.
% Adaptations: box-normalised coordinates (exact on a uniform box), absolute tests
% in problem units, f(m) at the clamped mean, and a negative fitness is divided
% by the penalty instead of multiplied, so the penalty always worsens it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rb_ipop_cmaes(problem)

    n     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    w     = ub - lb;              % search coordinates y = (x - lb) ./ w

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = (lb + ub)' / 2;

    lambda0     = 4 * n;
    sigma0      = 7 / 200;                    % control default sigma = 7 on [-100, 100]
    max_dx      = 1 / 5;                      % (ub - lb) / 5 in box units
    last_its    = 10 + ceil(30 * n / (4 * n)); % the release divides by 4n, not by lambda
    max_repairs = 5;
    chi_n       = sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ^ 2));

    iter = 0;                 % never reset: drives the LastIts schedule across restarts
    run  = -1;
    m    = rand(n, 1);        % cecb passes runif(n, -100, 100) as par
    done = false;

    while ~done && FE < maxFE
        run = run + 1;
        lambda = ceil(2 ^ run * lambda0);
        if run > 0
            m = rand(n, 1);
        end
        mu      = floor(lambda / 2);
        weights = log(mu + 1) - log(1:mu)';
        weights = weights / sum(weights);
        mueff   = sum(weights) ^ 2 / sum(weights .^ 2);

        cs   = (mueff + 2) / (n + mueff + 3);
        ds   = 1 + 2 * max(0, sqrt((mueff - 1) / (n + 1)) - 1) + cs;
        cc   = 4 / (n + 4);
        cmu  = mueff;
        ccov = (1 / cmu) * 2 / (n + 1.4) ^ 2 + (1 - 1 / cmu) * ((2 * cmu - 1) / ((n + 2) ^ 2 + 2 * cmu));

        sigma = sigma0;
        pc = zeros(n, 1);
        ps = zeros(n, 1);
        B  = eye(n);
        BD = eye(n);
        C  = eye(n);

        refFit  = inf;
        oldRef  = inf;
        repairs = 0;

        restarting = false;
        while ~restarting
            if FE >= maxFE
                done = true;
                break;
            end
            iter = iter + 1;

            nsample = min(lambda, maxFE - FE);
            arz = randn(n, lambda);
            ary = BD * arz;
            arx = m + sigma * ary;
            arxr = min(max(arx, 0), 1);
            % Eq. 51 penalty, distance measured in problem units
            pen = 1 + sum((w .* (arx - arxr)) .^ 2, 1);
            pen(isinf(pen)) = realmax / 2;

            Xe = min(max(lb + w .* arxr(:, 1:nsample), lb), ub);
            [fr, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
                evalCols(Xe, zeros(n, 0), zeros(1, 0), problem, FE, maxFE, curve, ...
                         population_history, fitness_history, history_index, bsf, bsfx);
            if nsample < lambda
                done = true;
                break;
            end

            fitn = penalise(fr, pen);
            [fitn_s, order] = sort(fitn);
            sel = order(1:mu);
            m = arx(:, sel) * weights;

            if mod(iter, last_its) == 0
                if FE >= maxFE
                    done = true;
                    break;
                end
                oldRef = refFit;
                xm = min(max(lb + w .* m, lb), ub);
                [refFit, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
                    evalCols(xm, Xe, fr, problem, FE, maxFE, curve, ...
                             population_history, fitness_history, history_index, bsf, bsfx);
            end

            y_w = ary(:, sel) * weights;
            z_w = arz(:, sel) * weights;

            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (B * z_w);
            % FE stands for the release's n.evals, which spans all restarts
            hsig = norm(ps) / sqrt(1 - (1 - cs) ^ (2 * FE / lambda)) / chi_n < 1.4 + 2 / (n + 1);
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * y_w;
            y  = ary(:, sel);

            C = (1 - ccov) * C ...
                + ccov * (1 / cmu) * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
                + ccov * (1 - 1 / cmu) * (y .* weights') * y';

            sigma = sigma * exp((norm(ps) / chi_n - 1) * cs / ds);

            % RB-IPOP sigma repair
            if any(sigma * sqrt(diag(C)) > max_dx)
                sigma = min(max_dx ./ sqrt(diag(C))) / 4;
                repairs = repairs + 1;
            end

            % R's eigen() fails on a non-finite C; it is taken as indefCovMat
            if any(~isfinite(C(:)))
                restarting = true;
                continue;
            end
            [B, e] = eigDesc(C);
            if any(isnan(e)) || any(e <= sqrt(eps) * abs(e(1)))
                restarting = true;                                % indefCovMat
                continue;
            end
            dvec = sqrt(e);
            BD = B .* dvec';

            % Escape flat fitness, the RB-IPOP doubling included
            if fitn_s(1) == fitn_s(min(1 + floor(lambda / 2), 2 + ceil(lambda / 4)))
                sigma = 2 * sigma * exp(0.2 + cs / ds);
            end

            xm = lb + w .* m;
            ii = mod(iter, n) + 1;
            ax = w .* (0.3 * sigma * dvec(ii) * B(:, ii));
            if all(dvec < 1e-12)
                restarting = true;                                % tolX
            elseif max(e) / min(e) > 1e14
                restarting = true;                                % conditionCov
            elseif sum((xm - (xm + ax)) .^ 2) < eps
                restarting = true;                                % noEffectAxis
            elseif sum((xm - (xm + 0.2 * sigma * w)) .^ 2) < eps
                restarting = true;                                % noEffectCoord
            elseif mod(iter, last_its) == 0 && abs(oldRef - refFit) < 1e-8
                restarting = true;                                % lastIts
            elseif repairs > max_repairs
                restarting = true;                                % sigSupress
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [f, FE, curve, ph, fh, hidx, bsf, bsfx] = evalCols(X, ctxX, ctxF, problem, FE, maxFE, ...
                                                           curve, ph, fh, hidx, bsf, bsfx)
% Evaluates the columns of X; the recorded population is the context plus these points
    k = size(X, 2);
    [f, FE] = calculate_fitness(X, problem, FE);
    f = f(:)';
    rows = [ctxX, X]';
    fall = [ctxF, f]';
    for j = 1:k
        if f(j) < bsf
            bsf  = f(j);
            bsfx = X(:, j)';
        end
        ec = FE - k + j;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hidx] = record_history(ec, rows, fall, ph, fh, hidx, maxFE);
        end
    end
end

function fp = penalise(f, pen)
% The release multiplies by the penalty; a negative fitness is divided so it still worsens
    fp = f .* pen;
    neg = f < 0;
    fp(neg) = f(neg) ./ pen(neg);
end

function [V, e] = eigDesc(C)
% R's eigen(symmetric = TRUE): reads the lower triangle, values in decreasing order
    Cs = tril(C) + tril(C, -1)';
    [V, E] = eig(Cs);
    [e, o] = sort(diag(E), 'descend');
    V = V(:, o);
end

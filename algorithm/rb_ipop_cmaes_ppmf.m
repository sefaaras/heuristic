% ----------------------------------------------------------------------- %
% RB-IPOP-CMA-ES with PPMF step-size adaptation (RB-IPOP-CMA-ES-PPMF)
% CEC 2021 competition -- 9th (last) place in all five rankings
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 2^run * 4*N, mu = lambda/2   % IPOP doubling of the release's 4*N
%   sigma0 = 0.035*(ub - lb)              % sigma = 7 on the release's [-100, 100] box
%   p_target = 0.2, d = 2                 % PPMF target success rate and damping
%   LastIts = 10 + ceil(30*N/(4*N)) = 18  % Midpoint check period in iterations
%   midpoint tolerance = 1e-8             % Midpoint fitness change that restarts
%   flat-fitness boost: sigma = 20*sigma*exp(0.2 + cs/ds)
%
% Algorithm Concept:
%   - CMA-ES rank-one/rank-mu covariance update; IPOP restarts from a uniform
%     point with the population doubled
%   - The midpoint (mean of the projected population) is evaluated every generation
%   - PPMF step size: sigma *= exp(d*(p_succ - p_target)/(1 - p_target)), with
%     p_succ the share of offspring beating the previous generation's midpoint
%   - Restart triggers: tolX, condition, no-effect axis/coordinate, indefinite C,
%     and a midpoint fitness change below 1e-8 over LastIts iterations
%   - Out-of-box samples are evaluated at their projection, fitness multiplied
%     by 1 + squared distance to the box
%
% Reference:
% Eryk Warchulski, Jaroslaw Arabas,
% A New Step-Size Adaptation Rule for CMA-ES Based on the Population Midpoint
% Fitness,
% 2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. 825-831.
% https://doi.org/10.1109/CEC45853.2021.9504829
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' R release, github.com/ewarchul/cec2021
% R/rb-ipop-cma-esr-ppmf.R, in the competition configuration of
% configs/reproduction/ppmf/*.yml: last_its_type = "ave", p_target = 0.2.
% It is rb_ipop_cmaes with PPMF in place of CSA and without the sigma repair.
% Release quirks kept: a run's first generation compares with an infinite
% midpoint, so p_succ = 1 and sigma grows by e^2; tolX reads an undefined p.c
% and tests sqrt(eig(C)) < 1e-12 alone; iter and h_sigma's count span restarts.
% Adaptations: box-normalised coordinates (exact on a uniform box), absolute tests
% in problem units, and a negative fitness is divided by the penalty instead of
% multiplied, so the penalty always worsens it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rb_ipop_cmaes_ppmf(problem)

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
    last_its    = 10 + ceil(30 * n / (4 * n)); % the release divides by 4n, not by lambda
    d_param     = 2;
    p_target    = 0.2;                        % reproduction configs; the R default is 0.1
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

        refFit    = inf;
        oldRef    = inf;
        fMid      = inf;
        fMidOld   = inf;

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

            % Midpoint of the projected population, evaluated every generation
            if FE >= maxFE
                done = true;
                break;
            end
            fMidOld = fMid;
            xmid = min(max(lb + w .* mean(arxr, 2), lb), ub);
            [fMid, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
                evalCols(xmid, Xe, fr, problem, FE, maxFE, curve, ...
                         population_history, fitness_history, history_index, bsf, bsfx);

            if mod(iter, last_its) == 0
                oldRef = refFit;
                refFit = fMid;                                    % last_its_type = "ave"
            end

            % PPMF: success = offspring (penalised) better than the previous midpoint
            p_succ = sum(fitn_s < fMidOld) / lambda;
            sigma  = sigma * exp(d_param * (p_succ - p_target) / (1 - p_target));

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

            % Escape flat fitness; the PPMF release boosts by 20 where RB-IPOP uses 2
            if fitn_s(1) == fitn_s(min(1 + floor(lambda / 2), 2 + ceil(lambda / 4)))
                sigma = 20 * sigma * exp(0.2 + cs / ds);
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

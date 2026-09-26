% ----------------------------------------------------------------------- %
% Matrix Adaptation Evolution Strategy (MA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4 + floor(3*log(D))              % Offspring per generation
%   mu = floor(lambda/2)                      % Parents, weights log(lambda/2+0.5) - log(m)
%   c_s = (mu_eff+2)/(D+mu_eff+5)             % Cumulation of the search path s
%   c_1 = 2/((D+1.3)^2 + mu_eff)              % Rank-one rate of M
%   c_mu = min(1-c_1, 2*(mu_eff-2+1/mu_eff)/((D+2)^2+mu_eff))  % Rank-mu rate of M
%   d_s = 1 + c_s + 2*max(0, sqrt((mu_eff-1)/(D+1)) - 1)       % Step-size damping
%   sigma0 = 0.3, M0 = I                      % Initial step, in box widths per coordinate
%
% Algorithm Concept:
%   - Offspring y + sigma*M*z with z ~ N(0, I): the transformation matrix M
%     replaces CMA-ES's covariance, so no square root or eigendecomposition
%   - The mean moves by sigma times the weighted mean of the best mu M*z
%   - The path s accumulates the weighted mean z; sigma follows ||s|| (CSA)
%   - M <- M*(I + c_1/2*(s*s' - I) + c_mu/2*(<z*z'>_w - I)), the rank-one and
%     rank-mu updates written directly on the matrix square root
%
% Reference:
% Hans-Georg Beyer, Bernhard Sendhoff,
% Simplify Your Covariance Matrix Adaptation Evolution Strategy,
% IEEE Transactions on Evolutionary Computation 21 (5) (2017) 746-759.
% https://doi.org/10.1109/TEVC.2017.2680320
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Beyer's MAES.m (ForDistributionFastMAES.tar from homepages.fhv.at/
% hgb, via the Wayback Machine), cross-checked with PyPop7 MAES; its E||N(0,I)||
% keeps the -1/(21 D^2) sign. Its per-generation evaluation of the parent is
% statistics/termination only and is dropped; the budget is the only terminator.
% Steps are in box widths. Samples are clamped and z back-calculated with
% pinv(M, 1e-12) as in Hellwig and Beyer's epsMAg-ES, capped at the drawn |z|
% (uncapped it reached 4e11 on CEC2020RW; capped won 20 of 33 decided problems).
% M is rescaled to unit largest row norm, sigma absorbing the factor (exact);
% sigma stays in [eps, 0.5] box widths; a non-finite M resets to I. cond(M) is
% not capped, as in the reference: it reached 1e25 on converged RW runs, finite.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = maes(problem)

    D     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    span  = ub - lb;

    lambda = 4 + floor(3 * log(D));
    mu     = floor(lambda / 2);
    wi_raw = log(lambda / 2 + 0.5) - log(1:mu);
    wi     = wi_raw / sum(wi_raw);
    mu_eff = 1 / sum(wi .^ 2);
    c_s    = (mu_eff + 2) / (D + mu_eff + 5);
    c_1    = 2 / ((D + 1.3) ^ 2 + mu_eff);
    c_mu   = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2) ^ 2 + mu_eff));
    d_s    = 1 + c_s + 2 * max(0, sqrt((mu_eff - 1) / (D + 1)) - 1);
    sqrt_s = sqrt(c_s * (2 - c_s) * mu_eff);
    echi   = sqrt(D) * (1 - 1 / D / 4 - 1 / D / D / 21);   % sign as in MAES.m
    I      = eye(D);

    sig_lo = eps;             % in box widths, for the widest coordinate
    sig_hi = 0.5;             % half the box, Hansen's maxdx rule

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    y     = lb + rand(D, 1) .* span;
    sigma = 0.3;
    s     = zeros(D, 1);
    M     = I;

    bsf          = inf;
    bsf_solution = y';

    while FE < maxFE
        n = min(lambda, maxFE - FE);
        Z = randn(D, n);
        Dm = M * Z;
        [X, Dm, moved] = into_box(y + span .* (sigma * Dm), Dm, y, lb, ub, span);
        if any(moved)
            % Back-calculation of z for repaired offspring (epsMAg-ES), capped at the drawn |z|
            Zc = pinv(M, 1e-12) * Dm(:, moved);
            Z(:, moved) = Zc .* min(1, sqrt(sum(Z(:, moved) .^ 2, 1)) ./ sqrt(sum(Zc .^ 2, 1)));
        end

        [fit, FE] = calculate_fitness(X, problem, FE);
        fit = fit(:)';

        popRows = X';
        for k = 1:n
            if fit(k) < bsf
                bsf = fit(k);
                bsf_solution = popRows(k, :);
            end
            ec = FE - n + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                ec, popRows, fit', population_history, fitness_history, history_index, maxFE);
        end

        % A truncated final generation cannot drive the update
        if n < lambda
            break;
        end

        [~, ranks] = sort(fit);
        sel = ranks(1:mu);
        sum_z  = Z(:, sel) * wi';
        sum_d  = Dm(:, sel) * wi';
        sum_zz = (Z(:, sel) .* wi) * Z(:, sel)';

        y = y + span .* (sigma * sum_d);
        s = (1 - c_s) * s + sqrt_s * sum_z;
        M = M + 0.5 * M * (c_1 * (s * s' - I) + c_mu * (sum_zz - I));
        sigma = sigma * exp(c_s / d_s * (norm(s) / echi - 1));

        r = max(sqrt(sum(M .^ 2, 2)));
        if ~(isfinite(r) && r > 0) || any(~isfinite(s))
            M = I;
            s = zeros(D, 1);
            r = 1;
        end
        % Only sigma*M is sampled and the M update is right-multiplicative, so this rescale is exact
        M = M / r;
        sigma = min(max(sigma * r, sig_lo), sig_hi);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [X, step, moved] = into_box(Xraw, step, m, lb, ub, span)
% Clamp to the box; a clamped coordinate keeps the share t of its step that stays inside
    bad = ~isfinite(Xraw);
    if any(bad(:))
        draw = lb + rand(size(Xraw)) .* span;
        Xraw(bad) = draw(bad);
        step(bad) = 0;
    end
    X = min(max(Xraw, lb), ub);
    hit = X ~= Xraw;
    moved = any(hit | bad, 1);
    if any(hit(:))
        Mr = repmat(m, 1, size(Xraw, 2));
        t = (X(hit) - Mr(hit)) ./ (Xraw(hit) - Mr(hit));
        t(~(t >= 0)) = 0;                  % the mean sits an ulp outside, or 0/0
        step(hit) = step(hit) .* min(t, 1);
    end
end

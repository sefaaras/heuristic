% ----------------------------------------------------------------------- %
% Restart CMA-ES with population reduction (RCMAES)
% CEC 2026 bound-constrained single-objective track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda0 = D*(10*log10(maxFe/D) - 20)  % Initial offspring (2*D if log10(maxFe/D) <= 2), in [lambda_min, 2000]
%   lambda = lambda0 -> max(lambda_min, D)  % Scaled by 1-(1-t)^r, r = max(0.5, 1.7-0.01*D), t = FE/maxFe
%   lambda_min = 4 + ceil(3*ln(D))        % Floor of the offspring count
%   mu = lambda/2                         % Parents, positive log-rank weights
%   sigma0 = 0.2                          % Initial step in the [0,1]-normalised box; restarts alternate 0.1 / 0.2
%   restart = 1e-8                        % Threshold on a generation's (fmax - fmin)/|fmean|
%   exclusion = best +/- 0.1              % Box per restart, normalised units; restart samples avoid them all
%
% Algorithm Concept:
%   - Active CMA-ES in the box normalised to [0,1]^D: the best half moves the mean,
%     and every offspring enters the rank-mu update with a signed log-rank weight
%   - Negative weights are rescaled by D/||C^(-1/2) y||^2, so poor steps shrink the
%     covariance only along their own Mahalanobis direction
%   - The offspring count shrinks nonlinearly with the spent budget, faster in low D
%   - A generation whose relative fitness range collapses restarts the run: C, sigma
%     and the evolution paths are reset
%   - The restart generation is drawn uniformly outside boxes around every stored
%     best, coordinate by coordinate, and its mean becomes the new centre
%   - Out-of-box coordinates are redrawn next to the violated bound, within the violation
%
% Reference:
% Khoirul Faiq Muzakka, Soren Moller, Martin Finsterbusch,
% RCMAES: A Robust CMA-ES Variant for CEC2026 Competition,
% arXiv preprint arXiv:2604.27138 (2026).
% https://doi.org/10.48550/arXiv.2604.27138
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' Minion library as released with the paper (tags
% v1.2.0 to v1.6.1, minion/src/rcmaes.cpp, github.com/khoirulmuzakka/Minion).
% From v1.7.0 (July 2026) RCMAES was reworked there -- another population rule,
% a step-size restart trigger, a centred start -- and that version is not ported.
% Where the paper and this code differ the code is followed: sigma0 = 0.2 (paper
% 0.3), alternating with 0.1 over restarts, and the exclusion box is +/-10 % of
% each range around the best so far (paper: 10 % wide, around the converged mean).
% Numerical guard: a non-finite mean or step size is handled as a restart, and a
% non-finite covariance falls back to the identity, as the release's eigensolver
% failure path does.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rcmaes(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    lambda_min = max(4, 4 + ceil(3 * log(dim)));
    logeta = log10(maxFE / dim);
    if logeta > 2
        mult = 10 * logeta - 20;
    else
        mult = 2;
    end
    lambda0   = min(2000, max(4, floor(min(max(dim * mult, lambda_min), 2000))));
    mu        = max(ceil(0.5 * lambda0), 1);
    lam_floor = max(lambda_min, dim);
    pp        = max(0.5, 1.7 - 0.01 * dim);
    sigma0    = 0.2;
    rel_range_threshold = 1e-8;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = lb + rand(1, dim) .* span;

    % The release starts from a uniform point of the normalised box
    m = rand(dim, 1);
    best_u = m';
    best_f = inf;

    lam = lambda0;
    P = strategy_params(lam, mu, dim);
    sigma_eff = sigma0;
    sigma = sigma_eff;
    ps = zeros(dim, 1);
    pc = zeros(dim, 1);
    C = eye(dim);
    B = eye(dim);
    Dg = ones(dim, 1);
    Cis = eye(dim);
    iter = 0;

    use_restart_set = false;
    restart_set = zeros(0, dim);
    box_lo = zeros(0, dim);
    box_hi = zeros(0, dim);

    while FE < maxFE
        t = min(FE / maxFE, 1);
        lam_t = round(lambda0 - (lambda0 - lam_floor) * (1 - (1 - t) ^ pp));
        lam_t = max([lam_t, lambda_min, 4]);
        if lam_t ~= lam
            lam = lam_t;
            mu = min(max(round(0.5 * lam), 1), lam);
            P = strategy_params(lam, mu, dim);
        end

        if use_restart_set
            if size(restart_set, 1) < lam
                restart_set = [restart_set; restart_samples(lam - size(restart_set, 1), box_lo, box_hi)]; %#ok<AGROW>
            end
            U = restart_set(1:lam, :)';
            use_restart_set = false;
        else
            U = m + sigma * ((B .* Dg') * randn(dim, lam));
            U = reflect_random(U);
        end
        Y = (U - m) / sigma;

        n_eval = min(lam, maxFE - FE);
        Xr = lb + U(:, 1:n_eval)' .* span;
        [fe, FE, bsf, bsf_solution, curve] = evaluate_rows(Xr, problem, FE, bsf, bsf_solution, curve);
        fe(isnan(fe)) = inf;
        [population_history, fitness_history, history_index] = record_history( ...
            FE, Xr, fe, population_history, fitness_history, history_index, maxFE);
        f_off = inf(lam, 1);
        f_off(1:n_eval) = fe;

        [~, order] = sort(f_off);
        % The release's own best only centres the exclusion boxes; it skips a non-finite leader
        if isfinite(f_off(order(1))) && f_off(order(1)) < best_f
            best_f = f_off(order(1));
            best_u = U(:, order(1))';
        end

        Yr = Y(:, order);
        y_mean = Yr(:, 1:mu) * P.w(1:mu);
        m = m + sigma * y_mean;

        ps = (1 - P.cs) * ps + P.ps_fact * (Cis * y_mean);
        hsig = norm(ps) < (1.4 + 2 / (dim + 1)) * sqrt(1 - (1 - P.cs) ^ (2 * (iter + 1))) * P.chi;
        pc = (1 - P.cc) * pc + hsig * P.pc_fact * y_mean;

        % Negative weights rescaled by D/||C^(-1/2) y||^2 (the active update)
        w_var = P.w;
        neg = P.w < 0;
        if any(neg)
            den = sum((Cis * Yr(:, neg)) .^ 2, 1)';
            wv = P.w(neg) * dim ./ den;
            wv(~(den > 0)) = 0;
            w_var(neg) = wv;
        end
        h1 = (1 - hsig) * P.cc * (2 - P.cc);
        C = (1 + P.c1 * h1 - P.c1 - P.cmu * sum(P.w)) * C + P.c1 * (pc * pc') + ...
            P.cmu * ((Yr .* w_var') * Yr');
        C = (C + C') / 2;
        [B, Dg, Cis, C] = eigen_update(C);
        sigma = sigma * exp(P.cs / P.ds * (norm(ps) / P.chi - 1));

        fmean = mean(f_off);
        rel_range = 0;
        if abs(fmean) > 1e-12
            rel_range = (max(f_off) - min(f_off)) / abs(fmean);
        end
        broken = ~all(isfinite(m)) || ~isfinite(sigma) || sigma <= 0;
        if rel_range < rel_range_threshold || broken
            box_lo(end + 1, :) = max(0, best_u - 0.1); %#ok<AGROW>
            box_hi(end + 1, :) = min(1, best_u + 0.1); %#ok<AGROW>
            restart_set = restart_samples(lam, box_lo, box_hi);
            m = mean(restart_set, 1)';
            if sigma_eff == sigma0
                sigma_eff = 0.5 * sigma0;
            else
                sigma_eff = sigma0;
            end
            P = strategy_params(lam, mu, dim);
            sigma = sigma_eff;
            ps = zeros(dim, 1);
            pc = zeros(dim, 1);
            C = eye(dim);
            B = eye(dim);
            Dg = ones(dim, 1);
            Cis = eye(dim);
            iter = 0;
            use_restart_set = true;
        end
        iter = iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Weights and learning rates for lam offspring and mu parents (Parameter::resize in the release)
function P = strategy_params(lam, mu, dim)
    w = log((lam + 1) / 2) - log((1:lam)');
    w_pos_sum = sum(w(w >= 0));
    w_neg_sum = sum(w(w < 0));
    wp = w(1:mu);
    den = sum(wp .^ 2);
    if den <= 0
        den = 1e-12;
    end
    mueff = sum(wp) ^ 2 / den;

    P.cs  = (mueff + 2) / (dim + mueff + 5);
    P.cc  = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim);
    P.c1  = 2 / ((dim + 1.3) ^ 2 + mueff);
    P.cmu = min(1 - P.c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ^ 2 + mueff));
    P.ds  = 1 + P.cs + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1);
    P.chi = sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ^ 2));
    P.ps_fact = sqrt(P.cs * (2 - P.cs) * mueff);
    P.pc_fact = sqrt(P.cc * (2 - P.cc) * mueff);

    a_min = min([1 + P.c1 / max(1e-12, P.cmu), 1 + 2 * mueff, ...
                 (1 - P.c1 - P.cmu) / (dim * max(1e-12, P.cmu))]);
    pos = w >= 0;
    w(pos)  = w(pos) / max(1e-12, w_pos_sum);
    w(~pos) = a_min * w(~pos) / max(1e-12, abs(w_neg_sum));
    P.w = w;
end

% Eigendecomposition with the release's 1e-30 eigenvalue floor; C itself is kept
function [B, Dg, Cis, C] = eigen_update(C)
    dim = size(C, 1);
    if ~all(isfinite(C(:)))
        C = eye(dim);
        B = eye(dim);
        Dg = ones(dim, 1);
        Cis = eye(dim);
        return;
    end
    [B, L] = eig(C);
    L = max(real(diag(L)), 1e-30);
    B = real(B);
    Dg = sqrt(L);
    Cis = (B ./ Dg') * B';
end

% reflect-random in [0,1]: a violating coordinate is redrawn between the bound and bound +/- min(violation, 1)
function U = reflect_random(U)
    bad = ~isfinite(U);
    U(bad) = rand(nnz(bad), 1);
    idx = find(U < 0 | U > 1);
    for q = 1:numel(idx)
        v = U(idx(q));
        if v < 0
            U(idx(q)) = rand * min(-v, 1);
        else
            d = min(v - 1, 1);
            U(idx(q)) = (1 - d) + rand * d;
        end
    end
end

% Restart generation: every coordinate is drawn uniformly outside the union of the boxes' intervals
function S = restart_samples(nrows, box_lo, box_hi)
    dim = size(box_lo, 2);
    gaps = cell(1, dim);
    for d = 1:dim
        iv = sortrows([box_lo(:, d), box_hi(:, d)], 1);
        merged = iv(1, :);
        for r = 2:size(iv, 1)
            if iv(r, 1) <= merged(end, 2)
                merged(end, 2) = max(merged(end, 2), iv(r, 2));
            else
                merged(end + 1, :) = iv(r, :); %#ok<AGROW>
            end
        end
        valid = zeros(0, 2);
        prev = 0;
        for r = 1:size(merged, 1)
            if merged(r, 1) > prev
                valid(end + 1, :) = [prev, merged(r, 1)]; %#ok<AGROW>
            end
            prev = min(merged(r, 2), 1);
        end
        if prev < 1
            valid(end + 1, :) = [prev, 1]; %#ok<AGROW>
        end
        if isempty(valid)
            valid = [0, 1];
        end
        gaps{d} = valid;
    end

    S = zeros(nrows, dim);
    for j = 1:nrows
        for d = 1:dim
            g = gaps{d};
            len = g(:, 2) - g(:, 1);
            c = cumsum(len) / sum(len);
            r = find(rand <= c, 1);
            if isempty(r)
                r = size(g, 1);
            end
            S(j, d) = g(r, 1) + rand * len(r);
        end
    end
end

% Evaluates the rows of X (already cut to the budget) and tracks the best per evaluation
function [f, FE, bsf, bsf_solution, curve] = evaluate_rows(X, problem, FE, bsf, bsf_solution, curve)
    [fv, FE_new] = calculate_fitness(X', problem, FE);
    f = fv(:);
    for q = 1:numel(f)
        if f(q) < bsf
            bsf = f(q);
            bsf_solution = X(q, :);
        end
        curve(FE + q) = bsf;
    end
    FE = FE_new;
end

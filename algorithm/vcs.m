% ----------------------------------------------------------------------- %
% Virus Colony Search (VCS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N      = 50                  % Viruses = host cells (Section 4.2), 3N FE per generation
%   lambda = floor(N/2)          % Best viruses recombined into the CMA-ES mean, Eq. (4)
%   sigma0 = 0.3/200*(ub - lb)   % Initial CMA-ES step, the paper's 0.3 on [-100, 100]
%   tau    = log(g)/g*(V_i - G)  % Per-coordinate deviation of the diffusion walk, Eq. (1)
%   Pr     = (N - rank + 1)/N    % Chance a coordinate survives the immune response, Eq. (11)
%   c_sigma, c_c, d_sigma, c_1, c_lambda as Eqs. (5)-(10), with n = D
%
% Algorithm Concept:
%   - Viruses diffusion: V_i' = N(G, tau) + r1*G - r2*V_i, a Gaussian walk around
%     the best virus G plus a search direction, Eq. (1)
%   - Host cells infection: N host cells drawn from N(m, sigma^2*C); m, sigma and C
%     then follow CMA-ES on the lambda best VIRUSES, Eqs. (2)-(10)
%   - Immune response: each coordinate of a rank-r virus is rebuilt, with
%     probability 1 - Pr, as V_k - rand*(V_h - V_i), Eqs. (11)-(12)
%   - Each phase replaces virus i by candidate i only if it is better
%   - Out-of-box coordinates are redrawn uniformly inside the box (Section 3.3)
%
% Reference:
% Mu Dong Li, Hui Zhao, Xing Wei Weng, Tong Han,
% A novel nature-inspired algorithm for optimization: Virus colony search,
% Advances in Engineering Software 92 (2016) 65-88.
% https://doi.org/10.1016/j.advengsoft.2015.11.004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Sections 3.1-3.4, Eqs. (1)-(12), Algorithm 1); no code is public.
% Readings, tested on CEC2014 D = 30 (rows below): Eq. (4) is CMA-ES's normalised weighted mean,
% as printed (1/lambda, i-free weight) it scales m by 0.139 toward x = 0; N in Eqs. (5)-(10) is D
% as in [30]; "update Vpop" is one-to-one greedy; g counts from 1 (log(0)/0); r1, r2, k, h per
% virus, r and rand per coordinate. sigma0 = the paper's absolute 0.3 on [-100, 100] (CMA-ES runs
% box-normalised); as m tracks the viruses, not the host cells, sigma overshoots the box and host
% cells are uniform redraws in 5-98 % of generations (CEC2014 F1/F5/F10, D = 10-100). Unchanged
% immune copies are re-evaluated (3N FE per generation). Guards: symmetric C, eig >= 1e-14*max,
% max(eig(C)) folded into sigma (exact), non-finite state restarts CMA-ES. Centre pull KEPT: Eq. (1)'s
% r1*G - r2*V_i jitters a converged colony by (r1 - r2)*G, zero only at x = 0 (F23/F28 error 200).
%     CEC2014 D=30, 3e5 FE   F1       F2     F9     F11    F17     (5-run mean error)
%     paper, Tables 8-10     5.1e-10  0      133    2071   910
%     this port              0        0      183    2036   660
%     Eq. (4) as printed     7.5e5    1036   180    1643   8.6e4
%     sigma0 = 0.3 of box    0        0      43.2   2925   1434
%     N = 50 in (5)-(10)     8.36     0      171    2035   1736
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = vcs(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N      = 50;
    lam    = floor(N / 2);
    sigma0 = 0.3 / 200;   % box-normalised units: 0.3 on the paper's [-100, 100] box

    w     = log(lam + 1) - log(1:lam);   % Eq. (4), normalised as a weighted mean
    w     = w / sum(w);
    mueff = 1 / sum(w .^ 2);   % lambda_w
    cs    = (mueff + 2) / (D + mueff + 3);   % Eq. (5)
    cc    = 4 / (D + 4);   % Eq. (6)
    ds    = 1 + cs + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1);   % Eq. (7)
    c1    = ((1 - 1 / mueff) * min(1, (2 * mueff - 1) / ((D + 2)^2 + mueff)) ...
             + 2 / (mueff * (D + sqrt(2))^2)) / mueff;   % Eq. (9)
    clam  = (mueff - 1) * c1;   % Eq. (10)
    chiN  = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D^2));   % E||N(0, I)||
    hth   = (1.5 + 1 / (D - 0.5)) * chiN;   % h_sigma = 0 above this, as in [30]

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X  = lb + rand(N, D) .* span;
    nt = min(N, maxFE);
    X  = X(1:nt, :);   % only a budget below N trims the colony, and then no generation runs
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);
    for e = 1:nt
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    m        = mean((X - lb) ./ span, 1);   % Eq. (3)
    sigma    = sigma0;
    C        = eye(D);
    B        = eye(D);
    dd       = ones(1, D);   % square roots of the eigenvalues of C
    invsqrtC = eye(D);
    ps       = zeros(1, D);
    pc       = zeros(1, D);

    g = 0;
    while FE < maxFE
        g = g + 1;   % 1-based: log(g)/g at Algorithm 1's g = 0 is 0/0

        % Viruses diffusion
        [~, ib] = min(fit);
        G   = X(ib, :);
        tau = log(g) / g * (X - G);   % signed deviation; randn is symmetric
        Xc  = G + tau .* randn(N, D) + (rand(N, 1) .* G - rand(N, 1) .* X);   % Eq. (1)
        Xc  = redraw_outside(Xc, lb, ub);
        [X, fit, FE, best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            evaluate_and_replace(Xc, X, fit, FE, best_fitness, best_solution, curve, ...
                                 population_history, fitness_history, history_index, problem, maxFE);
        if FE >= maxFE, break; end

        % Host cells infection
        Yc = m + sigma * (randn(N, D) .* dd) * B';   % Eq. (2), N(0, C) = B*diag(dd)*z
        Xc = redraw_outside(lb + Yc .* span, lb, ub);
        [X, fit, FE, best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            evaluate_and_replace(Xc, X, fit, FE, best_fitness, best_solution, curve, ...
                                 population_history, fitness_history, history_index, problem, maxFE);
        if FE >= maxFE, break; end

        [~, order] = sort(fit);   % NaN sorts last
        Ysel = (X(order(1:lam), :) - lb) ./ span;   % lambda best of Vpop, Algorithm 1 line 16
        mnew = w * Ysel;   % Eq. (4)
        y    = (mnew - m) / sigma;
        ps   = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (y * invsqrtC);   % Eq. (5)
        hsig = norm(ps) / sqrt(1 - (1 - cs)^(2 * g)) < hth;
        pc   = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * y;   % Eq. (6)
        Yd   = (Ysel - m) / sigma;   % steps over sigma^g, Eq. (8)
        C    = (1 - c1 - clam) * C + c1 * (pc' * pc) + clam * (Yd' * (w' .* Yd));   % Eq. (8)
        sigma = sigma * exp(cs / ds * (norm(ps) / chiN - 1));   % Eq. (7)
        m    = mnew;

        C  = triu(C) + triu(C, 1)';
        ok = isfinite(sigma) && sigma > 0 && all(isfinite(C(:))) && all(isfinite(ps)) && all(isfinite(pc));
        if ok
            [B, E] = eig(C);
            ev   = diag(E)';
            emax = max(ev);
            ok   = emax > 0;
        end
        if ok
            % sigma*C^(1/2), C^(-1/2)/sigma and pc/sigma are invariant, so max(eig(C)) = 1 is exact
            ev    = max(ev, emax * 1e-14) / emax;
            sigma = sigma * sqrt(emax);
            pc    = pc / sqrt(emax);
            ok    = isfinite(sigma) && sigma > 0;
        end
        if ok
            dd       = sqrt(ev);
            C        = (B .* ev) * B';
            C        = triu(C) + triu(C, 1)';
            invsqrtC = (B ./ dd) * B';
        else
            % degenerate state: restart CMA-ES around the current mean
            sigma    = sigma0;
            C        = eye(D);
            B        = eye(D);
            dd       = ones(1, D);
            invsqrtC = eye(D);
            ps       = zeros(1, D);
            pc       = zeros(1, D);
        end

        % Immune response
        [~, order] = sort(fit);
        rk = zeros(1, N);
        rk(order) = 1:N;
        Pr = (N - rk + 1) / N;   % Eq. (11)
        Xc = X;
        for i = 1:N
            kh = randperm(N - 1, 2);
            kh(kh >= i) = kh(kh >= i) + 1;   % k ~= h ~= i
            mask = rand(1, D) > Pr(i);
            Xc(i, mask) = X(kh(1), mask) - rand(1, nnz(mask)) .* (X(kh(2), mask) - X(i, mask));   % Eq. (12)
        end
        Xc = redraw_outside(Xc, lb, ub);
        [X, fit, FE, best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            evaluate_and_replace(Xc, X, fit, FE, best_fitness, best_solution, curve, ...
                                 population_history, fitness_history, history_index, problem, maxFE);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Section 3.3 boundary rule; NaN/Inf coordinates fail both tests and are redrawn too
function Xc = redraw_outside(Xc, lb, ub)
    bad = ~(Xc >= lb & Xc <= ub);
    if any(bad(:))
        R = lb + rand(size(Xc)) .* (ub - lb);
        Xc(bad) = R(bad);
    end
end

% Evaluate (last batch truncated to the budget); candidate e replaces virus e when better
function [X, fit, FE, bestf, bestx, curve, ph, fh, hi] = evaluate_and_replace(Xc, X, fit, FE, bestf, bestx, ...
                                                                             curve, ph, fh, hi, problem, maxFE)
    nt = min(size(Xc, 1), maxFE - FE);
    if nt <= 0, return; end
    FE0 = FE;
    [fc, FE] = calculate_fitness(Xc(1:nt, :)', problem, FE);
    fc = fc(:)';
    for e = 1:nt
        if fc(e) < fit(e) || isnan(fit(e))
            X(e, :) = Xc(e, :);
            fit(e)  = fc(e);
        end
        if fc(e) < bestf || isnan(bestf)
            bestf = fc(e);
            bestx = Xc(e, :);
        end
        curve(FE0 + e) = bestf;
        [ph, fh, hi] = record_history(FE0 + e, X, fit, ph, fh, hi, maxFE);
    end
end

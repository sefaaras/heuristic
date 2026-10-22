% ----------------------------------------------------------------------- %
% Nuclear Reaction Optimization (NRO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N            = 100           % Population size (nuclei)
%   P_Fi, P_beta = rand, rand    % Fission and beta-decay probabilities, redrawn each generation
%   freq         = 0.05          % Frequency of the sinusoidal non-fusion step
%   alpha, beta  = 0.01, 1.5     % Levy flight scale and exponent
%   Gmax         = (maxFe - N)/(3*N)   % Generations: three evaluations per nucleus each
%
% Algorithm Concept:
%   - Fission (rand <= P_Fi): a Gaussian walk around x_best (if rand <= P_beta)
%     or x, plus randn*x_best - Pne*Ne, Ne the mean of x and a random nucleus
%   - Non-fission: a Gaussian walk around x; every walk has the step
%     log(g)/g*|x_r - x_best|, which shrinks with the generation g
%   - Ionization, likelier for worse nuclei: x_r1 +- rand*(x_r2 - x) per
%     dimension; the others x + round(rand)*rand*(x_worst - x_best)
%   - Fusion, likelier for worse ions: x + rand*(x_r1 - x_best) + rand*(x_r2 -
%     x_best) - exp(-|d|)*d, d = x_r1 - x_r2; the others a sinusoidal step on d
%   - A Levy flight replaces a move whose difference is zero; out-of-box
%     coordinates are re-drawn uniformly; greedy acceptance in all three stages
%
% Reference:
% Zhenglei Wei, Changqiang Huang, Xiaofei Wang, Tong Han, Yintong Li,
% Nuclear Reaction Optimization: A Novel and Powerful Physics-Based Algorithm
% for Global Optimization,
% IEEE Access 7 (2019) 66084-66109.
% https://doi.org/10.1109/ACCESS.2019.2918406
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper's Eqs. (1)-(23), Algorithm 1 and Fig. 7; no author
% code was found (File Exchange, GitHub; the IEEE supplement holds tables only).
% mealpy's port deviates (rand for randn in Eqs. 3/6, argsort indices as ranks)
% and was not followed. Where the paper is silent or contradicts itself: P_Fi,
% P_beta drawn per generation; randn in Eqs. (3)/(6) one scalar; j, r, r1, r2
% never i, r1 ~= r2; Euclidean norm in Eq. (15); in-place updates, x_best and
% x_worst refreshed between stages; Pa < rand ionizes, as in Fig. 7. Acceptance
% is <= in fission and ionization, strict in fusion, as worded; Eq. (23) also
% re-draws NaN. The comparison runs use P_Fi = P_beta = rand; the paper's
% parameter study later recommends 0.75 and 0.1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nro(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 100;
    freq = 0.05;
    alpha = 0.01;
    beta = 1.5;
    sigma_mu = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);   % Eq. (20)
    Gmax = max(1, ceil((maxFE - N) / (3 * N)));

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);   % Eq. (1)
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);

    for e = 1:min(N, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    g = 0;
    while FE < maxFE
        g = g + 1;
        PFi = rand;
        Pbeta = rand;
        sc = log(g) / g;   % Eqs. (4) and (7)

        % NFi phase
        [~, ib] = min(fit);
        Xbest = X(ib, :);
        for i = 1:N
            Ne = (X(i, :) + X(other(N, i), :)) / 2;   % Eq. (2)
            if rand <= PFi
                if rand <= Pbeta
                    sigma1 = sc * abs(X(i, :) - Xbest);
                    Pne = round(rand + 1);   % Eq. (5)
                    Xn = Xbest + sigma1 .* randn(1, dim) + (randn * Xbest - Pne * Ne);   % Eq. (3), secondary fission
                else
                    sigma2 = sc * abs(X(other(N, i), :) - Xbest);
                    Pne = round(rand + 2);   % Eq. (8)
                    Xn = X(i, :) + sigma2 .* randn(1, dim) + (randn * Xbest - Pne * Ne);   % Eq. (6), primary fission
                end
            else
                sigma2 = sc * abs(X(other(N, i), :) - Xbest);
                Xn = X(i, :) + sigma2 .* randn(1, dim);   % Eq. (9), even-even nucleus
            end
            Xn = boundary(Xn, lb, ub);
            [fn, FE] = calculate_fitness(Xn', problem, FE);
            if fn <= fit(i)
                X(i, :) = Xn;
                fit(i) = fn;
                if fn < best_fitness
                    best_fitness = fn;
                    best_solution = Xn;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % NFu phase, ionization stage
        [~, ib] = min(fit);
        Xbest = X(ib, :);
        [~, iw] = max(fit);
        Xworst = X(iw, :);
        Pa = rank_desc(fit) / N;   % Eq. (10)
        for i = 1:N
            if Pa(i) < rand
                r = two_others(N, i);
                sgn = 1 - 2 * (rand(1, dim) > 0.5);
                Xn = X(r(1), :) + sgn .* rand(1, dim) .* (X(r(2), :) - X(i, :));   % Eqs. (11)-(12)
                same = X(r(2), :) == X(i, :);
                L = levy(dim, beta, sigma_mu);
                Xn(same) = X(i, same) + alpha * L(same) .* (X(i, same) - Xbest(same));   % Eq. (18)
            else
                Xn = X(i, :) + round(rand(1, dim)) .* rand(1, dim) .* (Xworst - Xbest);   % Eq. (13)
                same = Xworst == Xbest;
                L = levy(dim, beta, sigma_mu);
                Xn(same) = X(i, same) + alpha * L(same) .* (ub(same) - lb(same));   % Eq. (21)
            end
            Xn = boundary(Xn, lb, ub);
            [fn, FE] = calculate_fitness(Xn', problem, FE);
            if fn <= fit(i)
                X(i, :) = Xn;
                fit(i) = fn;
                if fn < best_fitness
                    best_fitness = fn;
                    best_solution = Xn;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % NFu phase, fusion stage
        [~, ib] = min(fit);
        Xbest = X(ib, :);
        Pc = rank_desc(fit) / N;   % Eq. (14)
        for i = 1:N
            r = two_others(N, i);
            d = X(r(1), :) - X(r(2), :);
            if Pc(i) < rand
                Xn = X(i, :) + rand * (X(r(1), :) - Xbest) + rand * (X(r(2), :) - Xbest) - exp(-norm(d)) * d;   % Eq. (15)
            elseif all(d == 0)
                Xn = X(i, :) + alpha * levy(dim, beta, sigma_mu) .* (X(i, :) - Xbest);   % Eq. (22)
            elseif rand > 0.5
                Xn = X(i, :) - 0.5 * (sin(2 * pi * freq * g + pi) * (Gmax - g) / Gmax + 1) * d;   % Eq. (16)
            else
                Xn = X(i, :) - 0.5 * (sin(2 * pi * freq * g + pi) * g / Gmax + 1) * d;   % Eq. (17)
            end
            Xn = boundary(Xn, lb, ub);
            [fn, FE] = calculate_fitness(Xn', problem, FE);
            if fn < fit(i)
                X(i, :) = Xn;
                fit(i) = fn;
                if fn < best_fitness
                    best_fitness = fn;
                    best_solution = Xn;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% A random index in 1..N other than i
function j = other(N, i)
    j = randi(N - 1);
    if j >= i
        j = j + 1;
    end
end

% Two distinct random indices in 1..N, both other than i
function r = two_others(N, i)
    r = randperm(N - 1, 2);
    r(r >= i) = r(r >= i) + 1;
end

% Rank 1 for the largest fitness, N for the smallest, as Eqs. (10) and (14) order them
function rk = rank_desc(fit)
    [~, order] = sort(fit, 'descend');
    rk = zeros(size(fit));
    rk(order) = 1:numel(fit);
end

% Mantegna Levy step, Eqs. (19)-(20)
function L = levy(dim, beta, sigma_mu)
    L = randn(1, dim) * sigma_mu ./ abs(randn(1, dim)).^(1 / beta);
end

% Eq. (23): an out-of-box (or NaN) coordinate is re-drawn uniformly in the box
function x = boundary(x, lb, ub)
    bad = ~(x >= lb & x <= ub);
    if any(bad)
        x(bad) = lb(bad) + rand(1, nnz(bad)) .* (ub(bad) - lb(bad));
    end
end

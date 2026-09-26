% ----------------------------------------------------------------------- %
% Red-billed Blue Magpie Optimizer (RBMO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                      % Population size
%   Epsilon = 0.5               % Chance of the small-group mean over the cluster mean
%   p = randi([2, 5])           % Size of a small group, redrawn per member
%   q = randi([10, N])          % Size of a cluster, redrawn per member
%   CF = (1-t)^(2*t)            % Step scale of the attack, t the spent budget
%
% Algorithm Concept:
%   - Search for food: each member adds rand times the gap between the mean of a
%     random small group (or cluster) and a random member (Eq. 3-4)
%   - Attack: each member restarts from the food position Xfood and steps by CF
%     times its gap to a group or cluster mean, scaled by N(0,1) per coordinate (Eq. 5-6)
%   - Both phases update members in place, so later members see earlier moves
%   - Food storage after each phase keeps a member's previous position if it was
%     better (Eq. 7); each generation evaluates the population twice
%   - Coordinates are clamped to the box
%
% Reference:
% Shengwei Fu, Ke Li, Haisong Huang, Chi Ma, Qingsong Fan, Yunwei Zhu,
% Red-billed blue magpie optimizer: a novel metaheuristic algorithm for 2D/3D
% UAV path planning and engineering design problems,
% Artificial Intelligence Review 57 (2024) 134.
% https://doi.org/10.1007/s10462-024-10716-3
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/ShengweiFu/RBMO, RBMO.zip); its
% RBMO_CEC2014 and RBMO_CEC2017 folders carry the same RBMO.m, which is ported.
% CF runs on the spent budget FE/maxFe instead of t/T, and FE >= maxFe is the
% only terminator. Kept as released: Xfood is the best position evaluated
% inside the loop, the initial sweep never enters it. Food storage selects by
% index where the release blends with 0/1 weights, which is the same on finite
% fitness but turns an Inf into NaN.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rbmo(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    N = 30;
    Epsilon = 0.5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    fit = inf(N, 1);
    bsf = inf;
    bsf_solution = X(1, :);

    n = min(N, maxFE);
    [fv, FE] = calculate_fitness(X(1:n, :)', problem, FE);
    fit(1:n) = fv(:);
    for k = 1:n
        if fit(k) < bsf
            bsf = fit(k);
            bsf_solution = X(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, X(1:n, :), fit(1:n), population_history, fitness_history, history_index, maxFE);
    end

    food_val = inf;
    Xfood = zeros(1, dim);

    while FE < maxFE
        for phase = 1:2
            if FE >= maxFE
                break;
            end
            X_old = X;
            fit_old = fit;
            if phase == 2
                t = FE / maxFE;
                CF = (1 - t) ^ (2 * t);
            end
            for i = 1:N
                p = randi([2, 5]);
                Xpmean = mean(X(randperm(N, p), :), 1);
                q = randi([10, N]);
                Xqmean = mean(X(randperm(N, q), :), 1);
                if phase == 1
                    R1 = randi(N);
                    if rand < Epsilon
                        X(i, :) = X(i, :) + (Xpmean - X(R1, :)) .* rand;              % Eq. (3)
                    else
                        X(i, :) = X(i, :) + (Xqmean - X(R1, :)) .* rand;              % Eq. (4)
                    end
                else
                    if rand < Epsilon
                        X(i, :) = Xfood + CF * (Xpmean - X(i, :)) .* randn(1, dim);   % Eq. (5)
                    else
                        X(i, :) = Xfood + CF * (Xqmean - X(i, :)) .* randn(1, dim);   % Eq. (6)
                    end
                end
            end

            bad = ~isfinite(X);
            if any(bad(:))
                R = lb + rand(N, dim) .* (ub - lb);
                X(bad) = R(bad);
            end
            X = min(max(X, lb), ub);

            n = min(N, maxFE - FE);
            [fv, FE] = calculate_fitness(X(1:n, :)', problem, FE);
            fit(1:n) = fv(:);
            for k = 1:n
                if fit(k) < food_val
                    food_val = fit(k);
                    Xfood = X(k, :);
                end
                if fit(k) < bsf
                    bsf = fit(k);
                    bsf_solution = X(k, :);
                end
                curve(FE - n + k) = bsf;
            end

            % Eq. (7); rows the budget left unevaluated go back to their stored member
            keep_old = [fit_old(1:n) < fit(1:n); true(N - n, 1)];
            X(keep_old, :) = X_old(keep_old, :);
            fit(keep_old) = fit_old(keep_old);

            for k = 1:n
                [population_history, fitness_history, history_index] = record_history(...
                    FE - n + k, X, fit, population_history, fitness_history, history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness = bsf;
    best_solution = bsf_solution;
end

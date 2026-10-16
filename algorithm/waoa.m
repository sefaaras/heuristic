% ----------------------------------------------------------------------- %
% Walrus Optimization Algorithm (WaOA)
% Not wo (Walrus Optimizer, Han et al. 2024): a different method
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30   % Population size (walruses; the paper's runs used 20)
%
% Algorithm Concept:
%   - Phase 1 (feeding, exploration): move under the strongest walrus,
%     x + r*(SW - I*x) with r per dimension and I in {1, 2}
%   - Phase 2 (migration): toward a random other walrus when it is better,
%     away from it otherwise
%   - Phase 3 (escaping predators, exploitation): a random step inside the
%     local box [lb/t, ub/t] that shrinks with the iteration count t
%   - Greedy acceptance after each phase, one walrus at a time
%
% Reference:
% Pavel Trojovsky, Mohammad Dehghani,
% A new bio-inspired metaheuristic algorithm for solving optimization
% problems based on walruses behavior,
% Scientific Reports 13 (2023) 8775.
% https://doi.org/10.1038/s41598-023-35863-5
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 133272 v1.0.0 (WOA.m,
% 5 Aug 2023), not from the paper; Phases 2 and 3 draw one r per move.
% Phase 3 follows the release, x + lb/t + r*(ub - lb)/t, not the printed
% Eq. (7), x + lb/t + (ub/t - r*lb/t), which on a symmetric box only steps up.
% Reference defect kept: the Phase 3 candidate is clamped to the local box
% [lb/t, ub/t] before the global one, so for lb < 0 < ub it lands in a box
% around the origin, not around the walrus as Eq. (7) intends.
% The release draws an unused I in Phase 3; the draw is kept for RNG parity.
% The release refreshes its best only when an iteration starts; the guide
% keeps that, while the reported best is updated at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = waoa(problem)

    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = initialization(SearchAgents, dim, upperbound, lowerbound);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);

    for e = 1:min(SearchAgents, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    t = 0;
    while FE < maxFE
        t = t + 1;

        % The strongest walrus SW is refreshed once per iteration, as in the release
        [best, location] = min(fit);
        if t == 1 || best < fbest
            fbest = best;
            Xbest = X(location, :);
        end
        SW = Xbest;

        for i = 1:SearchAgents
            if FE >= maxFE, break; end

            % Phase 1: feeding strategy
            I = round(1 + rand(1, 1));
            Xnew = X(i, :) + rand(1, dim) .* (SW - I .* X(i, :));   % Eq. (3)
            Xnew = max(Xnew, lowerbound);
            Xnew = min(Xnew, upperbound);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < fit(i)   % Eq. (4)
                X(i, :) = Xnew;
                fit(i) = fnew;
                if fnew < best_fitness
                    best_fitness = fnew;
                    best_solution = Xnew;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            % Phase 2: migration to a random walrus other than i
            I = round(1 + rand(1, 1));
            K = randperm(SearchAgents);
            K(K == i) = [];
            X_K = X(K(1), :);
            F_RAND = fit(K(1));
            if fit(i) > F_RAND   % Eq. (5)
                Xnew = X(i, :) + rand(1, 1) .* (X_K - I .* X(i, :));
            else
                Xnew = X(i, :) + rand(1, 1) .* (X(i, :) - X_K);
            end
            Xnew = max(Xnew, lowerbound);
            Xnew = min(Xnew, upperbound);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < fit(i)   % Eq. (6)
                X(i, :) = Xnew;
                fit(i) = fnew;
                if fnew < best_fitness
                    best_fitness = fnew;
                    best_solution = Xnew;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            % Phase 3: escaping and fighting against predators, local bounds of Eq. (8)
            LO_LOCAL = lowerbound ./ t;
            HI_LOCAL = upperbound ./ t;
            rand(1, 1);   % the release's unused I, drawn so a seeded run matches it
            Xnew = X(i, :) + LO_LOCAL + rand(1, 1) .* (HI_LOCAL - LO_LOCAL);   % Eq. (7) as released
            Xnew = max(Xnew, LO_LOCAL);
            Xnew = min(Xnew, HI_LOCAL);
            Xnew = max(Xnew, lowerbound);
            Xnew = min(Xnew, upperbound);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < fit(i)   % Eq. (9)
                X(i, :) = Xnew;
                fit(i) = fnew;
                if fnew < best_fitness
                    best_fitness = fnew;
                    best_solution = Xnew;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Initialization
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

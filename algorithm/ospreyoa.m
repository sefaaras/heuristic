% ----------------------------------------------------------------------- %
% Osprey Optimization Algorithm (OOA)
% Stored as ospreyoa; the acronym OOA collides with ooa (Octopus)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30   % Population size (ospreys)
%
% Algorithm Concept:
%   - Phase 1 (exploration): each osprey attacks a fish taken from the better
%     ospreys or the best one, x + r*(SF - I*x) with I in {1, 2}
%   - Phase 2 (exploitation): carries the fish by a random point of the box
%     scaled by 1/t, x + (lb + r*(ub - lb))/t
%   - Greedy acceptance after each phase, one osprey at a time
%
% Reference:
% Mohammad Dehghani, Pavel Trojovsky,
% Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm
% for solving engineering optimization problems,
% Frontiers in Mechanical Engineering 8 (2023) 1126450.
% https://doi.org/10.3389/fmech.2022.1126450
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 124555 v1.0.0 (OOA.m,
% 10 Feb 2023), not from the paper. The release takes the best osprey with
% probability 0.5 and otherwise a uniform better one (the paper draws
% uniformly from FP_i), and uses one r and one I per move, not per dimension.
% Reference defect kept: the better osprey is read as X(k) with a linear
% index, i.e. only its FIRST coordinate, broadcast to every dimension, so
% those moves head for the diagonal point (s, ..., s) instead of a fish.
% The release refreshes its best only when an iteration starts; the guide
% keeps that, while the reported best is updated at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ospreyoa(problem)

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

        % The guiding best is refreshed once per iteration, as in the release
        [Fbest, blocation] = min(fit);
        if t == 1 || Fbest < fbest
            fbest = Fbest;
            xbest = X(blocation, :);
        end

        for i = 1:SearchAgents
            if FE >= maxFE, break; end

            % Phase 1: the fish set FP_i of Eq. (4) is every better osprey plus the best
            fish_position = find(fit < fit(i));
            if isempty(fish_position) || rand < 0.5
                selected_fish = xbest;
            else
                k = randperm(numel(fish_position), 1);
                selected_fish = X(fish_position(k), 1);   % release reads X(k) linearly: first coordinate only
            end
            I = round(1 + rand);
            Xnew = X(i, :) + rand(1, 1) .* (selected_fish - I .* X(i, :));   % Eq. (5)
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

            % Phase 2: the offset is a random box point divided by the iteration count
            Xnew = X(i, :) + (lowerbound + rand * (upperbound - lowerbound)) / t;   % Eq. (7)
            Xnew = max(Xnew, lowerbound);
            Xnew = min(Xnew, upperbound);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < fit(i)   % Eq. (8)
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

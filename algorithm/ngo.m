% ----------------------------------------------------------------------- %
% Northern Goshawk Optimization (NGO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Search_Agents = 30                % Population size (goshawks)
%   R             = 0.02*(1 - t/T)    % Chase radius relative to |x|, Eq. (6), 0.02 -> 0
%
% Algorithm Concept:
%   - Phase 1 (prey identification, exploration): a random prey P; toward it,
%     x + r.*(P - I*x) with I in {1, 2}, when it is better, else x + r.*(x - P)
%   - Phase 2 (chase and escape, exploitation): x + (2r - 1)*R.*x, a
%     neighbourhood proportional to x that shrinks to zero at the end of the run
%   - Greedy acceptance after each phase, one goshawk at a time
%
% Reference:
% Mohammad Dehghani, Stepan Hubalovsky, Pavel Trojovsky,
% Northern Goshawk Optimization: A New Swarm-Based Algorithm for Solving
% Optimization Problems,
% IEEE Access 9 (2021) 162059-162080.
% https://doi.org/10.1109/ACCESS.2021.3133286
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 106665 v1.0.0 (NGO.m,
% 12 Feb 2022). T is counted from the budget, ceil((maxFe - N)/(2N)) at two
% evaluations per goshawk per iteration, so R reaches 0 at maxFe. Kept as
% released: the prey index k = randperm(N, 1) may be the goshawk itself, and
% at R = 0 the last iteration's chase cannot move it; both candidates are
% still evaluated. The release refreshes its best when an iteration starts
% (it never guides); here the reported best is updated at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ngo(problem)

    dimensions = problem.dimension;
    Lowerbound = problem.lb;
    Upperbound = problem.ub;
    maxFE = problem.maxFe;

    Search_Agents = 30;
    Max_iterations = max(1, ceil((maxFE - Search_Agents) / (2 * Search_Agents)));   % two evaluations per goshawk

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % One column per dimension, as released
    X = zeros(Search_Agents, dimensions);
    for i = 1:dimensions
        X(:, i) = Lowerbound(i) + rand(Search_Agents, 1) .* (Upperbound(i) - Lowerbound(i));
    end
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);
    for e = 1:min(Search_Agents, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    t = 0;
    while FE < maxFE
        t = t + 1;

        for i = 1:Search_Agents
            % Phase 1: prey identification and attack
            I = round(1 + rand);
            k = randperm(Search_Agents, 1);
            P = X(k, :);   % Eq. (3)
            F_P = fit(k);
            if fit(i) > F_P
                X_new = X(i, :) + rand(1, dimensions) .* (P - I .* X(i, :));   % Eq. (4)
            else
                X_new = X(i, :) + rand(1, dimensions) .* (X(i, :) - P);   % Eq. (4)
            end
            X_new = max(X_new, Lowerbound);
            X_new = min(X_new, Upperbound);

            [fit_new, FE] = calculate_fitness(X_new', problem, FE);
            if fit_new < fit(i)   % Eq. (5)
                X(i, :) = X_new;
                fit(i) = fit_new;
                if fit_new < best_fitness
                    best_fitness = fit_new;
                    best_solution = X_new;
                end
            end
            curve(FE) = best_fitness;   % one evaluation per step, so FE <= maxFE here
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            % Phase 2: chase and escape
            R = 0.02 * (1 - t / Max_iterations);   % Eq. (6)
            X_new = X(i, :) + (-R + 2 * R * rand(1, dimensions)) .* X(i, :);   % Eq. (7)
            X_new = max(X_new, Lowerbound);
            X_new = min(X_new, Upperbound);

            [fit_new, FE] = calculate_fitness(X_new', problem, FE);
            if fit_new < fit(i)   % Eq. (8)
                X(i, :) = X_new;
                fit(i) = fit_new;
                if fit_new < best_fitness
                    best_fitness = fit_new;
                    best_solution = X_new;
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

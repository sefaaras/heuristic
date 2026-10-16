% ----------------------------------------------------------------------- %
% Tasmanian Devil Optimization (TDO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30            % Population size (Tasmanian devils)
%   P_carrion    = 0.5           % Per devil: strategy 1 if rand > 0.5, else strategy 2
%   R            = 0.01 -> 0     % Prey-chasing radius relative to |x|, linear, Eq. (9)
%
% Algorithm Concept:
%   - Strategy 1 (carrion, exploration): toward a random other devil when it
%     is better, x + r.*(C - I*x) with I in {1, 2}, away from it otherwise
%   - Strategy 2 (prey, exploitation): the same move toward a random prey,
%     then a chase x + (2r - 1).*R.*x in a neighbourhood shrinking to zero
%   - Greedy acceptance after every move, one devil at a time
%
% Reference:
% Mohammad Dehghani, Stepan Hubalovsky, Pavel Trojovsky,
% Tasmanian Devil Optimization: A New Bio-Inspired Optimization Algorithm
% for Solving Optimization Algorithm,
% IEEE Access 10 (2022) 19599-19620.
% https://doi.org/10.1109/ACCESS.2022.3151641
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 111380 v1.0.0 (TDO.m,
% 8 May 2022), not from the paper. The release accepts ties (<=) where
% Eqs. (5), (8), (11) ask for strict improvement, and draws r per dimension.
% Reference defect kept: a draw k == i becomes k = i + 1 (wrapping), so in
% Eqs. (3), (6) devil i + 1 is picked twice as often as any other.
% The release's best never guides the search and is refreshed only when an
% iteration starts; the reported best is updated at every evaluation.
% T is set from the budget at the expected 1.5*N evaluations per iteration,
% with t/T clamped at 1; at R = 0 a chase cannot move the devil, so that
% evaluation (spent by the release in its last iteration) is skipped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tdo(problem)

    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents = 30;
    % Every devil spends one evaluation, the half that hunts prey a second one
    Max_iterations = max(1, ceil((maxFE - SearchAgents) / (1.5 * SearchAgents)));

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lowerbound + rand(SearchAgents, dim) .* (upperbound - lowerbound);
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
        R = 0.01 * (1 - min(t, Max_iterations) / Max_iterations);   % Eq. (9)

        for i = 1:SearchAgents
            if FE >= maxFE, break; end

            Pr = rand;
            % Eqs. (3), (6): carrion and prey are both a random devil other than i
            k = randperm(SearchAgents, 1);
            if k == i
                k = i + 1;
                if k > SearchAgents
                    k = 1;
                end
            end
            I = round(1 + rand(1, 1));
            if fit(i) > fit(k)   % Eqs. (4), (7)
                X_new = X(i, :) + rand(1, dim) .* (X(k, :) - I .* X(i, :));
            else
                X_new = X(i, :) + rand(1, dim) .* (X(i, :) - X(k, :));
            end
            X_new = max(X_new, lowerbound);
            X_new = min(X_new, upperbound);
            [f_new, FE] = calculate_fitness(X_new', problem, FE);
            if f_new <= fit(i)   % Eqs. (5), (8)
                X(i, :) = X_new;
                fit(i) = f_new;
                if f_new < best_fitness
                    best_fitness = f_new;
                    best_solution = X_new;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            % Strategy 2, stage 2: chasing the prey around the devil's new position
            if Pr <= 0.5
                X_new = X(i, :) + (-R + 2 * R * rand(1, dim)) .* X(i, :);   % Eq. (10)
                % At R = 0 the chase returns the devil itself, so it is not scored
                if R > 0
                    X_new = max(X_new, lowerbound);
                    X_new = min(X_new, upperbound);
                    [f_new, FE] = calculate_fitness(X_new', problem, FE);
                    if f_new <= fit(i)   % Eq. (11)
                        X(i, :) = X_new;
                        fit(i) = f_new;
                        if f_new < best_fitness
                            best_fitness = f_new;
                            best_solution = X_new;
                        end
                    end
                    curve(FE) = best_fitness;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% ----------------------------------------------------------------------- %
% Ali Baba and the Forty Thieves (AFT) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   noThieves = 30   % Population size (thieves)
%
% Algorithm Concept:
%   - Thieves search for Ali Baba's position guided by Marjaneh's plans
%     (per-thief best) and the global best
%   - Perception potential Pp (probability of an intelligent search) and a
%     tracking distance Td that shrinks over iterations drive the moves
%
% Reference:
% Malik Braik, Mohammad Hashem Ryalat, Hussein Al-Zoubi,
% A novel meta-heuristic algorithm for solving numerical optimization
% problems: Ali Baba and the forty thieves,
% Neural Computing and Applications 34 (1) (2022) 409-455.
% https://doi.org/10.1007/s00521-021-06392-x
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = aft(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    noThieves = 30;
    itemax = ceil(maxFE / noThieves);

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, noThieves, dim);
    fitness_history = zeros(history_size, noThieves);
    history_index = 1;

    % Position of the thieves in the space
    xth = zeros(noThieves, dim);
    for i = 1:noThieves
        for j = 1:dim
            xth(i, j) = lb(j) - rand() * (lb(j) - ub(j));
        end
    end

    % Evaluate the initial population
    [fit, FE] = calculate_fitness(xth', problem, FE);
    fit = fit(:);

    fitness = fit;
    [sorted_thieves_fitness, sorted_indexes] = sort(fit);
    Sorted_thieves = zeros(noThieves, dim);
    for index = 1:noThieves
        Sorted_thieves(index, :) = xth(sorted_indexes(index), :);
    end

    gbest = Sorted_thieves(1, :);
    fit0 = sorted_thieves_fitness(1);

    best = xth;   % per-thief best (Marjaneh's plans)
    xab = xth;    % position of Ali Baba

    for eval_count = 1:noThieves
        if eval_count <= maxFE
            curve(eval_count) = fit0;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, xth, fit', population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    %% Start running AFT
    for ite = 1:itemax
        if FE >= maxFE, break; end

        Pp = 0.1 * log(2.75 * (ite / itemax)^0.1);   % Perception potential
        Td = 2 * exp(-2 * (ite / itemax)^2);          % Tracking distance

        a = ceil((noThieves - 1) .* rand(noThieves, 1))';

        % Generation of new positions for the thieves
        for i = 1:noThieves
            if (rand >= 0.5)
                if rand > Pp
                    xth(i, :) = gbest + (Td * (best(i, :) - xab(i, :)) * rand + Td * (xab(i, :) - best(a(i), :)) * rand) * sign(rand - 0.50);
                else
                    for j = 1:dim
                        xth(i, j) = Td * ((ub(j) - lb(j)) * rand + lb(j));
                    end
                end
            else
                for j = 1:dim
                    xth(i, j) = gbest(j) - (Td * (best(i, j) - xab(i, j)) * rand + Td * (xab(i, j) - best(a(i), j)) * rand) * sign(rand - 0.50);
                end
            end
        end

        % Update the global, best position of the thieves and Ali Baba
        for i = 1:noThieves
            [fit(i, 1), FE] = calculate_fitness(xth(i, :)', problem, FE);

            % Handling the boundary conditions (reject out-of-bounds thieves)
            if and(~(xth(i, :) - lb <= 0), ~(xth(i, :) - ub >= 0))
                xab(i, :) = xth(i, :);

                if fit(i) < fitness(i)
                    best(i, :) = xth(i, :);
                    fitness(i) = fit(i);
                end

                if fitness(i) < fit0
                    fit0 = fitness(i);
                    gbest = best(i, :);
                end
            end

            if FE <= maxFE
                curve(FE) = fit0;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xth, fit', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = fit0;

    bestThieves = find(fitness == min(fitness));
    best_solution = best(bestThieves(1), :);
    best_fitness = min(fitness);
end

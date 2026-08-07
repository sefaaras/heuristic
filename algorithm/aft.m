% ----------------------------------------------------------------------- %
% Ali Baba and the Forty Thieves (AFT)
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
% Implementation Note:
% Thieves are clamped to the box before evaluation. The reference evaluates
% first and only then rejects an out-of-bounds thief, so the illegal point is
% still paid for out of the FE budget; the rejection itself survives, because a
% clamped coordinate sits exactly on the bound and the acceptance test requires
% every coordinate to be strictly inside. curve and best_fitness therefore track
% the best point evaluated separately from fit0, which stays the thieves'
% attractor and keeps that rejection.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
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
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Position of the thieves in the space
    xth = zeros(noThieves, dim);
    for i = 1:noThieves
        for j = 1:dim
            xth(i, j) = lb(j) - rand() * (lb(j) - ub(j));
        end
    end

    % Evaluate the initial population
    xth = min(max(xth, lb), ub);
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

    % Best point evaluated so far; a clamped thief can never enter fit0
    bsf_fit = fit0;
    bsf_x = gbest;

    best = xth;   % per-thief best (Marjaneh's plans)
    xab = xth;    % position of Ali Baba

    for eval_count = 1:noThieves
        if eval_count <= maxFE
            curve(eval_count) = bsf_fit;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, xth, fit', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Start running AFT
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
        xth = min(max(xth, lb), ub);
        for i = 1:noThieves
            [fit(i, 1), FE] = calculate_fitness(xth(i, :)', problem, FE);
            if fit(i, 1) < bsf_fit
                bsf_fit = fit(i, 1);
                bsf_x = xth(i, :);
            end

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
                curve(FE) = bsf_fit;
            end
            % xth is moved for the whole swarm before this loop, so fit only
            % describes it again once the last thief has been re-evaluated
            if i == noThieves && FE <= maxFE
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xth, fit', population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = bsf_fit;

    best_solution = bsf_x;
    best_fitness = bsf_fit;
end

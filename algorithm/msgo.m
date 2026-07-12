% ----------------------------------------------------------------------- %
% Modified Social Group Optimization (MSGO) for unconstrained benchmark
% problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pn = 30               % Population size (group members)
%   c = 0.2               % Self-introspection factor (improving phase)
%   SAP = 0.7             % Self-adaptive probability (acquiring phase)
%
% Algorithm Concept:
%   - Models human social behaviour of solving problems in a group
%   - Improving phase: every member improves by learning from the best
%     member (the "guru")
%   - Acquiring phase: members interact with a random peer and the current
%     group best; with probability (1-SAP) a member is reinitialised
%
% Reference:
% Suresh Chandra Satapathy, Anima Naik,
% Social group optimization (SGO): a new population evolutionary
% optimization technique,
% Complex & Intelligent Systems 2 (2016) 173-203
% https://doi.org/10.1007/s40747-016-0022-8
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = msgo(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    pn = 30;        % Population size
    c = 0.2;        % Self-introspection factor
    SAP = 0.7;      % Self-adaptive probability

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, pn, dim);
    fitness_history = zeros(history_size, pn);
    history_index = 1;

    % Initialize population
    pop = zeros(pn, dim);
    for i = 1:pn
        pop(i, :) = rand(1, dim) .* (ub - lb) + lb;
    end

    % Evaluate initial population
    [f, FE] = calculate_fitness(pop', problem, FE);
    f = f(:);   % pn x 1

    best_fitness = min(f);

    for eval_count = 1:pn
        curve(eval_count) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, pop, f, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    while FE < maxFE

        % ---------------- Improving phase (guru) ----------------
        [~, ibest] = min(f);
        guru = pop(ibest, :);

        newpop = zeros(pn, dim);
        for i = 1:pn
            newpop(i, :) = c * pop(i, :) + rand(1, dim) .* (guru - pop(i, :));
            newpop(i, :) = min(max(newpop(i, :), lb), ub);
        end

        FE_before = FE;
        [f1, FE] = calculate_fitness(newpop', problem, FE);
        f1 = f1(:);
        for i = 1:pn
            if f1(i) < f(i)
                pop(i, :) = newpop(i, :);
                f(i) = f1(i);
            end
        end
        best_fitness = min(f);
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, f, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE
            break;
        end

        % ---------------- Acquiring phase ----------------
        [~, b] = min(f);
        gpop = pop(b, :);

        popnew = zeros(pn, dim);
        for i = 1:pn
            r1 = floor(1 + rand * pn);
            while (r1 == i)
                r1 = floor(1 + rand * pn);
            end
            if (f(i) < f(r1))
                if rand > SAP
                    popnew(i, :) = pop(i, :) + rand(1, dim) .* (pop(i, :) - pop(r1, :)) + rand(1, dim) .* (gpop - pop(i, :));
                    popnew(i, :) = min(max(popnew(i, :), lb), ub);
                else
                    popnew(i, :) = lb + rand * (ub - lb);
                end
            else
                popnew(i, :) = pop(i, :) + rand(1, dim) .* (pop(r1, :) - pop(i, :)) + rand(1, dim) .* (gpop - pop(i, :));
                popnew(i, :) = min(max(popnew(i, :), lb), ub);
            end
        end

        FE_before = FE;
        [f2, FE] = calculate_fitness(popnew', problem, FE);
        f2 = f2(:);
        for i = 1:pn
            if f2(i) < f(i)
                pop(i, :) = popnew(i, :);
                f(i) = f2(i);
            end
        end
        best_fitness = min(f);
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, f, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    [best_fitness, position] = min(f);
    best_solution = pop(position, :);

end

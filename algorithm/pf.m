% ----------------------------------------------------------------------- %
% Pathfinder Algorithm (PFA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 50   % Population size
%
% Algorithm Concept:
%   - Inspired by collective movement and leadership of animal swarms
%   - One pathfinder (leader) explores the search space
%   - Followers move relative to the leader and to their neighbours
%   - Adaptive vibration/fluctuation terms shrink over the run
%   - Elite memory keeps each member's best position
%
% Reference:
% Hamza Yapici, Nurettin Cetinkaya,
% A new meta-heuristic optimizer: Pathfinder algorithm,
% Applied Soft Computing 78 (2019) 545-568
% https://doi.org/10.1016/j.asoc.2019.03.012
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = pf(problem)

    % Extract problem parameters
    problem_size = problem.dimension;
    Lb    = problem.lb;
    Ub    = problem.ub;
    maxFE = problem.maxFe;

    pop_size = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, pop_size, problem_size);
    fitness_history = zeros(history_size, pop_size);
    history_index = 1;

    % Initial population
    pop = zeros(pop_size, problem_size);
    for i = 1:pop_size
        pop(i, :) = rand(1, problem_size) .* (Ub - Lb) + Lb;
    end

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:)';
    fit_old = fitness;
    [fit_global, index] = min(fitness);
    pop_old = pop;
    global_pop = pop(index, :);
    path_ = global_pop;
    path__old = path_;

    for eval_count = 1:pop_size
        curve(eval_count) = fit_global;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, pop, fitness', population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    while FE < maxFE
        FE_before = FE;

        % Update parameters
        u1 = -1 + 2 .* rand(1, problem_size);
        u2 = -1 + 2 .* rand(pop_size, problem_size);
        alpha = 1 + rand;
        beta  = 1 + rand;

        r3 = rand(1, problem_size);
        r1 = rand(pop_size, problem_size);
        r2 = rand(pop_size, problem_size);

        A   = u1 .* exp(-2 * FE / maxFE);
        eps = (1 - FE / (maxFE + 1)) .* u2;

        pop = pop_old;

        % Movement of the pathfinder
        path_(1, :) = path_(1, :) + (2 .* r3(1, :)) .* (path__old(1, :) - path_(1, :)) + A(1, :);
        if all(path_(1, :) < Lb)
            path_(1, :) = Lb;
        end
        if all(path_(1, :) > Ub)
            path_(1, :) = Ub;
        end
        path__old = path_;

        [fitx, FE] = calculate_fitness(path_(1, :)', problem, FE);
        fitx = fitx(1);

        [a, index] = sort(fitness);
        pop = pop(index, :);
        if fitx < a(1)
            fitness(1) = fitx;
            pop(1, :)  = path_(1, :);
        end

        % Movement of the followers
        for j = 2:pop_size
            d = sqrt(((pop(j, :)).^2 + (pop(j-1, :)).^2));
            pop(j, :) = pop(j, :) ...
                + ((alpha) .* r1(j, :)) .* (global_pop(1, :) - pop(j, :)) ...
                + ((beta) .* r2(j, :)) .* (pop_old(j-1, :) - pop(j, :)) + 0.1 .* (eps(j, :) .* d ./ 2);
        end

        % Bound the whole population and evaluate
        for i = 1:pop_size
            I = pop(i, :) < Lb; pop(i, I) = Lb(I);
            I = pop(i, :) > Ub; pop(i, I) = Ub(I);
        end
        [fitness, FE] = calculate_fitness(pop', problem, FE);
        fitness = fitness(:)';

        % Greedy update of the elite memory
        for i = 1:pop_size
            if fitness(i) < fit_old(i)
                fit_old(i)    = fitness(i);
                pop_old(i, :) = pop(i, :);
            end
        end

        [fit_global, index] = min(fit_old);
        global_pop = pop_old(index, :);
        path_ = global_pop;

        % Record convergence curve and history for this generation
        for eval_count = (FE_before + 1):FE
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = fit_global;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, fitness', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness  = fit_global;
    best_solution = global_pop;
end

% ----------------------------------------------------------------------- %
% Snow Ablation Optimizer (SAO)
% Stored as snowao; the acronym SAO collides with the Smell Agent Optimizer
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30                     % Population size
%   Na = N/2 -> N               % Size of the exploration subpopulation, +1 per iteration
%   N1 = floor(N/2)             % Members averaged into the elite pool's fourth slot
%   DDF = 0.35*(1+(5/7)*(e^t-1)/(e-1))  % Degree-day factor, t the spent budget ratio
%
% Algorithm Concept:
%   - The population splits into two subpopulations: one explores, one exploits,
%     and the split moves one individual per iteration from the second to the first
%   - Exploration steps from a random member of a four-slot elite pool (best,
%     second, third, mean of the best half) along a Brownian-weighted mix of the
%     pulls towards the best and towards the population centroid
%   - Exploitation steps from the snowmelt rate M times the best position, with
%     the same Brownian mix but a signed random weight
%   - M = DDF * exp(-t) is the snowmelt rate, which decays as the budget is spent
%   - Coordinates outside the box are clamped to it
%
% Reference:
% Lingyun Deng, Sanyang Liu,
% Snow ablation optimizer: A novel metaheuristic technique for numerical
% optimization and engineering design,
% Expert Systems with Applications 225 (2023) 120069.
% https://doi.org/10.1016/j.eswa.2023.120069
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released code (github.com/denglingyun123/
% SAO-snow-ablation-optimizer, SAO.m). It schedules T and DDF on the iteration
% ratio l/Max_iter against a fixed 1000 iterations; here they run on the spent
% budget FE/maxFe, so the same schedule is traversed whatever the budget, and
% FE >= maxFe is the only terminator.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = snowao(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    N1 = floor(N * 0.5);       % how many of the best are averaged into the elite pool

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:N
        if fitness(i) < best_fitness
            best_fitness = fitness(i);
            best_solution = X(i, :);
        end
        if i <= maxFE
            curve(i) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, fitness', population_history, fitness_history, history_index, maxFE);
        end
    end

    elite = elite_pool(X, fitness, best_solution, N1);

    Na = round(N / 2);          % exploration subpopulation, grows by one per iteration
    Nb = N - Na;

    while FE < maxFE
        t = FE / maxFE;                                   % spent budget, the schedule's clock
        RB = randn(N, dim);                               % Brownian steps
        DDF = 0.35 * (1 + (5 / 7) * (exp(t) - 1) / (exp(1) - 1));
        M = DDF * exp(-t);                                % snowmelt rate
        centroid = mean(X, 1);

        idx_a = randperm(N, Na);
        idx_b = setdiff(1:N, idx_a);

        for i = 1:Na
            r1 = rand;
            k1 = randi(4);
            row = idx_a(i);
            X(row, :) = elite(k1, :) + RB(row, :) .* (r1 * (best_solution - X(row, :)) ...
                        + (1 - r1) * (centroid - X(row, :)));
        end

        if Na < N
            Na = Na + 1;
            Nb = Nb - 1;
        end

        for i = 1:Nb
            r2 = 2 * rand - 1;
            row = idx_b(i);
            X(row, :) = M * best_solution + RB(row, :) .* (r2 * (best_solution - X(row, :)) ...
                        + (1 - r2) * (centroid - X(row, :)));
        end

        X = min(max(X, repmat(lb, N, 1)), repmat(ub, N, 1));

        [fitness, FE] = calculate_fitness(X', problem, FE);
        fitness = fitness(:);

        for i = 1:N
            if fitness(i) < best_fitness
                best_fitness = fitness(i);
                best_solution = X(i, :);
            end
            eval_count = FE - N + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        elite = elite_pool(X, fitness, best_solution, N1);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Best, second, third and the mean of the best half
function elite = elite_pool(X, fitness, best_solution, N1)
    [~, order] = sort(fitness);
    elite = [best_solution; X(order(2), :); X(order(3), :); mean(X(order(1:N1), :), 1)];
end

% ----------------------------------------------------------------------- %
% Barnacles Mating Optimizer (BMO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30   % Population size (number of barnacles)
%   pl = 7    % Penis length (maximum mating distance)
%
% Algorithm Concept:
%   - Bio-inspired by the mating behaviour of barnacles
%   - Offspring produced by Hardy-Weinberg reproduction within pl range
%   - Sperm-cast reproduction when partners are beyond pl
%   - Each generation keeps the best N of parents plus offspring
%
% Reference:
% Mohd Herwan Sulaiman, Zuriani Mustaffa, Mohd Mawardi Saari, Hamdan Daniyal,
% Barnacles Mating Optimizer: A new bio-inspired algorithm for solving engineering optimization problems,
% Engineering Applications of Artificial Intelligence 87 (2020) 103330
% https://doi.org/10.1016/j.engappai.2019.103330
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bmo(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N  = 30;   % Population size
    pl = 7;    % Penis length

    % Needs an even dimension: an odd one gets a dummy variable, ignored during evaluation
    flag = 0;
    eval_dim = dim;
    if rem(dim, 2) ~= 0
        dim = dim + 1;
        ub = [ub, ub(1)];
        lb = [lb, lb(1)];
        flag = 1;
    end

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the population of barnacles
    BarnaclesPositions = initialization(N, dim, ub, lb);

    % Evaluate initial population
    [BarnaclesFitness, FE] = evaluate_pop(BarnaclesPositions, flag, problem, FE);

    [sorted_fitness, sorted_indexes] = sort(BarnaclesFitness);
    BarnaclesPositions = BarnaclesPositions(sorted_indexes, :);
    BarnaclesFitness   = sorted_fitness;
    TargetPosition = BarnaclesPositions(1, :);
    TargetFitness  = sorted_fitness(1);

    % Record initial evaluations
    for eval_count = 1:N
        curve(eval_count) = TargetFitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, BarnaclesPositions(:, 1:eval_dim), BarnaclesFitness', population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Main loop
    while FE < maxFE

        % Selection (barnacles find their mate within penis-length range)
        k1 = randperm(N);
        k2 = randperm(N);
        k1x = BarnaclesPositions(k1, :);
        k2x = BarnaclesPositions(k2, :);

        select = [k1', k2'];
        kurang_7 = abs(select(:, 1) - select(:, 2));
        id_over7 = find(kurang_7 > pl);   % beyond pl -> sperm cast (no mating)

        % Reproduction (Eq. 11 & 12)
        p = randn(N, dim);
        S_i = p .* k1x + (1 - p) .* k2x;
        X_new = S_i;

        if ~isempty(id_over7)
            for k = 1:size(id_over7, 1)
                X_new(id_over7(k), :) = rand() .* k2x(id_over7(k), :);
            end
        end

        Barnaclesoffspring = X_new;
        Barnaclescolony = [BarnaclesPositions; Barnaclesoffspring];

        % Relocate barnacles that leave the search space
        for i = 1:size(Barnaclescolony, 1)
            Tp = Barnaclescolony(i, :) > ub;
            Tm = Barnaclescolony(i, :) < lb;
            Barnaclescolony(i, :) = (Barnaclescolony(i, :) .* (~(Tp + Tm))) + ub .* Tp + lb .* Tm;
        end

        % Evaluate whole colony (2N)
        [colonyFitness, FE] = evaluate_pop(Barnaclescolony, flag, problem, FE);

        % Sort and keep the best N
        [sorted_fitness, sorted_indexes] = sort(colonyFitness);
        Barnaclescolony = Barnaclescolony(sorted_indexes, :);
        BarnaclesPositions = Barnaclescolony(1:N, :);
        BarnaclesFitness   = sorted_fitness(1:N);

        if BarnaclesFitness(1) < TargetFitness
            TargetPosition = BarnaclesPositions(1, :);
            TargetFitness  = BarnaclesFitness(1);
        end

        % Record convergence curve and history for this generation (2N evals)
        for eval_idx = 1:(2 * N)
            eval_count = FE - 2 * N + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = TargetFitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, BarnaclesPositions(:, 1:eval_dim), BarnaclesFitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    if flag == 1
        TargetPosition = TargetPosition(1:eval_dim);
    end
    best_fitness  = TargetFitness;
    best_solution = TargetPosition;
end

% Evaluate a population (row-wise), honouring the dummy-variable flag
function [fit, FE] = evaluate_pop(P, flag, problem, FE)
    if flag == 1
        [fit, FE] = calculate_fitness(P(:, 1:end-1)', problem, FE);
    else
        [fit, FE] = calculate_fitness(P', problem, FE);
    end
    fit = fit(:)';
end

% Initialization Function
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

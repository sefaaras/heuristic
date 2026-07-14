% ----------------------------------------------------------------------- %
% Rat Swarm Optimizer (RSO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Search_Agents = 30    % Population size (number of rats)
%   x = 1, y = 5          % Bounds for the random parameter R
%
% Algorithm Concept:
%   - Inspired by the chasing and fighting behaviour of rats in nature
%   - Parameter A decreases linearly from R towards 0 controlling the
%     transition between exploration and exploitation
%   - Rats update their position relative to the best rat found so far
%
% Reference:
% Gaurav Dhiman, Meenakshi Garg, Atulya Nagar, Vijay Kumar, Mohammad Dehghani,
% A novel algorithm for global optimization: Rat Swarm Optimizer,
% Journal of Ambient Intelligence and Humanized Computing 12 (2021) 8457-8482
% https://doi.org/10.1007/s12652-020-02580-0
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = rso(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    Search_Agents = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, Search_Agents, dim);
    fitness_history = zeros(history_size, Search_Agents);
    history_index = 1;

    Position = zeros(1, dim);
    Score = inf;

    Positions = init(Search_Agents, dim, ub, lb);

    Max_iterations = (maxFE / Search_Agents) + 1;
    l = 0;
    x = 1;
    y = 5;
    R = floor((y - x) .* rand(1, 1) + x);

    while FE < maxFE
        FE_before = FE;

        % Boundary control
        for i = 1:size(Positions, 1)
            Flag4Upper_bound = Positions(i, :) > ub;
            Flag4Lower_bound = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4Upper_bound + Flag4Lower_bound))) + ub .* Flag4Upper_bound + lb .* Flag4Lower_bound;
        end

        % Evaluate the whole swarm
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        fitness = fitness(:);

        for i = 1:size(Positions, 1)
            if fitness(i) < Score
                Score = fitness(i);
                Position = Positions(i, :);
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE
            break;
        end

        % Update the position of search agents
        A = R - l * ((R) / Max_iterations);
        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)
                C = 2 * rand();
                P_vec = A * Positions(i, j) + abs(C * ((Position(j) - Positions(i, j))));
                P_final = Position(j) - P_vec;
                Positions(i, j) = P_final;
            end
        end

        l = l + 1;
    end

    best_solution = Position;
    best_fitness = Score;

end

%% --- Initialization Function ---
function Pos = init(SearchAgents, dimension, upperbound, lowerbound)
    Boundary = size(upperbound, 2);
    if Boundary == 1
        Pos = rand(SearchAgents, dimension) .* (upperbound - lowerbound) + lowerbound;
    end
    if Boundary > 1
        for i = 1:dimension
            ub_i = upperbound(i);
            lb_i = lowerbound(i);
            Pos(:, i) = rand(SearchAgents, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

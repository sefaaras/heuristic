% ----------------------------------------------------------------------- %
% Tunicate Swarm Algorithm (TSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Search_Agents = 30    % Population size (number of tunicates)
%   xmin = 1, xmax = 4    % Bounds for the random social-force parameter
%
% Algorithm Concept:
%   - Inspired by jet propulsion and swarm behaviour of tunicates when
%     searching for a food source (the optimum)
%   - Avoids conflicts between search agents, moves towards the best agent
%     and keeps a swarm cohesion by averaging with the previous agent
%
% Reference:
% Satnam Kaur, Lalit K. Awasthi, A.L. Sangal, Gaurav Dhiman,
% Tunicate Swarm Algorithm: A new bio-inspired based metaheuristic paradigm
% for global optimization,
% Engineering Applications of Artificial Intelligence 90 (2020) 103541
% https://doi.org/10.1016/j.engappai.2020.103541
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = tsa(problem)

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

    while FE < maxFE
        FE_before = FE;

        % Boundary control then evaluate
        for i = 1:size(Positions, 1)
            Flag4Upperbound = Positions(i, :) > ub;
            Flag4Lowerbound = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4Upperbound + Flag4Lowerbound))) + ub .* Flag4Upperbound + lb .* Flag4Lowerbound;
        end

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
        xmin = 1;
        xmax = 4;
        xr = xmin + rand() * (xmax - xmin);
        xr = fix(xr);

        Pos = Positions;   % pre-memory for the shifted update
        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)
                A1 = ((rand() + rand()) - (2 * rand())) / xr;
                c2 = rand();
                if (i == 1)
                    c3 = rand();
                    if (c3 >= 0)
                        d_pos = abs(Position(j) - c2 * Positions(i, j));
                        Positions(i, j) = Position(j) + A1 * d_pos;
                    else
                        d_pos = abs(Position(j) - c2 * Positions(i, j));
                        Positions(i, j) = Position(j) - A1 * d_pos;
                    end
                else
                    c3 = rand();
                    if (c3 >= 0)
                        d_pos = abs(Position(j) - c2 * Positions(i, j));
                        Pos(i, j) = Position(j) + A1 * d_pos;
                    else
                        Pos(i, j) = Position(j) - A1 * d_pos;
                    end
                    Positions(i, j) = (Pos(i, j) + Positions(i - 1, j)) / 2;
                end
            end
        end
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

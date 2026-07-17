% ----------------------------------------------------------------------- %
% Seagull Optimization Algorithm (SOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Search_Agents = 30   % Population size (seagulls)
%
% Algorithm Concept:
%   - Migration: collision avoidance + movement toward the best seagull,
%     governed by a control parameter Fc decaying linearly from 2 to 0
%   - Attacking: spiral (logarithmic) dive toward the prey
%
% Reference:
% Gaurav Dhiman, Vijay Kumar,
% Seagull optimization algorithm: Theory and its applications for
% large-scale industrial engineering problems,
% Knowledge-Based Systems 165 (2019) 169-196.
% https://doi.org/10.1016/j.knosys.2018.11.024
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = soa(problem)

    % Extract problem parameters
    dimension = problem.dimension;
    Lower_bound = problem.lb;
    Upper_bound = problem.ub;
    maxFE = problem.maxFe;

    Search_Agents = 30;
    Max_iterations = ceil(maxFE / Search_Agents);

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, Search_Agents, dimension);
    fitness_history = zeros(history_size, Search_Agents);
    history_index = 1;

    Position = zeros(1, dimension);
    Score = inf;

    Positions = init(Search_Agents, dimension, Upper_bound, Lower_bound);
    fitness_all = inf(1, Search_Agents);

    l = 0;
    while l < Max_iterations
        if FE >= maxFE, break; end

        for i = 1:size(Positions, 1)
            Flag4Upper_bound = Positions(i, :) > Upper_bound;
            Flag4Lower_bound = Positions(i, :) < Lower_bound;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4Upper_bound + Flag4Lower_bound))) + Upper_bound .* Flag4Upper_bound + Lower_bound .* Flag4Lower_bound;

            [fitness, FE] = calculate_fitness(Positions(i, :)', problem, FE);
            fitness_all(i) = fitness;

            if fitness < Score
                Score = fitness;
                Position = Positions(i, :);
            end

            if FE <= maxFE
                curve(FE) = Score;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Positions, fitness_all, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end

        Fc = 2 - l * ((2) / Max_iterations);

        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)
                r1 = rand();
                r2 = rand();
                A1 = 2 * Fc * r1 - Fc;
                C1 = 2 * r2;
                b = 1;
                ll = (Fc - 1) * rand() + 1;
                D_alphs = Fc * Positions(i, j) + A1 * ((Position(j) - Positions(i, j)));
                X1 = D_alphs * exp(b .* ll) .* cos(ll .* 2 * pi) + Position(j);
                Positions(i, j) = X1;
            end
        end
        l = l + 1;
    end

    curve(min(FE, maxFE):end) = Score;

    best_fitness = Score;
    best_solution = Position;
end

%% --- Initialization ---
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

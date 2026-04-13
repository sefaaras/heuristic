% ----------------------------------------------------------------------- %
% Butterfly Optimization Algorithm (BOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 50                      % Population size
%   p = 0.8                     % Probability switch
%   power_exponent = 0.1        % Power exponent for fragrance
%   sensory_modality = 0.01     % Initial sensory modality
%
% Algorithm Concept:
%   - Butterflies move based on fragrance (fitness)
%   - Global search: move towards best butterfly
%   - Local search: move between random butterflies
%   - Sensory modality increases over iterations
%
% Reference:
% Sankalap Arora and Satvir Singh,
% Butterfly optimization algorithm: a novel approach for global optimization,
% Soft Computing 23 (2019) 715-734
% https://doi.org/10.1007/s00500-018-3102-4
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = boa(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    n = 50;                       % Population size
    p = 0.8;                      % Probability switch
    power_exponent = 0.1;         % Power exponent
    sensory_modality = 0.01;      % Initial sensory modality
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, n, dim);
    fitness_history = zeros(history_size, n);
    history_index = 1;
    
    % Initialize the positions of butterflies
    Sol = initialization(n, dim, ub, lb);
    
    % Evaluate initial population
    [Fitness, FE] = calculate_fitness(Sol', problem, FE);
    
    % Find the current best position
    [fmin, I] = min(Fitness);
    best_pos = Sol(I, :);
    S = Sol;
    
    % Record initial population
    for eval_count = 1:n
        curve(eval_count) = fmin;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Sol, Fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    N_iter = ceil((maxFE - n) / (n * 2));  % Each iteration evaluates 2*n solutions
    
    for t = 1:N_iter
        
        % Update each butterfly
        for i = 1:n
            
            % Calculate fragrance based on fitness
            Fnew = Fitness(i);
            FP = sensory_modality * (Fnew^power_exponent);
            
            % Global or local search
            if rand < p
                % Global search: move towards best butterfly
                dis = rand * rand * best_pos - Sol(i, :);
                S(i, :) = Sol(i, :) + dis * FP;
            else
                % Local search: move between random butterflies
                epsilon = rand;
                JK = randperm(n);
                dis = epsilon * epsilon * Sol(JK(1), :) - Sol(JK(2), :);
                S(i, :) = Sol(i, :) + dis * FP;
            end
            
            % Apply boundary constraints
            S(i, :) = bound(S(i, :), ub, lb);
            
        end
        
        % Evaluate new solutions
        [Fnew_array, FE] = calculate_fitness(S', problem, FE);
        
        % Update solutions if fitness improves
        for i = 1:n
            if Fnew_array(i) <= Fitness(i)
                Sol(i, :) = S(i, :);
                Fitness(i) = Fnew_array(i);
            end
            
            % Update global best
            if Fnew_array(i) <= fmin
                best_pos = S(i, :);
                fmin = Fnew_array(i);
            end
        end
        
        % Record convergence curve and history (first n evaluations)
        for eval_idx = 1:n
            eval_count = FE - n + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = fmin;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Sol, Fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Update sensory modality
        sensory_modality = sensory_modality_NEW(sensory_modality, N_iter);
        
        % Check if we've reached maxFE
        if FE >= maxFE
            break;
        end
    end
    
    % Return best solution
    best_fitness = fmin;
    best_solution = best_pos;
    
end

%% --- Helper Functions ---

function Positions = initialization(popsize, dim, ub, lb)
    Boundary_no = size(ub, 2);
    
    if Boundary_no == 1
        Positions = rand(popsize, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(popsize, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

function y = sensory_modality_NEW(x, Ngen)
    y = x + (0.025 / (x * Ngen));
end


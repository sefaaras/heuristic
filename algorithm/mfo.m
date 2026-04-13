% ----------------------------------------------------------------------- %
% Moth-Flame Optimization (MFO) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                  % Population size (number of moths)
%   b = 1                   % Shape constant for logarithmic spiral
%   
% Algorithm Concept:
%   - Inspired by moth navigation using moon (transverse orientation)
%   - Moths spiral around flames (best solutions found so far)
%   - Number of flames decreases over iterations
%   - Logarithmic spiral flight path
%
% Reference:
% S. Mirjalili,
% Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm,
% Knowledge-Based Systems, Volume 89, 2015, Pages 228-249
% https://doi.org/10.1016/j.knosys.2015.07.006
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = mfo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    N = 30;                       % Population size (number of moths)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize the positions of moths
    Moth_pos = initialization(N, dim, ub, lb);
    
    % Evaluate initial population
    [Moth_fitness, FE] = calculate_fitness(Moth_pos', problem, FE);
    
    % Sort the first population of moths
    [fitness_sorted, I] = sort(Moth_fitness);
    sorted_population = Moth_pos(I, :);
    
    % Initialize flames (best solutions)
    best_flames = sorted_population;
    best_flame_fitness = fitness_sorted;
    
    % Best solution so far
    Best_flame_score = fitness_sorted(1);
    Best_flame_pos = sorted_population(1, :);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:N
        curve(eval_count) = Best_flame_score;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Moth_pos, Moth_fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        % Number of flames Eq. (3.14) in the paper
        Flame_no = round(N - Iteration * ((N - 1) / Max_iteration));
        
        % Store previous population
        previous_population = Moth_pos;
        previous_fitness = Moth_fitness;
        
        % a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + Iteration * ((-1) / Max_iteration);
        
        % Update moth positions
        for i = 1:N
            for j = 1:dim
                if i <= Flame_no
                    % Update the position of the moth with respect to its corresponding flame
                    distance_to_flame = abs(sorted_population(i, j) - Moth_pos(i, j));
                    b = 1;
                    t = (a - 1) * rand + 1;
                    
                    % Eq. (3.12) - Logarithmic spiral
                    Moth_pos(i, j) = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + sorted_population(i, j);
                else
                    % Update the position of the moth with respect to one flame (best flame)
                    distance_to_flame = abs(sorted_population(Flame_no, j) - Moth_pos(i, j));
                    b = 1;
                    t = (a - 1) * rand + 1;
                    
                    % Eq. (3.12)
                    Moth_pos(i, j) = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + sorted_population(Flame_no, j);
                end
            end
            
            % Apply boundary constraints
            Moth_pos(i, :) = bound(Moth_pos(i, :), ub, lb);
        end
        
        % Evaluate new moth positions
        [Moth_fitness, FE] = calculate_fitness(Moth_pos', problem, FE);
        
        % Combine previous population with current flames
        double_population = [previous_population; best_flames];
        double_fitness = [previous_fitness(:)', best_flame_fitness(:)'];
        
        % Sort the combined population
        [double_fitness_sorted, I] = sort(double_fitness);
        double_sorted_population = double_population(I, :);
        
        % Select top N solutions as new flames
        fitness_sorted = double_fitness_sorted(1:N);
        sorted_population = double_sorted_population(1:N, :);
        
        % Update the flames
        best_flames = sorted_population;
        best_flame_fitness = fitness_sorted;
        
        % Update the position of best flame obtained so far
        Best_flame_score = fitness_sorted(1);
        Best_flame_pos = sorted_population(1, :);
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = Best_flame_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Moth_pos, Moth_fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        Iteration = Iteration + 1;
    end
    
    % Return best solution
    best_fitness = Best_flame_score;
    best_solution = Best_flame_pos;
    
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);  % Number of boundaries
    
    % If the boundaries of all variables are equal
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    
    % If each variable has a different lb and ub
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end


% ----------------------------------------------------------------------- %
% Sine Cosine Algorithm (SCA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30  % Population size (number of solutions)
%   
% Algorithm Concept:
%   - Uses sine and cosine mathematical functions
%   - Solutions update positions based on sine and cosine functions
%   - Exploration vs exploitation controlled by adaptive parameter
%
% Reference:
% Seyedali Mirjalili,
% SCA: A Sine Cosine Algorithm for solving optimization problems,
% Knowledge-Based Systems 96 (2016) 120-133
% http://dx.doi.org/10.1016/j.knosys.2015.12.022
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = sca(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    N = 30;                       % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize the positions of solutions
    X = initialization(N, dim, ub, lb);
    
    % Initialize destination position and fitness
    Destination_position = zeros(1, dim);
    Destination_fitness = inf;
    
    % Evaluate initial population
    [Objective_values, FE] = calculate_fitness(X', problem, FE);
    
    % Find the best solution in initial population
    for i = 1:N
        if Objective_values(i) < Destination_fitness
            Destination_position = X(i, :);
            Destination_fitness = Objective_values(i);
        end
    end
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:N
        curve(eval_count) = Destination_fitness;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, Objective_values, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    t = 0;  % Loop counter
    
    while FE < maxFE && t < Max_iteration
        % Eq. (3.4) - r1 decreases linearly from a to 0
        a = 2;
        r1 = a - (t + 1) * ((a) / Max_iteration);
        
        % Update the position of solutions with respect to destination
        for i = 1:N  % for i-th solution
            for j = 1:dim  % for j-th dimension
                % Update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * pi) * rand();
                r3 = 2 * rand;
                r4 = rand();
                
                % Eq. (3.3)
                if r4 < 0.5
                    % Eq. (3.1) - Sine-based position update
                    X(i, j) = X(i, j) + (r1 * sin(r2) * abs(r3 * Destination_position(j) - X(i, j)));
                else
                    % Eq. (3.2) - Cosine-based position update
                    X(i, j) = X(i, j) + (r1 * cos(r2) * abs(r3 * Destination_position(j) - X(i, j)));
                end
            end
            
            % Apply boundary constraints
            X(i, :) = bound(X(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [Objective_values, FE] = calculate_fitness(X', problem, FE);
        
        % Update the destination if there is a better solution
        for i = 1:N
            if Objective_values(i) < Destination_fitness
                Destination_position = X(i, :);
                Destination_fitness = Objective_values(i);
            end
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = Destination_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, Objective_values, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        t = t + 1;
    end
    
    % Return best solution
    best_fitness = Destination_fitness;
    best_solution = Destination_position;
    
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


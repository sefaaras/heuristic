% ----------------------------------------------------------------------- %
% Particle Swarm Optimization (PSO) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                  % Population size (number of particles)
%   w = 0.9 -> 0.4          % Inertia weight (linearly decreasing)
%   c1 = 2                  % Cognitive parameter (personal best)
%   c2 = 2                  % Social parameter (global best)
%   
% Algorithm Concept:
%   - Inspired by social behavior of bird flocking or fish schooling
%   - Each particle has position and velocity
%   - Particles move based on their own experience (pbest) and swarm experience (gbest)
%   - Velocity update equation balances exploration and exploitation
%
% Reference:
% J. Kennedy and R. Eberhart,
% Particle swarm optimization,
% Proceedings of ICNN'95 - International Conference on Neural Networks,
% Perth, WA, Australia, 1995, pp. 1942-1948 vol.4
% https://doi.org/10.1109/ICNN.1995.488968
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = pso(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    N = 30;                       % Population size (number of particles)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % PSO parameters
    w_max = 0.9;                  % Maximum inertia weight
    w_min = 0.4;                  % Minimum inertia weight
    c1 = 2;                       % Cognitive parameter (personal best)
    c2 = 2;                       % Social parameter (global best)
    
    % Initialize particle positions and velocities
    X = initialization(N, dim, ub, lb);  % Particle positions
    V = zeros(N, dim);                   % Particle velocities
    
    % Velocity limits (20% of search space)
    vmax = 0.2 * (ub - lb);
    vmin = -vmax;
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(X', problem, FE);
    
    % Initialize personal best positions and fitness
    pbest = X;                    % Personal best positions
    pbest_fitness = fitness;      % Personal best fitness values
    
    % Initialize global best
    [gbest_fitness, best_idx] = min(fitness);
    gbest = X(best_idx, :);       % Global best position
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:N
        curve(eval_count) = gbest_fitness;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        % Update inertia weight (linearly decreasing)
        w = w_max - ((w_max - w_min) * Iteration / Max_iteration);
        
        % Update velocity and position for each particle
        for i = 1:N
            % Update velocity
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            
            % Velocity update equation
            V(i, :) = w * V(i, :) + ...
                      c1 * r1 .* (pbest(i, :) - X(i, :)) + ...
                      c2 * r2 .* (gbest - X(i, :));
            
            % Apply velocity limits
            V(i, :) = max(min(V(i, :), vmax), vmin);
            
            % Update position
            X(i, :) = X(i, :) + V(i, :);
            
            % Apply boundary constraints
            X(i, :) = bound(X(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [fitness, FE] = calculate_fitness(X', problem, FE);
        
        % Update personal best
        for i = 1:N
            if fitness(i) < pbest_fitness(i)
                pbest(i, :) = X(i, :);
                pbest_fitness(i) = fitness(i);
            end
        end
        
        % Update global best
        [min_fitness, min_idx] = min(fitness);
        if min_fitness < gbest_fitness
            gbest_fitness = min_fitness;
            gbest = X(min_idx, :);
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = gbest_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        Iteration = Iteration + 1;
    end
    
    % Return best solution
    best_fitness = gbest_fitness;
    best_solution = gbest;
    
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


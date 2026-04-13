% ----------------------------------------------------------------------- %
% Artificial Electric Field Algorithm (AEFA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Population size (number of charged particles)
%   alfa = 30               % Coulomb constant decay parameter
%   K0 = 500                % Initial Coulomb constant
%   Rpower = 1              % Distance power in force calculation
%   Rnorm = 2               % Euclidean norm type
%   fper = 3                % Final percentage of charges applying force
%   
% Algorithm Concept:
%   - Inspired by Coulomb's law and electric field theory
%   - Each particle represents a charged particle in search space
%   - Particles move based on electric forces between them
%   - Charge of each particle depends on its fitness value
%   - Better solutions have higher charge and exert more force
%   - Electric force decreases over time (exploration to exploitation)
%
% Reference:
% Anita and Yadav, A. (2019),
% AEFA: Artificial electric field algorithm for global optimization,
% Swarm and Evolutionary Computation, 48, 93-108.
% https://doi.org/10.1016/j.swevo.2019.03.013
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = aefa(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % AEFA Parameters
    N = 50;                       % Population size (number of charged particles)
    alfa = 30;                    % Coulomb constant decay parameter
    K0 = 500;                     % Initial Coulomb constant
    Rpower = 1;                   % Distance power in force calculation
    Rnorm = 2;                    % Euclidean norm type
    fper = 3;                     % Final percentage of charges applying force
    FCheck = 1;                   % Flag to check force application
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize charged particles
    X = initialization(N, dim, ub, lb);  % Particle positions
    V = zeros(N, dim);                   % Particle velocities
    E = zeros(N, dim);                   % Electric field
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(X', problem, FE);
    
    % Initialize best solution
    [best_fitness_current, best_idx] = min(fitness);
    best_solution_current = X(best_idx, :);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:N
        curve(eval_count) = best_fitness_current;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        % Calculate charges based on fitness values
        Fmax = max(fitness);
        Fmin = min(fitness);
        
        if Fmax == Fmin
            Q = ones(N, 1);
        else
            best = Fmin;
            worst = Fmax;
            Q = exp((fitness - worst) ./ (best - worst));  % Charge calculation
        end
        Q = Q ./ sum(Q);  % Normalize charges
        
        % Determine number of best charges to apply force
        if FCheck == 1
            cbest = fper + (1 - Iteration / Max_iteration) * (100 - fper);
            cbest = round(N * cbest / 100);
        else
            cbest = N;
        end
        
        % Sort particles by charge (descending)
        [~, s] = sort(Q, 'descend');
        
        % Calculate total electric force for each particle
        for i = 1:N
            E(i, :) = zeros(1, dim);
            for ii = 1:cbest
                j = s(ii);
                if j ~= i
                    % Calculate Euclidean distance
                    R = norm(X(i, :) - X(j, :), Rnorm);
                    % Calculate electric force for each dimension
                    for k = 1:dim
                        E(i, k) = E(i, k) + rand * Q(j) * ((X(j, k) - X(i, k)) / (R^Rpower + eps));
                    end
                end
            end
        end
        
        % Calculate Coulomb constant (decreases over iterations)
        K = K0 * exp(-alfa * Iteration / Max_iteration);
        
        % Calculate acceleration from electric field
        a = E * K;
        
        % Update velocity and position
        V = rand(N, dim) .* V + a;
        X = X + V;
        
        % Apply boundary constraints
        for i = 1:N
            X(i, :) = bound(X(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [fitness, FE] = calculate_fitness(X', problem, FE);
        
        % Update best solution
        [min_fitness, min_idx] = min(fitness);
        if min_fitness < best_fitness_current
            best_fitness_current = min_fitness;
            best_solution_current = X(min_idx, :);
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        Iteration = Iteration + 1;
    end
    
    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;
    
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


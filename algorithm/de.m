% ----------------------------------------------------------------------- %
% Differential Evolution (DE) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50               % Population size
%   beta_min = 0.2          % Lower Bound of Scaling Factor
%   beta_max = 0.8          % Upper Bound of Scaling Factor
%   pCR = 0.2               % Crossover Probability
%   
% Algorithm Concept:
%   - Population-based stochastic optimization algorithm
%   - Uses mutation, crossover, and selection operations
%   - Mutation: Creates trial vector using difference of random population members
%   - Crossover: Combines target and mutant vectors
%   - Selection: Greedy selection between parent and offspring
%
% Reference:
% Storn, R. and Price, K. (1997),
% Differential evolution – a simple and efficient heuristic for global 
% optimization over continuous spaces,
% Journal of Global Optimization, 11(4), 341-359.
% https://doi.org/10.1023/A:1008202821328
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = de(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % DE Parameters
    nPop = 50;                    % Population size
    beta_min = 0.2;               % Lower Bound of Scaling Factor
    beta_max = 0.8;               % Upper Bound of Scaling Factor
    pCR = 0.2;                    % Crossover Probability
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, nPop, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, nPop);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize population
    X = initialization(nPop, dim, ub, lb);  % Population positions
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(X', problem, FE);
    
    % Find initial best solution
    [best_fitness_current, best_idx] = min(fitness);
    best_solution_current = X(best_idx, :);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:nPop
        curve(eval_count) = best_fitness_current;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - nPop) / nPop);
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        % Create new population
        X_new = zeros(nPop, dim);
        
        for i = 1:nPop
            % Select three random individuals (different from current)
            indices = 1:nPop;
            indices(i) = [];  % Remove current individual
            selected = indices(randperm(length(indices), 3));
            a = selected(1);
            b = selected(2);
            c = selected(3);
            
            % Mutation: Create mutant vector
            beta = unifrnd(beta_min, beta_max, 1, dim);
            mutant = X(a, :) + beta .* (X(b, :) - X(c, :));
            
            % Apply boundary constraints to mutant
            mutant = bound(mutant, ub, lb);
            
            % Crossover: Create trial vector
            trial = zeros(1, dim);
            j_rand = randi(dim);  % Ensure at least one dimension from mutant
            
            for j = 1:dim
                if j == j_rand || rand <= pCR
                    trial(j) = mutant(j);
                else
                    trial(j) = X(i, j);
                end
            end
            
            X_new(i, :) = trial;
        end
        
        % Evaluate new population
        [fitness_new, FE] = calculate_fitness(X_new', problem, FE);
        
        % Selection: Greedy selection between parent and offspring
        for i = 1:nPop
            if fitness_new(i) < fitness(i)
                X(i, :) = X_new(i, :);
                fitness(i) = fitness_new(i);
            end
        end
        
        % Update best solution
        [min_fitness, min_idx] = min(fitness);
        if min_fitness < best_fitness_current
            best_fitness_current = min_fitness;
            best_solution_current = X(min_idx, :);
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
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


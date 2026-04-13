% ----------------------------------------------------------------------- %
% Crow Search Algorithm (CSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Flock (population) size
%   AP = 0.1                % Awareness probability
%   fl = 2                  % Flight length
%   
% Algorithm Concept:
%   - Inspired by intelligent behavior of crows
%   - Crows hide food in hiding places and retrieve it when needed
%   - Crows follow each other to discover hiding places
%   - If a crow is aware of being followed, it deceives the follower
%   - Memory stores the best position each crow has found
%   - Balance between following (exploitation) and random search (exploration)
%
% Reference:
% Askarzadeh, A. (2016),
% A novel metaheuristic method for solving constrained engineering 
% optimization problems: Crow search algorithm,
% Computers & Structures, 169, 1-12.
% https://doi.org/10.1016/j.compstruc.2016.03.001
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = csa(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % CSA Parameters
    N = 50;                       % Flock (population) size
    AP = 0.1;                     % Awareness probability
    fl = 2;                       % Flight length
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;
    
    % Initialize crow positions
    x = initialization(N, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(x', problem, FE);
    
    % Memory initialization (best positions found by each crow)
    mem = x;
    fit_mem = fitness;
    
    % Find initial best
    [best_fitness_current, best_idx] = min(fit_mem);
    best_solution_current = mem(best_idx, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:N
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, x, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    t = 1;
    
    while FE < maxFE && t <= Max_iteration
        
        xnew = zeros(N, dim);
        
        % Generate random candidate crows for following
        num = ceil(N * rand(1, N));
        
        for i = 1:N
            if rand > AP
                % State 1: Crow i follows crow num(i) to its memory position
                xnew(i, :) = x(i, :) + fl * rand * (mem(num(i), :) - x(i, :));
            else
                % State 2: Crow num(i) is aware, crow i moves randomly
                xnew(i, :) = lb + (ub - lb) .* rand(1, dim);
            end
        end
        
        % Apply boundary constraints
        for i = 1:N
            xnew(i, :) = bound(xnew(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [ft, FE] = calculate_fitness(xnew', problem, FE);
        
        % Update position and memory
        for i = 1:N
            % Update position
            x(i, :) = xnew(i, :);
            fitness(i) = ft(i);
            
            % Update memory if new position is better
            if ft(i) < fit_mem(i)
                mem(i, :) = xnew(i, :);
                fit_mem(i) = ft(i);
            end
        end
        
        % Update global best
        [min_fit, min_idx] = min(fit_mem);
        if min_fit < best_fitness_current
            best_fitness_current = min_fit;
            best_solution_current = mem(min_idx, :);
        end
        
        % Record convergence curve for each evaluation
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, x, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        t = t + 1;
    end
    
    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;
    
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        X = zeros(SearchAgents_no, dim);
        for i = 1:dim
            X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end


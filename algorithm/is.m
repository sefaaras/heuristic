% ----------------------------------------------------------------------- %
% Interior Search Algorithm (ISA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 50                  % Population size
%   alpha = iteration/max   % Adaptive parameter (increases over time)
%   
% Algorithm Concept:
%   - Inspired by interior design and decoration
%   - Uses mirror concept to reflect solutions around a center point
%   - Center is the best solution found so far
%   - Early iterations: more exploration (random search)
%   - Later iterations: more exploitation (mirror-based search)
%   - Evolutionary boundary handling with center-based repair
%
% Reference:
% Gandomi, A.H. (2014),
% Interior search algorithm (ISA): A novel approach for global optimization,
% ISA Transactions, 53(4), 1168-1183.
% https://doi.org/10.1016/j.isatra.2014.03.018
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = is(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % ISA Parameters
    n = 50;                       % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, n, dim);
    fitness_history = zeros(history_size, n);
    history_index = 1;
    
    % Initialize population
    ns = initialization(n, dim, ub, lb);
    
    % Evaluate initial population
    [fvalue, FE] = calculate_fitness(ns', problem, FE);
    
    % Find initial best (center)
    [best_fitness_current, l] = min(fvalue);
    best_solution_current = ns(l, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:n
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, ns, fvalue, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - n) / n);
    j = 1;
    
    while FE < maxFE && j <= Max_iteration
        
        % Adaptive parameter (increases from 0 to 1)
        alpha = j / Max_iteration;
        
        % Current search space bounds
        LX = min(ns);
        UX = max(ns);
        
        % Find current best (center)
        [~, l] = min(fvalue);
        center = ns(l, :);
        
        x_new = zeros(n, dim);
        
        for i = 1:n
            if i == l
                % Best solution: small perturbation
                x_new(i, :) = ns(i, :) + randn(1, dim) .* (ub - lb) .* 0.01;
            else
                if rand < alpha
                    % Mirror-based search (exploitation)
                    beta = rand;
                    mirror = beta * ns(i, :) + (1 - beta) * center;
                    x_new(i, :) = 2 * mirror - ns(i, :);
                else
                    % Random search (exploration)
                    x_new(i, :) = LX + (UX - LX) .* rand(1, dim);
                end
            end
            
            % Evolutionary boundary handling with center-based repair
            ns_tmp = x_new(i, :);
            I = ns_tmp < lb;
            A = rand;
            ns_tmp(I) = A * lb(I) + (1 - A) * center(I);
            J = ns_tmp > ub;
            B = rand;
            ns_tmp(J) = B * ub(J) + (1 - B) * center(J);
            x_new(i, :) = ns_tmp;
        end
        
        % Evaluate new solutions
        [fvalue_new, FE] = calculate_fitness(x_new', problem, FE);
        
        % Greedy selection
        for i = 1:n
            if fvalue_new(i) < fvalue(i)
                ns(i, :) = x_new(i, :);
                fvalue(i) = fvalue_new(i);
            end
        end
        
        % Update global best
        [min_fit, min_idx] = min(fvalue);
        if min_fit < best_fitness_current
            best_fitness_current = min_fit;
            best_solution_current = ns(min_idx, :);
        end
        
        % Record convergence curve for each evaluation
        for eval_idx = 1:n
            eval_count = FE - n + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, ns, fvalue, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        j = j + 1;
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


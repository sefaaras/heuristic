% ----------------------------------------------------------------------- %
% Cuckoo Search (CS) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 25                  % Population size (number of nests)
%   pa = 0.25               % Discovery rate of alien eggs/solutions
%   beta = 1.5              % Levy flight parameter
%   
% Algorithm Concept:
%   - Inspired by brood parasitism of cuckoo birds
%   - Cuckoos lay eggs in host bird nests
%   - Each egg represents a solution
%   - Levy flights for global random walk
%   - Fraction of worst nests abandoned and rebuilt
%   - Balance between random walk and local search
%
% Reference:
% Yang, X.S. and Deb, S. (2009),
% Cuckoo search via Lévy flights,
% World Congress on Nature & Biologically Inspired Computing (NaBIC),
% IEEE, pp. 210-214.
% https://doi.org/10.1109/NABIC.2009.5393690
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = cs(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % CS Parameters
    n = 25;                       % Population size (number of nests)
    pa = 0.25;                    % Discovery rate of alien eggs
    beta = 1.5;                   % Levy flight parameter
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, n, dim);
    fitness_history = zeros(history_size, n);
    history_index = 1;
    
    % Initialize nests
    nest = initialization(n, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(nest', problem, FE);
    
    % Find initial best
    [fmin, K] = min(fitness);
    bestnest = nest(K, :);
    best_fitness_current = fmin;
    best_solution_current = bestnest;
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:n
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, nest, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Precompute Levy flight sigma
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    
    % Main loop
    while FE < maxFE
        
        % Generate new solutions via Levy flights
        new_nest = get_cuckoos(nest, bestnest, lb, ub, sigma, beta);
        
        % Evaluate new solutions
        [new_fitness, FE] = calculate_fitness(new_nest', problem, FE);
        
        % Update nests if new solution is better
        for j = 1:n
            if new_fitness(j) <= fitness(j)
                fitness(j) = new_fitness(j);
                nest(j, :) = new_nest(j, :);
            end
        end
        
        % Find current best
        [fmin_temp, K] = min(fitness);
        if fmin_temp < best_fitness_current
            best_fitness_current = fmin_temp;
            best_solution_current = nest(K, :);
            bestnest = nest(K, :);
        end
        
        % Record convergence curve
        for eval_idx = 1:n
            eval_count = FE - n + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, nest, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        if FE >= maxFE
            break;
        end
        
        % Discovery and randomization (empty nests)
        new_nest = empty_nests(nest, lb, ub, pa);
        
        % Evaluate new solutions
        [new_fitness, FE] = calculate_fitness(new_nest', problem, FE);
        
        % Update nests if new solution is better
        for j = 1:n
            if new_fitness(j) <= fitness(j)
                fitness(j) = new_fitness(j);
                nest(j, :) = new_nest(j, :);
            end
        end
        
        % Find current best
        [fmin_temp, K] = min(fitness);
        if fmin_temp < best_fitness_current
            best_fitness_current = fmin_temp;
            best_solution_current = nest(K, :);
            bestnest = nest(K, :);
        end
        
        % Record convergence curve
        for eval_idx = 1:n
            eval_count = FE - n + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, nest, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
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

%% --- Get Cuckoos via Levy Flights ---
function nest = get_cuckoos(nest, best, lb, ub, sigma, beta)
    n = size(nest, 1);
    for j = 1:n
        s = nest(j, :);
        % Levy flights by Mantegna's algorithm
        u = randn(size(s)) * sigma;
        v = randn(size(s));
        step = u ./ abs(v).^(1 / beta);
        % Step size relative to difference from best
        stepsize = 0.01 * step .* (s - best);
        % Update position
        s = s + stepsize .* randn(size(s));
        nest(j, :) = simplebounds(s, lb, ub);
    end
end

%% --- Empty Nests (Discovery) ---
function new_nest = empty_nests(nest, lb, ub, pa)
    n = size(nest, 1);
    % Discovered or not -- a status vector
    K = rand(size(nest)) > pa;
    % Random walk with biased step sizes
    stepsize = rand * (nest(randperm(n), :) - nest(randperm(n), :));
    new_nest = nest + stepsize .* K;
    for j = 1:size(new_nest, 1)
        new_nest(j, :) = simplebounds(new_nest(j, :), lb, ub);
    end
end

%% --- Boundary Handling ---
function s = simplebounds(s, lb, ub)
    % Apply lower bound
    I = s < lb;
    s(I) = lb(I);
    % Apply upper bound
    J = s > ub;
    s(J) = ub(J);
end


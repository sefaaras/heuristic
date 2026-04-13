% ----------------------------------------------------------------------- %
% Artificial Ecosystem-based Optimization (AEO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50                % Population size
%   
% Algorithm Concept:
%   - Inspired by energy flow in artificial ecosystems
%   - Population sorted by fitness (worst to best)
%   - Three types of organisms: Production, Herbivore, Carnivore/Omnivore
%   - Production: Interaction between worst and random solutions
%   - Herbivore: Grazing behavior consuming producers
%   - Carnivore/Omnivore: Predation on herbivores with probabilistic switching
%   - Decomposition phase improves exploitation around best solution
%
% Reference:
% Zhao, W., Wang, L., & Zhang, Z. (2020).
% Artificial ecosystem-based optimization: a novel nature-inspired 
% meta-heuristic algorithm.
% Neural Computing and Applications, 32(13), 9383-9425.
% https://doi.org/10.1007/s00521-019-04452-x
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = aeo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    nPop = 50;                    % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, nPop, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, nPop);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize population
    PopPos = initialization(nPop, dim, ub, lb);
    
    % Evaluate initial population
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    
    % Sort population by fitness in descending order (worst to best)
    [PopFit, indF] = sort(PopFit, 'descend');
    PopPos = PopPos(indF, :);
    
    % Best solution is at the end (after descending sort)
    BestF = PopFit(end);
    BestX = PopPos(end, :);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:nPop
        curve(eval_count) = BestF;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, PopPos, PopFit, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % For equation (9) - determines whether to update in 1D or full dimension
    Matr = [1, dim];
    
    % Main loop
    Max_iteration = ceil((maxFE - nPop) / (nPop * 2));
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        newPopPos = zeros(nPop, dim);
        
        % ===== PRODUCTION PHASE =====
        % Equation (1): Production operator - worst individual interacts with random solution
        r1 = rand;
        a = (1 - Iteration / Max_iteration) * r1;
        xrand = rand(1, dim) .* (ub - lb) + lb;
        newPopPos(1, :) = (1 - a) * PopPos(nPop, :) + a * xrand;
        
        % ===== CONSUMPTION PHASE =====
        % Equation (6): Herbivore consumes producer (second individual)
        u = randn(1, dim);
        v = randn(1, dim);
        C = 0.5 * u ./ abs(v);  % Equation (4): Levy-like coefficient
        newPopPos(2, :) = PopPos(2, :) + C .* (PopPos(2, :) - newPopPos(1, :));
        
        % Carnivore/Omnivore behavior for remaining individuals
        for i = 3:nPop
            u = randn(1, dim);
            v = randn(1, dim);
            C = 0.5 * u ./ abs(v);
            
            r = rand;
            if r < 1/3
                % Equation (6): Herbivore - consumes producer
                newPopPos(i, :) = PopPos(i, :) + C .* (PopPos(i, :) - newPopPos(1, :));
            elseif r < 2/3
                % Equation (7): Carnivore - consumes random herbivore
                randIdx = randi([2, i-1]);
                newPopPos(i, :) = PopPos(i, :) + C .* (PopPos(i, :) - PopPos(randIdx, :));
            else
                % Equation (8): Omnivore - consumes both producer and herbivore
                r2 = rand;
                randIdx = randi([2, i-1]);
                newPopPos(i, :) = PopPos(i, :) + C .* (r2 * (PopPos(i, :) - newPopPos(1, :)) + ...
                                  (1 - r2) * (PopPos(i, :) - PopPos(randIdx, :)));
            end
        end
        
        % Apply boundary constraints and evaluate new positions
        for i = 1:nPop
            newPopPos(i, :) = bound(newPopPos(i, :), ub, lb);
        end
        
        % Evaluate new population after consumption phase
        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);
        
        % Greedy selection: Update if new solution is better
        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopFit(i) = newPopFit(i);
                PopPos(i, :) = newPopPos(i, :);
            end
        end
        
        % Record convergence curve for consumption phase evaluations
        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = min(PopFit);
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Check if we've exceeded maxFE
        if FE >= maxFE
            break;
        end
        
        % ===== DECOMPOSITION PHASE =====
        % Find current best individual
        [~, indOne] = min(PopFit);
        
        % Equation (9): Decomposition - search around the best solution
        for i = 1:nPop
            r3 = rand;
            Ind = round(rand) + 1;  % Randomly choose between 1 or full dimension update
            newPopPos(i, :) = PopPos(indOne, :) + 3 * randn(1, Matr(Ind)) .* ...
                             ((r3 * randi([1, 2]) - 1) * PopPos(indOne, :) - (2 * r3 - 1) * PopPos(i, :));
        end
        
        % Apply boundary constraints
        for i = 1:nPop
            newPopPos(i, :) = bound(newPopPos(i, :), ub, lb);
        end
        
        % Evaluate new population after decomposition phase
        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);
        
        % Greedy selection: Update if new solution is better
        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopPos(i, :) = newPopPos(i, :);
                PopFit(i) = newPopFit(i);
            end
        end
        
        % Record convergence curve for decomposition phase evaluations
        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = min(PopFit);
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Sort population by fitness (descending - worst to best)
        [PopFit, indF] = sort(PopFit, 'descend');
        PopPos = PopPos(indF, :);
        
        % Update global best
        if PopFit(end) < BestF
            BestF = PopFit(end);
            BestX = PopPos(end, :);
        end
        
        Iteration = Iteration + 1;
    end
    
    % Return best solution
    best_fitness = BestF;
    best_solution = BestX;
    
end

%% --- Initialization Function ---
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);  % Number of boundaries
    
    % If the boundaries of all variables are equal
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    
    % If each variable has a different lb and ub
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

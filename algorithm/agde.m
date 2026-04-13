% ----------------------------------------------------------------------- %
% Adaptive Guided Differential Evolution (AGDE) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                 % Population size
%   F = 0.1 + 0.9*rand      % Adaptive scaling factor
%   CR = Adaptive           % Crossover rate (two pools: 0.05-0.15 or 0.9-1.0)
%   
% Algorithm Concept:
%   - Variant of Differential Evolution with adaptive crossover rate
%   - Uses two CR pools: low (0.05-0.15) and high (0.9-1.0)
%   - Pool selection probability adapts based on success rate
%   - Mutation uses best, worst, and middle individuals
%   - Balances exploration and exploitation through adaptive mechanism
%
% Reference:
% Mohamed, A.W., Hadi, A.A. and Jambi, K.M. (2019),
% Novel mutation strategy for enhancing SHADE and LSHADE algorithms 
% for global numerical optimization,
% Swarm and Evolutionary Computation, 50, 100455.
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = agde(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % AGDE Parameters
    NP = 50;                      % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, NP, dim);
    fitness_history = zeros(history_size, NP);
    history_index = 1;
    
    % Initialize population
    Pop = initialization(NP, dim, ub, lb);
    
    % Evaluate initial population
    [Fit, FE] = calculate_fitness(Pop', problem, FE);
    
    % Find initial best
    [best_fitness_current, iBest] = min(Fit);
    best_solution_current = Pop(iBest, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:NP
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Pop, Fit, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Adaptive CR parameters
    NW = [0.5, 0.5];  % Initial weights for CR pools
    
    % Main loop
    Max_gen = ceil((maxFE - NP) / NP);
    g = 1;
    
    while FE < maxFE && g <= Max_gen
        
        CrPriods_Index = zeros(1, NP);
        Sr = zeros(1, 2);
        CrPriods_Count = zeros(1, 2);
        
        for j = 1:NP
            % Adaptive CR Rule
            Ali = rand;
            if g <= 1
                if Ali <= 0.5
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            else
                if Ali <= NW(1)
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            end
            CrPriods_Count(CrPriods_Index(j)) = CrPriods_Count(CrPriods_Index(j)) + 1;
            
            % Sort population by fitness
            [~, in] = sort(Fit, 'ascend');
            
            % Select indices from best, worst, and middle groups
            AA = in(1:5);           % Best 5
            BB = in(46:50);         % Worst 5
            CC = in(6:45);          % Middle 40
            
            % Choose random individuals from each group
            r1 = AA(randi(length(AA)));
            r2 = BB(randi(length(BB)));
            r3 = CC(randi(length(CC)));
            
            % Adaptive scaling factor
            F = 0.1 + 0.9 * rand;
            
            % Mutation and Crossover
            X = zeros(1, dim);
            Rnd = randi(dim);
            for i = 1:dim
                if rand < CR || Rnd == i
                    X(i) = Pop(r3, i) + F * (Pop(r1, i) - Pop(r2, i));
                else
                    X(i) = Pop(j, i);
                end
            end
            
            % Boundary handling
            X = bound(X, ub, lb);
            
            % Evaluate trial vector
            [f_trial, FE] = calculate_fitness(X', problem, FE);
            
            % Selection
            if f_trial <= Fit(j)
                Sr(CrPriods_Index(j)) = Sr(CrPriods_Index(j)) + 1;
                Pop(j, :) = X;
                Fit(j) = f_trial;
                
                if f_trial <= best_fitness_current
                    best_fitness_current = f_trial;
                    best_solution_current = X;
                end
            end
            
            % Record convergence curve
            if FE <= maxFE
                curve(FE) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pop, Fit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Update CR pool weights
        CrPriods_Count(CrPriods_Count == 0) = 0.0001;
        Sr = Sr ./ CrPriods_Count;
        
        if sum(Sr) == 0
            W = [0.5, 0.5];
        else
            W = Sr / sum(Sr);
        end
        
        NW = (NW * (g - 1) + W) / g;
        g = g + 1;
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
    % Random reinitialization for out-of-bound values
    for i = 1:length(a)
        if a(i) < lb(min(i, length(lb))) || a(i) > ub(min(i, length(ub)))
            lb_i = lb(min(i, length(lb)));
            ub_i = ub(min(i, length(ub)));
            a(i) = lb_i + (ub_i - lb_i) * rand;
        end
    end
end


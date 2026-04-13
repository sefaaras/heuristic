% ----------------------------------------------------------------------- %
% Teaching-Learning-based Artificial Bee Colony (TLABC) Algorithm
% for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50           % Population size
%   limit = 200            % Abandonment limit for scout bee phase
%   CR = 0.5               % Crossover rate for diversity learning
%   TF = round(1+rand)     % Teaching factor (1 or 2)
%   F = rand               % Scale factor for differential evolution
%
% Algorithm Phases:
%   1. Teaching-based employed bee phase  % Combines TLBO teacher phase with DE
%   2. Learning-based onlooker bee phase  % Probabilistic selection and learning
%   3. Generalized oppositional scout bee % Scout with opposition-based learning
%
% Reference:
% Teaching-Learning-Based Artificial Bee Colony (TLABC) Algorithm
% Applied Energy, 2018
% https://www.sciencedirect.com/science/article/pii/S0306261917318391
% https://doi.org/10.1016/j.apenergy.2017.12.115
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = tlabc(problem)

    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    low = problem.lb;              % Lower bounds
    up = problem.ub;               % Upper bounds
    maxIteration = problem.maxFe;  % Maximum function evaluations
    
    % Algorithm parameters
    popsize = 50;
    trial = zeros(1, popsize);
    limit = 200;
    CR = 0.5;
    
    FE = 0;                         % Function Evaluation Counter
    curve = zeros(1, maxIteration); % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;           % Fixed history size
    sampling_interval = max(1, floor(maxIteration / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, popsize, dim);     % Store population at sampled FEs
    fitness_history = zeros(history_size, popsize);             % Store fitness values at sampled FEs
    history_index = 1;              % Current index in history arrays
    
    % Initialize population
    X = repmat(low, popsize, 1) + rand(popsize, dim) .* (repmat(up - low, popsize, 1));
    
    % Calculate initial fitness
    [val_X, FE] = calculate_fitness(X', problem, FE);
    
    [val_gBest, min_index] = min(val_X);
    gBest = X(min_index(1), :);
    
    % Record initial best fitness and store history
    for eval_count = 1:popsize
        [current_best, ~] = min(val_X);
        if eval_count <= maxIteration
            curve(eval_count) = current_best;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, val_X, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    while FE < maxIteration
        % ============ Teaching-based employed bee phase ============
        for i = 1:popsize
            [~, sortIndex] = sort(val_X);
            mean_result = mean(X);        % Calculate the mean
            Best = X(sortIndex(1), :);    % Identify the teacher
            TF = round(1 + rand * (1));
            Xi = X(i, :) + (Best - TF * mean_result) .* rand(1, dim);
            
            % Diversity learning
            r = generateR(popsize, i);
            F = rand;
            V = X(r(1), :) + F * (X(r(2), :) - X(r(3), :));
            flag = (rand(1, dim) <= CR);
            Xi(flag) = V(flag);
            Xi = boundary_repair(Xi, low, up, 'reflect');
            
            % Accept or Reject
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);
            
            if val_Xi < val_X(i)
                val_X(i) = val_Xi;
                X(i, :) = Xi;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
            
            % Record convergence curve and store history
            if FE <= maxIteration
                [current_best, ~] = min(val_X);
                curve(FE) = current_best;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, val_X, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            
            if FE >= maxIteration
                break;
            end
        end
        
        if FE >= maxIteration
            break;
        end
        
        % ============ Learning-based onlooker bee phase ============
        Fitness = calculateFitnessABC(val_X);
        prob = Fitness / sum(Fitness);
        cum_prob = cumsum(prob);
        
        for k = 1:popsize
            i = find(rand < cum_prob, 1);
            j = randi(popsize);
            while j == i
                j = randi(popsize);
            end
            
            if val_X(i) < val_X(j)
                Xi = X(i, :) + rand(1, dim) .* (X(i, :) - X(j, :));
            else
                Xi = X(i, :) + rand(1, dim) .* (X(j, :) - X(i, :));
            end
            Xi = boundary_repair(Xi, low, up, 'reflect');
            
            % Accept or Reject
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);
            
            if val_Xi < val_X(i)
                val_X(i) = val_Xi;
                X(i, :) = Xi;
            end
            
            % Record convergence curve and store history
            if FE <= maxIteration
                [current_best, ~] = min(val_X);
                curve(FE) = current_best;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, val_X, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            
            if FE >= maxIteration
                break;
            end
        end
        
        if FE >= maxIteration
            break;
        end
        
        % ============ Generalized oppositional scout bee phase ============
        ind = find(trial == max(trial));
        ind = ind(1);
        
        if (trial(ind) > limit)
            trial(ind) = 0;
            sol = (up - low) .* rand(1, dim) + low;
            solGOBL = (max(X) + min(X)) * rand - X(ind, :);
            newSol = [sol; solGOBL];
            newSol = boundary_repair(newSol, low, up, 'random');
            
            [val_sol, FE] = calculate_fitness(newSol', problem, FE);
            
            [~, min_index] = min(val_sol);
            X(ind, :) = newSol(min_index(1), :);
            val_X(ind) = val_sol(min_index(1));
            
            % Record convergence curve for scout phase evaluations
            for scout_idx = 1:2
                eval_count = FE - 2 + scout_idx;
                if eval_count <= maxIteration
                    [current_best, ~] = min(val_X);
                    curve(eval_count) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, X, val_X, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
        end
        
        % The best food source is memorized
        if min(val_X) < val_gBest
            [val_gBest, min_index] = min(val_X);
            gBest = X(min_index(1), :);
        end
    end
    
    % Final best solution
    best_fitness = val_gBest;
    best_solution = gBest;

end

function r = generateR(popsize, i)
    % Generate index r = [r1 r2 r3 r4 r5]
    r1 = randi(popsize);
    while r1 == i
        r1 = randi(popsize);
    end
    r2 = randi(popsize);
    while r2 == r1 || r2 == i
        r2 = randi(popsize);
    end
    r3 = randi(popsize);
    while r3 == r2 || r3 == r1 || r3 == i
        r3 = randi(popsize);
    end
    r4 = randi(popsize);
    while r4 == r3 || r4 == r2 || r4 == r1 || r4 == i
        r4 = randi(popsize);
    end
    r5 = randi(popsize);
    while r5 == r4 || r5 == r3 || r5 == r2 || r5 == r1 || r5 == i
        r5 = randi(popsize);
    end
    r = [r1 r2 r3 r4 r5];
end

function u = boundary_repair(v, low, up, str)
    [NP, D] = size(v);
    u = v;
    
    if strcmp(str, 'absorb')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j)
                    u(i, j) = up(j);
                elseif v(i, j) < low(j)
                    u(i, j) = low(j);
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
    
    if strcmp(str, 'random')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j) || v(i, j) < low(j)
                    u(i, j) = low(j) + rand * (up(j) - low(j));
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
    
    if strcmp(str, 'reflect')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j)
                    u(i, j) = max(2 * up(j) - v(i, j), low(j));
                elseif v(i, j) < low(j)
                    u(i, j) = min(2 * low(j) - v(i, j), up(j));
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
end

function fFitness = calculateFitnessABC(fObjV)
    fFitness = zeros(size(fObjV));
    ind = find(fObjV >= 0);
    fFitness(ind) = 1 ./ (fObjV(ind) + 1);
    ind = find(fObjV < 0);
    fFitness(ind) = 1 + abs(fObjV(ind));
end


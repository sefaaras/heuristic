% ----------------------------------------------------------------------- %
% Chaos Game Optimization (CGO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Seed_Number = 25         % Population size (number of seeds)
%   
% Algorithm Concept:
%   - Inspired by chaos game theory and fractal patterns
%   - Uses random groups and mean positions for exploration
%   - Generates 4 new solutions per seed using different strategies:
%     1. Current seed influenced by best and group mean
%     2. Best seed influenced by group mean and current seed
%     3. Group mean influenced by best and current seed
%     4. Random exploration
%   - Tournament selection keeps best solutions
%
% Reference:
% Talatahari, S., & Azizi, M. (2021).
% Chaos Game Optimization: a novel metaheuristic algorithm.
% Artificial Intelligence Review, 54(2), 917-1004.
% https://doi.org/10.1007/s10462-020-09867-w
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = cgo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    Seed_Number = 25;             % Population size (number of seeds)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, Seed_Number, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, Seed_Number);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize population
    Seed = initialization(Seed_Number, dim, ub, lb);
    
    % Evaluate initial population
    [Fun_eval, FE] = calculate_fitness(Seed', problem, FE);
    Fun_eval = Fun_eval(:);  % Ensure column vector
    
    % Find initial best
    [BestFitness, idbest] = min(Fun_eval);
    BestSeed = Seed(idbest, :);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:Seed_Number
        curve(eval_count) = BestFitness;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Seed, Fun_eval', population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Preallocate for new seeds
    NewSeed = zeros(4, dim);
    Alfa = zeros(4, dim);
    
    % Main loop
    while FE < maxFE
        
        for i = 1:Seed_Number
            
            if FE >= maxFE
                break;
            end
            
            % Update the best Seed
            [~, idbest] = min(Fun_eval);
            BestSeed = Seed(idbest, :);
            
            %% Generate New Solutions
            % Random Numbers
            I = randi([1, 2], 1, 12);  % Beta and Gamma
            Ir = randi([0, 1], 1, 5);
            
            % Random Groups
            RandGroupNumber = randperm(Seed_Number, 1);
            RandGroup = randperm(Seed_Number, RandGroupNumber);
            
            % Mean of Random Group
            if length(RandGroup) ~= 1
                MeanGroup = mean(Seed(RandGroup, :), 1);
            else
                MeanGroup = Seed(RandGroup(1), :);
            end
            
            % Alpha coefficients for different strategies
            Alfa(1, :) = rand(1, dim);
            Alfa(2, :) = 2 * rand(1, dim) - 1;
            Alfa(3, :) = Ir(1) * rand(1, dim) + 1;
            Alfa(4, :) = Ir(2) * rand(1, dim) + (~Ir(2));
            
            % Select random alpha for each new seed
            ii = randi([1, 4], 1, 3);
            SelectedAlfa = Alfa(ii, :);
            
            % Generate 4 new seeds using different strategies
            % Strategy 1: Current seed + Alpha * (Best - MeanGroup)
            NewSeed(1, :) = Seed(i, :) + SelectedAlfa(1, :) .* (I(1) * BestSeed - I(2) * MeanGroup);
            
            % Strategy 2: Best + Alpha * (MeanGroup - Current)
            NewSeed(2, :) = BestSeed + SelectedAlfa(2, :) .* (I(3) * MeanGroup - I(4) * Seed(i, :));
            
            % Strategy 3: MeanGroup + Alpha * (Best - Current)
            NewSeed(3, :) = MeanGroup + SelectedAlfa(3, :) .* (I(5) * BestSeed - I(6) * Seed(i, :));
            
            % Strategy 4: Random exploration
            NewSeed(4, :) = lb + rand(1, dim) .* (ub - lb);
            
            % Apply boundary constraints
            for j = 1:4
                NewSeed(j, :) = bound(NewSeed(j, :), ub, lb);
            end
            
            % Evaluate new solutions
            [Fun_evalNew, FE] = calculate_fitness(NewSeed', problem, FE);
            Fun_evalNew = Fun_evalNew(:);  % Ensure column vector
            
            % Add new seeds to population
            Seed = [Seed; NewSeed];
            Fun_eval = [Fun_eval; Fun_evalNew];
            
            % Update global best
            [minFit, minIdx] = min(Fun_eval);
            if minFit < BestFitness
                BestFitness = minFit;
                BestSeed = Seed(minIdx, :);
            end
            
            % Record convergence curve for each evaluation
            for eval_idx = 1:4
                eval_count = FE - 4 + eval_idx;
                if eval_count <= maxFE
                    curve(eval_count) = BestFitness;
                    % Use current top Seed_Number for history
                    [~, topIdx] = sort(Fun_eval);
                    topIdx = topIdx(1:min(Seed_Number, length(topIdx)));
                    topSeed = Seed(topIdx, :);
                    topFit = Fun_eval(topIdx)';
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, topSeed, topFit, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
        end
        
        % Sort and keep best Seed_Number solutions
        [Fun_eval, SortOrder] = sort(Fun_eval);
        Seed = Seed(SortOrder, :);
        Seed = Seed(1:Seed_Number, :);
        Fun_eval = Fun_eval(1:Seed_Number);
        
    end
    
    % Return best solution
    [best_fitness, idbest] = min(Fun_eval);
    best_solution = Seed(idbest, :);
    
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

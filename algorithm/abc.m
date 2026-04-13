% ----------------------------------------------------------------------- %
% Artificial Bee Colony (ABC) Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                     % Colony size (employed + onlooker bees)
%   FoodNumber = NP/2           % Number of food sources
%   limit = FoodNumber * D      % Abandonment limit
%
% Algorithm Concept:
%   - Three types of bees: employed, onlooker, and scout
%   - Employed bees: search around food sources
%   - Onlooker bees: select food sources based on probability
%   - Scout bees: abandon exhausted sources and explore new ones
%
% Reference:
% Dervis Karaboga, Bahriye Basturk,
% A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm,
% Journal of Global Optimization 39 (2007) 459-471
% https://doi.org/10.1007/s10898-007-9149-x
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = abc(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    NP = 50;                              % Colony size
    FoodNumber = NP / 2;                  % Number of food sources
    limit = FoodNumber * dim;             % Abandonment limit
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, FoodNumber, dim);
    fitness_history = zeros(history_size, FoodNumber);
    history_index = 1;
    
    % Initialize food sources
    Foods = initialization(FoodNumber, dim, ub, lb);
    
    % Evaluate initial population
    [ObjVal, FE] = calculate_fitness(Foods', problem, FE);
    Fitness = calculateFitness_ABC(ObjVal);
    
    % Reset trial counters
    trial = zeros(1, FoodNumber);
    
    % Find the best food source
    [GlobalMin, BestInd] = min(ObjVal);
    GlobalParams = Foods(BestInd, :);
    
    % Record initial population
    for eval_count = 1:FoodNumber
        curve(eval_count) = GlobalMin;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Foods, ObjVal, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    while FE < maxFE
        
        %%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1:FoodNumber
            if FE >= maxFE
                break;
            end
            
            % A randomly chosen solution is used in producing a mutant solution
            neighbour = randi(FoodNumber);
            while neighbour == i
                neighbour = randi(FoodNumber);
            end
            
            % Produce new solution for all dimensions: v_{ij} = x_{ij} + phi_{ij} * (x_{ij} - x_{kj})
            sol = Foods(i, :) + (Foods(i, :) - Foods(neighbour, :)) .* (rand(1, dim) - 0.5) * 2;
            
            % Apply boundary constraints
            sol = bound(sol, ub, lb);
            
            % Evaluate new solution
            [ObjValSol, FE] = calculate_fitness(sol', problem, FE);
            FitnessSol = calculateFitness_ABC(ObjValSol);
            
            % Greedy selection
            if FitnessSol > Fitness(i)
                Foods(i, :) = sol;
                Fitness(i) = FitnessSol;
                ObjVal(i) = ObjValSol;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
            
            % Record convergence curve and history
            if FE <= maxFE
                curve(FE) = GlobalMin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Foods, ObjVal, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate probability values for onlooker bees
        prob = (0.9 .* Fitness ./ max(Fitness)) + 0.1;
        
        %%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        i = 1;
        t = 0;
        while t < FoodNumber && FE < maxFE
            if rand < prob(i)
                t = t + 1;
                
                % A randomly chosen solution is used in producing a mutant solution
                neighbour = randi(FoodNumber);
                while neighbour == i
                    neighbour = randi(FoodNumber);
                end
                
                % Produce new solution for all dimensions
                sol = Foods(i, :) + (Foods(i, :) - Foods(neighbour, :)) .* (rand(1, dim) - 0.5) * 2;
                
                % Apply boundary constraints
                sol = bound(sol, ub, lb);
                
                % Evaluate new solution
                [ObjValSol, FE] = calculate_fitness(sol', problem, FE);
                FitnessSol = calculateFitness_ABC(ObjValSol);
                
                % Greedy selection
                if FitnessSol > Fitness(i)
                    Foods(i, :) = sol;
                    Fitness(i) = FitnessSol;
                    ObjVal(i) = ObjValSol;
                    trial(i) = 0;
                else
                    trial(i) = trial(i) + 1;
                end
                
                % Record convergence curve and history
                if FE <= maxFE
                    curve(FE) = GlobalMin;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Foods, ObjVal, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
            
            i = i + 1;
            if i == FoodNumber + 1
                i = 1;
            end
        end
        
        % Update the best food source
        [minVal, minInd] = min(ObjVal);
        if minVal < GlobalMin
            GlobalMin = minVal;
            GlobalParams = Foods(minInd, :);
        end
        
        %%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Determine the food source with maximum trial counter
        [maxTrial, maxInd] = max(trial);
        
        if maxTrial > limit && FE < maxFE
            % Abandon the exhausted food source and generate a new one
            trial(maxInd) = 0;
            sol = initialization(1, dim, ub, lb);
            
            % Evaluate new solution
            [ObjValSol, FE] = calculate_fitness(sol', problem, FE);
            FitnessSol = calculateFitness_ABC(ObjValSol);
            
            Foods(maxInd, :) = sol;
            Fitness(maxInd) = FitnessSol;
            ObjVal(maxInd) = ObjValSol;
            
            % Record convergence curve and history
            if FE <= maxFE
                curve(FE) = GlobalMin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Foods, ObjVal, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
    end
    
    % Fill remaining curve values with best fitness
    curve(FE:end) = GlobalMin;
    
    % Return best solution
    best_fitness = GlobalMin;
    best_solution = GlobalParams;
    
end

%% --- Helper Functions ---

function Positions = initialization(popsize, dim, ub, lb)
    Boundary_no = size(ub, 2);
    
    if Boundary_no == 1
        Positions = rand(popsize, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(popsize, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

function fFitness = calculateFitness_ABC(fObjV)
    % ABC-specific fitness calculation
    % Converts objective values to fitness values for probability calculation
    fFitness = zeros(size(fObjV));
    ind = find(fObjV >= 0);
    fFitness(ind) = 1 ./ (fObjV(ind) + 1);
    ind = find(fObjV < 0);
    fFitness(ind) = 1 + abs(fObjV(ind));
end


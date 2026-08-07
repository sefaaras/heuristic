% ----------------------------------------------------------------------- %
% Artificial Bee Colony (ABC)
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
%
% Implementation Note:
%   curve and best_fitness track the best point the run has evaluated. The
%   reference refreshes its global best once per cycle, after the onlooker
%   phase, which leaves a source found mid-cycle unreported and drops whatever
%   the final, partial cycle finds; that per-cycle update fed nothing but the
%   report, so it is replaced rather than kept alongside.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = abc(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    % Algorithm parameters
    NP = 50;                              % Colony size
    FoodNumber = NP / 2;                  % Number of food sources
    limit = FoodNumber * dim;             % Abandonment limit
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
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

    % Best point evaluated so far; GlobalMin only refreshes once per cycle
    bsf_fit = GlobalMin;
    bsf_x = GlobalParams;

    % Record initial population
    for eval_count = 1:FoodNumber
        curve(eval_count) = bsf_fit;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Foods, ObjVal, population_history, fitness_history, ...
            history_index, maxFE);
    end
    
    % Main loop
    while FE < maxFE
        
        % Employed bee phase
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
            if ObjValSol < bsf_fit
                bsf_fit = ObjValSol;
                bsf_x = sol;
            end
            
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
                curve(FE) = bsf_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Foods, ObjVal, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        % Onlooker selection probabilities
        prob = (0.9 .* Fitness ./ max(Fitness)) + 0.1;
        
        % Onlooker bee phase
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
                if ObjValSol < bsf_fit
                    bsf_fit = ObjValSol;
                    bsf_x = sol;
                end
                
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
                    curve(FE) = bsf_fit;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Foods, ObjVal, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
            
            i = i + 1;
            if i == FoodNumber + 1
                i = 1;
            end
        end
        
        % Scout bee phase: the food source with the highest trial counter
        [maxTrial, maxInd] = max(trial);
        
        if maxTrial > limit && FE < maxFE
            % Abandon the exhausted food source and generate a new one
            trial(maxInd) = 0;
            sol = initialization(1, dim, ub, lb);
            
            % Evaluate new solution
            [ObjValSol, FE] = calculate_fitness(sol', problem, FE);
            FitnessSol = calculateFitness_ABC(ObjValSol);
            if ObjValSol < bsf_fit
                bsf_fit = ObjValSol;
                bsf_x = sol;
            end
            
            Foods(maxInd, :) = sol;
            Fitness(maxInd) = FitnessSol;
            ObjVal(maxInd) = ObjValSol;
            
            % Record convergence curve and history
            if FE <= maxFE
                curve(FE) = bsf_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Foods, ObjVal, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
    end
    
    % Fill remaining curve values with best fitness
    curve(FE:end) = bsf_fit;
    
    % Return best solution
    best_fitness = bsf_fit;
    best_solution = bsf_x;
    
end

% Helper Functions

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
    % Objective values mapped to the positive fitness the probabilities need
    fFitness = zeros(size(fObjV));
    ind = find(fObjV >= 0);
    fFitness(ind) = 1 ./ (fObjV(ind) + 1);
    ind = find(fObjV < 0);
    fFitness(ind) = 1 + abs(fObjV(ind));
end


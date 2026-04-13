% ----------------------------------------------------------------------- %
% Atom Search Optimization (ASO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Atom_Num = 50               % Population size (number of atoms)
%   alpha = 50                  % Depth weight coefficient
%   beta = 0.2                  % Multiplier weight coefficient
%
% Algorithm Concept:
%   - Atoms interact through Lennard-Jones potential
%   - Mass of atoms calculated based on fitness
%   - Acceleration computed using interaction forces
%   - Best atom attracts others through potential field
%
% Reference:
% Weiguo Zhao, Liying Wang, Zhenxing Zhang,
% Atom search optimization and its application to solve a hydrogeologic parameter estimation problem,
% Knowledge-Based Systems 163 (2019) 283-304
% https://doi.org/10.1016/j.knosys.2018.08.030
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = aso(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    Atom_Num = 50;                % Population size
    alpha = 50;                   % Depth weight
    beta = 0.2;                   % Multiplier weight
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, Atom_Num, dim);
    fitness_history = zeros(history_size, Atom_Num);
    history_index = 1;
    
    % Initialize positions and velocities of atoms
    Atom_Pop = initialization(Atom_Num, dim, ub, lb);
    Atom_V = initialization(Atom_Num, dim, ub, lb);
    
    % Evaluate initial population
    [Fitness, FE] = calculate_fitness(Atom_Pop', problem, FE);
    
    % Find the best atom
    [best_fitness_val, Index] = min(Fitness);
    X_Best = Atom_Pop(Index, :);
    
    % Record initial population
    for eval_count = 1:Atom_Num
        curve(eval_count) = best_fitness_val;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Atom_Pop, Fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Calculate initial acceleration
    Max_Iteration = ceil((maxFE - Atom_Num) / Atom_Num);
    Iteration = 1;
    Atom_Acc = Acceleration(Atom_Pop, Fitness, Iteration, Max_Iteration, dim, Atom_Num, X_Best, alpha, beta);
    
    % Main iteration loop
    for Iteration = 2:Max_Iteration
        % Update velocity and position
        Atom_V = rand(Atom_Num, dim) .* Atom_V + Atom_Acc;
        Atom_Pop = Atom_Pop + Atom_V;
        
        % Apply boundary constraints
        for i = 1:Atom_Num
            Atom_Pop(i, :) = bound(Atom_Pop(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [Fitness, FE] = calculate_fitness(Atom_Pop', problem, FE);
        
        % Update best solution
        [Min_Fitness, Index] = min(Fitness);
        
        if Min_Fitness < best_fitness_val
            best_fitness_val = Min_Fitness;
            X_Best = Atom_Pop(Index, :);
        else
            % Replace a random atom with best solution
            r = randi(Atom_Num);
            Atom_Pop(r, :) = X_Best;
        end
        
        % Record convergence curve and history
        for eval_idx = 1:Atom_Num
            eval_count = FE - Atom_Num + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = best_fitness_val;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Atom_Pop, Fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Calculate acceleration for next iteration
        Atom_Acc = Acceleration(Atom_Pop, Fitness, Iteration, Max_Iteration, dim, Atom_Num, X_Best, alpha, beta);
        
        % Check if we've reached maxFE
        if FE >= maxFE
            break;
        end
    end
    
    % Return best solution
    best_fitness = best_fitness_val;
    best_solution = X_Best;
    
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

function Acc = Acceleration(Atom_Pop, Fitness, Iteration, Max_Iteration, Dim, Atom_Num, X_Best, alpha, beta)
    % Calculate mass based on fitness
    M = exp(-(Fitness - max(Fitness)) ./ (max(Fitness) - min(Fitness) + eps));
    M = M ./ sum(M);
    
    % Interaction strength (gravity)
    G = exp(-20 * Iteration / Max_Iteration);
    
    % Calculate number of best atoms to consider
    Kbest = Atom_Num - (Atom_Num - 2) * (Iteration / Max_Iteration)^0.5;
    Kbest = floor(Kbest) + 1;
    
    % Sort atoms by mass
    [~, Index_M] = sort(M, 'descend');
    
    % Initialize force and acceleration arrays
    E = zeros(Atom_Num, Dim);
    a = zeros(Atom_Num, Dim);
    
    for i = 1:Atom_Num
        E(i, :) = zeros(1, Dim);
        
        % Calculate mean position of Kbest atoms
        MK = sum(Atom_Pop(Index_M(1:Kbest), :), 1) / Kbest;
        Distance = norm(Atom_Pop(i, :) - MK, 2);
        
        % Calculate interaction force from Kbest atoms
        for ii = 1:Kbest
            j = Index_M(ii);
            
            % Calculate Lennard-Jones potential
            Potential = LJPotential(Atom_Pop(i, :), Atom_Pop(j, :), Iteration, Max_Iteration, Distance);
            
            % Accumulate force
            direction = (Atom_Pop(j, :) - Atom_Pop(i, :)) / (norm(Atom_Pop(i, :) - Atom_Pop(j, :)) + eps);
            E(i, :) = E(i, :) + rand(1, Dim) .* Potential .* direction;
        end
        
        % Add attraction to best solution
        E(i, :) = alpha * E(i, :) + beta * (X_Best - Atom_Pop(i, :));
        
        % Calculate acceleration
        a(i, :) = E(i, :) ./ (M(i) + eps);
    end
    
    % Apply gravity coefficient
    Acc = a .* G;
end

function Potential = LJPotential(Atom1, Atom2, Iteration, Max_Iteration, s)
    % Calculate Lennard-Jones potential
    r = norm(Atom1 - Atom2, 2);
    c = (1 - (Iteration - 1) / Max_Iteration)^3;
    
    % Calculate depth parameter
    rsmin = 1.1 + 0.1 * sin(Iteration / Max_Iteration * pi / 2);
    rsmax = 1.24;
    
    if r / (s + eps) < rsmin
        rs = rsmin;
    elseif r / (s + eps) > rsmax
        rs = rsmax;
    else
        rs = r / (s + eps);
    end
    
    % Lennard-Jones potential formula
    Potential = c * (12 * (-rs)^(-13) - 6 * (-rs)^(-7));
end


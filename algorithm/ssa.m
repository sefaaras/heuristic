% ----------------------------------------------------------------------- %
% Salp Swarm Algorithm (SSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30  % Population size (number of salps)
%   
% Algorithm Concept:
%   - Salp chain: Leader salp guides followers
%   - Leader updates position based on food source
%   - Followers follow the salp in front of them
%
% Reference:
% Seyedali Mirjalili, Amir H. Gandomi, Seyedeh Zahra Mirjalili,
% Shahrzad Saremi, Hossam Faris, Seyed Mohammad Mirjalili,
% Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems,
% Advances in Engineering Software 114 (2017) 163-191
% https://doi.org/10.1016/j.advengsoft.2017.07.002
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ssa(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    N = 30;                       % Population size (number of salps)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize the positions of salps
    SalpPositions = initialization(N, dim, ub, lb);
    
    % Evaluate initial population
    [SalpFitness, FE] = calculate_fitness(SalpPositions', problem, FE);
    
    % Sort salps based on fitness
    [sorted_salps_fitness, sorted_indexes] = sort(SalpFitness);
    Sorted_salps = SalpPositions(sorted_indexes, :);
    
    % Best solution (Food position)
    FoodPosition = Sorted_salps(1, :);
    FoodFitness = sorted_salps_fitness(1);
    
    % Record best fitness for each initial evaluation and store history
    for eval_count = 1:N
        curve(eval_count) = FoodFitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, SalpPositions, SalpFitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iter = ceil((maxFE - N) / N);
    l = 0;  % Loop counter
    
    while FE < maxFE && l < Max_iter
        % Update parameter c1 (Eq. 3.2 in the paper)
        c1 = 2 * exp(-((4 * (l + 1)) / Max_iter)^2);
        
        % Update the position of salps
        for i = 1:N
            if i <= N/2
                % Leader salps (first half)
                for j = 1:dim
                    c2 = rand();
                    c3 = rand();
                    
                    % Eq. (3.1) in the paper
                    if c3 < 0.5
                        SalpPositions(i, j) = FoodPosition(j) + c1 * ((ub(j) - lb(j)) * c2 + lb(j));
                    else
                        SalpPositions(i, j) = FoodPosition(j) - c1 * ((ub(j) - lb(j)) * c2 + lb(j));
                    end
                end
            else
                % Follower salps (second half)
                % Eq. (3.4) in the paper - follow the salp in front
                SalpPositions(i, :) = (SalpPositions(i, :) + SalpPositions(i-1, :)) / 2;
            end
            
            % Apply boundary constraints
            SalpPositions(i, :) = bound(SalpPositions(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [SalpFitness, FE] = calculate_fitness(SalpPositions', problem, FE);
        
        % Update Food Position if there is a better solution
        for i = 1:N
            if SalpFitness(i) < FoodFitness
                FoodPosition = SalpPositions(i, :);
                FoodFitness = SalpFitness(i);
            end
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = FoodFitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, SalpPositions, SalpFitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        l = l + 1;
    end
    
    % Return best solution
    best_fitness = FoodFitness;
    best_solution = FoodPosition;
    
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


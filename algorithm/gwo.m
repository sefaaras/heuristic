% ----------------------------------------------------------------------- %
% Grey Wolf Optimizer (GWO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 30  % Population size (number of wolves)
%   
% Algorithm Concept:
%   - Social hierarchy: Alpha (best), Beta (2nd best), Delta (3rd best)
%   - Hunting behavior: Wolves encircle and attack prey
%   - Position update based on three best solutions
%
% Reference:
% Seyedali Mirjalili, Seyed Mohammad Mirjalili, Andrew Lewis,
% Grey Wolf Optimizer,
% Advances in Engineering Software 69 (2014) 46-61
% http://dx.doi.org/10.1016/j.advengsoft.2013.12.007
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = gwo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    SearchAgents_no = 30;         % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, SearchAgents_no, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, SearchAgents_no);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize alpha, beta, and delta positions and scores
    Alpha_pos = zeros(1, dim);
    Alpha_score = inf;
    
    Beta_pos = zeros(1, dim);
    Beta_score = inf;
    
    Delta_pos = zeros(1, dim);
    Delta_score = inf;
    
    % Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(Positions', problem, FE);
    
    % Find initial alpha, beta, delta
    for i = 1:SearchAgents_no
        if fitness(i) < Alpha_score
            Alpha_score = fitness(i);
            Alpha_pos = Positions(i, :);
        end
        
        if fitness(i) > Alpha_score && fitness(i) < Beta_score
            Beta_score = fitness(i);
            Beta_pos = Positions(i, :);
        end
        
        if fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score
            Delta_score = fitness(i);
            Delta_pos = Positions(i, :);
        end
    end
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:SearchAgents_no
        curve(eval_count) = Alpha_score;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Positions, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iter = ceil((maxFE - SearchAgents_no) / SearchAgents_no);
    l = 0;  % Loop counter
    
    while FE < maxFE && l < Max_iter
        % Linearly decreased parameter a from 2 to 0
        a = 2 - l * (2 / Max_iter);
        
        % Update the Position of search agents including omegas
        for i = 1:SearchAgents_no
            for j = 1:dim
                
                % Alpha influence
                r1 = rand();
                r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;
                
                % Beta influence
                r1 = rand();
                r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;
                
                % Delta influence
                r1 = rand();
                r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;
                
                % Update position
                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
            
            % Apply boundary constraints
            Positions(i, :) = bound(Positions(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        
        % Update Alpha, Beta, and Delta
        for i = 1:SearchAgents_no
            if fitness(i) < Alpha_score
                Alpha_score = fitness(i);
                Alpha_pos = Positions(i, :);
            end
            
            if fitness(i) > Alpha_score && fitness(i) < Beta_score
                Beta_score = fitness(i);
                Beta_pos = Positions(i, :);
            end
            
            if fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score
                Delta_score = fitness(i);
                Delta_pos = Positions(i, :);
            end
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:SearchAgents_no
            eval_count = FE - SearchAgents_no + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = Alpha_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        l = l + 1;
    end
    
    % Return best solution
    best_fitness = Alpha_score;
    best_solution = Alpha_pos;
    
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


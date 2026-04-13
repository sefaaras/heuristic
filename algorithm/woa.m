% ----------------------------------------------------------------------- %
% Whale Optimization Algorithm (WOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 30  % Population size (number of whales)
%   
% Algorithm Concept:
%   - Inspired by humpback whales' bubble-net hunting strategy
%   - Three operators: encircling prey, bubble-net attacking, search for prey
%   - Spiral updating position mimics the helix-shaped movement
%
% Reference:
% Seyedali Mirjalili, Andrew Lewis,
% The Whale Optimization Algorithm,
% Advances in Engineering Software 95 (2016) 51-67
% http://dx.doi.org/10.1016/j.advengsoft.2016.01.008
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = woa(problem)
    
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
    
    % Initialize position vector and score for the leader
    Leader_pos = zeros(1, dim);
    Leader_score = inf;
    
    % Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(Positions', problem, FE);
    
    % Find initial leader
    for i = 1:SearchAgents_no
        if fitness(i) < Leader_score
            Leader_score = fitness(i);
            Leader_pos = Positions(i, :);
        end
    end
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:SearchAgents_no
        curve(eval_count) = Leader_score;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Positions, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    Max_iter = ceil((maxFE - SearchAgents_no) / SearchAgents_no);
    t = 0;  % Loop counter
    
    while FE < maxFE && t < Max_iter
        % a decreases linearly from 2 to 0 (Eq. 2.3)
        a = 2 - t * (2 / Max_iter);
        
        % a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / Max_iter);
        
        % Update the Position of search agents
        for i = 1:SearchAgents_no
            r1 = rand();  % r1 is a random number in [0,1]
            r2 = rand();  % r2 is a random number in [0,1]
            
            A = 2 * a * r1 - a;  % Eq. (2.3) in the paper
            C = 2 * r2;          % Eq. (2.4) in the paper
            
            b = 1;               % Parameters in Eq. (2.5)
            l = (a2 - 1) * rand + 1;  % Parameters in Eq. (2.5)
            
            p = rand();          % p in Eq. (2.6)
            
            for j = 1:dim
                if p < 0.5
                    if abs(A) >= 1
                        % Search for prey (exploration)
                        rand_leader_index = floor(SearchAgents_no * rand() + 1);
                        X_rand = Positions(rand_leader_index, :);
                        D_X_rand = abs(C * X_rand(j) - Positions(i, j));  % Eq. (2.7)
                        Positions(i, j) = X_rand(j) - A * D_X_rand;       % Eq. (2.8)
                    elseif abs(A) < 1
                        % Encircle prey (exploitation)
                        D_Leader = abs(C * Leader_pos(j) - Positions(i, j));  % Eq. (2.1)
                        Positions(i, j) = Leader_pos(j) - A * D_Leader;       % Eq. (2.2)
                    end
                elseif p >= 0.5
                    % Spiral updating position (bubble-net attacking)
                    distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                    Positions(i, j) = distance2Leader * exp(b .* l) .* cos(l .* 2 * pi) + Leader_pos(j);  % Eq. (2.5)
                end
            end
            
            % Apply boundary constraints
            Positions(i, :) = bound(Positions(i, :), ub, lb);
        end
        
        % Evaluate new positions
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        
        % Update the leader
        for i = 1:SearchAgents_no
            if fitness(i) < Leader_score
                Leader_score = fitness(i);
                Leader_pos = Positions(i, :);
            end
        end
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:SearchAgents_no
            eval_count = FE - SearchAgents_no + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = Leader_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        t = t + 1;
    end
    
    % Return best solution
    best_fitness = Leader_score;
    best_solution = Leader_pos;
    
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


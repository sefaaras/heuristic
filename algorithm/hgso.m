% ----------------------------------------------------------------------- %
% Henry Gas Solubility Optimization (HGSO) Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP = 50                      % Population size (number of gases)
%   nTypes = 5                   % Number of gas types (clusters)
%   l1, l2, l3                   % Constants in Eq.(7)
%   alpha, beta                  % Constants in Eq.(10)
%   M1, M2                       % Constants in Eq.(11) for worst agents
%
% Algorithm Concept:
%   - Based on Henry's law of gas solubility
%   - Population divided into clusters with different Henry's constants
%   - Uses gas solubility behavior for optimization
%
% Reference:
% Fatemeh Ahmadi Hashim, Essam H. Houssein, Maimonah S. Mabrouk, 
% Walid Al-Atabany, Seyedali Mirjalili,
% Henry gas solubility optimization: A novel physics-based algorithm,
% Future Generation Computer Systems, Volume 101, 2019, Pages 646-667
% DOI: https://doi.org/10.1016/j.future.2019.07.015
% https://www.sciencedirect.com/science/article/pii/S0167739X19306557
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = hgso(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    nP = 50;                      % Population size (number of gases)
    nTypes = 5;                   % Number of gas types (clusters)
    nPerType = nP / nTypes;       % Number of gases per type
    
    % Constants in Eq.(7)
    l1 = 5E-03;
    l2 = 100;
    l3 = 1E-02;
    
    % Constants in Eq.(10)
    alpha = 1;
    beta = 1;
    
    % Constants in Eq.(11)
    M1 = 0.1;
    M2 = 0.2;
    
    MaxIt = floor(maxFE / nP);    % Maximum iterations
    
    FE = 0;                       % Function Evaluation Counter
    curve = zeros(1, maxFE);      % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nP, dim);
    fitness_history = zeros(history_size, nP);
    history_index = 1;
    
    % Parameters setting in Eq.(7)
    K = l1 * rand(nTypes, 1);
    P = l2 * rand(nP, 1);
    C = l3 * rand(nTypes, 1);
    
    % Initialize population
    X = initialization(nP, dim, ub, lb);
    
    % Evaluate initial population
    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:)';  % Ensure row vector
    
    % Create groups (divide population into clusters)
    Group = cell(1, nTypes);
    GroupFitness = cell(1, nTypes);
    idx = 1;
    for j = 1:nTypes
        Group{j} = X(idx:idx + nPerType - 1, :);
        GroupFitness{j} = Cost(idx:idx + nPerType - 1);
        idx = idx + nPerType;
    end
    
    % Find best in each group and global best
    best_fit = zeros(1, nTypes);
    best_pos = cell(1, nTypes);
    for i = 1:nTypes
        [best_fit(i), best_idx] = min(GroupFitness{i});
        best_pos{i} = Group{i}(best_idx, :);
    end
    
    [Gbest, gbest_idx] = min(best_fit);
    Xbest = best_pos{gbest_idx};
    
    % Record initial population
    for eval_count = 1:nP
        if eval_count <= maxFE
            curve(eval_count) = Gbest;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Cost, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    %% Main Loop
    it = 0;
    while FE < maxFE
        it = it + 1;
        
        % Update variables (temperature and solubility)
        T = exp(-it / MaxIt);
        T0 = 298.15;
        S = zeros(nP, 1);
        
        idx = 1;
        for j = 1:nTypes
            K(j) = K(j) * exp(-C(j) * (1/T - 1/T0));
            S(idx:idx + nPerType - 1) = P(idx:idx + nPerType - 1) * K(j);
            idx = idx + nPerType;
        end
        
        % Update positions
        vec_flag = [1, -1];
        GroupNew = cell(1, nTypes);
        
        for i = 1:nTypes
            GroupNew{i} = Group{i};
            for j = 1:nPerType
                gamma = beta * exp(-(Gbest + 0.05) / (GroupFitness{i}(j) + 0.05));
                flag_index = floor(2 * rand() + 1);
                var_flag = vec_flag(flag_index);
                
                S_idx = (i - 1) * nPerType + j;
                for k = 1:dim
                    GroupNew{i}(j, k) = Group{i}(j, k) + var_flag * rand * gamma * (best_pos{i}(k) - Group{i}(j, k)) + ...
                                        rand * alpha * var_flag * (S(S_idx) * Xbest(k) - Group{i}(j, k));
                end
            end
        end
        
        % Check boundaries
        for j = 1:nTypes
            GroupNew{j} = bound(GroupNew{j}, ub, lb);
        end
        
        % Evaluate new positions and update groups
        for i = 1:nTypes
            if FE >= maxFE
                break;
            end
            
            % Evaluate new positions
            [newFitness, FE] = calculate_fitness(GroupNew{i}', problem, FE);
            newFitness = newFitness(:)';
            
            % Greedy selection
            for j = 1:nPerType
                if newFitness(j) < GroupFitness{i}(j)
                    Group{i}(j, :) = GroupNew{i}(j, :);
                    GroupFitness{i}(j) = newFitness(j);
                end
            end
            
            % Update best in group
            [best_fit(i), best_idx] = min(GroupFitness{i});
            best_pos{i} = Group{i}(best_idx, :);
            
            % Handle worst agents (Eq. 11)
            Group{i} = worst_agents(Group{i}, GroupFitness{i}, M1, M2, dim, ub, lb, nPerType);
        end
        
        % Update global best
        [Ybest, idx_best] = min(best_fit);
        if Ybest < Gbest
            Gbest = Ybest;
            Xbest = best_pos{idx_best};
        end
        
        % Reconstruct full population for history recording
        X_full = zeros(nP, dim);
        Cost_full = zeros(1, nP);
        idx = 1;
        for j = 1:nTypes
            X_full(idx:idx + nPerType - 1, :) = Group{j};
            Cost_full(idx:idx + nPerType - 1) = GroupFitness{j};
            idx = idx + nPerType;
        end
        
        % Record convergence curve and history
        if FE <= maxFE
            curve(FE) = Gbest;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X_full, Cost_full, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    % Fill remaining curve values with best fitness
    curve(FE:end) = Gbest;
    
    % Return best solution
    best_fitness = Gbest;
    best_solution = Xbest;
    
end

%% --- Helper Functions ---

% Initialize population
function Positions = initialization(popsize, dim, ub, lb)
    Boundary_no = size(ub, 2);
    
    if Boundary_no == 1
        Positions = rand(popsize, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(popsize, dim);
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(popsize, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

% Boundary constraint handling
function a = bound(a, ub, lb)
    for i = 1:size(a, 1)
        a(i, a(i, :) > ub) = ub(a(i, :) > ub);
        a(i, a(i, :) < lb) = lb(a(i, :) < lb);
    end
end

% Handle worst agents (Eq. 11)
function X = worst_agents(X, fitness, M1, M2, dim, ub, lb, nPerType)
    % Rank and select number of worst agents
    [~, X_index] = sort(fitness, 'descend');
    M1N = M1 * nPerType;
    M2N = M2 * nPerType;
    Nw = round((M2N - M1N) * rand() + M1N);
    
    for k = 1:Nw
        if Nw >= 1 && k <= length(X_index)
            X(X_index(k), :) = lb + rand(1, dim) .* (ub - lb);
        end
    end
end

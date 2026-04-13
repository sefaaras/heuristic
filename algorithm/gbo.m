% ----------------------------------------------------------------------- %
% Gradient-Based Optimizer (GBO) Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP = 50                      % Population size
%   pr = 0.5                     % Probability Parameter
%   beta = adaptive              % Movement parameter (Eq. 14.2)
%   alpha = adaptive             % Movement parameter (Eq. 14.1)
%
% Algorithm Concept:
%   - Uses gradient-based search rules for optimization
%   - Combines global and local search strategies
%   - Includes Local Escaping Operator (LEO)
%
% Reference:
% Iman Ahmadianfar, Omid Bozorg-Haddad, Xuefeng Chu,
% Gradient-Based Optimizer: A New Metaheuristic Optimization Algorithm,
% Information Sciences, 2020
% DOI: https://doi.org/10.1016/j.ins.2020.06.037
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = gbo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    nP = 50;                      % Population size
    pr = 0.5;                     % Probability Parameter
    MaxIt = floor(maxFE / nP);    % Maximum iterations
    
    FE = 0;                       % Function Evaluation Counter
    curve = zeros(1, maxFE);      % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nP, dim);
    fitness_history = zeros(history_size, nP);
    history_index = 1;
    
    % Initialize population
    X = initialization(nP, dim, ub, lb);
    
    % Evaluate initial population
    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:)';  % Ensure row vector
    
    % Sort and find best/worst
    [~, Ind] = sort(Cost);     
    Best_Cost = Cost(Ind(1));        % Determine the value of Best Fitness
    Best_X = X(Ind(1), :);
    Worst_Cost = Cost(Ind(end));     % Determine the value of Worst Fitness
    Worst_X = X(Ind(end), :);
    
    % Record initial population
    for eval_count = 1:nP
        if eval_count <= maxFE
            curve(eval_count) = Best_Cost;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Cost, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    %% Main Loop
    it = 0;
    while FE < maxFE
        it = it + 1;
        
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / MaxIt)^3)^2;              % Eq.(14.2)
        alpha = abs(beta .* sin((3*pi/2 + sin(3*pi/2 * beta))));        % Eq.(14.1)
        
        for i = 1:nP
            if FE >= maxFE
                break;
            end
            
            % Four positions randomly selected from population
            A1 = randperm(nP, 4);
            r1 = A1(1); r2 = A1(2);   
            r3 = A1(3); r4 = A1(4);        
            
            % Average of four positions randomly selected from population
            Xm = (X(r1, :) + X(r2, :) + X(r3, :) + X(r4, :)) / 4;
            
            ro = alpha .* (2 * rand - 1);
            ro1 = alpha .* (2 * rand - 1);        
            eps = 5e-3 * rand;                                           % Randomization Epsilon
            
            % Direction of Movement Eq.(18)
            DM = rand .* ro .* (Best_X - X(r1, :));
            Flag = 1;
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X(i, :), X(r1, :), DM, eps, Xm, Flag);      
            DM = rand .* ro .* (Best_X - X(r1, :));
            X1 = X(i, :) - GSR + DM;                                     % Eq.(25)
            
            DM = rand .* ro .* (X(r1, :) - X(r2, :));
            Flag = 2;
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X(i, :), X(r1, :), DM, eps, Xm, Flag); 
            DM = rand .* ro .* (X(r1, :) - X(r2, :));
            X2 = Best_X - GSR + DM;                                      % Eq.(26)            
            
            Xnew = zeros(1, dim);
            for j = 1:dim                                                  
                ro = alpha .* (2 * rand - 1);                       
                X3 = X(i, j) - ro .* (X2(j) - X1(j));           
                ra = rand; rb = rand;
                Xnew(j) = ra .* (rb .* X1(j) + (1 - rb) .* X2(j)) + (1 - ra) .* X3;  % Eq.(27)          
            end
            
            % Local escaping operator (LEO) Eq.(28)
            if rand < pr           
                k = randi(nP);
                f1 = -1 + 2 * rand();
                f2 = -1 + 2 * rand();         
                ro = alpha .* (2 * rand - 1);
                Xk = lb + (ub - lb) .* rand(1, dim);                      % Eq.(28.8)
                
                L1 = rand < 0.5;
                u1 = L1 * 2 * rand + (1 - L1) * 1;
                u2 = L1 * rand + (1 - L1) * 1;
                u3 = L1 * rand + (1 - L1) * 1;                                    
                L2 = rand < 0.5;            
                Xp = (1 - L2) .* X(k, :) + L2 .* Xk;                      % Eq.(28.7)
                                                             
                if u1 < 0.5
                    Xnew = Xnew + f1 .* (u1 .* Best_X - u2 .* Xp) + f2 .* ro .* (u3 .* (X2 - X1) + u2 .* (X(r1, :) - X(r2, :))) / 2;     
                else
                    Xnew = Best_X + f1 .* (u1 .* Best_X - u2 .* Xp) + f2 .* ro .* (u3 .* (X2 - X1) + u2 .* (X(r1, :) - X(r2, :))) / 2;   
                end
            end
            
            % Check if solutions go outside the search space and bring them back
            Xnew = bound(Xnew, ub, lb);
            
            % Evaluate new solution
            [Xnew_Cost, FE] = calculate_fitness(Xnew', problem, FE);
            
            % Update the Best Position        
            if Xnew_Cost < Cost(i)
                X(i, :) = Xnew;
                Cost(i) = Xnew_Cost;
                if Cost(i) < Best_Cost
                    Best_X = X(i, :);
                    Best_Cost = Cost(i);
                end            
            end
            
            % Update the Worst Position 
            if Cost(i) > Worst_Cost
                Worst_X = X(i, :);
                Worst_Cost = Cost(i);
            end
            
            % Record convergence curve and history
            if FE <= maxFE
                curve(FE) = Best_Cost;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Cost, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end
    
    % Fill remaining curve values with best fitness
    curve(FE:end) = Best_Cost;
    
    % Return best solution
    best_fitness = Best_Cost;
    best_solution = Best_X;
    
end

%% --- Helper Functions ---

% Gradient Search Rule
function GSR = GradientSearchRule(ro1, Best_X, Worst_X, X, Xr1, DM, eps, Xm, Flag)
    nV = size(X, 2);
    Delta = 2 .* rand .* abs(Xm - X);                            % Eq.(16.2)
    Step = ((Best_X - Xr1) + Delta) / 2;                         % Eq.(16.1)
    DelX = rand(1, nV) .* abs(Step);                             % Eq.(16)
    
    GSR = randn .* ro1 .* (2 * DelX .* X) ./ (Best_X - Worst_X + eps);  % Gradient search rule Eq.(15)
    if Flag == 1
        Xs = X - GSR + DM;                                       % Eq.(21)
    else
        Xs = Best_X - GSR + DM;
    end    
    yp = rand .* (0.5 * (Xs + X) + rand .* DelX);                % Eq.(22.6)
    yq = rand .* (0.5 * (Xs + X) - rand .* DelX);                % Eq.(22.7)
    GSR = randn .* ro1 .* (2 * DelX .* X) ./ (yp - yq + eps);    % Eq.(23)   
end

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
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

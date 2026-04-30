% ----------------------------------------------------------------------- %
% Fitness-Distance Balance Stochastic Fractal Search (FDB-SFS) Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Population size (Start_Point)
%   MDN = 1                 % Maximum Diffusion Number
%   Walk = 1                % Walk probability (Gaussian walk selection)
%
% Algorithm Concept:
%   - Extension of standard SFS with Fitness-Distance Balance (FDB)
%     guide selection in the updating processes.
%   - The diffusion process is identical to original SFS.
%   - In the first and second updating processes, ONE of the random
%     guide indices is replaced by an FDB-selected index. The FDB
%     method picks a candidate that simultaneously has good fitness
%     and is sufficiently far from the current best (Euclidean
%     distance), which improves diversity and helps avoid local
%     optima while preserving exploitation.
%
% Reference:
% Aras, S., Gedikli, E., Kahraman, H. T. (2021),
% A novel stochastic fractal search algorithm with
% fitness-distance balance for global numerical optimization,
% Swarm and Evolutionary Computation, 61, 100821.
% https://doi.org/10.1016/j.swevo.2020.100821
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_sfs(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;               % Lower bounds
    ub = problem.ub;               % Upper bounds
    maxFE = problem.maxFe;         % Maximum function evaluations
    
    % FDB-SFS Parameters
    N = 50;                        % Population size (Start_Point)
    MDN = 1;                       % Maximum Diffusion Number
    Walk = 1;                      % Walk probability
    
    FE = 0;                            % Function Evaluation Counter
    curve = zeros(1, maxFE);           % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;              % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;
    
    % Initialize population
    point = initialization(N, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(point', problem, FE);
    
    % Sort population based on fitness
    [sorted_fitness, indices] = sort(fitness);
    point = point(indices, :);
    fitness = sorted_fitness;
    
    % Find initial best
    best_fitness_current = fitness(1);
    best_solution_current = point(1, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:N
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, point, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Calculate maximum generations
    Max_Generation = ceil((maxFE - N) / ((MDN + 2) * N + N));
    
    % Main loop
    G = 1;
    while FE < maxFE && G <= Max_Generation
        
        New_Point = zeros(N, dim);
        FitVector = zeros(1, N);
        
        % Diffusion process occurs for all points in the group
        for i = 1:N
            if FE >= maxFE
                break;
            end
            % Creating new points based on diffusion process
            [NP, fit, FE] = Diffusion_Process(point(i, :), dim, lb, ub, G, MDN, Walk, point(1, :), problem, FE);
            New_Point(i, :) = NP;
            FitVector(i) = fit;
            
            % Update convergence curve
            if fit < best_fitness_current
                best_fitness_current = fit;
                best_solution_current = NP;
            end
            
            % Record for each diffusion (MDN+1 evaluations per point)
            for eval_idx = 1:(MDN + 1)
                eval_count = FE - (MDN + 1) + eval_idx;
                if eval_count > 0 && eval_count <= maxFE
                    curve(eval_count) = best_fitness_current;
                end
            end
        end
        
        if FE >= maxFE
            break;
        end
        
        % Update sorting
        fit = FitVector';
        [~, sortIndex] = sort(fit);
        
        % Starting The First Updating Process (FDB-guided)
        Pa = zeros(1, N);
        for i = 1:N
            Pa(sortIndex(i)) = (N - i + 1) / N;
        end
        
        % Random guide indices (kept for one of the two guides)
        RandVec2 = randperm(N);
        
        % FDB-selected guide index for the first updating process.
        % Replaces the second random guide (RandVec1) of the original SFS
        % with a candidate that balances fitness and distance to the best.
        fdbIndex1 = fitnessDistanceBalance(New_Point, fit);
        
        P = zeros(N, dim);
        for i = 1:N
            for j = 1:dim
                if rand > Pa(i)
                    % Original SFS: guide_a = New_Point(RandVec1(i), :)
                    % FDB-SFS:      guide_a = New_Point(fdbIndex1, :)
                    P(i, j) = New_Point(fdbIndex1, j) - rand * (New_Point(RandVec2(i), j) - New_Point(i, j));
                else
                    P(i, j) = New_Point(i, j);
                end
            end
        end
        
        % Check bounds
        P = Bound_Checking(P, lb, ub);
        
        % Evaluate first process
        [Fit_FirstProcess, FE] = calculate_fitness(P', problem, FE);
        
        % Update population based on first process
        for i = 1:N
            if Fit_FirstProcess(i) <= fit(i)
                New_Point(i, :) = P(i, :);
                fit(i) = Fit_FirstProcess(i);
            end
            
            % Update best
            if fit(i) < best_fitness_current
                best_fitness_current = fit(i);
                best_solution_current = New_Point(i, :);
            end
        end
        
        % Record convergence curve for first process evaluations
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, New_Point, fit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        FitVector = fit;
        
        % Sort and update best point
        [~, SortedIndex] = sort(FitVector);
        New_Point = New_Point(SortedIndex, :);
        FitVector = FitVector(SortedIndex);
        BestPoint = New_Point(1, :);
        
        point = New_Point;
        fitness = FitVector;
        
        % FDB-selected guide index for the second updating process,
        % computed on the freshly updated population.
        fdbIndex2 = fitnessDistanceBalance(point, fitness);
        
        % Starting The Second Updating Process (FDB-guided)
        Pa = sort(SortedIndex / N, 'descend');
        
        for i = 1:N
            if FE >= maxFE
                break;
            end
            
            if rand > Pa(i)
                % Original SFS: R1 (random) and R2 (random) guides.
                % FDB-SFS: R1 is replaced with the FDB-selected guide
                % to provide a high-quality, well-spread reference.
                R1 = fdbIndex2;
                R2 = ceil(rand * N);
                while R2 == R1 || R2 == i
                    R2 = ceil(rand * N);
                end
                if R2 == 0, R2 = 1; end
                
                if rand < 0.5
                    ReplacePoint = point(i, :) - rand * (point(R2, :) - BestPoint);
                else
                    ReplacePoint = point(i, :) + rand * (point(R2, :) - point(R1, :));
                end
                
                ReplacePoint = Bound_Checking(ReplacePoint, lb, ub);
                
                % Evaluate replacement point
                [fit_replace, FE] = calculate_fitness(ReplacePoint', problem, FE);
                
                if fit_replace < fitness(i)
                    point(i, :) = ReplacePoint;
                    fitness(i) = fit_replace;
                    
                    % Update best
                    if fit_replace < best_fitness_current
                        best_fitness_current = fit_replace;
                        best_solution_current = ReplacePoint;
                    end
                end
                
                % Record convergence
                if FE <= maxFE
                    curve(FE) = best_fitness_current;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, point, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
        end
        
        G = G + 1;
    end
    
    % Fill remaining curve values
    for i = FE+1:maxFE
        curve(i) = best_fitness_current;
    end
    
    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;
    
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        X = zeros(SearchAgents_no, dim);
        for i = 1:dim
            X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Bound Checking Function ---
function p = Bound_Checking(p, lowB, upB)
    for i = 1:size(p, 1)
        upper = double(gt(p(i, :), upB));
        lower = double(lt(p(i, :), lowB));
        up = find(upper == 1);
        lo = find(lower == 1);
        if (size(up, 2) + size(lo, 2) > 0)
            for j = 1:size(up, 2)
                p(i, up(j)) = (upB(up(j)) - lowB(up(j))) * rand() + lowB(up(j));
            end
            for j = 1:size(lo, 2)
                p(i, lo(j)) = (upB(lo(j)) - lowB(lo(j))) * rand() + lowB(lo(j));
            end
        end
    end
end

%% --- Diffusion Process Function ---
function [createPoint, best_fitness, FE] = Diffusion_Process(Point, dim, lb, ub, g, MDN, Walk, BestPoint, problem, FE)
    % Creating new points based on diffusion process
    NumDiffusion = MDN;
    New_Point = zeros(NumDiffusion + 1, dim);
    New_Point(1, :) = Point;
    
    % Diffusing Part
    for i = 1:NumDiffusion
        % Consider which walks should be selected
        if rand < Walk
            % Gaussian walk 1 (Equation 11)
            sigma = (log(g) / g) * (abs(Point - BestPoint));
            sigma(sigma == 0) = 1e-10;  % Avoid zero std
            GeneratePoint = normrnd(BestPoint, sigma, [1 dim]) + (randn * BestPoint - randn * Point);
        else
            % Gaussian walk 2 (Equation 12)
            sigma = (log(g) / g) * (abs(Point - BestPoint));
            sigma(sigma == 0) = 1e-10;  % Avoid zero std
            GeneratePoint = normrnd(Point, sigma, [1 dim]);
        end
        New_Point(i + 1, :) = GeneratePoint;
    end
    
    % Check bounds of New Point
    New_Point = Bound_Checking(New_Point, lb, ub);
    
    % Evaluate all points
    [fitness, FE] = calculate_fitness(New_Point', problem, FE);
    
    % Find best point from diffusion
    [best_fitness, best_idx] = min(fitness);
    createPoint = New_Point(best_idx, :);
end

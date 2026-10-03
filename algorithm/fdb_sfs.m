% ----------------------------------------------------------------------- %
% Fitness-Distance Balance based Stochastic Fractal Search (FDB-SFS)
% Variant of sfs: FDB-selected point replaces point i as base of one second-update move
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Population size (Start_Point)
%   MDN = 1                 % Maximum Diffusion Number
%   Walk = 1                % Walk probability (Gaussian walk selection)
%
% Algorithm Concept:
%   - Diffusion: Gaussian walks around the best point, step log(G)/G*|x - best|
%   - First updating process: rank-dependent component-wise moves between random points
%   - Second updating process, rank-dependent per point: x_i - rand*(x_R2 - best),
%     or x_FDB + rand*(x_R2 - x_R1) where sfs moves from x_i
%   - x_FDB maximises FDB score = normalised fitness + normalised L1 distance to the best
%   - Greedy replacement in both updating processes
%
% Reference:
% Sefa Aras, Eyup Gedikli, Hamdi Tolga Kahraman,
% A novel stochastic fractal search algorithm with fitness-distance balance for
% global numerical optimization,
% Swarm and Evolutionary Computation 61 (2021) 100821.
% https://doi.org/10.1016/j.swevo.2020.100821
% Components:
%   SFS - Salimi, H. (2015), Stochastic Fractal Search: A powerful metaheuristic
%     algorithm, Knowledge-Based Systems, 75, 1-18,
%     https://doi.org/10.1016/j.knosys.2014.07.025
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange package FDB-SFS 1.0.1 (fdb_sfs.m), whose only
% change to the group's sfs.m is the FDB base point; applied to this repository's
% sfs.m. Kept as released: FDB pairs the points, sorted after the first updating
% process, with fit, still in pre-sort order and not refreshed by the second process;
% scoring the sorted fitness instead moved the mean error both ways (better on 3 of 5
% CEC2014 D=10 functions, worse on 1). Charged the release's re-evaluation of point i
% per second-process test and without sfs.m's 1e-10 sigma floor, this file reproduces
% the release exactly (15/15 runs). Added: diffusion records history, so a budget
% ending there fills the last row (sfs.m fails port_check on CEC2020RW F5 for that).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_sfs(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    % SFS Parameters
    N = 50;                       % Population size (Start_Point)
    MDN = 1;                      % Maximum Diffusion Number
    Walk = 1;                     % Walk probability
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
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
            history_index, maxFE);
    end
    
    G = 1;
    while FE < maxFE
        
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
            
            % Record each diffusion evaluation; points after i are still the parents
            pop_now = [New_Point(1:i, :); point(i+1:N, :)];
            fit_now = [reshape(FitVector(1:i), [], 1); reshape(fitness(i+1:N), [], 1)];
            for eval_idx = 1:(MDN + 1)
                eval_count = FE - (MDN + 1) + eval_idx;
                if eval_count > 0 && eval_count <= maxFE
                    curve(eval_count) = best_fitness_current;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, pop_now, fit_now, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
        
        if FE >= maxFE
            break;
        end
        
        % Update sorting
        fit = FitVector';
        [~, sortIndex] = sort(fit);
        
        % Starting The First Updating Process
        Pa = zeros(1, N);
        for i = 1:N
            Pa(sortIndex(i)) = (N - i + 1) / N;
        end
        
        RandVec1 = randperm(N);
        RandVec2 = randperm(N);
        
        P = zeros(N, dim);
        for i = 1:N
            for j = 1:dim
                if rand > Pa(i)
                    P(i, j) = New_Point(RandVec1(i), j) - rand * (New_Point(RandVec2(i), j) - New_Point(i, j));
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
                    history_index, maxFE);
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
        
        % Starting The Second Updating Process
        Pa = sort(SortedIndex / N, 'descend');
        
        for i = 1:N
            if FE >= maxFE
                break;
            end
            
            if rand > Pa(i)
                % Selecting two different points in the group
                R1 = ceil(rand * N);
                R2 = ceil(rand * N);
                while R1 == R2
                    R2 = ceil(rand * N);
                end
                if R1 == 0, R1 = 1; end
                if R2 == 0, R2 = 1; end
                
                if rand < 0.5
                    ReplacePoint = point(i, :) - rand * (point(R2, :) - BestPoint);
                else
                    % FDB winner as base; as released, scored against fit in its pre-sort order
                    ReplacePoint = point(fitnessDistanceBalance(point, fit), :) + rand * (point(R2, :) - point(R1, :));
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
                        history_index, maxFE);
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

% Initialization Function
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

% Bound Checking Function
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

% Diffusion Process Function
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


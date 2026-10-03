% ----------------------------------------------------------------------- %
% Dynamic Fitness-Distance Balance based Stochastic Fractal Search (dFDB-SFS)
% Variant of sfs: dFDB-selected point replaces point i as base of one second-update move
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50, MDN = 1, Walk = 1             % Population, diffusion number, walk choice (sfs)
%   frequency = 100                       % dFDB weight cycles over the budget (case 6)
%   w = 0.6 -> 0 (sawtooth)               % dFDB distance weight, period round(maxFE/100)
%
% Algorithm Concept:
%   - Diffusion by Gaussian walks around BestPoint with step log(g)/g*|x - BestPoint|,
%     g the FE count as released; the parent competes with its offspring unevaluated
%   - First updating process: rank-dependent component-wise moves, greedy survivor
%   - Second updating process, rank-dependent per point: x_i - rand*(x_R2 - BestPoint),
%     or x_dFDB + rand*(x_R2 - x_R1) where sfs moves from x_i
%   - dFDB score = (1-w)*normalised fitness + w*normalised L1 distance to the best
%   - The second process moves BestPoint to any point that beats fbest
%
% Reference:
% Hamdi Tolga Kahraman, Mohamed H. Hassan, Mehmet Kati, Marcos Tostado-Veliz,
% Serhat Duman, Salah Kamel,
% Dynamic-fitness-distance-balance stochastic fractal search (dFDB-SFS algorithm):
% an effective metaheuristic for global optimization and accurate photovoltaic
% modeling,
% Soft Computing 28(9-10) (2024) 6447-6474.
% https://doi.org/10.1007/s00500-023-09505-x
% Components:
%   SFS - Salimi, H. (2015), Stochastic Fractal Search: A powerful metaheuristic
%     algorithm, Knowledge-Based Systems, 75, 1-18,
%     https://doi.org/10.1016/j.knosys.2014.07.025
%   dFDB - Hamdi Tolga Kahraman, Huseyin Bakir, Serhat Duman, Mehmet Kati, Sefa Aras,
%     Ugur Guvenc, Dynamic FDB selection method and its application: modeling and
%     optimizing of directional overcurrent relays coordination, Applied
%     Intelligence 52(5) (2022) 4873-4908, https://doi.org/10.1007/s10489-021-02629-3
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange package dFDB-SFS 1.0.0: case_1-3 drop the
% diffusion and use dFDB in the first updating process, case_4-6 in the second, with
% 1/10/100 weight cycles; the README names the sixth the proposed one, so case_6.m.
% Its SFS base is the group's later copy (as in nsm_sfs), whose changes are kept: g =
% the FE count after initialisation in log(g)/g, the parent's stored fitness reused in
% diffusion, BestPoint moved to any second-process point beating fbest (read from the
% unsorted vector). One budget counts the 50 initial evaluations the release leaves
% out; the reported best is the running best of all evaluations; diffusion records
% history (as in fdb_sfs). Without sfs.m's 1e-10 sigma floor it reproduces the
% release exactly (15/15 runs); the dFDB period is floored at 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dfdb_sfs(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    % SFS Parameters
    N = 50;                       % Population size (Start_Point)
    MDN = 1;                      % Maximum Diffusion Number
    Walk = 1;                     % Walk probability
    frequency = 100;              % dFDB weight cycles over the budget (case_6)
    
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
    
    % Walk centre of the release, carried across generations and moved in the second process
    BestPoint = point(1, :);
    while FE < maxFE
        
        New_Point = zeros(N, dim);
        FitVector = zeros(1, N);
        
        % Diffusion process occurs for all points in the group
        for i = 1:N
            if FE >= maxFE
                break;
            end
            % g is the release's post-initialisation FE counter (starts at 1), not the generation
            nfeval = FE - N + 1;
            [NP, fit, FE] = Diffusion_Process(point(i, :), fitness(i), dim, lb, ub, nfeval, MDN, Walk, BestPoint, problem, FE);
            New_Point(i, :) = NP;
            FitVector(i) = fit;
            
            % Update convergence curve
            if fit < best_fitness_current
                best_fitness_current = fit;
                best_solution_current = NP;
            end
            
            % Record each diffusion evaluation (MDN per point); points after i are still the parents
            pop_now = [New_Point(1:i, :); point(i+1:N, :)];
            fit_now = [reshape(FitVector(1:i), [], 1); reshape(fitness(i+1:N), [], 1)];
            for eval_idx = 1:MDN
                eval_count = FE - MDN + eval_idx;
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
        fbest = FitVector(1);   % the release reads fbest before sorting, so it is row 1's fitness
        
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
                    % dFDB winner as base; its weight cycle runs on the release's FE counter
                    fdbIndex = dynamicFitnessDistanceBalance(point, fitness, frequency, FE - N + 1, maxFE);
                    ReplacePoint = point(fdbIndex, :) + rand * (point(R2, :) - point(R1, :));
                end
                
                ReplacePoint = Bound_Checking(ReplacePoint, lb, ub);
                
                % Evaluate replacement point
                [fit_replace, FE] = calculate_fitness(ReplacePoint', problem, FE);
                
                % The release moves the walk centre to any replacement that beats fbest
                if fit_replace < fbest
                    fbest = fit_replace;
                    BestPoint = ReplacePoint;
                end
                
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
function [createPoint, best_fitness, FE] = Diffusion_Process(Point, PointFit, dim, lb, ub, g, MDN, Walk, BestPoint, problem, FE)
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
    
    % Evaluate only the diffused points; the parent keeps its stored fitness (as released)
    [fnew, FE] = calculate_fitness(New_Point(2:end, :)', problem, FE);
    fitness = [PointFit, fnew(:)'];
    
    % Find best point from diffusion; a tie keeps the parent
    [best_fitness, best_idx] = min(fitness);
    createPoint = New_Point(best_idx, :);
end

% Dynamic FDB: the distance weight w falls 0.6 -> 0 and resets every round(Gen/frequency) iterations
function index = dynamicFitnessDistanceBalance(population, fitness, frequency, iter, Gen)
    fx = max(1, round(Gen / frequency));
    y = mod(iter, fx);
    w = (y/fx * -0.6) + 0.6;
    fitness = fitness(:)';
    populationSize = numel(fitness);
    if min(fitness) == max(fitness)
        index = randi(populationSize);
        return;
    end
    [~, bestIndex] = min(fitness);
    best = population(bestIndex, :);
    distances = sum(abs(best - population), 2)';
    minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
    minDistance = min(distances); maxMinDistance = max(distances) - minDistance;
    normFitness = 1 - ((fitness - minFitness) / maxMinFitness);
    normDistances = (distances - minDistance) / maxMinDistance;
    [~, index] = max((1-w)*normFitness + w*normDistances);
end

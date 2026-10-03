% ----------------------------------------------------------------------- %
% Fitness-Distance Balance based Adaptive Guided Differential Evolution (FDB-AGDE)
% Variant of agde: roulette-FDB pick replaces the bottom-5 donor x_pworst
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                 % Population size
%   F = 0.1 + 0.9*rand      % Scaling factor, drawn per trial
%   CR = Adaptive           % Crossover rate (two pools: 0.05-0.15 or 0.9-1.0)
%
% Algorithm Concept:
%   - Mutation v = x_r + F*(x_pbest - x_pworst): x_r from the middle 40 and
%     x_pbest from the best 5 of the fitness-sorted population
%   - x_pworst is drawn by roulette FDB over the whole population, score =
%     normalised fitness + normalised L1 distance to the best (agde: worst 5)
%   - CR from a low or a high pool; the pool probability follows each pool's
%     success rate averaged over the generations
%   - Out-of-box genes re-sampled uniformly; greedy one-to-one selection
%
% Reference:
% Ugur Guvenc, Serhat Duman, Hamdi Tolga Kahraman, Sefa Aras, Mehmet Kati,
% Fitness-Distance Balance based adaptive guided differential evolution
% algorithm for security-constrained optimal power flow problem incorporating
% renewable energy sources,
% Applied Soft Computing 108 (2021) 107421.
% https://doi.org/10.1016/j.asoc.2021.107421
% Components:
%   AGDE - Ali Wagdy Mohamed, Ali Khater Mohamed, Adaptive guided differential
%     evolution algorithm with novel mutation for numerical optimization,
%     International Journal of Machine Learning and Cybernetics 10(2) (2019)
%     253-277, https://doi.org/10.1007/s13042-017-0711-7
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange package fdb_agde 1.0.1 (FX 90601),
% FDB_AGDE_Case_2.m, as a one-line delta on this repository's agde.m. The package
% numbers its cases differently from the paper: its Case_2 replaces x_pworst,
% which is the paper's Case 3 (Table 3, Eq. 47), the variation the paper carries
% forward ("the FDBAGDE (Case 3) will be used to optimize the SCOPF problem").
% As in the package, the worst-5 draw is still made and discarded, the roulette
% runs once per trial on the current population and may return the target, x_r
% or x_pbest itself, with L1 distance and equal weights (the paper writes the
% Euclidean distance, Eq. 43). Added: the roulette falls back to the last index
% when rounding or a NaN score leaves the cumulative sum short of the draw.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_agde(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    % AGDE Parameters
    NP = 50;                      % Population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize population
    Pop = initialization(NP, dim, ub, lb);
    
    % Evaluate initial population
    [Fit, FE] = calculate_fitness(Pop', problem, FE);
    
    % Find initial best
    [best_fitness_current, iBest] = min(Fit);
    best_solution_current = Pop(iBest, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:NP
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Pop, Fit, population_history, fitness_history, ...
            history_index, maxFE);
    end
    
    % Adaptive CR parameters
    NW = [0.5, 0.5];  % Initial weights for CR pools
    
    % Main loop
    Max_gen = ceil((maxFE - NP) / NP);
    g = 1;
    
    while FE < maxFE && g <= Max_gen
        
        CrPriods_Index = zeros(1, NP);
        Sr = zeros(1, 2);
        CrPriods_Count = zeros(1, 2);
        
        for j = 1:NP
            % Adaptive CR Rule
            Ali = rand;
            if g <= 1
                if Ali <= 0.5
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            else
                if Ali <= NW(1)
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            end
            CrPriods_Count(CrPriods_Index(j)) = CrPriods_Count(CrPriods_Index(j)) + 1;
            
            % Sort population by fitness
            [~, in] = sort(Fit, 'ascend');
            
            % Select indices from best, worst, and middle groups
            AA = in(1:5);           % Best 5
            BB = in(46:50);         % Worst 5
            CC = in(6:45);          % Middle 40
            
            % Choose random individuals from each group
            r1 = AA(randi(length(AA)));
            r2 = BB(randi(length(BB)));
            r3 = CC(randi(length(CC)));
            % x_pworst by roulette FDB (paper Case 3); the worst-5 draw above only keeps the RNG stream
            r2 = rouletteFitnessDistanceBalance(Pop, Fit);
            
            % Adaptive scaling factor
            F = 0.1 + 0.9 * rand;
            
            % Mutation and Crossover
            X = zeros(1, dim);
            Rnd = randi(dim);
            for i = 1:dim
                if rand < CR || Rnd == i
                    X(i) = Pop(r3, i) + F * (Pop(r1, i) - Pop(r2, i));
                else
                    X(i) = Pop(j, i);
                end
            end
            
            % Boundary handling
            X = bound(X, ub, lb);
            
            % Evaluate trial vector
            [f_trial, FE] = calculate_fitness(X', problem, FE);
            
            % Selection
            if f_trial <= Fit(j)
                Sr(CrPriods_Index(j)) = Sr(CrPriods_Index(j)) + 1;
                Pop(j, :) = X;
                Fit(j) = f_trial;
                
                if f_trial <= best_fitness_current
                    best_fitness_current = f_trial;
                    best_solution_current = X;
                end
            end
            
            % Record convergence curve
            if FE <= maxFE
                curve(FE) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pop, Fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        % Update CR pool weights
        CrPriods_Count(CrPriods_Count == 0) = 0.0001;
        Sr = Sr ./ CrPriods_Count;
        
        if sum(Sr) == 0
            W = [0.5, 0.5];
        else
            W = Sr / sum(Sr);
        end
        
        NW = (NW * (g - 1) + W) / g;
        g = g + 1;
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

% Boundary Handling
function a = bound(a, ub, lb)
    % Random reinitialization for out-of-bound values
    for i = 1:length(a)
        if a(i) < lb(min(i, length(lb))) || a(i) > ub(min(i, length(ub)))
            lb_i = lb(min(i, length(lb)));
            ub_i = ub(min(i, length(ub)));
            a(i) = lb_i + (ub_i - lb_i) * rand;
        end
    end
end

% Roulette-wheel FDB selection: score = normalised fitness + normalised distance to the best
function index = rouletteFitnessDistanceBalance(population, fitness)
    fitness = fitness(:)';
    populationSize = numel(fitness);
    [~, bestIndex] = min(fitness);
    best = population(bestIndex, :);
    if (min(fitness) == max(fitness)) || (sum(fitness) >= Inf) || ~(sum(best) < Inf)
        index = randi(populationSize);
        return;
    end
    distances = sum(abs(best - population), 2)';
    minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
    minDistance = min(distances); maxMinDistance = max(distances) - minDistance;
    normFitness = 1 - ((fitness - minFitness) / maxMinFitness);
    normDistances = (distances - minDistance) / maxMinDistance;
    divDistances = normFitness + normDistances;
    r = rand * sum(divDistances);
    index = find(r <= cumsum(divDistances), 1, 'first');
    % Rounding (or a NaN score) can leave the cumulative sum short of r
    if isempty(index), index = populationSize; end
end

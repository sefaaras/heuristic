% ----------------------------------------------------------------------- %
% Fitness-Distance Balance based Artificial Ecosystem-based Optimization (FDB-AEO)
% Variant of aeo: roulette-FDB-selected individual replaces the best as base of Eq. (9)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50                % Population size
%
% Algorithm Concept:
%   - Population sorted by fitness (worst to best)
%   - Production: the worst individual moves toward a random point, Eq. (1)
%   - Consumption: herbivore, carnivore or omnivore moves with a Levy-like factor C
%   - Decomposition, Eq. (9): each individual is placed around a roulette-FDB-selected
%     individual (aeo: the best); the best still enters the step term
%   - Roulette FDB: probability proportional to normalised fitness + normalised L1
%     distance to the best
%   - Greedy replacement after each phase
%
% Reference:
% Yusuf Sonmez, Serhat Duman, Hamdi T. Kahraman, Mehmet Kati, Sefa Aras, Ugur Guvenc,
% Fitness-distance balance based artificial ecosystem optimisation to solve
% transient stability constrained optimal power flow problem,
% Journal of Experimental & Theoretical Artificial Intelligence 36(5) (2024) 745-784.
% https://doi.org/10.1080/0952813X.2022.2104388
% Components:
%   AEO - Zhao, W., Wang, L., & Zhang, Z. (2020). Artificial ecosystem-based
%     optimization: a novel nature-inspired meta-heuristic algorithm. Neural
%     Computing and Applications, 32(13), 9383-9425,
%     https://doi.org/10.1007/s00521-019-04452-x
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange package FDB-AEO 1.0.0 (FDB_AEO.m and its
% rouletteFitnessDistanceBalance.m) as a delta on this repository's aeo.m. As
% released, the roulette is re-drawn for every individual, after r3 and Ind, on the
% population as it stood after the consumption phase. aeo.m's column-wise
% initialisation, clamp bounds and reachable Eq. (7) branch (the release's
% 1/3<r<2/3 never fires) are kept; with those three reverted this file reproduces
% the release exactly (15/15 runs). Added: the roulette falls back to the last index
% when rounding or a NaN score leaves the cumulative sum short of the draw (the
% find() empty-index crash fixed in sdo).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_aeo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    nPop = 50;
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize population
    PopPos = initialization(nPop, dim, ub, lb);
    
    % Evaluate initial population
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    
    % Sort population by fitness in descending order (worst to best)
    [PopFit, indF] = sort(PopFit, 'descend');
    PopPos = PopPos(indF, :);
    
    % Best point EVALUATED so far, updated at every evaluation
    bsf = PopFit(end);
    bsf_sol = PopPos(end, :);

    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:nPop
        curve(eval_count) = bsf;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, PopPos, PopFit, population_history, fitness_history, ...
            history_index, maxFE);
    end
    
    % For equation (9) - determines whether to update in 1D or full dimension
    Matr = [1, dim];
    
    % Main loop
    Max_iteration = ceil((maxFE - nPop) / (nPop * 2));
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        newPopPos = zeros(nPop, dim);
        
        % Production phase, Eq. (1): the worst individual interacts with a random solution
        r1 = rand;
        a = (1 - Iteration / Max_iteration) * r1;
        xrand = rand(1, dim) .* (ub - lb) + lb;
        newPopPos(1, :) = (1 - a) * PopPos(nPop, :) + a * xrand;
        
        % Consumption phase, Eq. (6): the herbivore consumes the producer
        u = randn(1, dim);
        v = randn(1, dim);
        C = 0.5 * u ./ abs(v);  % Equation (4): Levy-like coefficient
        newPopPos(2, :) = PopPos(2, :) + C .* (PopPos(2, :) - newPopPos(1, :));
        
        % Carnivore/Omnivore behavior for remaining individuals
        for i = 3:nPop
            u = randn(1, dim);
            v = randn(1, dim);
            C = 0.5 * u ./ abs(v);
            
            r = rand;
            if r < 1/3
                % Equation (6): Herbivore - consumes producer
                newPopPos(i, :) = PopPos(i, :) + C .* (PopPos(i, :) - newPopPos(1, :));
            elseif r < 2/3
                % Equation (7): Carnivore - consumes random herbivore
                randIdx = randi([2, i-1]);
                newPopPos(i, :) = PopPos(i, :) + C .* (PopPos(i, :) - PopPos(randIdx, :));
            else
                % Equation (8): Omnivore - consumes both producer and herbivore
                r2 = rand;
                randIdx = randi([2, i-1]);
                newPopPos(i, :) = PopPos(i, :) + C .* (r2 * (PopPos(i, :) - newPopPos(1, :)) + ...
                                  (1 - r2) * (PopPos(i, :) - PopPos(randIdx, :)));
            end
        end
        
        % Apply boundary constraints and evaluate new positions
        for i = 1:nPop
            newPopPos(i, :) = bound(newPopPos(i, :), ub, lb);
        end
        
        % Evaluate new population after consumption phase
        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);
        [mF, mI] = min(newPopFit);
        if mF < bsf
            bsf = mF; bsf_sol = newPopPos(mI, :);
        end

        % Greedy selection: Update if new solution is better
        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopFit(i) = newPopFit(i);
                PopPos(i, :) = newPopPos(i, :);
            end
        end
        
        % Record convergence curve for consumption phase evaluations
        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        % Check if we've exceeded maxFE
        if FE >= maxFE
            break;
        end
        
        % Decomposition phase: the best still sets the Eq. (9) step
        [~, indOne] = min(PopFit);
        
        % Equation (9): Decomposition around a roulette-FDB individual instead of the best
        for i = 1:nPop
            r3 = rand;
            Ind = round(rand) + 1;  % Randomly choose between 1 or full dimension update
            fdbIndex = rouletteFitnessDistanceBalance(PopPos, PopFit);  % re-drawn per i, as released
            newPopPos(i, :) = PopPos(fdbIndex, :) + 3 * randn(1, Matr(Ind)) .* ...
                             ((r3 * randi([1, 2]) - 1) * PopPos(indOne, :) - (2 * r3 - 1) * PopPos(i, :));
        end
        
        % Apply boundary constraints
        for i = 1:nPop
            newPopPos(i, :) = bound(newPopPos(i, :), ub, lb);
        end
        
        % Evaluate new population after decomposition phase
        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);
        [mF, mI] = min(newPopFit);
        if mF < bsf
            bsf = mF; bsf_sol = newPopPos(mI, :);
        end

        % Greedy selection: Update if new solution is better
        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopPos(i, :) = newPopPos(i, :);
                PopFit(i) = newPopFit(i);
            end
        end
        
        % Record convergence curve for decomposition phase evaluations
        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        % Sort population by fitness (descending - worst to best)
        [PopFit, indF] = sort(PopFit, 'descend');
        PopPos = PopPos(indF, :);
        
        Iteration = Iteration + 1;
    end
    
    curve(min(max(FE, 1), maxFE):end) = bsf;

    % Return best solution
    best_fitness = bsf;
    best_solution = bsf_sol;
    
end

% Initialization Function
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

% Boundary Handling
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

% Roulette FDB of the release: P(i) ~ normalised fitness + normalised L1 distance to the best
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

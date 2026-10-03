% ----------------------------------------------------------------------- %
% Dynamic Fitness-Distance Balance based Manta Ray Foraging Optimization (dFDB-MRFO)
% Variant of mrfo: dFDB-selected guide replaces the best in half the chain-foraging moves
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50                 % Population size
%   frequency = 10            % Number of dFDB weight cycles over the run
%   w = 0.6 -> 0 (sawtooth)   % Distance weight in the dFDB score, period MaxIt/10
%
% Algorithm Concept:
%   - Inspired by the foraging behavior of manta rays
%   - Chain foraging: individuals follow the one ahead toward the food; for
%     i >= 2 the food is the dFDB guide with probability 0.5, else the best
%   - Cyclone foraging: spiral movement toward the best solution
%   - Somersault foraging: random somersault around the best solution
%   - dFDB guide: argmax of (1-w)*normFitness + w*normDistance, once per iteration
%
% Reference:
% Hamdi Tolga Kahraman, Huseyin Bakir, Serhat Duman, Mehmet Kati, Sefa Aras,
% Ugur Guvenc,
% Dynamic FDB selection method and its application: modeling and optimizing of
% directional overcurrent relays coordination,
% Applied Intelligence 52(5) (2022) 4873-4908.
% https://doi.org/10.1007/s10489-021-02629-3
% Components:
%   MRFO - Weiguo Zhao, Zhenxing Zhang, Liying Wang, Manta ray foraging
%     optimization: An effective bio-inspired optimizer for engineering
%     applications, Engineering Applications of Artificial Intelligence 87
%     (2020) 103300, https://doi.org/10.1016/j.engappai.2019.103300
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released package (dFDB_MRFO.m + dFDB.m, 2024-mso copy)
% as a delta on this repository's mrfo.m. As released,
% dFDB runs once per iteration, after the first manta's move, on the population
% of the iteration start, and the extra rand < 0.5 is drawn right after Alpha.
% The weight period round(MaxIt/10) uses mrfo.m's MaxIt, which nets out the
% initial population, and is floored at 1 so a run of under 5 iterations does
% not divide by zero (the release would turn w into -Inf there).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dfdb_mrfo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;

    FE = 0;
    curve = nan(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    PopPos = initialization(nPop, dim, ub, lb);
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);

    [BestF, best_idx] = min(PopFit);
    BestX = PopPos(best_idx, :);

    for eval_count = 1:min(nPop, maxFE)
        curve(eval_count) = BestF;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, PopPos, PopFit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    MaxIt = ceil((maxFE - nPop) / (2 * nPop));

    for It = 1:MaxIt
        if FE >= maxFE, break; end

        Coef = It / MaxIt;
        newPopPos = zeros(nPop, dim);

        % Phase 1: Chain foraging or Cyclone foraging
        if rand < 0.5
            r1 = rand;
            Beta = 2 * exp(r1 * ((MaxIt - It + 1) / MaxIt)) * sin(2 * pi * r1);
            if Coef > rand
                newPopPos(1,:) = BestX + rand(1, dim) .* (BestX - PopPos(1,:)) + Beta * (BestX - PopPos(1,:));
            else
                IndivRand = rand(1, dim) .* (ub - lb) + lb;
                newPopPos(1,:) = IndivRand + rand(1, dim) .* (IndivRand - PopPos(1,:)) + Beta * (IndivRand - PopPos(1,:));
            end
        else
            Alpha = 2 * rand(1, dim) .* (-log(rand(1, dim))).^0.5;
            newPopPos(1,:) = PopPos(1,:) + rand(1, dim) .* (BestX - PopPos(1,:)) + Alpha .* (BestX - PopPos(1,:));
        end

        % One dFDB guide per iteration, picked where the release picks it
        fdbIndex = dynamicFitnessDistanceBalance(PopPos, PopFit, 10, It, MaxIt);
        for i = 2:nPop
            if rand < 0.5
                r1 = rand;
                Beta = 2 * exp(r1 * ((MaxIt - It + 1) / MaxIt)) * sin(2 * pi * r1);
                if Coef > rand
                    newPopPos(i,:) = BestX + rand(1, dim) .* (PopPos(i-1,:) - PopPos(i,:)) + Beta * (BestX - PopPos(i,:));
                else
                    IndivRand = rand(1, dim) .* (ub - lb) + lb;
                    newPopPos(i,:) = IndivRand + rand(1, dim) .* (PopPos(i-1,:) - PopPos(i,:)) + Beta * (IndivRand - PopPos(i,:));
                end
            else
                Alpha = 2 * rand(1, dim) .* (-log(rand(1, dim))).^0.5;
                if rand < 0.5
                    newPopPos(i,:) = PopPos(i,:) + rand(1, dim) .* (PopPos(i-1,:) - PopPos(i,:)) + Alpha .* (PopPos(fdbIndex,:) - PopPos(i,:));
                else
                    newPopPos(i,:) = PopPos(i,:) + rand(1, dim) .* (PopPos(i-1,:) - PopPos(i,:)) + Alpha .* (BestX - PopPos(i,:));
                end
            end
        end

        for i = 1:nPop
            newPopPos(i,:) = space_bound(newPopPos(i,:), ub, lb);
        end

        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);

        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopFit(i) = newPopFit(i);
                PopPos(i,:) = newPopPos(i,:);
            end
        end

        [min_fit, min_idx] = min(PopFit);
        if min_fit < BestF
            BestF = min_fit;
            BestX = PopPos(min_idx,:);
        end

        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = BestF;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE, break; end

        % Phase 2: Somersault foraging
        S = 2;
        for i = 1:nPop
            newPopPos(i,:) = PopPos(i,:) + S * (rand * BestX - rand * PopPos(i,:));
            newPopPos(i,:) = space_bound(newPopPos(i,:), ub, lb);
        end

        [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);

        for i = 1:nPop
            if newPopFit(i) < PopFit(i)
                PopFit(i) = newPopFit(i);
                PopPos(i,:) = newPopPos(i,:);
            end
        end

        [min_fit, min_idx] = min(PopFit);
        if min_fit < BestF
            BestF = min_fit;
            BestX = PopPos(min_idx,:);
        end

        for eval_idx = 1:nPop
            eval_count = FE - nPop + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = BestF;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    % NaN marks a slot no evaluation reached; 0 is a legal fitness value
    for idx = 2:maxFE
        if isnan(curve(idx))
            curve(idx) = curve(idx - 1);
        end
    end

    best_fitness = BestF;
    best_solution = BestX;

end

% Initialization Function
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

% Boundary Handling (Random Replacement)
function X = space_bound(X, ub, lb)
    D = length(X);
    S = (X > ub) + (X < lb);
    X = (rand(1, D) .* (ub - lb) + lb) .* S + X .* (~S);
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

% ----------------------------------------------------------------------- %
% Fitness-Distance Balance based Supply-Demand-Based Optimization (FDB-SDO)
% Variant of sdo: roulette-FDB-selected quantity replaces the equilibrium one in Eq. (13)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   MarketSize = 50           % Population size (number of market agents)
%
% Algorithm Concept:
%   - Inspired by supply-demand mechanism in economics
%   - Each agent has a commodity price (position) and commodity quantity
%   - Supply function: new quantity = roulette-FDB-selected quantity plus a
%     sine-weighted price difference (sdo starts from the equilibrium quantity)
%   - Demand function updates price based on quantity differences
%   - Market equilibrium drives convergence to optimal solution
%
% Reference:
% Mehmet Kati, Hamdi Tolga Kahraman,
% Improving supply-demand-based optimization algorithm with FDB method: a
% comprehensive research on engineering design problems (in Turkish),
% Muhendislik Bilimleri ve Tasarim Dergisi (Journal of Engineering Sciences and
% Design) 8(5) (2020) 156-172, special issue ICAIAME 2020.
% https://doi.org/10.21923/jesd.829508
% Components:
%   SDO - Weiguo Zhao, Liying Wang, Zhenxing Zhang, Supply-Demand-Based
%     Optimization: A Novel Economics-Inspired Algorithm for Global
%     Optimization, IEEE Access 7 (2019) 73182-73206,
%     https://doi.org/10.1109/ACCESS.2019.2918753
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' working copy (37-sdo/fdb_sdo.m) as a delta on this
% repository's sdo.m: only the base point of the supply update (Eq. 13) changes.
% That is the paper's Case-2 (Table 1), the variation it carries forward; its
% Algorithm 2 pseudocode also swaps the demand term, which neither Table 1 nor the
% code does. The FQ-roulette equilibrium quantity is still drawn and enters the demand
% update (Eq. 14), as in that copy. The copy draws Alpha and Beta before the FQ
% roulette; this file keeps sdo.m's draw order (same distribution). The roulette
% FDB is re-run for every agent on the current quantities and falls back to the
% last index when rounding or a NaN score leaves the cumulative sum short.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_sdo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    MarketSize = 50;

    FE = 0;
    curve = nan(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize commodity prices and quantities
    CommPrice = initialization(MarketSize, dim, ub, lb);
    [CommPriceFit, FE] = calculate_fitness(CommPrice', problem, FE);

    CommQuantity = initialization(MarketSize, dim, ub, lb);
    [CommQuantityFit, FE] = calculate_fitness(CommQuantity', problem, FE);

    % Replace price with quantity where quantity is better
    for i = 1:MarketSize
        if CommQuantityFit(i) <= CommPriceFit(i)
            CommPriceFit(i) = CommQuantityFit(i);
            CommPrice(i,:) = CommQuantity(i,:);
        end
    end

    [BestF, best_idx] = min(CommPriceFit);
    BestX = CommPrice(best_idx,:);

    init_evals = min(2 * MarketSize, maxFE);
    for eval_count = 1:init_evals
        curve(eval_count) = BestF;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, CommPrice, CommPriceFit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    MaxIt = ceil((maxFE - 2 * MarketSize) / (2 * MarketSize));
    Matr = [1, dim];

    for Iter = 1:MaxIt
        if FE >= maxFE, break; end

        a = 2 * (MaxIt - Iter + 1) / MaxIt;

        F = zeros(MarketSize, 1);
        MeanQuantityFit = mean(CommQuantityFit);
        for i = 1:MarketSize
            F(i) = abs(CommQuantityFit(i) - MeanQuantityFit) + 1e-15;
        end
        FQ = F / sum(F);

        MeanPriceFit = mean(CommPriceFit);
        for i = 1:MarketSize
            F(i) = abs(CommPriceFit(i) - MeanPriceFit) + 1e-15;
        end
        FP = F / sum(F);
        MeanPrice = mean(CommPrice, 1);

        for i = 1:MarketSize
            if FE >= maxFE, break; end

            Ind = round(rand) + 1;
            k = find(rand <= cumsum(FQ), 1, 'first');
            if isempty(k), k = MarketSize; end
            CommQuantityEqu = CommQuantity(k,:);

            Alpha = a * sin(2 * pi * rand(1, Matr(Ind)));
            Beta = 2 * cos(2 * pi * rand(1, Matr(Ind)));

            if rand > 0.5
                CommPriceEqu = rand * MeanPrice;
            else
                k2 = find(rand <= cumsum(FP), 1, 'first');
                if isempty(k2), k2 = MarketSize; end
                CommPriceEqu = CommPrice(k2,:);
            end

            % Supply function (Eq. 13) from a roulette-FDB-selected quantity
            fdbIndex = rouletteFitnessDistanceBalance(CommQuantity, CommQuantityFit);
            NewCommQuantity = CommQuantity(fdbIndex,:) + Alpha .* (CommPrice(i,:) - CommPriceEqu);
            NewCommQuantity = space_bound(NewCommQuantity, ub, lb);
            [NewCommQuantityFit, FE] = calculate_fitness(NewCommQuantity', problem, FE);

            if NewCommQuantityFit <= CommQuantityFit(i)
                CommQuantityFit(i) = NewCommQuantityFit;
                CommQuantity(i,:) = NewCommQuantity;
            end

            if NewCommQuantityFit < BestF
                BestF = NewCommQuantityFit;
                BestX = NewCommQuantity;
            end
            if FE <= maxFE
                curve(FE) = BestF;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, CommPrice, CommPriceFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end

            if FE >= maxFE, break; end

            % Demand function
            NewCommPrice = CommPriceEqu - Beta .* (NewCommQuantity - CommQuantityEqu);
            NewCommPrice = space_bound(NewCommPrice, ub, lb);
            [NewCommPriceFit, FE] = calculate_fitness(NewCommPrice', problem, FE);

            if NewCommPriceFit <= CommPriceFit(i)
                CommPriceFit(i) = NewCommPriceFit;
                CommPrice(i,:) = NewCommPrice;
            end

            if NewCommPriceFit < BestF
                BestF = NewCommPriceFit;
                BestX = NewCommPrice;
            end
            if FE <= maxFE
                curve(FE) = BestF;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, CommPrice, CommPriceFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Replacement: update price with quantity where quantity is better
        for i = 1:MarketSize
            if CommQuantityFit(i) <= CommPriceFit(i)
                CommPriceFit(i) = CommQuantityFit(i);
                CommPrice(i,:) = CommQuantity(i,:);
            end
        end

        [min_fit, min_idx] = min(CommPriceFit);
        if min_fit < BestF
            BestF = min_fit;
            BestX = CommPrice(min_idx,:);
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

% ----------------------------------------------------------------------- %
% Supply-Demand-Based Optimization (SDO) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   MarketSize = 50           % Population size (number of market agents)
%
% Algorithm Concept:
%   - Inspired by supply-demand mechanism in economics
%   - Each agent has a commodity price (position) and commodity quantity
%   - Supply function updates quantity based on price differences
%   - Demand function updates price based on quantity differences
%   - Market equilibrium drives convergence to optimal solution
%
% Reference:
% Weiguo Zhao, Liying Wang, Zhenxing Zhang,
% Supply-Demand-Based Optimization: A Novel Economics-Inspired Algorithm
% for Global Optimization,
% IEEE Access 7 (2019) 73182-73206
% https://doi.org/10.1109/ACCESS.2019.2918753
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = sdo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    MarketSize = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, MarketSize, dim);
    fitness_history = zeros(history_size, MarketSize);
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
            history_index, sampling_interval, history_size);
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
            CommQuantityEqu = CommQuantity(k,:);

            Alpha = a * sin(2 * pi * rand(1, Matr(Ind)));
            Beta = 2 * cos(2 * pi * rand(1, Matr(Ind)));

            if rand > 0.5
                CommPriceEqu = rand * MeanPrice;
            else
                k2 = find(rand <= cumsum(FP), 1, 'first');
                CommPriceEqu = CommPrice(k2,:);
            end

            % Supply function
            NewCommQuantity = CommQuantityEqu + Alpha .* (CommPrice(i,:) - CommPriceEqu);
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
                    history_index, sampling_interval, history_size);
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
                    history_index, sampling_interval, history_size);
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

    for idx = 2:maxFE
        if curve(idx) == 0
            curve(idx) = curve(idx - 1);
        end
    end

    best_fitness = BestF;
    best_solution = BestX;

end

%% --- Initialization Function ---
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

%% --- Boundary Handling (Random Replacement) ---
function X = space_bound(X, ub, lb)
    D = length(X);
    S = (X > ub) + (X < lb);
    X = (rand(1, D) .* (ub - lb) + lb) .* S + X .* (~S);
end

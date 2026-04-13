% ----------------------------------------------------------------------- %
% Manta Ray Foraging Optimization (MRFO) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50                 % Population size
%
% Algorithm Concept:
%   - Inspired by the foraging behavior of manta rays
%   - Three foraging strategies: chain, cyclone, and somersault foraging
%   - Chain foraging: individuals follow the one ahead toward the food
%   - Cyclone foraging: spiral movement toward the best solution
%   - Somersault foraging: random somersault around the best solution
%
% Reference:
% Wenyin Zhao, Zhenxing Zhang, Liying Wang,
% Manta ray foraging optimization: An effective bio-inspired optimizer
% for engineering applications,
% Engineering Applications of Artificial Intelligence 87 (2020) 103300
% https://doi.org/10.1016/j.engappai.2019.103300
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = mrfo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nPop, dim);
    fitness_history = zeros(history_size, nPop);
    history_index = 1;

    PopPos = initialization(nPop, dim, ub, lb);
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);

    [BestF, best_idx] = min(PopFit);
    BestX = PopPos(best_idx, :);

    for eval_count = 1:min(nPop, maxFE)
        curve(eval_count) = BestF;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, PopPos, PopFit, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
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
                newPopPos(i,:) = PopPos(i,:) + rand(1, dim) .* (PopPos(i-1,:) - PopPos(i,:)) + Alpha .* (BestX - PopPos(i,:));
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
                    history_index, sampling_interval, history_size);
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
                    history_index, sampling_interval, history_size);
            end
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

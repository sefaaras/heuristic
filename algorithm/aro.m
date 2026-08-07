% ----------------------------------------------------------------------- %
% Artificial Rabbits Optimization (ARO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50   % Population size (rabbits)
%
% Algorithm Concept:
%   - Detour foraging (exploration): move toward a random rabbit (A > 1)
%   - Random hiding (exploitation): jump into one of d burrows (A <= 1)
%   - Energy factor A shrinks over time, switching between the two behaviors
%
% Reference:
% Liying Wang, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili,
% Weiguo Zhao,
% Artificial rabbits optimization: A new bio-inspired meta-heuristic
% algorithm for solving engineering optimization problems,
% Engineering Applications of Artificial Intelligence 114 (2022) 105082.
% https://doi.org/10.1016/j.engappai.2022.105082
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = aro(problem)

    Dim = problem.dimension;
    Low = problem.lb;
    Up = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;
    MaxIt = ceil(maxFE / nPop);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    PopPos = initialization(nPop, Dim, Up, Low);
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    PopFit = PopFit(:);

    [BestF, bidx] = min(PopFit);
    BestX = PopPos(bidx, :);

    for e = 1:nPop
        if e <= maxFE
            curve(e) = BestF;
            [population_history, fitness_history, history_index] = record_history(...
                e, PopPos, PopFit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    It = 0;
    while FE < maxFE
        It = It + 1;
        theta = 2 * (1 - It / MaxIt);
        newPop = PopPos;
        for i = 1:nPop
            L = (exp(1) - exp(((It - 1) / MaxIt)^2)) * (sin(2 * pi * rand)); % Eq.(3)
            rd = ceil(rand * Dim);
            Direct1 = zeros(1, Dim);
            Direct1(randperm(Dim, rd)) = 1;                 % Eq.(4)
            R = L .* Direct1;                               % Eq.(2)
            A = 2 * log(1 / rand) * theta;                  % Eq.(15)
            if A > 1
                K = [1:i - 1, i + 1:nPop];
                RandInd = K(randi(nPop - 1));
                newPop(i, :) = PopPos(RandInd, :) + R .* (PopPos(i, :) - PopPos(RandInd, :)) ...
                    + round(0.5 * (0.05 + rand)) * randn;   % Eq.(1)
            else
                Direct2 = zeros(1, Dim);
                Direct2(ceil(rand * Dim)) = 1;              % Eq.(12)
                gr = Direct2;
                H = ((MaxIt - It + 1) / MaxIt) * randn;      % Eq.(8)
                b = PopPos(i, :) + H * gr .* PopPos(i, :);   % Eq.(13)
                newPop(i, :) = PopPos(i, :) + R .* (rand * b - PopPos(i, :)); % Eq.(11)
            end
            newPop(i, :) = SpaceBound(newPop(i, :), Up, Low);
        end

        [newFit, FE] = calculate_fitness(newPop', problem, FE);
        newFit = newFit(:);

        for i = 1:nPop
            if newFit(i) < PopFit(i)
                PopFit(i) = newFit(i);
                PopPos(i, :) = newPop(i, :);
                if PopFit(i) < BestF
                    BestF = PopFit(i);
                    BestX = PopPos(i, :);
                end
            end
            ec = FE - nPop + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = BestF;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = BestF;
    best_fitness = BestF;
    best_solution = BestX;
end

% Boundary relocation (random re-init of out-of-range dims)
function X = SpaceBound(X, Up, Low)
    Dim = length(X);
    S = (X > Up) + (X < Low);
    X = (rand(1, Dim) .* (Up - Low) + Low) .* S + X .* (~S);
end

% Initialization
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

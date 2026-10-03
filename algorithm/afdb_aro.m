% ----------------------------------------------------------------------- %
% Adaptive Fitness-Distance Balance based Artificial Rabbits Optimization (AFDB-ARO)
% Variant of aro: a sinus-switched FDB rabbit replaces the random base rabbit in detour foraging
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50                        % Population size (rabbits)
%   p_fdb = |sin(2*pi*FE/maxFE)|     % Probability the detour base rabbit is the FDB pick
%
% Algorithm Concept:
%   - Detour foraging (A > 1): move relative to another rabbit, Eq.(1)
%   - Adaptive FDB: with probability p_fdb the base rabbit of Eq.(1) is the
%     FDB winner instead of a random one (variant AFDB-ARO9 of the paper)
%   - FDB score: normalised fitness + normalised Manhattan distance to the best
%   - Random hiding (A <= 1): jump into one of d burrows
%   - Energy factor A shrinks with FE/maxFE, switching between the two behaviors
%
% Reference:
% Burcin Ozkaya, Serhat Duman, Hamdi Tolga Kahraman, Ugur Guvenc,
% Optimal solution of the combined heat and power economic dispatch problem
% by adaptive fitness-distance balance based artificial rabbits optimization
% algorithm, Expert Systems with Applications 238 (2024) 122272.
% https://doi.org/10.1016/j.eswa.2023.122272
% Components:
%   ARO - Liying Wang, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili,
%     Weiguo Zhao, Artificial rabbits optimization: A new bio-inspired
%     meta-heuristic algorithm for solving engineering optimization problems,
%     Engineering Applications of Artificial Intelligence 114 (2022) 105082,
%     https://doi.org/10.1016/j.engappai.2022.105082
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' package (FX 136846, afdb_aro-1.0.1, AFDB_ARO9.m) as a
% delta on this repository's aro.m. Case: Sec. 5.3 (p. 24) "the AFDB-ARO9
% algorithm was named as AFDB-ARO"; Table 3 gives ARO9 the sinus switch with the
% FDB rabbit in slot A (base point). Kept from the release: theta, L and H run on
% FE/maxFE, not It/MaxIt; rabbit i of a batch uses the count FE+i-1 the release
% has there. The released switch fires FDB with probability |sin(2*pi*FE/maxFE)|
% (rand < (rand < |sin|)), not Algorithm-1's "threshold < rand" with frequency f.
% FDB scores the iteration-start population, since aro.m evaluates an iteration
% as one batch; the release, rabbit by rabbit, scores the population updated so far.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = afdb_aro(problem)

    Dim = problem.dimension;
    Low = problem.lb;
    Up = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;

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

    while FE < maxFE
        theta = 2 * (1 - FE / maxFE);
        newPop = PopPos;
        fdbIdx = [];
        for i = 1:nPop
            fe_i = FE + i - 1;                              % evaluation count the release has at rabbit i
            L = (exp(1) - exp(((fe_i - 1) / maxFE)^2)) * (sin(2 * pi * rand)); % Eq.(3)
            rd = ceil(rand * Dim);
            Direct1 = zeros(1, Dim);
            Direct1(randperm(Dim, rd)) = 1;                 % Eq.(4)
            R = L .* Direct1;                               % Eq.(2)
            A = 2 * log(1 / rand) * theta;                  % Eq.(15)
            if A > 1
                rand_num = rand;
                if rand_num < sinSwitch(maxFE, fe_i)
                    % Deterministic on an unchanged population unless fitness is flat
                    if isempty(fdbIdx) || min(PopFit) == max(PopFit)
                        fdbIdx = fitnessDistanceBalance(PopPos, PopFit);
                    end
                    K = [1:i - 1, i + 1:nPop];
                    RandInd = K(randi(nPop - 1));
                    newPop(i, :) = PopPos(fdbIdx, :) + R .* (PopPos(i, :) - PopPos(RandInd, :)) ...
                        + round(0.5 * (0.05 + rand)) * randn;   % Eq.(23), guide A from FDB
                else
                    K = [1:i - 1, i + 1:nPop];
                    RandInd = K(randi(nPop - 1));
                    newPop(i, :) = PopPos(RandInd, :) + R .* (PopPos(i, :) - PopPos(RandInd, :)) ...
                        + round(0.5 * (0.05 + rand)) * randn;   % Eq.(1)
                end
            else
                Direct2 = zeros(1, Dim);
                Direct2(ceil(rand * Dim)) = 1;              % Eq.(12)
                gr = Direct2;
                H = ((maxFE - fe_i + 1) / maxFE) * randn;   % Eq.(8)
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

% Released ActFun_Sin_SF_Decrease: 1 with probability |sin| of one full turn over the budget
function out = sinSwitch(maxFE, fe)
    if fe > maxFE
        out = 0;
    else
        out = rand < abs(sin((pi / 180) * ((fe * 360) / maxFE)));
    end
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

% ----------------------------------------------------------------------- %
% Fitness-Distance Balance based Levy Flight Distribution (FDB-LFD)
% Variant of lfd: FDB-selected agent replaces the target TP as each agent's Levy guide
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50          % Population size
%   threshold = 2   % Neighbourhood radius (comfort zone, first two coordinates)
%   CSV = 0.5       % Probability that a neighbour Levy-flies instead of re-sampling
%
% Algorithm Concept:
%   - Neighbours inside the comfort zone either Levy-fly toward one of the two
%     least-crowded agents or are re-sampled uniformly in the box
%   - Each agent moves to the target plus a neighbour-weighted sum, then takes a
%     Levy flight guided by P_FDB instead of the target TP (Eq. 31)
%   - P_FDB maximises FDB score = normalised fitness + normalised L1 distance
%     to the best, scored on the current population
%   - Levy steps (Mantegna, beta = 1.5) give occasional long exploratory jumps
%
% Reference:
% Huseyin Bakir, Ugur Guvenc, Hamdi Tolga Kahraman, Serhat Duman,
% Improved Levy flight distribution algorithm with FDB-based guiding mechanism
% for AVR system optimal design,
% Computers & Industrial Engineering 168 (2022) 108032.
% https://doi.org/10.1016/j.cie.2022.108032
% Components:
%   LFD - Essam H. Houssein, Mohammed R. Saad, Fatma A. Hashim, Hassan Shaban,
%     M. Hassaballah, Levy flight distribution: A new metaheuristic algorithm for
%     solving engineering optimization problems, Engineering Applications of
%     Artificial Intelligence 94 (2020) 103731,
%     https://doi.org/10.1016/j.engappai.2020.103731
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange package fdb_lfd 1.0.2 (FX 107090),
% FDB_LFD_Case_1.m, as a delta on this repository's lfd.m (bound fixes and N = 50
% kept; the paper and the package's driver run N = 35). The paper adopts Case-1
% (Table 1, Eq. 31: P_FDB instead of TP in Eq. 24): "Case-1 was the most successful
% of the FDB versions and will be referred to as FDB-LFD in the following sections"
% (Sec. 5.1); package Case_1 makes exactly that change. Also as released, lfd's
% never-true l==2 test becomes FES==NN (FES = evaluations after initialisation, NN =
% neighbour counts): in generation 1 only, agent i >= 3 leaves its neighbours unmoved
% while every earlier agent had none. P_FDB is scored once per generation; on flat
% fitness each agent draws its own uniform index, as the release's per-agent call does.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_lfd(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 50;
    threshold = 2;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the population
    Positions = Initialization(N, dim, ub, lb);
    Positions_temp = Positions;

    [PositionsFitness, FE] = calculate_fitness(Positions', problem, FE);
    PositionsFitness = PositionsFitness(:)';

    [sorted_fitness, sorted_indexes] = sort(PositionsFitness);
    Sorted_Positions = Positions(sorted_indexes, :);
    TargetPosition = Sorted_Positions(1, :);
    TargetFitness  = sorted_fitness(1);

    for eval_count = 1:N
        curve(eval_count) = TargetFitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Positions, PositionsFitness', population_history, fitness_history, ...
            history_index, maxFE);
    end

    vec_flag = [1, -1];
    NN = [0, 1];
    FES = 0;   % the release's FE counter, initial evaluations excluded (read only by FES==NN)

    while FE < maxFE
        [~, ll] = sort(NN);
        % P_FDB (Eq. 30) is fixed within a generation; a flat fitness redraws it per agent below
        flatFit = min(PositionsFitness) == max(PositionsFitness);
        if ~flatFit
            fdbIndex = fitnessDistanceBalance(Positions, PositionsFitness);
        end
        for i = 1:N
            S_i = zeros(1, dim);
            NeighborN = 0;
            var_flag = vec_flag(1);
            D = 0;
            pos_temp_nei = {};
            for j = 1:N
                flag_index = floor(2 * rand() + 1);
                var_flag = vec_flag(flag_index);
                if i ~= j
                    dis = Distance(Positions(i, :), Positions(j, :));
                    if dis < threshold
                        NeighborN = NeighborN + 1;
                        D = (PositionsFitness(j) / (PositionsFitness(i) + eps));
                        D(NeighborN) = ((0.9 * (D - min(D))) ./ (max(D(:)) - min(D) + eps)) + 0.1;
                        if all(FES == NN)   % release's first-generation test (lfd: l == 2)
                            rand_leader_index = floor(N * rand() + 1);
                            X_rand = Positions(rand_leader_index, :); %#ok<NASGU>
                        else
                            R = rand(); CSV = 0.5;
                            if R < CSV
                                rand_leader_index = floor(2 * rand() + 1);
                                X_rand = Positions(ll(rand_leader_index), :);
                                Positions_temp(j, :) = LevyFlights(Positions(j, :), X_rand, lb, ub);
                            else
                                Positions_temp(j, :) = lb + rand(1, dim) .* (ub - lb);
                            end
                        end
                        pos_temp_nei{NeighborN} = Positions(j, :);
                    end
                end
            end
            for p = 1:NeighborN
                s_ij = var_flag * D(NeighborN) .* (pos_temp_nei{p}) / NeighborN;
                S_i = S_i + s_ij;
            end
            S_i_total = S_i;
            rand_leader_index = floor(N * rand() + 1);
            X_rand = Positions(rand_leader_index, :);
            X_new = TargetPosition + 10 * S_i_total + rand * 0.00005 * ((TargetPosition + 0.005 * X_rand) / 2 - Positions(i, :));
            if flatFit
                fdbIndex = fitnessDistanceBalance(Positions, PositionsFitness);
            end
            % Eq. (31): the Levy guide is P_FDB instead of the target TP (paper Case-1)
            X_new = LevyFlights(X_new, Positions(fdbIndex, :), lb, ub);
            Positions_temp(i, :) = X_new;
            NN(i) = NeighborN;
        end

        Positions = min(max(Positions_temp, lb), ub);
        [PositionsFitness, FE] = calculate_fitness(Positions', problem, FE);
        PositionsFitness = PositionsFitness(:)';
        FES = FES + N;

        [xminn, x_pos_min] = min(PositionsFitness);
        if xminn < TargetFitness
            TargetPosition = Positions(x_pos_min, :);
            TargetFitness  = xminn;
        end

        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = TargetFitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, PositionsFitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    best_fitness  = TargetFitness;
    best_solution = TargetPosition;
end

% Levy Flights (Mantegna's algorithm)
function CP = LevyFlights(CP, DP, Lb, Ub)
    n = size(CP, 1);
    beta = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    for j = 1:n
        s = CP(j, :);
        u = randn(size(s)) * sigma;
        v = randn(size(s));
        step = u ./ abs(v) .^ (1 / beta);
        stepsize = 0.01 * step .* (s - DP);
        s = s + stepsize .* randn(size(s));
        CP(j, :) = simplebounds(s, Lb, Ub);
    end
end

% Simple bounds
function s = simplebounds(s, Lb, Ub)
    I = s < Lb; s(I) = Lb(I);
    J = s > Ub; s(J) = Ub(J);
end

% Initialization Function
function X = Initialization(N, dim, up, down)
    X = zeros(N, dim);
    for i = 1:dim
        X(:, i) = rand(N, 1) .* (up(i) - down(i)) + down(i);
    end
end

% Distance (comfort-zone metric, first two coordinates)
function d = Distance(a, b)
    d = sqrt((a(1) - b(1)) ^ 2 + (a(2) - b(2)) ^ 2);
end

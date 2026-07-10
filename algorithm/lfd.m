% ----------------------------------------------------------------------- %
% Levy Flight Distribution (LFD) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50          % Population size
%   threshold = 2   % Neighbourhood radius (comfort zone)
%
% Algorithm Concept:
%   - Inspired by Levy-flight random walks for exploring large spaces
%   - Agents interact with neighbours inside a comfort-zone threshold
%   - Levy flights generate long jumps toward target / random leaders
%   - Balances local refinement with occasional long exploratory steps
%
% Reference:
% Essam H. Houssein, Mohammed R. Saad, Fatma A. Hashim, Hassan Shaban, M. Hassaballah,
% Levy flight distribution: A new metaheuristic algorithm for solving engineering optimization problems,
% Engineering Applications of Artificial Intelligence 94 (2020) 103731
% https://doi.org/10.1016/j.engappai.2020.103731
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = lfd(problem)

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
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
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
            history_index, sampling_interval, history_size);
    end

    vec_flag = [1, -1];
    NN = [0, 1];
    l = 1;   % mirrors the reference FE counter (used only by the l==2 branch)

    while FE < maxFE
        [~, ll] = sort(NN);
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
                        if l == 2
                            rand_leader_index = floor(N * rand() + 1);
                            X_rand = Positions(rand_leader_index, :); %#ok<NASGU>
                        else
                            R = rand(); CSV = 0.5;
                            if R < CSV
                                rand_leader_index = floor(2 * rand() + 1);
                                X_rand = Positions(ll(rand_leader_index), :);
                                Positions_temp(j, :) = LevyFlights(Positions(j, :), X_rand, lb, ub);
                            else
                                Positions_temp(j, :) = lb(1) + rand(1, dim) * (ub(1) - lb(1));
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
            X_new = LevyFlights(X_new, TargetPosition, lb, ub);
            Positions_temp(i, :) = X_new;
            NN(i) = NeighborN;
        end

        Positions = Positions_temp;
        [PositionsFitness, FE] = calculate_fitness(Positions', problem, FE);
        PositionsFitness = PositionsFitness(:)';
        l = l + N;

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
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness  = TargetFitness;
    best_solution = TargetPosition;
end

%% --- Levy Flights (Mantegna's algorithm) ---
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

%% --- Simple bounds ---
function s = simplebounds(s, Lb, Ub)
    I = s < Lb; s(I) = Lb(I);
    J = s > Ub; s(J) = Ub(J);
end

%% --- Initialization Function ---
function X = Initialization(N, dim, up, down)
    X = zeros(N, dim);
    for i = 1:dim
        X(:, i) = rand(N, 1) .* (up(i) - down(i)) + down(i);
    end
end

%% --- Distance (comfort-zone metric, first two coordinates) ---
function d = Distance(a, b)
    d = sqrt((a(1) - b(1)) ^ 2 + (a(2) - b(2)) ^ 2);
end

% ----------------------------------------------------------------------- %
% Golden Jackal Optimization (GJO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 50   % Population size (jackals)
%
% Algorithm Concept:
%   - A male / female golden jackal pair (best / second-best) guide the pack
%   - Evading energy E1 decreases linearly; |E|>=1 exploration, |E|<1 exploitation
%   - Levy-flight based moves toward the jackal pair
%
% Reference:
% Nitish Chopra, Muhammad Mohsin Ansari,
% Golden jackal optimization: A novel nature-inspired optimizer for
% engineering applications,
% Expert Systems with Applications 198 (2022) 116924.
% https://doi.org/10.1016/j.eswa.2022.116924
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gjo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 50;
    Max_iter = ceil(maxFE / SearchAgents_no);

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    Male_Jackal_pos = zeros(1, dim);   Male_Jackal_score = inf;
    Female_Jackal_pos = zeros(1, dim); Female_Jackal_score = inf;

    Positions = initialization(SearchAgents_no, dim, ub, lb);

    % Evaluate the initial population and set the jackal pair
    [fitness, FE] = calculate_fitness(Positions', problem, FE);
    fitness = fitness(:)';
    for i = 1:SearchAgents_no
        if fitness(i) < Male_Jackal_score
            Male_Jackal_score = fitness(i);
            Male_Jackal_pos = Positions(i, :);
        end
        if fitness(i) > Male_Jackal_score && fitness(i) < Female_Jackal_score
            Female_Jackal_score = fitness(i);
            Female_Jackal_pos = Positions(i, :);
        end
    end
    best_pos = Male_Jackal_pos;

    for e = 1:SearchAgents_no
        if e <= maxFE
            curve(e) = Male_Jackal_score;
            [population_history, fitness_history, history_index] = record_history(...
                e, Positions, fitness, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    l = 0;
    while FE < maxFE
        l = l + 1;
        E1 = 1.5 * (1 - (l / Max_iter));
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5);

        NewPos = Positions;
        for i = 1:SearchAgents_no
            for j = 1:dim
                r1 = rand();
                E0 = 2 * r1 - 1;
                E = E1 * E0;                       % Evading energy
                if abs(E) < 1
                    % EXPLOITATION
                    D_male = abs(RL(i, j) * Male_Jackal_pos(j) - Positions(i, j));
                    Male_p = Male_Jackal_pos(j) - E * D_male;
                    D_female = abs(RL(i, j) * Female_Jackal_pos(j) - Positions(i, j));
                    Female_p = Female_Jackal_pos(j) - E * D_female;
                else
                    % EXPLORATION
                    D_male = abs(Male_Jackal_pos(j) - RL(i, j) * Positions(i, j));
                    Male_p = Male_Jackal_pos(j) - E * D_male;
                    D_female = abs(Female_Jackal_pos(j) - RL(i, j) * Positions(i, j));
                    Female_p = Female_Jackal_pos(j) - E * D_female;
                end
                NewPos(i, j) = (Male_p + Female_p) / 2;
            end
            NewPos(i, :) = bound(NewPos(i, :), ub, lb);
        end

        Positions = NewPos;
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        fitness = fitness(:)';

        for i = 1:SearchAgents_no
            if fitness(i) < Male_Jackal_score
                Male_Jackal_score = fitness(i);
                Male_Jackal_pos = Positions(i, :);
                best_pos = Male_Jackal_pos;
            end
            if fitness(i) > Male_Jackal_score && fitness(i) < Female_Jackal_score
                Female_Jackal_score = fitness(i);
                Female_Jackal_pos = Positions(i, :);
            end
            ec = FE - SearchAgents_no + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Male_Jackal_score;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = Male_Jackal_score;
    best_fitness = Male_Jackal_score;
    best_solution = best_pos;
end

%% --- Levy flight (n x m) ---
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = randn(n, m) * sigma_u;
    v = randn(n, m);
    z = u ./ (abs(v).^(1 / beta));
end

%% --- Initialization ---
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

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

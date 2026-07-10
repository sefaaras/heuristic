% ----------------------------------------------------------------------- %
% Archimedes Optimization Algorithm (AOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Materials_no = 30       % Population size (number of objects)
%   C1 = 2; C2 = 6; C3 = 1; C4 = 2;  % Paper default constants
%   u = 0.9; l = 0.1;       % Acceleration normalization range (Eq. 12)
%
% Algorithm Concept:
%   - Physics-based, inspired by Archimedes' principle of buoyancy
%   - Each object has a position, density, volume and acceleration
%   - Densities/volumes move toward the best; collision vs. equilibrium
%     phases are switched by the transfer operator TF
%   - Acceleration is normalized to balance exploration and exploitation
%
% Reference:
% Fatma A. Hashim, Kashif Hussain, Essam H. Houssein, Mai S. Mabrouk, Walid Al-Atabany,
% Archimedes optimization algorithm: a new metaheuristic algorithm for solving optimization problems,
% Applied Intelligence 51 (2021) 1531-1551
% https://doi.org/10.1007/s10489-020-01893-z
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = aoa(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    % AOA parameters
    Materials_no = 30;
    C1 = 2; C2 = 6; C3 = 1; C4 = 2;
    u = 0.9; l = 0.1;   % parameters in Eq. (12)

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage (1/10000 sampling, consistent with the framework)
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, Materials_no, dim);
    fitness_history = zeros(history_size, Materials_no);
    history_index = 1;

    % Initialization
    X = zeros(Materials_no, dim);
    acc = zeros(Materials_no, dim);
    for i = 1:Materials_no
        X(i, :) = rand(1, dim) .* (ub - lb) + lb;
        acc(i, :) = and(1, dim) .* (ub - lb) + lb;   % Eq. (initial acceleration)
    end

    den = rand(Materials_no, dim);   % Eq. (5)
    vol = rand(Materials_no, dim);

    % Evaluate initial population
    [Y, FE] = calculate_fitness(X', problem, FE);
    Y = Y(:)';

    [Scorebest, Score_index] = min(Y);
    Xbest    = X(Score_index, :);
    den_best = den(Score_index, :);
    vol_best = vol(Score_index, :);
    acc_best = acc(Score_index, :);
    acc_norm = acc;

    % Record initial evaluations
    for eval_count = 1:Materials_no
        curve(eval_count) = Scorebest;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, Y', population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    % Main loop (t used only for the transfer/density schedules)
    Max_iter = max(1, ceil((maxFE - Materials_no) / Materials_no));
    t = 0;

    while FE < maxFE
        t = t + 1;
        TF = exp((t - Max_iter) / Max_iter);   % Eq. (8)
        if TF > 1
            TF = 1;
        end
        d = exp((Max_iter - t) / Max_iter) - (t / Max_iter);   % Eq. (9)
        acc = acc_norm;
        r = rand();

        acc_temp = zeros(Materials_no, dim);
        for i = 1:Materials_no
            den(i, :) = den(i, :) + r * (den_best - den(i, :));   % Eq. (7)
            vol(i, :) = vol(i, :) + r * (vol_best - vol(i, :));
            if TF < 0.45   % collision
                mr = randi(Materials_no);
                acc_temp(i, :) = ((den(mr, :) .* vol(mr, :) .* acc(mr, :))) ./ (den(i, :) .* vol(i, :)) * rand;   % Eq. (10)
            else
                acc_temp(i, :) = (den_best .* vol_best .* acc_best) ./ (den(i, :) .* vol(i, :)) * rand;   % Eq. (11)
            end
        end

        acc_norm = ((u * (acc_temp - min(acc_temp(:)))) ./ (max(acc_temp(:)) - min(acc_temp(:)))) + l;   % Eq. (12)

        Xnew = zeros(Materials_no, dim);
        for i = 1:Materials_no
            if TF < 0.4
                for j = 1:dim
                    mrand = randi(Materials_no);
                    Xnew(i, j) = X(i, j) + C1 * rand * acc_norm(i, j) .* (X(mrand, j) - X(i, j)) * d;   % Eq. (13)
                end
            else
                for j = 1:dim
                    p = 2 * rand - C4;   % Eq. (15)
                    T = C3 * TF;
                    if T > 1
                        T = 1;
                    end
                    if p < 0.5
                        Xnew(i, j) = Xbest(j) + C2 * rand * acc_norm(i, j) .* (T * Xbest(j) - X(i, j)) * d;   % Eq. (14)
                    else
                        Xnew(i, j) = Xbest(j) - C2 * rand * acc_norm(i, j) .* (T * Xbest(j) - X(i, j)) * d;
                    end
                end
            end
        end

        Xnew = fun_checkpositions(Xnew, Materials_no, lb, ub);

        % Evaluate all candidates then apply greedy selection
        [Ynew, FE] = calculate_fitness(Xnew', problem, FE);
        Ynew = Ynew(:)';
        for i = 1:Materials_no
            if Ynew(i) < Y(i)
                X(i, :) = Xnew(i, :);
                Y(i)    = Ynew(i);
            end
        end

        [var_Ybest, var_index] = min(Y);
        if var_Ybest < Scorebest
            Scorebest = var_Ybest;
            Score_index = var_index;
            Xbest    = X(var_index, :);
            den_best = den(Score_index, :);
            vol_best = vol(Score_index, :);
            acc_best = acc_norm(Score_index, :);
        end

        % Record convergence curve and history for this generation
        for eval_idx = 1:Materials_no
            eval_count = FE - Materials_no + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = Scorebest;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, Y', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness  = Scorebest;
    best_solution = Xbest;
end

%% --- Boundary Handling ---
function vec_pos = fun_checkpositions(vec_pos, var_no_group, Lb, Ub)
    for i = 1:var_no_group
        isBelow    = vec_pos(i, :) < Lb;
        isAboveMax = vec_pos(i, :) > Ub;
        vec_pos(i, isBelow)    = Lb(isBelow);
        vec_pos(i, isAboveMax) = Ub(isAboveMax);
    end
end

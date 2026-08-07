% ----------------------------------------------------------------------- %
% Chaotic Evolution Optimization (CEO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Np = 30    % Population size (forced even for pairing)
%   N  = 10    % Number of chaotic candidates generated per individual
%   k  = 2.66  % Parameter of the exponential discrete memristor map
%   chaotic domains: low_c = [-0.5 -0.25], up_c = [0.5 0.25]
%
% Algorithm Concept:
%   - Random pairing splits the population into two halves P1 and P2
%   - Interval mapping (Eq. 4) sends both halves into the chaotic domain
%   - The exponential discrete memristor map (Eq. 2) is iterated N times to
%     produce N chaotic candidates per individual
%   - Inverse mapping (Eq. 5) brings them back to the search space
%   - Mutation (Eq. 7-8) blends the chaotic candidate with either the target
%     or the current best, followed by binomial crossover (Eq. 9)
%   - Greedy selection (Eq. 10) keeps the best of the N candidates
%
% Reference:
% Yingchao Dong, Shaohua Zhang, Hongli Zhang, Xiaojun Zhou, Jiading Jiang,
% Chaotic evolution optimization: A novel metaheuristic algorithm inspired
% by chaotic dynamics,
% Chaos, Solitons and Fractals 192 (2025) 116049.
% https://doi.org/10.1016/j.chaos.2025.116049
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ceo(problem)

    Dim    = problem.dimension;
    Varmin = problem.lb;
    Varmax = problem.ub;
    MaxFES = problem.maxFe;

    Np = 30;
    N  = 10;
    if mod(Np, 2) ~= 0
        Np = Np + 1;
    end

    FE    = 0;
    curve = zeros(1, MaxFES);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialise the population
    Population = Varmin + rand(Np, Dim) .* (Varmax - Varmin);
    [fit, FE] = calculate_fitness(Population', problem, FE);
    fit = fit(:);

    [fBest, idx] = min(fit);
    Best = Population(idx, :);
    bsf  = fBest;

    for eval_count = 1:min(Np, MaxFES)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Population, fit, population_history, fitness_history, ...
            history_index, MaxFES);
    end

    % Chaotic map parameters
    low_c = [-0.5 -0.25];
    up_c  = [ 0.5  0.25];

    % Main loop
    while FE < MaxFES

        % 1. Random pairing
        ridx = randperm(Np);
        idx1 = ridx(1:2:end);
        idx2 = ridx(2:2:end);
        P1 = Population(idx1, :);
        P2 = Population(idx2, :);

        % 2. Interval mapping -- Eq. (4)
        lo    = min(Population);
        delta = max(Population) - lo + eps;
        P1_dot = (P1 - lo) ./ delta * (up_c(1) - low_c(1)) + low_c(1);
        P2_dot = (P2 - lo) ./ delta * (up_c(2) - low_c(2)) + low_c(2);

        % 3. Chaotic iterations -- Eq. (2)
        [CX, CY] = EDM_Matrix(P1_dot, P2_dot, N);

        % 4. Inverse mapping -- Eq. (5)
        Targets = [P1; P2];
        idx_c = [ones(1, Np/2), 2 * ones(1, Np/2)];
        L_mat = reshape(low_c(idx_c), 1, [], 1);
        U_mat = reshape(up_c(idx_c),  1, [], 1);

        ChaosDot = (cat(2, CX, CY) - L_mat) ./ (U_mat - L_mat);
        ChaosDot = ChaosDot .* reshape(delta, 1, 1, []) + reshape(lo, 1, 1, []);
        ChaosDot = boundConstraint(ChaosDot, Varmin, Varmax);

        % 5. Mutation -- Eq. (7) and (8)
        Mask  = rand(1, Np) < 0.5;
        T_Exp = reshape(Targets, 1, Np, Dim) + zeros(N, 1, 1);
        B_Exp = reshape(Best, 1, 1, Dim) + zeros(N, Np, 1);
        XY_Hat = zeros(N, Np, Dim);
        if any(Mask)
            XY_Hat(:, Mask, :) = T_Exp(:, Mask, :) + ...
                rand(N, sum(Mask), 1) .* (ChaosDot(:, Mask, :) - T_Exp(:, Mask, :));
        end
        if any(~Mask)
            XY_Hat(:, ~Mask, :) = B_Exp(:, ~Mask, :) + ...
                rand(N, sum(~Mask), 1) .* (ChaosDot(:, ~Mask, :) - T_Exp(:, ~Mask, :));
        end

        % 6. Binomial crossover -- Eq. (9)
        Trials = crossover_binomial(T_Exp, XY_Hat, N, Np, Dim);
        Trials = boundConstraint(Trials, Varmin, Varmax);

        % 7. Selection -- Eq. (10)
        Trials_Flat = reshape(Trials, [], Dim);
        [fflat, FE] = calculate_fitness(Trials_Flat', problem, FE);
        Fit_Mat = reshape(fflat(:), N, Np);

        [min_val, min_idx] = min(Fit_Mat, [], 1);

        Update = min_val < fit([idx1, idx2])';
        idx_up = [idx1, idx2];
        idx_up = idx_up(Update);

        best_flat = (0:Np-1)' * N + min_idx';
        Population(idx_up, :) = Trials_Flat(best_flat(Update), :);
        fit(idx_up) = min_val(Update)';

        % Update the global best
        [fnew, idx] = min(fit);
        if fnew < fBest
            fBest = fnew;
            Best  = Population(idx, :);
        end
        if fBest < bsf
            bsf = fBest;
        end

        nAdd = N * Np;
        for k = 1:nAdd
            ec = FE - nAdd + k;
            if ec >= 1 && ec <= MaxFES
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Population, fit, population_history, fitness_history, ...
                    history_index, MaxFES);
            end
        end
    end

    curve(min(FE, MaxFES):end) = bsf;

    best_fitness  = fBest;
    best_solution = Best;
end

% Binomial crossover (at least one dimension from the mutant)
function Trials = crossover_binomial(Targets, Mutants, N, Np, Dim)
    Trials = Targets;
    CR_Mask = rand(N, Np, Dim) < rand(1, Np);
    ind = (1:N)' + (0:Np-1) * N + (randi(Dim, N, Np) - 1) * N * Np;
    CR_Mask(ind) = true;
    Trials(CR_Mask) = Mutants(CR_Mask);
end

% Vectorised exponential discrete memristor map -- Eq. (2)
function [X, Y] = EDM_Matrix(x, y, iter)
    k = 2.66;
    X = zeros(iter, size(x, 1), size(x, 2));
    Y = X;
    for j = 1:iter
        x_old = x;
        x = k * (exp(-cos(pi * y)) - 1) .* x;
        y = y + x_old;
        X(j, :, :) = x;
        Y(j, :, :) = y;
    end
end

% Mirror-reflection bound handling
function v = boundConstraint(v, Varmin, Varmax)
    sz = size(v);
    v = reshape(v, [], sz(end));
    low = Varmin + zeros(size(v, 1), 1);
    up  = Varmax + zeros(size(v, 1), 1);
    v(v < low) = min(up(v < low), 2 * low(v < low) - v(v < low));
    v(v > up)  = max(low(v > up), 2 * up(v > up)  - v(v > up));
    v = reshape(v, sz);
end

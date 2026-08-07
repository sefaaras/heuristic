% ----------------------------------------------------------------------- %
% Harris Hawks Optimization with Joint Opposite Selection (HHO-JOS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30     % Population size (hawks)
%   Jr = 0.25   % Jumping rate for Dynamic Opposite (DO)
%
% Algorithm Concept:
%   - Standard Harris Hawks Optimization (soft/hard besiege, team dives)
%   - Joint Opposite Selection (JOS) = Selective Leading Opposition (SLO)
%     applied each iteration + Dynamic Opposite (DO) applied with prob. Jr
%   - Energy E1 decays 2 -> 0 over the evaluation budget
%
% Reference:
% Fedhila Arini, Sirapat Chiewchanwattana, Chitsutha Soomlek, Khamron Sunat,
% Joint Opposite Selection (JOS): premiere joint of selective leading
% opposition and dynamic opposite enhanced Harris' hawks optimization,
% Expert Systems with Applications 188 (2022) 116001.
% https://doi.org/10.1016/j.eswa.2021.116001
% Base HHO: Heidari et al., Future Generation Computer Systems 97 (2019) 849-872.
% ----------------------------------------------------------------------- %
% Implementation Note:
% All evaluations go through a helper that clamps to the box and returns the
% clamped point, so the stored individual matches what was evaluated. The HHO
% reference leaves the rabbit-dive candidates X1 and X2 unbounded, and the
% numeric CEC mex evaluates outside the box without complaint.
% A hawk that moves without being evaluated in the same pass is clamped only
% before the next evaluation, so a mid-pass sample can hold out-of-box rows;
% clamping it at move time would feed a different mean(X) into Eq. (3).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hho_jos(problem)

    dim = problem.dimension;
    ub = problem.ub;
    lb = problem.lb;
    maxFE = problem.maxFe;

    N = 30;
    Jr = 0.25;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Rabbit_Location = zeros(1, dim);
    Rabbit_Energy = inf;
    Rabbit_Row_Id = 1;
    bsf = inf;
    best_pos = zeros(1, dim);

    upper = ub; lower = lb;   % opposition bounds tracking (stays [lb,ub] since X is clamped)

    X = initialization(N, dim, ub, lb);

    % Dynamic Opposite (DO) on the initial population
    X = dynamic_opposite(X, ub, lb, dim, N);

    fitness = inf(1, N);
    % Initial evaluation
    for i = 1:N
        X(i, :) = clampv(X(i, :), ub, lb);
        [f, FE, X(i, :)] = evalb(X(i, :), problem, FE);
        fitness(i) = f;
        if fitness(i) < Rabbit_Energy
            Rabbit_Energy = fitness(i);
            Rabbit_Location = X(i, :);
            Rabbit_Row_Id = i;
        end
        if fitness(i) < bsf
            bsf = fitness(i); best_pos = X(i, :);
        end
        if FE <= maxFE
            curve(FE) = bsf;
        end
        % fitness is still unset past i, so the population is only worth
        % recording once every hawk has been evaluated
        if i == N && FE <= maxFE
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end
    upper = max(upper, max(X, [], 1)); lower = min(lower, min(X, [], 1));

    while FE < maxFE
        E1 = 2 * (1 - FE / maxFE);   % decaying energy over the budget
        threshold = E1;

        % Hawks as they stood when fitness was measured; an entry moves only once
        % that hawk has been re-evaluated below, so the pair stays whole
        Xrec = X;

        % Selective Leading Opposition (SLO)
        X = corOppose2(X, ub, lb, upper, lower, threshold, Rabbit_Row_Id);

        % HHO movement
        for i = 1:N
            E0 = 2 * rand() - 1;
            Escaping_Energy = E1 * E0;

            if abs(Escaping_Energy) >= 1
                % Exploration
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);
                if q < 0.5
                    X(i, :) = X_rand - rand() * abs(X_rand - 2 * rand() * X(i, :));
                else
                    X(i, :) = (Rabbit_Location - mean(X)) - rand() * ((ub - lb) * rand + lb);
                end
            else
                % Exploitation
                r = rand();
                if r >= 0.5 && abs(Escaping_Energy) < 0.5
                    X(i, :) = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X(i, :));
                end
                if r >= 0.5 && abs(Escaping_Energy) >= 0.5
                    Jump_strength = 2 * (1 - rand());
                    X(i, :) = (Rabbit_Location - X(i, :)) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                end
                if r < 0.5 && abs(Escaping_Energy) >= 0.5
                    Jump_strength = 2 * (1 - rand());
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                    [fX1, FE, X1] = evalb(X1, problem, FE);
                    [fXi, FE, X(i, :)] = evalb(X(i, :), problem, FE);
                    [bsf, best_pos, curve] = ...
                        trackfe(fX1, X1, fXi, X(i, :), bsf, best_pos, curve, FE, maxFE);
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE, X2] = evalb(X2, problem, FE);
                        [fXi2, FE, X(i, :)] = evalb(X(i, :), problem, FE);
                        [bsf, best_pos, curve] = ...
                            trackfe(fX2, X2, fXi2, X(i, :), bsf, best_pos, curve, FE, maxFE);
                        if fX2 < fXi2
                            X(i, :) = X2;
                        end
                    end
                end
                if r < 0.5 && abs(Escaping_Energy) < 0.5
                    Jump_strength = 2 * (1 - rand());
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X));
                    [fX1, FE, X1] = evalb(X1, problem, FE);
                    [fXi, FE, X(i, :)] = evalb(X(i, :), problem, FE);
                    [bsf, best_pos, curve] = ...
                        trackfe(fX1, X1, fXi, X(i, :), bsf, best_pos, curve, FE, maxFE);
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE, X2] = evalb(X2, problem, FE);
                        [fXi2, FE, X(i, :)] = evalb(X(i, :), problem, FE);
                        [bsf, best_pos, curve] = ...
                            trackfe(fX2, X2, fXi2, X(i, :), bsf, best_pos, curve, FE, maxFE);
                        if fX2 < fXi2
                            X(i, :) = X2;
                        end
                    end
                end
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Dynamic Opposite (DO)
        if rand < Jr && FE + N < maxFE
            X = dynamic_opposite(X, ub, lb, dim, N);
        end

        % Evaluate the population
        X = min(max(X, lb), ub);
        for i = 1:N
            [f, FE, X(i, :)] = evalb(X(i, :), problem, FE);
            fitness(i) = f;
            Xrec(i, :) = X(i, :);
            if fitness(i) < Rabbit_Energy
                Rabbit_Energy = fitness(i);
                Rabbit_Location = X(i, :);
                Rabbit_Row_Id = i;
            end
            if fitness(i) < bsf
                bsf = fitness(i); best_pos = X(i, :);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Xrec, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        upper = max(upper, max(X, [], 1)); lower = min(lower, min(X, [], 1));
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

% Track best-so-far / curve for a pair of trial evaluations
function [bsf, best_pos, curve] = trackfe(fA, A, fB, B, bsf, best_pos, curve, FE, maxFE)
    % Two evaluations were just performed in order (A at FE-1, then B at FE).
    e1 = FE - 1; e2 = FE;
    if fA < bsf, bsf = fA; best_pos = A; end   % first eval (A) at FE-1
    if e1 >= 1 && e1 <= maxFE
        curve(e1) = bsf;
    end
    if fB < bsf, bsf = fB; best_pos = B; end   % second eval (B) at FE
    if e2 >= 1 && e2 <= maxFE
        curve(e2) = bsf;
    end
end

% Dynamic Opposite (DO)
function X = dynamic_opposite(X, ub, lb, dim, N)
    for i = 1:N
        OP = ((ub - lb) .* rand(size(lb))) + lb - X(i, :);
        RO = rand * OP;
        Dov = X(i, :) + rand * (RO - X(i, :));
        for j = 1:dim
            if Dov(j) < lb(1, j), Dov(j) = lb(1, j); end
            if Dov(j) > ub(1, j), Dov(j) = ub(1, j); end
        end
        X(i, :) = Dov;
    end
end

% Selective Leading Opposition (SLO)
function Positions = corOppose2(Positions, ~, ~, upper, lower, threshold, Rabbit_Row_Id)
    [n, b] = size(Positions);
    for i = 1:n
        if i ~= Rabbit_Row_Id
            ssum = 0; greater = []; less = [];
            y = 1; z = 1;
            for j = 1:b
                dd = abs(Positions(Rabbit_Row_Id, j) - Positions(i, j));
                if dd < threshold
                    greater(y) = j; y = y + 1; %#ok<AGROW>
                else
                    less(z) = j; z = z + 1; %#ok<AGROW>
                end
                ssum = ssum + dd * dd;
            end
            src = 1 - (double(6 * ssum)) / (double(n * (n * n - 1)));
            if src <= 0
                if size(greater) < size(less) %#ok<SZARLOG>
                    % (no operation, per reference)
                else
                    for jj = 1:numel(greater)
                        d2 = greater(jj);
                        Positions(i, d2) = (upper(1, d2) + lower(1, d2) - Positions(i, d2));
                    end
                end
            end
        end
    end
end

% Objective evaluation: clamps to the box and returns the clamped position
function [z, FE, pos] = evalb(pos, problem, FE)
    pos = min(max(pos, problem.lb), problem.ub);
    [z, FE] = calculate_fitness(pos', problem, FE);
end

% Levy flight (1 x d)
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma; v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

% Initialization
function X = initialization(N, dim, up, down)
    if size(up, 1) == 1
        X = rand(N, dim) .* repmat((up - down), N, 1) + repmat(down, N, 1);
    else
        X = zeros(N, dim);
        for i = 1:dim
            X(:, i) = rand(1, N) .* (up(i) - down(i)) + down(i);
        end
    end
end

% Clamp
function a = clampv(a, ub, lb)
    FU = a > ub; FL = a < lb;
    a = (a .* (~(FU + FL))) + ub .* FU + lb .* FL;
end

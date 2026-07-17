% ----------------------------------------------------------------------- %
% Leader Harris Hawks Optimization (LHHO) for unconstrained problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30   % Population size (hawks)
%
% Algorithm Concept:
%   - Standard Harris Hawks Optimization (rabbit = best-so-far, escaping
%     energy, soft/hard besiege, Levy dives) enhanced with:
%       (a) an adaptive perch probability (replacing HHO's fixed q), and
%       (b) a leader-based mutation & selection using the top-3 hawks.
%
% Reference:
% Manoj Kumar Naik, Rutuparna Panda, Aneesh Wunnava, Bibekananda Jena,
% Ajith Abraham,
% A leader Harris hawks optimization for 2-D Masi entropy-based multilevel
% image thresholding,
% Multimedia Tools and Applications 80 (2021) 35543-35583.
% https://doi.org/10.1007/s11042-020-10467-7
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = lhho(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    T = ceil(maxFE / (N * 3.5));

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    Rabbit_Location = zeros(1, dim);
    Rabbit_Energy = inf;
    fitnessAll = zeros(N, 1);

    % Running best-so-far (for a gap-free, monotone convergence curve)
    bsf = inf; bsf_sol = zeros(1, dim);

    X = initialization(N, dim, ub, lb);
    t = 0;

    for i = 1:size(X, 1)
        FU = X(i, :) > ub; FL = X(i, :) < lb;
        X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
        [fitness, FE] = calculate_fitness(X(i, :)', problem, FE);
        fitnessAll(i, 1) = fitness;
        if fitness < Rabbit_Energy
            Rabbit_Energy = fitness;
            Rabbit_Location = X(i, :);
        end
        if fitness < bsf, bsf = fitness; bsf_sol = X(i, :); end
        if FE <= maxFE
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fitnessAll', population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    while t < T && FE < maxFE
        sortedfitness = sort(fitnessAll);
        E1 = 2 * (1 - (t / T));

        for i = 1:size(X, 1)
            E0 = 2 * rand() - 1;
            Escaping_Energy = E1 * (E0);
            pepr = (abs(fitnessAll(i) - sortedfitness(1))) / abs((sortedfitness(end) - sortedfitness(1)));

            if abs(Escaping_Energy) >= 1
                %% Exploration
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);
                if q < pepr
                    X(i, :) = (Rabbit_Location(1, :) - mean(X)) - rand() * ((ub - lb) * rand + lb);
                elseif q >= pepr
                    X(i, :) = X_rand - rand() * abs(X_rand - 2 * rand() * X(i, :));
                end

            elseif abs(Escaping_Energy) < 1
                %% Exploitation
                r = rand();

                if r >= 0.5 && abs(Escaping_Energy) < 0.5   % Hard besiege
                    X(i, :) = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X(i, :));
                end

                if r >= 0.5 && abs(Escaping_Energy) >= 0.5  % Soft besiege
                    Jump_strength = 2 * (1 - rand());
                    X(i, :) = (Rabbit_Location - X(i, :)) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                end

                if r < 0.5 && abs(Escaping_Energy) >= 0.5   % Soft besiege, team dives
                    Jump_strength = 2 * (1 - rand());
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                    [fX1, FE] = calculate_fitness(X1', problem, FE);
                    if fX1 < bsf, bsf = fX1; bsf_sol = X1; end
                    if FE <= maxFE, curve(FE) = bsf; end
                    [fXi, FE] = calculate_fitness(X(i, :)', problem, FE);
                    if fXi < bsf, bsf = fXi; bsf_sol = X(i, :); end
                    if FE <= maxFE, curve(FE) = bsf; end
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE] = calculate_fitness(X2', problem, FE);
                        if fX2 < bsf, bsf = fX2; bsf_sol = X2; end
                        if FE <= maxFE, curve(FE) = bsf; end
                        [fXi2, FE] = calculate_fitness(X(i, :)', problem, FE);
                        if fXi2 < bsf, bsf = fXi2; bsf_sol = X(i, :); end
                        if FE <= maxFE, curve(FE) = bsf; end
                        if fX2 < fXi2
                            X(i, :) = X2;
                        end
                    end
                end

                if r < 0.5 && abs(Escaping_Energy) < 0.5    % Hard besiege, team dives
                    Jump_strength = 2 * (1 - rand());
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X));
                    [fX1, FE] = calculate_fitness(X1', problem, FE);
                    if fX1 < bsf, bsf = fX1; bsf_sol = X1; end
                    if FE <= maxFE, curve(FE) = bsf; end
                    [fXi, FE] = calculate_fitness(X(i, :)', problem, FE);
                    if fXi < bsf, bsf = fXi; bsf_sol = X(i, :); end
                    if FE <= maxFE, curve(FE) = bsf; end
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE] = calculate_fitness(X2', problem, FE);
                        if fX2 < bsf, bsf = fX2; bsf_sol = X2; end
                        if FE <= maxFE, curve(FE) = bsf; end
                        [fXi2, FE] = calculate_fitness(X(i, :)', problem, FE);
                        if fXi2 < bsf, bsf = fXi2; bsf_sol = X(i, :); end
                        if FE <= maxFE, curve(FE) = bsf; end
                        if fX2 < fXi2
                            X(i, :) = X2;
                        end
                    end
                end
            end

            if FE >= maxFE, break; end
        end

        if FE >= maxFE, break; end

        for i = 1:size(X, 1)
            FU = X(i, :) > ub; FL = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
            [fitness, FE] = calculate_fitness(X(i, :)', problem, FE);
            fitnessAll(i, 1) = fitness;
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness;
                Rabbit_Location = X(i, :);
            end
            if fitness < bsf, bsf = fitness; bsf_sol = X(i, :); end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fitnessAll', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end

        if FE >= maxFE, break; end

        [~, fitSind] = sort(fitnessAll);
        Xa = X(fitSind(1), :); Xb = X(fitSind(2), :); Xd = X(fitSind(3), :);
        for i = 1:size(X, 1)
            % Leader-based mutation and selection
            Xnew = X(i, :) + E1 * (2 * rand - 1) * (2 * Xa - Xb - Xd) + (2 * rand - 1) * (Xa - X(i, :));
            [fitness, FE] = calculate_fitness(Xnew', problem, FE);
            if fitness < fitnessAll(i)
                X(i, :) = Xnew;
            end
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness;
                Rabbit_Location = Xnew;
            end
            if fitness < bsf, bsf = fitness; bsf_sol = Xnew; end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fitnessAll', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end

        t = t + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_solution = bsf_sol;
    best_fitness = bsf;
end

%% --- Initialization ---
function [X] = initialization(N, dim, up, down)
    for i = 1:dim
        high = up(i); low = down(i);
        X(:, i) = rand(1, N) .* (high - low) + low;
    end
end

%% --- Levy flight ---
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma; v = randn(1, d); step = u ./ abs(v).^(1 / beta);
    o = step;
end

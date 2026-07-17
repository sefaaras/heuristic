% ----------------------------------------------------------------------- %
% Honey Badger Algorithm (HBA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 30    % Population size (honey badgers)
%   beta = 6     % Ability of HB to get the food (Eq. 4)
%   C    = 2     % Constant in density factor (Eq. 3)
%
% Algorithm Concept:
%   - Two foraging behaviours: "digging" (smell-intensity driven) and
%     "honey" phase (following the honey guide)
%   - Prey intensity I (Eq. 2) and time-decaying density factor alpha (Eq. 3)
%     balance exploration and exploitation around the prey (best-so-far).
%
% Reference:
% Fatma A. Hashim, Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk,
% Walid Al-Atabany,
% Honey Badger Algorithm: New metaheuristic algorithm for solving
% optimization problems,
% Mathematics and Computers in Simulation 192 (2022) 84-110.
% https://doi.org/10.1016/j.matcom.2021.08.013
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = hba(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;                       % Population size
    Max_iter = ceil(maxFE / N);  % Maximum number of iterations
    beta = 6;                     % Ability of HB to get the food (Eq. 4)
    C = 2;                        % Constant in Eq. (3)
    vec_flag = [1, -1];

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    % Initialization
    X = initialization(N, dim, ub, lb);
    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);
    [GYbest, gbest] = min(fitness);
    Xprey = X(gbest, :);

    % Running best-so-far (for the convergence curve only; never fed back)
    bsf = GYbest;

    % Record initial evaluations
    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, fitness, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    for t = 1:Max_iter
        if FE >= maxFE, break; end
        alpha = C * exp(-t / Max_iter);   % density factor (Eq. 3)
        I = Intensity(N, Xprey, X);       % intensity (Eq. 2)
        for i = 1:N
            r = rand();
            F = vec_flag(floor(2 * rand() + 1));
            Xnew = X(i, :);
            for j = 1:dim
                di = ((Xprey(j) - X(i, j)));
                if r < .5
                    r3 = rand; r4 = rand; r5 = rand;
                    Xnew(j) = Xprey(j) + F * beta * I(i) * Xprey(j) + F * r3 * alpha * (di) * abs(cos(2 * pi * r4) * (1 - cos(2 * pi * r5)));
                else
                    r7 = rand;
                    Xnew(j) = Xprey(j) + F * r7 * alpha * di;
                end
            end
            FU = Xnew > ub; FL = Xnew < lb;
            Xnew = (Xnew .* (~(FU + FL))) + ub .* FU + lb .* FL;

            [tempFitness, FE] = calculate_fitness(Xnew', problem, FE);
            if tempFitness < fitness(i)
                fitness(i) = tempFitness;
                X(i, :) = Xnew;
            end

            if tempFitness < bsf
                bsf = tempFitness;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end
        for bi = 1:N
            FU = X(i, :) > ub;
            FL = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
        end
        [Ybest, index] = min(fitness);
        if Ybest < GYbest
            GYbest = Ybest;
            Xprey = X(index, :);
        end
    end

    % Fill remaining curve values
    curve(min(FE, maxFE):end) = bsf;

    best_fitness = GYbest;
    best_solution = Xprey;
end

%% --- Prey Intensity (Eq. 2) ---
function I = Intensity(N, Xprey, X)
    for i = 1:N - 1
        di(i) = (norm((X(i, :) - Xprey + eps))).^2;
        S(i) = (norm((X(i, :) - X(i + 1, :) + eps))).^2;
    end
    di(N) = (norm((X(N, :) - Xprey + eps))).^2;
    S(N) = (norm((X(N, :) - X(1, :) + eps))).^2;
    for i = 1:N
        r2 = rand;
        I(i) = r2 * S(i) / (4 * pi * di(i));
    end
end

%% --- Initialization ---
function [X] = initialization(N, dim, up, down)
    if size(up, 2) == 1
        X = rand(N, dim) .* (up - down) + down;
    end
    if size(up, 2) > 1
        for i = 1:dim
            high = up(i); low = down(i);
            X(:, i) = rand(N, 1) .* (high - low) + low;
        end
    end
end

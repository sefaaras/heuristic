% ----------------------------------------------------------------------- %
% Nonlinear-based Chaotic Harris Hawks Optimizer (NCHHO) for unconstrained
% benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 100               % Population size (number of hawks)
%   a1 = 4, teta = 0.7    % Chaotic (sine) map parameters
%
% Algorithm Concept:
%   - Harris Hawks Optimizer enhanced with a chaotic sine map (to enrich
%     exploration) and a nonlinear control parameter (adjusting the
%     exploration/exploitation balance)
%
% NOTE ON DIRECTION:
%   The reference implementation is written for maximization
%   (Rabbit_Energy = 0 and ">" comparisons). It has been converted here to
%   minimization (Rabbit_Energy = inf and "<" comparisons, as in the
%   canonical HHO) so that it is consistent with the CEC minimization
%   benchmark suite. All update equations and parameters are unchanged.
%
% Reference:
% Amin Abdollahi Dehkordi, Ali Safaa Sadiq, Seyedali Mirjalili,
% Kayhan Zrar Ghafoor,
% Nonlinear-based Chaotic Harris Hawks Optimizer: Algorithm and Internet of
% Vehicles application,
% Applied Soft Computing 109 (2021) 107574
% https://doi.org/10.1016/j.asoc.2021.107574
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = nchho(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 100;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    T = maxFE / (N * 2.5);

    % initialize the location and Energy of the rabbit (minimization)
    Rabbit_Location = zeros(1, dim);
    Rabbit_Energy = inf;

    % best-so-far (framework convention)
    bf = inf;
    bs = zeros(1, dim);

    % Initialize the locations of Harris' hawks
    X = initialization(N, dim, ub, lb);

    fitness_all = inf(1, N);
    t = 0;
    while FE < maxFE
        FE_before = FE;

        for i = 1:size(X, 1)
            % Check boundaries
            FU = X(i, :) > ub; FL = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
            % fitness of locations
            [fitness, FE, bf, bs] = fobj(X(i, :), problem, FE, bf, bs);
            fitness_all(i) = fitness;
            % Update the location of Rabbit (minimization)
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness;
                Rabbit_Location = X(i, :);
            end
            if FE >= maxFE, break; end
        end

        % Consistent (position, fitness) snapshot for history
        X_snap = X;

        E1 = abs(2 * (1 - (t / T)) - 2);   % factor showing the decreasing energy of rabbit
        a1 = 4;                            % chaotic map parameter
        teta = 0.7;                        % chaotic map parameter

        % Update the location of Harris' hawks
        for i = 1:size(X, 1)
            if FE >= maxFE, break; end
            Cm = zeros(1, 4);
            for ii = 1:4
                Cm(1, ii) = abs((a1 / 4) * sin(pi * teta));
                teta = Cm(1, ii);
            end
            E0 = 2 * rand() - 1;              % -1<E0<1
            Escaping_Energy = E1 * (E0);      % escaping energy of rabbit

            if abs(Escaping_Energy) >= 1
                %% Exploration
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);
                if q < 0.5
                    X(i, :) = X_rand - Cm(1, 1) * abs(X_rand - 2 * Cm(1, 2) * X(i, :));
                elseif q >= 0.5
                    X(i, :) = (Rabbit_Location(1, :) - mean(X)) - Cm(1, 3) * ((ub - lb) * Cm(1, 4) + lb);
                end

            elseif abs(Escaping_Energy) < 1
                %% Exploitation
                r = rand();   % probability of each event

                if r >= 0.5 && abs(Escaping_Energy) < 0.5   % Hard besiege
                    X(i, :) = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X(i, :));
                end

                if r >= 0.5 && abs(Escaping_Energy) >= 0.5  % Soft besiege
                    Jump_strength = 2 * (1 - rand());
                    X(i, :) = (Rabbit_Location - X(i, :)) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                end

                %% phase 2: team rapid dives (leapfrog movements)
                if r < 0.5 && abs(Escaping_Energy) >= 0.5   % Soft besiege
                    w1 = 2 * exp(-(8 * t / T)^2);
                    Jump_strength = 2 * (1 - rand());
                    X1 = w1 * Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :));
                    [fX1, FE, bf, bs] = fobj(X1, problem, FE, bf, bs);
                    [fXi, FE, bf, bs] = fobj(X(i, :), problem, FE, bf, bs);
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = w1 * Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i, :)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE, bf, bs] = fobj(X2, problem, FE, bf, bs);
                        if fX2 < fXi
                            X(i, :) = X2;
                        end
                    end
                end

                if r < 0.5 && abs(Escaping_Energy) < 0.5   % Hard besiege
                    w1 = 2 * exp(-(8 * t / T)^2);
                    Jump_strength = 2 * (1 - rand());
                    X1 = w1 * Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X));
                    [fX1, FE, bf, bs] = fobj(X1, problem, FE, bf, bs);
                    [fXi, FE, bf, bs] = fobj(X(i, :), problem, FE, bf, bs);
                    if fX1 < fXi
                        X(i, :) = X1;
                    else
                        X2 = w1 * Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X)) + rand(1, dim) .* Levy(dim);
                        [fX2, FE, bf, bs] = fobj(X2, problem, FE, bf, bs);
                        if fX2 < fXi
                            X(i, :) = X2;
                        end
                    end
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = bf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X_snap, fitness_all, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        t = t + 1;
    end

    best_solution = bs;
    best_fitness = bf;

end

%% --- Objective evaluation: threads FE and best-so-far ---
function [z, FE, bf, bs] = fobj(pos, problem, FE, bf, bs)
    [z, FE] = calculate_fitness(pos', problem, FE);
    if z < bf
        bf = z;
        bs = pos;
    end
end

%% --- Initialization ---
function [X] = initialization(N, dim, up, down)
    X = zeros(N, dim);
    for i = 1:dim
        high = up(i); low = down(i);
        X(:, i) = rand(1, N) .* (high - low) + low;
    end
end

%% --- Levy Flight ---
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

% ----------------------------------------------------------------------- %
% Aquila Optimizer (AO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20                % Population size
%   alpha = 0.1, delta = 0.1   % Exploitation adjustment parameters
%
% Algorithm Concept:
%   - Inspired by the four hunting strategies of the Aquila (eagle)
%   - Expanded/narrowed exploration (high soar, contour flight) for the
%     first 2/3 of the run, then expanded/narrowed exploitation (low flight,
%     walk-and-grab) afterwards
%
% Reference:
% Laith Abualigah, Dalia Yousri, Mohamed Abd Elaziz, Ahmed A. Ewees,
% Mohammed A.A. Al-qaness, Amir H. Gandomi,
% Aquila Optimizer: A novel meta-heuristic optimization algorithm,
% Computers & Industrial Engineering 157 (2021) 107250
% https://doi.org/10.1016/j.cie.2021.107250
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ao(problem)

    % Extract problem parameters
    Dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    N = 20;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, Dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    T = ceil(maxFE / (N * 2));

    Best_P = zeros(1, Dim);
    Best_FF = inf;

    X = initialization(N, Dim, UB, LB);
    Xnew = X;
    Ffun = zeros(1, size(X, 1));
    Ffun_new = zeros(1, size(Xnew, 1));

    t = 1;
    alpha = 0.1;
    delta = 0.1;

    while FE < maxFE
        FE_before = FE;

        for i = 1:size(X, 1)
            F_UB = X(i, :) > UB;
            F_LB = X(i, :) < LB;
            X(i, :) = (X(i, :) .* (~(F_UB + F_LB))) + UB .* F_UB + LB .* F_LB;
            [Ffun(1, i), FE] = calculate_fitness(X(i, :)', problem, FE);
            if Ffun(1, i) < Best_FF
                Best_FF = Ffun(1, i);
                Best_P = X(i, :);
            end
            if FE >= maxFE, break; end
        end

        G2 = 2 * rand() - 1;         % Eq. (16)
        G1 = 2 * (1 - (t / T));      % Eq. (17)
        to = 1:Dim;
        u = .0265;
        r0 = 10;
        r = r0 + u * to;
        omega = .005;
        phi0 = 3 * pi / 2;
        phi = -omega * to + phi0;
        x = r .* sin(phi);           % Eq. (9)
        y = r .* cos(phi);           % Eq. (10)
        QF = t^((2 * rand() - 1) / (1 - T)^2);   % Eq. (15)

        for i = 1:size(X, 1)
            if FE >= maxFE, break; end
            if t <= (2 / 3) * T
                if rand < 0.5
                    Xnew(i, :) = Best_P(1, :) * (1 - t / T) + (mean(X(i, :)) - Best_P(1, :)) * rand();   % Eq. (3) and Eq. (4)
                    [Ffun_new(1, i), FE] = calculate_fitness(Xnew(i, :)', problem, FE);
                    if Ffun_new(1, i) < Ffun(1, i)
                        X(i, :) = Xnew(i, :);
                        Ffun(1, i) = Ffun_new(1, i);
                    end
                else
                    Xnew(i, :) = Best_P(1, :) .* Levy(Dim) + X((floor(N * rand() + 1)), :) + (y - x) * rand;   % Eq. (5)
                    [Ffun_new(1, i), FE] = calculate_fitness(Xnew(i, :)', problem, FE);
                    if Ffun_new(1, i) < Ffun(1, i)
                        X(i, :) = Xnew(i, :);
                        Ffun(1, i) = Ffun_new(1, i);
                    end
                end
            else
                if rand < 0.5
                    Xnew(i, :) = (Best_P(1, :) - mean(X)) * alpha - rand + ((UB - LB) * rand + LB) * delta;   % Eq. (13)
                    [Ffun_new(1, i), FE] = calculate_fitness(Xnew(i, :)', problem, FE);
                    if Ffun_new(1, i) < Ffun(1, i)
                        X(i, :) = Xnew(i, :);
                        Ffun(1, i) = Ffun_new(1, i);
                    end
                else
                    Xnew(i, :) = QF * Best_P(1, :) - (G2 * X(i, :) * rand) - G1 .* Levy(Dim) + rand * G2;   % Eq. (14)
                    [Ffun_new(1, i), FE] = calculate_fitness(Xnew(i, :)', problem, FE);
                    if Ffun_new(1, i) < Ffun(1, i)
                        X(i, :) = Xnew(i, :);
                        Ffun(1, i) = Ffun_new(1, i);
                    end
                end
            end
            if Ffun(1, i) < Best_FF
                Best_FF = Ffun(1, i);
                Best_P = X(i, :);
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Best_FF;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, Ffun, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        t = t + 1;
    end

    best_solution = Best_P;
    best_fitness = Best_FF;

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

%% --- Initialization Function ---
function X = initialization(N, Dim, UB, LB)
    B_no = size(UB, 2);
    if B_no == 1
        X = rand(N, Dim) .* (UB - LB) + LB;
    end
    if B_no > 1
        for i = 1:Dim
            Ub_i = UB(i);
            Lb_i = LB(i);
            X(:, i) = rand(N, 1) .* (Ub_i - Lb_i) + Lb_i;
        end
    end
end

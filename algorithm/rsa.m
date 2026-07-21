% ----------------------------------------------------------------------- %
% Reptile Search Algorithm (RSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 10     % Population size
%   Alpha = 0.1    % Sensitive parameter (hunting cooperation)
%   Beta  = 0.005  % Sensitive parameter (encircling accuracy)
%
% Algorithm Concept:
%   - Four behaviors split across the iteration budget T:
%       t<T/4 high walking, T/4<=t<T/2 belly walking (exploration),
%       T/2<=t<3T/4 hunting coordination, t>=3T/4 hunting cooperation
%   - Evolutionary Sense ES = 2*randn*(1 - t/T) drives the belly-walking phase
%
% Reference:
% Laith Abualigah, Mohamed Abd Elaziz, Putra Sumari, Zong Woo Geem,
% Amir H. Gandomi,
% Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic
% optimizer, Expert Systems with Applications 191 (2022) 116158.
% https://doi.org/10.1016/j.eswa.2021.116158
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rsa(problem)

    Dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    N = 10;
    Alpha = 0.1;
    Beta = 0.005;
    % Only agents 2..N are updated each iteration -> N-1 evaluations per iter
    T = max(1, ceil((maxFE - N) / (N - 1)));   % iteration budget (drives the phases)

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, Dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    X = initialization(N, Dim, UB, LB);
    [Ffun, FE] = calculate_fitness(X', problem, FE);
    Ffun = Ffun(:)';

    [Best_F, bidx] = min(Ffun);
    Best_P = X(bidx, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = Best_F;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, Ffun, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    t = 0;
    lin = (0:Dim - 1) * N;   % column offsets for linear indexing of X
    while FE < maxFE
        t = t + 1;
        ES = 2 * randn * (1 - (t / T));   % Evolutionary Sense (probability ratio)

        Xnew = X;
        for i = 2:N
            Xrand1 = X(randi(N, 1, Dim) + lin);   % random reptile per dim (for R)
            Xrand2 = X(randi(N, 1, Dim) + lin);   % second random reptile (belly-walking)
            R = Best_P - Xrand1 ./ (Best_P + eps);
            P = Alpha + (X(i, :) - mean(X(i, :))) ./ (Best_P .* (UB - LB) + eps);
            Eta = Best_P .* P;
            r = rand(1, Dim);
            if t < T / 4
                Xnew(i, :) = Best_P - Eta * Beta - R .* r;
            elseif t < 2 * T / 4
                Xnew(i, :) = Best_P .* Xrand2 .* ES .* r;
            elseif t < 3 * T / 4
                Xnew(i, :) = Best_P .* P .* r;
            else
                Xnew(i, :) = Best_P - Eta * eps - R .* r;
            end
            Xnew(i, :) = bound(Xnew(i, :), UB, LB);
        end

        [Ffun_new, FE] = calculate_fitness(Xnew(2:N, :)', problem, FE);
        Ffun_new = Ffun_new(:)';

        for k = 1:(N - 1)
            i = k + 1;
            if Ffun_new(k) < Ffun(i)
                X(i, :) = Xnew(i, :);
                Ffun(i) = Ffun_new(k);
                if Ffun(i) < Best_F
                    Best_F = Ffun(i);
                    Best_P = X(i, :);
                end
            end
            ec = FE - (N - 1) + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Best_F;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, Ffun, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = Best_F;
    best_fitness = Best_F;
    best_solution = Best_P;
end

%% --- Initialization ---
function X = initialization(N, Dim, UB, LB)
    B_no = size(UB, 2);
    if B_no == 1
        X = rand(N, Dim) .* (UB - LB) + LB;
    else
        X = zeros(N, Dim);
        for i = 1:Dim
            X(:, i) = rand(N, 1) .* (UB(i) - LB(i)) + LB(i);
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

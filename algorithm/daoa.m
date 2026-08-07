% ----------------------------------------------------------------------- %
% Dynamic Arithmetic Optimization Algorithm (DAOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 5      % Population size
%   Mu    = 0.001  % Control parameter
%   alpha = 25     % Sensitive parameter of the DAF
%
% Algorithm Concept:
%   - Replaces the original AOA's MOA/MOP with a Dynamic Accelerated
%     Function (DAF) and a Dynamic Candidate Solution (DCS) factor
%   - DAF = (M_Iter + 1/C_Iter)^alpha  (division/multiplication phase gate)
%   - DCS = 0.99*(1 - (C_Iter/M_Iter)^0.5)  (shrinking step around best)
%
% Reference:
% Nima Khodadadi, Vaclav Snasel, Seyedali Mirjalili,
% Dynamic Arithmetic Optimization Algorithm for Truss Optimization Under
% Natural Frequency Constraints,
% IEEE Access, vol. 10, pp. 16188-16208, 2022.
% https://doi.org/10.1109/ACCESS.2022.3146374
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = daoa(problem)

    Dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    N = 5;
    M_Iter = ceil(maxFE / N);
    Mu = 0.001;
    alpha = 25;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = initialization(N, Dim, UB, LB);
    [Ffun, FE] = calculate_fitness(X', problem, FE);
    Ffun = Ffun(:)';

    [Best_FF, bidx] = min(Ffun);
    Best_P = X(bidx, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = Best_FF;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, Ffun, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    C_Iter = 1;
    while FE < maxFE
        DAF = (M_Iter + 1 / C_Iter)^(alpha);         % DAF2
        DCS = 0.99 * (1 - (C_Iter / M_Iter)^(0.5));   % DCS

        term = (UB - LB) .* Mu + LB;                  % 1 x Dim
        Xnew = X;
        for i = 1:N
            r1 = rand(1, Dim); r2 = rand(1, Dim); r3 = rand(1, Dim);
            branch1 = (r2 > 0.5) .* (Best_P ./ (DCS + eps) .* term) + (r2 <= 0.5) .* (Best_P .* DCS .* term);
            branch2 = (r3 > 0.5) .* (Best_P - DCS .* term)         + (r3 <= 0.5) .* (Best_P + DCS .* term);
            Xnew(i, :) = (r1 < DAF) .* branch1 + (r1 >= DAF) .* branch2;
            Xnew(i, :) = bound(Xnew(i, :), UB, LB);
        end

        [Ffun_new, FE] = calculate_fitness(Xnew', problem, FE);
        Ffun_new = Ffun_new(:)';

        for i = 1:N
            if Ffun_new(i) < Ffun(i)
                X(i, :) = Xnew(i, :);
                Ffun(i) = Ffun_new(i);
                if Ffun(i) < Best_FF
                    Best_FF = Ffun(i);
                    Best_P = X(i, :);
                end
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Best_FF;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, Ffun, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        C_Iter = C_Iter + 1;
        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = Best_FF;
    best_fitness = Best_FF;
    best_solution = Best_P;
end

% Initialization
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

% Boundary Handling
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

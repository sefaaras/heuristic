% ----------------------------------------------------------------------- %
% Emperor Penguin Optimizer (EPO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 80                     % Population size (penguins), Table 1
%   M            = 2                      % Movement parameter of Eq. (9)
%   f, l         = U(2, 3), U(1.5, 2)     % Social-force constants of Eq. (12)
%   T'           = T - Max/(x - Max)      % Temperature profile, 2 -> Max + 1 (T = 1)
%
% Algorithm Concept:
%   - Every coordinate of every penguin is redrawn each iteration around the
%     best ever evaluated position P; no selection, no other memory
%   - Collision avoidance A = M*(T' + |P - x|)*rand - T', and C = rand
%   - Social force S = (sqrt(f*e^(-x/l) - e^(-x)))^2 with x the iteration,
%     ~0 after a few dozen iterations
%   - Distance D = |S*P - C*x|; relocation x = P - A*D, clamped to the box
%
% Reference:
% Gaurav Dhiman, Vijay Kumar,
% Emperor penguin optimizer: A bio-inspired algorithm for engineering
% problems,
% Knowledge-Based Systems 159 (2018) 20-50.
% https://doi.org/10.1016/j.knosys.2018.06.001
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author code exists (the first author's site lists EPO code under an empty
% link; his File Exchange page has five other algorithms): ported from the
% paper's Eqs. (7)-(13) and Algorithm, opytimizer epo.py (2026-09) as a
% cross-check only. Literal where opytimizer is not: R = rand never exceeds 1,
% so T = 1 always, and S keeps its sqrt. Left open, chosen here: A, C, f, l
% drawn per penguin and coordinate, f and l uniform in their ranges; x counts
% 0..Max-1 as in the authors' STOA/SHO releases; clamping to the box. Line 4's
% FITNESS call re-evaluates the points line 21 just evaluated and is skipped,
% so Max = ceil((maxFe - N)/N) iterations of N evaluations, the last one
% truncated to the budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = epo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 80;
    M = 2;
    Max = max(1, ceil((maxFE - N) / N));   % Max_iteration: N new evaluations per iteration

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    nev = min(N, maxFE);
    [fit, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
    fit = fit(:)';

    fbest = Inf;
    P = X(1, :);
    for e = 1:nev
        if fit(e) < fbest
            fbest = fit(e);
            P = X(e, :);
        end
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X(1:nev, :), fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    Xrec = X;   % position at which each entry of fit was measured
    x = 0;
    while FE < maxFE
        R = rand();
        if R > 1   % Eq. (7) as printed: never true for R = rand
            T = 0;
        else
            T = 1;
        end
        Tp = T - Max / (x - Max);   % Eq. (7)

        for i = 1:N
            for j = 1:dim
                Pgrid = abs(P(j) - X(i, j));            % Eq. (10)
                A = (M * (Tp + Pgrid) * rand()) - Tp;   % Eq. (9)
                C = rand();                             % Eq. (11)
                f = 2 + rand();
                l = 1.5 + 0.5 * rand();
                S = (sqrt(f * exp(-x / l) - exp(-x)))^2;   % Eq. (12), x the current iteration
                Dep = abs(S * P(j) - C * X(i, j));         % Eq. (8)
                X(i, j) = P(j) - A * Dep;                  % Eq. (13)
            end
        end
        X = min(max(X, lb), ub);   % Algorithm line 20: amend agents beyond the search space
        x = x + 1;

        nev = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for i = 1:nev
            Xrec(i, :) = X(i, :);
            fit(i) = fnew(i);
            if fit(i) < fbest
                fbest = fit(i);
                P = X(i, :);
            end
            ec = FE - nev + i;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = P;
end

% Initialization, one column per dimension
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

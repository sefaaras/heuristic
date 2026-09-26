% ----------------------------------------------------------------------- %
% Arithmetic Optimization Algorithm (AOA)
% Stored as arithoa; the acronym AOA collides with aoa (Archimedes)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                      % Population size
%   Alpha = 5                   % Exponent of the MOP schedule
%   Mu = 0.499                  % Control parameter of the step scale
%   MOA = 0.2 -> 1 (linear)     % Math Optimizer Accelerated, t the spent budget
%   MOP = 1 - t^(1/Alpha)       % Math Optimizer Probability, 1 -> 0
%
% Algorithm Concept:
%   - Every coordinate of a new solution is built from the best solution alone;
%     the member's own position never enters the update
%   - With probability MOA per coordinate: division best/(MOP+eps)*c or
%     multiplication best*MOP*c, c = (ub-lb)*Mu + lb (Eq. 3)
%   - Otherwise subtraction or addition, best -+ MOP*c (Eq. 5)
%   - On a box symmetric about 0, c is ~0, so multiplication draws coordinates
%     to the origin and division throws them to the bounds as MOP -> 0
%   - Coordinates are clamped to the box; a new solution replaces its member only
%     if better, and the best is updated after every member
%
% Reference:
% Laith Abualigah, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz,
% Amir H. Gandomi,
% The Arithmetic Optimization Algorithm,
% Computer Methods in Applied Mechanics and Engineering 376 (2021) 113609.
% https://doi.org/10.1016/j.cma.2020.113609
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/laithabualigah/Arithmetic-
% Optimization-Algorithm-AOA-, AOA.m, per-dimension-bounds branch), which differs
% from the paper: D/M run when r1 < MOA where Algorithm 1 has r1 > MOA, so their
% share grows over the run instead of shrinking; MOA ends at 1, not 0.9; Mu is
% 0.499, not 0.5; and a new solution is kept only if it improves its member,
% a selection Algorithm 1 does not have. N = 30 is the paper's setting (the demo
% driver uses 20). C_Iter/M_Iter becomes FE/maxFe. The 0/1 bound blend is a
% clamp here, and a non-finite coordinate is redrawn uniformly.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = arithoa(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    N = 30;
    Alpha = 5;
    Mu = 0.499;
    MOP_Max = 1;
    MOP_Min = 0.2;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    fit = inf(N, 1);
    bsf = inf;
    bsf_solution = X(1, :);

    n = min(N, maxFE);
    [fv, FE] = calculate_fitness(X(1:n, :)', problem, FE);
    fit(1:n) = fv(:);
    for k = 1:n
        if fit(k) < bsf
            bsf = fit(k);
            bsf_solution = X(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, X(1:n, :), fit(1:n), population_history, fitness_history, history_index, maxFE);
    end

    c = (ub - lb) * Mu + lb;          % the step scale of Eq. (3) and (5), per dimension
    Best_FF = bsf;                    % the release's leader, apart from bsf only under a NaN parent
    Best_P = bsf_solution;

    while FE < maxFE
        t = FE / maxFE;
        MOP = 1 - t ^ (1 / Alpha);                        % Eq. (4)
        MOA = MOP_Min + t * (MOP_Max - MOP_Min);          % Eq. (2)

        for i = 1:N
            if FE >= maxFE
                break;
            end
            dm = rand(1, dim) < MOA;                      % the release's gate, the paper's is r1 > MOA
            r2 = rand(1, dim) > 0.5;
            r3 = rand(1, dim) > 0.5;
            xn = zeros(1, dim);
            div = dm & r2;
            mul = dm & ~r2;
            sub = ~dm & r3;
            add = ~dm & ~r3;
            xn(div) = Best_P(div) / (MOP + eps) .* c(div);         % Eq. (3), division
            xn(mul) = Best_P(mul) * MOP .* c(mul);                 % Eq. (3), multiplication
            xn(sub) = Best_P(sub) - MOP * c(sub);                  % Eq. (5), subtraction
            xn(add) = Best_P(add) + MOP * c(add);                  % Eq. (5), addition

            bad = ~isfinite(xn);
            if any(bad)
                r = lb + rand(1, dim) .* (ub - lb);
                xn(bad) = r(bad);
            end
            xn = min(max(xn, lb), ub);

            [fv, FE] = calculate_fitness(xn', problem, FE);
            fx = fv(1);
            if fx < bsf
                bsf = fx;
                bsf_solution = xn;
            end
            curve(FE) = bsf;
            if fx < fit(i)
                X(i, :) = xn;
                fit(i) = fx;
            end
            if fit(i) < Best_FF
                Best_FF = fit(i);
                Best_P = X(i, :);
            end
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness = bsf;
    best_solution = bsf_solution;
end

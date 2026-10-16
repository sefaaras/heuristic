% ----------------------------------------------------------------------- %
% Greylag Goose Optimization (GGO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30                 % Population size (geese), see Implementation Note
%   n1 = N/2 -> [1, N-1]    % Exploration group size, adapted every iteration
%   a  = 2 -> 0 (linear)    % Scale of A = 2a*r1 - a
%   z  = 1 - (t/T)^2        % Weight of the paddle and flock moves, Eq. (3)
%   b  = 1                  % Spiral constant of Eq. (4)
%   w, w1..w4 ~ U[0, 2]     % Random weights, redrawn for every goose
%
% Algorithm Concept:
%   - Odd iterations: every goose takes the flock step
%     X + D(1+z)*w*(X - X*) around the leader X*, Eq. (7)
%   - Even iterations, exploration group with r3 < 0.5: |A| < 1 encircles the
%     leader, |A| >= 1 combines three random paddles (Eqs. (1)-(2))
%   - Exploration group with r3 >= 0.5: weighted spiral about the leader,
%     w4|X* - X| e^(bl) cos(2*pi*l) + 2w1(r4 + r5) X*, Eq. (4)
%   - Even iterations, exploitation group: mean of three encircling moves
%     toward the sentries, the three best geese (Eqs. (5)-(6))
%   - The flock is replaced wholesale; the leader is kept, and the order is
%     shuffled each iteration so geese swap groups
%   - n1 shrinks by one per iteration, but grows by one when the best value is
%     the same as in the previous two iterations
%
% Reference:
% El-Sayed M. El-kenawy, Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A.
% Abdelhamid, Marwa M. Eid, Abdelhameed Ibrahim,
% Greylag Goose Optimization: Nature-inspired optimization algorithm,
% Expert Systems with Applications 238 (2024) 122147.
% https://doi.org/10.1016/j.eswa.2023.122147
% ----------------------------------------------------------------------- %
% Implementation Note:
%   No author code exists and the paper is closed access: ported from Algorithm 1
%   and the equations as the authors reprint them (doi 10.3389/fenvs.2024.1429410,
%   10.1038/s41598-025-99472-0); pyMetaheuristic ggo.py (2026-06) as cross-check;
%   FX 163321 is a third-party stub and was not used. Left open by the paper and
%   chosen here: N = 30; D = |X* - X|; sentries = three best geese; paddles = three
%   distinct others; b = 1 as in WOA; the lost leader replaces the worst goose; n1
%   in [1, N-1]. r, w and l are drawn once per goose in the WOA-derived Eqs. (1),
%   (2), (4), as |A| < 1 implies a scalar A; per-dimension draws (pyMetaheuristic)
%   move the shifted-Rastrigin D=10 median from 62 to 32 (20 seeds, 1e5 FE).
%   T = (maxFE - N)/N iterations: every iteration evaluates all N geese.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ggo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    b = 1;
    T = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    fbest = inf;
    xbest = X(1, :);
    for e = 1:N
        if fit(e) < fbest
            fbest = fit(e);
            xbest = X(e, :);
        end
        if e <= maxFE
            curve(e) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    n1 = floor(N / 2);
    stall = 0;
    t = 0;
    while FE < maxFE
        t = t + 1;
        a = 2 * (1 - t / T);
        z = 1 - (t / T)^2;                                         % Eq. (3)
        fprev = fbest;

        [~, ord] = sort(fit);
        S1 = X(ord(1), :);
        S2 = X(ord(2), :);
        S3 = X(ord(3), :);

        Xnew = X;
        for i = 1:N
            if mod(t, 2) == 1
                w = 2 * rand;
                D = abs(xbest - X(i, :));
                Xnew(i, :) = X(i, :) + (1 + z) * w * D .* (X(i, :) - xbest);       % Eq. (7)
            elseif i <= n1
                A = 2 * a * rand - a;
                C = 2 * rand;
                if rand < 0.5
                    if abs(A) < 1
                        Xnew(i, :) = xbest - A * abs(C * xbest - X(i, :));         % Eq. (1)
                    else
                        pd = randperm(N - 1, 3);
                        pd = pd + (pd >= i);
                        w1 = 2 * rand; w2 = 2 * rand; w3 = 2 * rand;
                        Xnew(i, :) = w1 * X(pd(1), :) + z * w2 * (X(pd(2), :) - X(pd(3), :)) ...
                                   + (1 - z) * w3 * (X(i, :) - X(pd(1), :));      % Eq. (2)
                    end
                else
                    w1 = 2 * rand; w4 = 2 * rand;
                    l = 2 * rand - 1;
                    Xnew(i, :) = w4 * abs(xbest - X(i, :)) * exp(b * l) * cos(2 * pi * l) ...
                               + 2 * w1 * (rand + rand) * xbest;                    % Eq. (4)
                end
            else
                A1 = 2 * a * rand(1, dim) - a; C1 = 2 * rand(1, dim);
                A2 = 2 * a * rand(1, dim) - a; C2 = 2 * rand(1, dim);
                A3 = 2 * a * rand(1, dim) - a; C3 = 2 * rand(1, dim);
                X1 = S1 - A1 .* abs(C1 .* S1 - X(i, :));                           % Eq. (5)
                X2 = S2 - A2 .* abs(C2 .* S2 - X(i, :));
                X3 = S3 - A3 .* abs(C3 .* S3 - X(i, :));
                Xnew(i, :) = (X1 + X2 + X3) / 3;                                    % Eq. (6)
            end
        end
        Xnew = min(max(Xnew, lb), ub);

        m = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(Xnew(1:m, :)', problem, FE);
        fnew = fnew(:)';
        for i = 1:m
            X(i, :) = Xnew(i, :);
            fit(i) = fnew(i);
            if fit(i) < fbest
                fbest = fit(i);
                xbest = X(i, :);
            end
            ec = FE - m + i;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE
            break;
        end

        % Elitism: the leader re-enters over the worst goose with its measured fitness
        if ~(min(fit) <= fbest)
            [~, worst] = max(fit);
            X(worst, :) = xbest;
            fit(worst) = fbest;
        end
        perm = randperm(N);
        X = X(perm, :);
        fit = fit(perm);

        if fbest < fprev
            stall = 0;
        else
            stall = stall + 1;
        end
        if stall >= 2
            n1 = min(N - 1, n1 + 1);
        else
            n1 = max(1, n1 - 1);
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = xbest;
end

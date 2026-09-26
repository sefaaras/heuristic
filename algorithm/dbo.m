% ----------------------------------------------------------------------- %
% Dung Beetle Optimizer (DBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 30                    % Population size
%   pNum = round(0.2*pop) = 6   % Ball-rolling beetles; brood balls 7-12, small 13-19, thieves 20-30
%   k = 0.1, b = 0.3            % Deflection and light-intensity weights of Eq. (1)
%   p_roll = 0.9                % Chance a generation's rollers go unobstructed (else dance)
%   R = 1 - t                   % Width of the spawning and foraging regions, t the spent budget
%
% Algorithm Concept:
%   - Ball rollers step from their best position away from the worst current member,
%     deflected by 0.1 of their previous best (Eq. 1), or dance at tan(theta) (Eq. 2)
%   - Brood balls sit around the generation's best inside [X*(1-R), X*(1+R)]
%     (Eq. 3-4); small beetles forage around the global best's region (Eq. 5-6)
%   - Thieves steal around the global best with a Gaussian step scaled by their
%     distances to the generation's and the global best (Eq. 7)
%   - Every member keeps its personal best; the global best is refreshed per generation
%   - Coordinates are clamped to the box, brood balls to their spawning region
%
% Reference:
% Jiankai Xue, Bo Shen,
% Dung beetle optimizer: a new meta-heuristic algorithm for global optimization,
% The Journal of Supercomputing 79 (2023) 7305-7336.
% https://doi.org/10.1007/s11227-022-04959-6
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/Lancephil/Dung-Beetle-Optimizer,
% Matlab/DBO.m), keeping its hard-coded role ranges for pop = 30. R = 1 - t/M
% runs on the spent budget FE/maxFe, and FE >= maxFe is the only terminator.
% Kept as released: for a negative coordinate of X* the spawning region
% [X*(1-R), X*(1+R)] is inverted, and the lower-then-upper clamp puts every brood
% ball on X*(1+R) in that coordinate. It stays inside the box (both ends are
% clamped to it first); sorting the ends would lower the cec14 D10 median error
% from about 7e5 to 3.6e5, a gain the published algorithm does not have. The global best
% the update uses stays per generation as released; only the reported best is
% tracked per evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dbo(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    pop = 30;
    P_percent = 0.2;
    pNum = round(pop * P_percent);
    brood_end = 12;                   % the release hard-codes 12 and 19 for pop = 30
    small_end = 19;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    x = lb + rand(pop, dim) .* (ub - lb);
    fit = inf(pop, 1);
    bsf = inf;
    bsf_solution = x(1, :);

    n = min(pop, maxFE);
    [fv, FE] = calculate_fitness(x(1:n, :)', problem, FE);
    fit(1:n) = fv(:);
    for k = 1:n
        if fit(k) < bsf
            bsf = fit(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:n, :), fit(1:n), population_history, fitness_history, history_index, maxFE);
    end

    pFit = fit;
    pX = x;
    XX = pX;                          % personal bests one generation back, x(t-1) of Eq. (1)
    fMin = bsf;
    bestX = bsf_solution;

    while FE < maxFE
        R = 1 - FE / maxFE;
        [~, B] = max(fit);
        worse = x(B, :);
        r2 = rand;

        cand = zeros(pNum, dim);
        for i = 1:pNum
            if r2 < 0.9
                if rand > 0.1
                    a = 1;
                else
                    a = -1;
                end
                cand(i, :) = pX(i, :) + 0.3 * abs(pX(i, :) - worse) + a * 0.1 * XX(i, :);   % Eq. (1)
            else
                theta = randi(180) * pi / 180;
                cand(i, :) = pX(i, :) + tan(theta) * abs(pX(i, :) - XX(i, :));             % Eq. (2)
            end
        end
        cand = into_box(cand, lb, ub);

        n = min(pNum, maxFE - FE);
        [fv, FE] = calculate_fitness(cand(1:n, :)', problem, FE);
        x(1:n, :) = cand(1:n, :);
        fit(1:n) = fv(:);
        for k = 1:n
            if fit(k) < bsf
                bsf = fit(k);
                bsf_solution = x(k, :);
            end
            curve(FE - n + k) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE - n + k, x, fit, population_history, fitness_history, history_index, maxFE);
        end
        if FE >= maxFE
            break;
        end

        [~, bestII] = min(fit);
        bestXX = x(bestII, :);

        Xnew1 = into_box(bestXX .* (1 - R), lb, ub);                                        % Eq. (3)
        Xnew2 = into_box(bestXX .* (1 + R), lb, ub);
        Xnew11 = into_box(bestX .* (1 - R), lb, ub);                                        % Eq. (5)
        Xnew22 = into_box(bestX .* (1 + R), lb, ub);

        m = pop - pNum;
        cand = zeros(m, dim);
        for i = pNum + 1:pop
            if i <= brood_end
                c = bestXX + rand(1, dim) .* (pX(i, :) - Xnew1) + rand(1, dim) .* (pX(i, :) - Xnew2);   % Eq. (4)
                c = min(max(into_box(c, lb, ub), Xnew1), Xnew2);   % released order: lower end first, so an inverted pair yields Xnew2
            elseif i <= small_end
                c = pX(i, :) + randn * (pX(i, :) - Xnew11) + rand(1, dim) .* (pX(i, :) - Xnew22);       % Eq. (6)
                c = into_box(c, lb, ub);
            else
                c = bestX + randn(1, dim) .* (abs(pX(i, :) - bestXX) + abs(pX(i, :) - bestX)) / 2;    % Eq. (7)
                c = into_box(c, lb, ub);
            end
            cand(i - pNum, :) = c;
        end

        n = min(m, maxFE - FE);
        rows = pNum + (1:n);
        [fv, FE] = calculate_fitness(cand(1:n, :)', problem, FE);
        x(rows, :) = cand(1:n, :);
        fit(rows) = fv(:);
        for k = 1:n
            if fit(rows(k)) < bsf
                bsf = fit(rows(k));
                bsf_solution = x(rows(k), :);
            end
            curve(FE - n + k) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE - n + k, x, fit, population_history, fitness_history, history_index, maxFE);
        end

        XX = pX;
        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness = bsf;
    best_solution = bsf_solution;
end

% Non-finite coordinates are redrawn uniformly, the rest clamped as the release's Bounds()
function X = into_box(X, lb, ub)
    bad = ~isfinite(X);
    if any(bad(:))
        R = lb + rand(size(X)) .* (ub - lb);
        X(bad) = R(bad);
    end
    X = min(max(X, lb), ub);
end

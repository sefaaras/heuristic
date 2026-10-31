% ----------------------------------------------------------------------- %
% Black Hole Algorithm (BH)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                     % Stars, the black hole among them (paper gives none)
%
% Algorithm Concept:
%   - The best star is the black hole; every other star moves toward it,
%     x_i <- x_i + rand*(x_BH - x_i), one rand in [0, 1] per star, Eq. (3)
%   - A moved star with a lower cost than the black hole becomes the black hole
%   - Event horizon R = f_BH / sum_i f_i, Eq. (4): a star closer than R to the
%     black hole is swallowed and reborn uniformly in the search space
%
% Reference:
% Abdolreza Hatamlou,
% Black hole: A new heuristic optimization approach for data clustering,
% Information Sciences 222 (2013) 175-184.
% https://doi.org/10.1016/j.ins.2012.08.023
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Section 4 of the paper; no author code exists. Left open there and
% chosen here: N = 50 (no population size is given), Euclidean distance in the
% swallow test, rebirth uniform in the box. rand in Eq. (3) is "a random number"
% (p. 177), one per star; a per-dimension draw would cut the cec14 D=10 median
% error on F1/F4/F10 from 7.7e6/79/1044 to 1.2e6/5.7/258 (10 seeds). The exchange
% follows the step list (p. 178), after all stars have moved. Each new position is
% evaluated once, so the unmoved black hole is not re-evaluated every iteration.
% Kept as printed: R in Eq. (4) is a fitness ratio used as a distance, unguarded,
% so f <= 0 makes it negative or NaN (nothing swallowed) or huge (all swallowed).
% Eq. (3) is a convex step and never leaves the box, so no bound handling.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bh(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    N = 50;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    n0 = min(N, maxFE);
    [f0, FE] = calculate_fitness(X(1:n0, :)', problem, FE);
    fit = inf(1, N);
    fit(1:n0) = f0(:)';
    Xrec = X;   % the point each fit(i) was measured at

    best_fitness  = inf;
    best_solution = X(1, :);
    for e = 1:n0
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = X(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, Xrec(1:n0, :), fit(1:n0), population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE
        [~, ibh] = min(fit);
        movers = [1:ibh-1, ibh+1:N];
        % Eq. (3)
        for i = movers
            X(i, :) = X(i, :) + rand * (X(ibh, :) - X(i, :));
        end
        evaluate(movers);
        if FE >= maxFE, break; end

        % Exchange: a star that beat the black hole takes its role
        [fbh, ibh] = min(fit);
        % Eq. (4)
        R = fbh / sum(fit);
        dist = sqrt(sum((X - X(ibh, :)).^2, 2))';
        sucked = find(dist < R);
        sucked(sucked == ibh) = [];
        if ~isempty(sucked)
            X(sucked, :) = lb + rand(numel(sucked), dim) .* (ub - lb);
            evaluate(sucked);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Evaluates stars idx in order (cut to the budget); history gets each star as measured
    function evaluate(idx)
        idx = idx(1:min(numel(idx), maxFE - FE));
        if isempty(idx)
            return;
        end
        [fv, FE_new] = calculate_fitness(X(idx, :)', problem, FE);
        fv = fv(:)';
        for q = 1:numel(idx)
            i = idx(q);
            fit(i) = fv(q);
            Xrec(i, :) = X(i, :);
            if fv(q) < best_fitness
                best_fitness  = fv(q);
                best_solution = X(i, :);
            end
            curve(FE + q) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE + q, Xrec, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
        FE = FE_new;
    end
end

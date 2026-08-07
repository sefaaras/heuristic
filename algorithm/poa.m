% ----------------------------------------------------------------------- %
% Pelican Optimization Algorithm (POA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30   % Population size (pelicans)
%
% Algorithm Concept:
%   - Phase 1 (exploration): move toward a randomly located prey
%   - Phase 2 (exploitation): wing on the water surface with a shrinking radius
%   - Greedy acceptance after each phase
%
% Reference:
% Pavel Trojovsky, Mohammad Dehghani,
% Pelican Optimization Algorithm: A Novel Nature-Inspired Algorithm for
% Engineering Applications,
% Sensors 2022, 22(3), 855.
% https://doi.org/10.3390/s22030855
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = poa(problem)

    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents = 30;
    Max_iterations = ceil(maxFE / (2 * SearchAgents));   % two evaluations per pelican per iteration

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = initialization(SearchAgents, dim, upperbound, lowerbound);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [fbest, bidx] = min(fit);
    Xbest = X(bidx, :);

    for e = 1:SearchAgents
        if e <= maxFE
            curve(e) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    t = 0;
    while FE < maxFE
        t = t + 1;

        % Update best candidate
        [best, location] = min(fit);
        if best < fbest
            fbest = best;
            Xbest = X(location, :);
        end

        % PHASE 1: Moving towards prey (exploration)
        k = randperm(SearchAgents, 1);
        X_FOOD = X(k, :);
        F_FOOD = fit(k);
        Xnew1 = X;
        for i = 1:SearchAgents
            I = round(1 + rand(1, 1));
            if fit(i) > F_FOOD
                Xnew1(i, :) = X(i, :) + rand(1, 1) .* (X_FOOD - I .* X(i, :)); % Eq(4)
            else
                Xnew1(i, :) = X(i, :) + rand(1, 1) .* (X(i, :) - 1 .* X_FOOD); % Eq(4)
            end
            Xnew1(i, :) = max(Xnew1(i, :), lowerbound);
            Xnew1(i, :) = min(Xnew1(i, :), upperbound);
        end
        [fnew1, FE] = calculate_fitness(Xnew1', problem, FE);
        fnew1 = fnew1(:)';
        for i = 1:SearchAgents
            if fnew1(i) <= fit(i)          % Eq(5)
                X(i, :) = Xnew1(i, :);
                fit(i) = fnew1(i);
                if fit(i) < fbest
                    fbest = fit(i);
                    Xbest = X(i, :);
                end
            end
            ec = FE - SearchAgents + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        % PHASE 2: Winging on the water surface (exploitation)
        Xnew2 = X;
        for i = 1:SearchAgents
            Xnew2(i, :) = X(i, :) + 0.2 * (1 - t / Max_iterations) .* (2 * rand(1, dim) - 1) .* X(i, :); % Eq(6)
            Xnew2(i, :) = max(Xnew2(i, :), lowerbound);
            Xnew2(i, :) = min(Xnew2(i, :), upperbound);
        end
        [fnew2, FE] = calculate_fitness(Xnew2', problem, FE);
        fnew2 = fnew2(:)';
        for i = 1:SearchAgents
            if fnew2(i) <= fit(i)          % Eq(7)
                X(i, :) = Xnew2(i, :);
                fit(i) = fnew2(i);
                if fit(i) < fbest
                    fbest = fit(i);
                    Xbest = X(i, :);
                end
            end
            ec = FE - SearchAgents + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness = fbest;
    best_solution = Xbest;
end

% Initialization
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

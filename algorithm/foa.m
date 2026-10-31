% ----------------------------------------------------------------------- %
% Fruit Fly Optimization Algorithm (FOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   sizepop = 20              % Flies per generation
%   init    = 10*rand(1, D)   % X_axis and Y_axis: one (X, Y) pair per decision variable
%   step    = 2*rand() - 1    % Fly offset in [-1, 1], one draw per fly and axis
%
% Algorithm Concept:
%   - Smell search: fly p is the swarm location plus one random offset in X and
%     one in Y, the same two offsets for every decision variable
%   - Decision value: S_j = 1/(X_j^2 + Y_j^2)^0.5, the reciprocal distance of the
%     j-th (X, Y) pair to the origin, so every candidate coordinate is positive
%   - Vision: the swarm moves to the generation's best fly only when it beats the
%     best smell kept so far; the first generation's best is taken outright
%
% Reference:
% Wen-Tsao Pan,
% A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example,
% Knowledge-Based Systems 26 (2012) 69-74.
% https://doi.org/10.1016/j.knosys.2011.07.001
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's book Fruit Fly Optimization Algorithm, Second Edition
% (Pan, Feb 2014; File Exchange 48834 v1.0.0.0); the KBS paper was not read. Sec. 2
% gives sizepop, the 10*rand start and the [-1, 1] offset, Sec. 3.3 (FOAZSCORE,
% which starts at rands(1,5) with 5 flies) the D-variable form; seeded runs are
% bit-identical to that loop. Kept as released: S = 1/Dist > 0, so no candidate
% leaves the positive orthant (Dai et al. 2014), and all variables share one
% offset per fly and axis. Harness: the release has no bounds, so the evaluated
% point is S clamped to [lb, ub] (a NaN goes to lb) while the swarm keeps the
% unclamped X, Y; maxgen = 100 becomes the budget, the last generation truncated,
% and the best is tracked per evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = foa(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    sizepop = 20;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X_axis = 10 * rand(1, dim);
    Y_axis = 10 * rand(1, dim);

    m = min(sizepop, maxFE);
    [X, Y] = smell_search(X_axis, Y_axis, m);
    S = 1 ./ (X.^2 + Y.^2).^0.5;
    Xrec = min(max(S, lb), ub);   % max() drops a NaN, so it lands on lb
    [Smell, FE] = calculate_fitness(Xrec', problem, FE);
    Smell = Smell(:)';
    frec = Smell;

    best_fitness = inf;
    best_solution = Xrec(1, :);
    for e = 1:m
        if Smell(e) < best_fitness
            best_fitness = Smell(e);
            best_solution = Xrec(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, Xrec, frec, population_history, fitness_history, ...
            history_index, maxFE);
    end

    [bestSmell, bestindex] = min(Smell);
    X_axis = X(bestindex, :);
    Y_axis = Y(bestindex, :);
    Smellbest = bestSmell;

    while FE < maxFE
        m = min(sizepop, maxFE - FE);
        [X, Y] = smell_search(X_axis, Y_axis, m);
        S = 1 ./ (X.^2 + Y.^2).^0.5;
        Xc = min(max(S, lb), ub);
        [Smell, FE] = calculate_fitness(Xc', problem, FE);
        Smell = Smell(:)';

        % Recorded population: this generation's flies overwrite the last one's as they are evaluated
        for e = 1:m
            Xrec(e, :) = Xc(e, :);
            frec(e) = Smell(e);
            if Smell(e) < best_fitness
                best_fitness = Smell(e);
                best_solution = Xc(e, :);
            end
            ec = FE - m + e;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, ...
                history_index, maxFE);
        end

        [bestSmell, bestindex] = min(Smell);
        if bestSmell < Smellbest
            X_axis = X(bestindex, :);
            Y_axis = Y(bestindex, :);
            Smellbest = bestSmell;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% X(p,:) = X_axis + 2*rand() - 1, then Y: one scalar per fly and axis, as in FOAZSCORE
function [X, Y] = smell_search(X_axis, Y_axis, n)
    X = zeros(n, numel(X_axis));
    Y = X;
    for p = 1:n
        X(p, :) = X_axis + 2 * rand() - 1;
        Y(p, :) = Y_axis + 2 * rand() - 1;
    end
end

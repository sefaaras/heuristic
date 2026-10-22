% ----------------------------------------------------------------------- %
% Social Ski-Driver Optimization (SSD)
% Stored as ssdo, the name third-party ports use for SSD
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   swarm_size = 20                 % Number of skiers
%   a          = 2 -> 0 (linear)    % Weight of the own-best term; 2 - a weights the mean term
%
% Algorithm Concept:
%   - Each skier moves x = x + v and is clamped to the box
%   - Velocity v = a*sin(r).*(p - x) + (2 - a)*sin(r').*(m - x), or the same
%     with cos; a coin per skier picks sin or cos, r, r' ~ U(0, 1) per dimension
%   - p is the skier's best position so far, m the mean of the three skiers
%     with the best current fitness
%   - a falls linearly over the run, moving the pull from p to m
%
% Reference:
% Alaa Tharwat, Thomas Gabel,
% Parameters optimization of support vector machines for imbalanced data
% using social ski driver algorithm,
% Neural Computing and Applications 32(11) (2020) 6925-6938.
% https://doi.org/10.1007/s00521-019-04159-z
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 71210 v1.0.0
% (SSDAlgorithm.m, 14 Apr 2019; same code as his GitHub copy
% EngAlaaOthman/SocialSkiDriver-Optimization-algorithm), a 2-D Rastrigin
% script generalised here to D dimensions. Kept as released, unlike the paper's
% velocity c*sin(r1)*(P - X) + sin(r1)*(M - X) with c <- alpha*c: the linear a
% and 2 - a weights, a fresh r per dimension and term, sin/cos chosen per skier.
% The release's unused r1, r2 draws stay so seeded runs match it; its
% correction_factor and inertia are set but never used. m ranks skiers by
% current fitness (the author's Python port ranks by personal best).
% T = ceil(maxFe/20): the last iteration moves only the skiers it can evaluate.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ssdo(problem)

    dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    swarm_size = 20;
    Max_iterations = ceil(maxFE / swarm_size);   % one evaluation per skier per iteration

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = rand(swarm_size, dim) .* (UB - LB) + LB;   % column by column, as the release fills it
    V = zeros(swarm_size, dim);
    P = X;
    Pfit = inf(1, swarm_size);
    Fval = inf(1, swarm_size);
    best_fitness = inf;
    best_solution = X(1, :);

    for iter = 1:Max_iterations
        % Iteration 1 evaluates the initial positions, since v starts at zero
        nev = min(swarm_size, maxFE - FE);
        X(1:nev, :) = X(1:nev, :) + V(1:nev, :);
        X(1:nev, :) = min(max(X(1:nev, :), LB), UB);
        [f, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
        Fval(1:nev) = f(:)';
        for i = 1:nev
            if Fval(i) < Pfit(i)
                P(i, :) = X(i, :);
                Pfit(i) = Fval(i);
            end
            if Fval(i) < best_fitness
                best_fitness = Fval(i);
                best_solution = X(i, :);
            end
            ec = FE - nev + i;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, Fval, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        a = 2 - iter * (2 / Max_iterations);

        % Mean of the three best current positions
        [~, ind] = sort(Fval);
        M = mean(X(ind(1:3), :), 1);
        rand(1, 2);   % the release draws r1, r2 here (for A1, C1) and never uses them

        for i = 1:swarm_size
            branch = rand;
            R = rand(2, dim);   % column j: the P-term draw, then the M-term draw, as the release's j loop
            if branch < 0.5
                V(i, :) = a * sin(R(1, :)) .* (P(i, :) - X(i, :)) + (2 - a) * sin(R(2, :)) .* (M - X(i, :));
            else
                V(i, :) = a * cos(R(1, :)) .* (P(i, :) - X(i, :)) + (2 - a) * cos(R(2, :)) .* (M - X(i, :));
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

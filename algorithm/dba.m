% ----------------------------------------------------------------------- %
% Detective Behavior Algorithm (DBA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP   = 50    % Population size (detectives)
%   beta = 1.5   % Levy exponent used in the directional search
%
% Algorithm Concept:
%   - A judgement criterion A = (10*rand-1)*sin(0.5*pi*it/Max_It) switches
%     between three core search mechanisms:
%       * large-area directional exploration around the global best,
%       * Levy-flight jumps from the global best,
%       * mid-range search from the centre of the search space,
%   - followed by an opposition-based check: the opposite point
%     (ub+lb)-x is adopted whenever it is better.
%
% Reference:
% Jun Cheng, Wim De Waele,
% Detective Behavior Algorithm (DBA): A New Metaheuristic for Design and
% Engineering Optimization,
% Knowledge-Based Systems 338 (2026) 115434.
% https://doi.org/10.1016/j.knosys.2026.115434
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dba(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    nP     = 50;
    Max_It = max(1, ceil(maxFE / (3 * nP)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    X = initialization(nP, dim, ub, lb);

    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:);

    pBest_X    = X;
    pBest_Cost = Cost;

    [Best_Cost, ind] = min(Cost);
    Best_X = X(ind, :);
    bsf    = Best_Cost;
    bsx    = Best_X;

    for eval_count = 1:nP
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    for it = 1:Max_It
        if FE >= maxFE, break; end

        for i = 1:nP
            if FE >= maxFE, break; end

            % Judgement criterion for exploration / exploitation
            A = (10.0 * rand - 1) * sin(0.5 * pi * it / Max_It);
            if A > 0.5
                if rand < 0.5
                    X(i, :) = Best_X(1, :) - rand * (Best_X(1, :) - X(i, :));
                else
                    levy = zeros(1, dim);
                    for j = 1:dim
                        beta = 1.5;
                        sigma1 = gamma(1 + beta) * sin(pi * beta / 2) / ...
                                 (beta * gamma(0.5 + 0.5 * beta) * 2 ^ (0.5 * beta - 0.5));
                        levy(j) = normrnd(0, sigma1 ^ 2) / abs(normrnd(0, 1)) ^ (-beta);
                        X(i, j) = Best_X(1, j) + levy(j);
                    end
                end
            else
                for j = 1:dim
                    X(i, j) = (ub(j) + lb(j)) * 0.5 + rand * (Best_X(1, j) - X(i, j));
                end
            end

            % Bound check
            FU = X(i, :) > ub;
            FL = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;

            [Cost(i), FE] = calculate_fitness(X(i, :)', problem, FE);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Cost(i), X(i, :), bsf, bsx, curve, X, Cost, population_history, ...
                      fitness_history, history_index);

            % Opposition-based check
            Opposite_X = (ub + lb) - X(i, :);
            if FE < maxFE
                [fx, FE] = calculate_fitness(X(i, :)', problem, FE);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, fx, X(i, :), bsf, bsx, curve, X, Cost, population_history, ...
                          fitness_history, history_index);
            else
                fx = Cost(i);
            end
            if FE < maxFE
                [fo, FE] = calculate_fitness(Opposite_X', problem, FE);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, fo, Opposite_X, bsf, bsx, curve, X, Cost, population_history, ...
                          fitness_history, history_index);
            else
                fo = inf;
            end

            if fx > fo
                X(i, :) = Opposite_X;
                if FE < maxFE
                    [Cost(i), FE] = calculate_fitness(X(i, :)', problem, FE);
                else
                    Cost(i) = fo;
                end
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, Cost(i), X(i, :), bsf, bsx, curve, X, Cost, population_history, ...
                          fitness_history, history_index);
            end

            % Update personal and global best
            if Cost(i) < pBest_Cost(i)
                pBest_X(i, :)  = X(i, :);
                pBest_Cost(i)  = Cost(i);

                if pBest_Cost(i) < Best_Cost
                    Best_X    = pBest_X(i, :);
                    Best_Cost = pBest_Cost(i);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Curve / history stamp for a single evaluation
function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, x, bsf, bsx, curve, X, Cost, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Cost, ph, fh, hi, maxFE);
    end
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

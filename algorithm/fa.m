% ----------------------------------------------------------------------- %
% Firefly Algorithm (FA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 20                       % Fireflies
%   alpha0 = 1.0, theta = 0.97   % Randomness strength and its decay per generation
%   beta0  = 1.0                 % Attractiveness at zero distance
%   gamma  = 0.01                % Light absorption coefficient
%
% Algorithm Concept:
%   - Every firefly is attracted to every BRIGHTER firefly, so there is no
%     single leader: the swarm splits into subgroups that each converge on a
%     local optimum, which is the paper's claim to natural multimodality
%   - Attractiveness falls off as light in an absorbing medium,
%     beta = beta0*exp(-gamma*r^2) with r the Euclidean distance, so gamma sets
%     how far a firefly can see: gamma -> 0 is one global PSO, gamma -> inf a
%     set of independent random walkers
%   - The move combines attraction with a geometrically annealed random kick:
%         x_i <- x_i + beta*(x_j - x_i) + alpha*(rand - 0.5).*|ub - lb|
%   - Both loops run over the full population and x_i is re-evaluated inside the
%     inner loop, so a firefly moves and is scored several times per generation
%
% Reference:
% Xin-She Yang,
% Firefly Algorithm, Stochastic Test Functions and Design Optimisation,
% International Journal of Bio-Inspired Computation, vol. 2, no. 2,
% pp. 78-84, 2010.
% https://doi.org/10.1504/IJBIC.2010.032124
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own MATLAB release (fa_ndim_new.m, revised Oct 2010).
% The released form calls the objective once per (i,j) PAIR, so a generation
% costs n^2 = 400 evaluations; that is deliberate, as x_i moves inside the loop.
% GAMMA IS SCALED TO THE BOX, not fixed at the released 0.01: at CEC scale
% (r ~ 250) gamma*r^2 ~ 600 and beta underflows to zero, leaving FA a set of
% annealed random walks. Yang's own book prescribes gamma = O(1/L^2), so
% 4/mean(ub-lb)^2 is used, reproducing beta ~ e^-9 at the release's own scale.
% The bound clamp is applied after each move rather than once per generation;
% the release otherwise scores, and can return, infeasible points.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fa(problem)

    d     = problem.dimension;
    Lb    = problem.lb(:)';
    Ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    n      = 20;
    alpha  = 1.0;
    beta0  = 1.0;
    gamma  = 4 / mean(Ub - Lb) ^ 2;   % scale-aware; see the SCALE NOTE in the header
    theta  = 0.97;
    scale  = abs(Ub - Lb);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    ns = repmat(Lb, n, 1) + rand(n, d) .* repmat(Ub - Lb, n, 1);

    [Lightn, FE] = calculate_fitness(ns', problem, FE);
    Lightn = Lightn(:)';

    bsf  = inf;
    bsfx = ns(1, :);
    for i = 1:n
        if Lightn(i) < bsf
            bsf  = Lightn(i);
            bsfx = ns(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, ns, Lightn', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        alpha = alpha * theta;

        for i = 1:n
            for j = 1:n
                if FE >= maxFE
                    break;
                end

                % Firefly i is re-scored at every j, as in the reference: it may already have moved
                [li, FE] = calculate_fitness(ns(i, :)', problem, FE);
                Lightn(i) = li(1);

                if Lightn(i) < bsf
                    bsf  = Lightn(i);
                    bsfx = ns(i, :);
                end
                if FE >= 1 && FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, ns, Lightn', population_history, fitness_history, ...
                        history_index, maxFE);
                end

                if Lightn(i) >= Lightn(j)       % j is brighter (or equal)
                    r     = sqrt(sum((ns(i, :) - ns(j, :)) .^ 2));
                    beta  = beta0 * exp(-gamma * r ^ 2);
                    steps = alpha .* (rand(1, d) - 0.5) .* scale;
                    ns(i, :) = ns(i, :) + beta * (ns(j, :) - ns(i, :)) + steps;
                    % findlimits applied per move, not per generation; see the header note
                    ns(i, :) = min(max(ns(i, :), Lb), Ub);
                end
            end
            if FE >= maxFE
                break;
            end
        end

        % Clamp, then rank by the (possibly one-move-stale) light intensities
        ns = min(max(ns, repmat(Lb, n, 1)), repmat(Ub, n, 1));
        [Lightn, Index] = sort(Lightn);
        ns = ns(Index, :);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

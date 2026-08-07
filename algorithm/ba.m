% ----------------------------------------------------------------------- %
% Bat Algorithm (BA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 20                       % Bats
%   A0 = 1,  alpha = 0.97        % Loudness and its decay, A <- alpha*A
%   r0 = 1,  gamma = 0.1         % Pulse rate, r = r0*(1 - exp(-gamma*t))
%   Freq = [0, 2]                % Frequency-tuning range
%
% Algorithm Concept:
%   - A PSO-like velocity swarm plus two state variables from the biology,
%     LOUDNESS A and PULSE RATE r, which move in opposite directions
%   - Each bat draws a random frequency and updates
%         v_i <- v_i + (x_i - x*) * f_i ,      x_i <- x_i + v_i
%     the difference being (bat - best) as published, so the frequency term
%     makes the bat oscillate around the best rather than fall into it
%   - With probability r the global move is replaced by a LOCAL walk around the
%     best, x = x* + 0.1*randn*A; since r rises towards 1 and A decays
%     geometrically, the search shifts from flights to fine local sampling
%   - Acceptance is doubly conditioned on no-worse AND rand > A, so loud early
%     bats reject most improvements and quiet late ones accept them
%   - The best is updated inside the bat loop, so the algorithm is sequential
%
% Reference:
% Xin-She Yang,
% A New Metaheuristic Bat-Inspired Algorithm,
% Nature Inspired Cooperative Strategies for Optimization (NICSO 2010),
% Studies in Computational Intelligence, vol. 284, pp. 65-74, Springer, 2010.
% https://doi.org/10.1007/978-3-642-12538-6_6
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own MATLAB release (bat_algorithm_new.m, 2010),
% including the (x_i - best) sign of the frequency term and the fact that A and
% r are GLOBAL scalars updated once per iteration rather than the per-bat A_i
% and r_i of the paper's pseudocode.
% What that schedule does: A decays as 0.97^t while r rises as 1-exp(-0.1t), so
% after about thirty iterations nearly every bat takes the local walk with an
% already-small A and the swarm collapses onto the incumbent. That is the
% published algorithm, and it is why the literature reports BA as a fast local
% refiner and a weak global searcher.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ba(problem)

    d     = problem.dimension;
    Lb    = problem.lb(:)';
    Ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    n        = 20;
    A        = 1;          % initial loudness
    r0       = 1;          % initial pulse rate
    alpha    = 0.97;
    gamma    = 0.1;
    Freq_min = 0;
    Freq_max = 2;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Sol = repmat(Lb, n, 1) + rand(n, d) .* repmat(Ub - Lb, n, 1);
    v   = zeros(n, d);

    [Fitness, FE] = calculate_fitness(Sol', problem, FE);
    Fitness = Fitness(:)';

    [fmin, I] = min(Fitness);
    best = Sol(I, :);

    for i = 1:n
        if i <= maxFE
            curve(i) = min(Fitness(1:i));
            [population_history, fitness_history, history_index] = record_history(...
                i, Sol, Fitness', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    t = 0;

    % Main loop: one evaluation per bat, and the best moves inside the loop
    while FE < maxFE
        r = r0 * (1 - exp(-gamma * t));
        A = alpha * A;

        for i = 1:n
            if FE >= maxFE
                break;
            end

            Freq  = Freq_min + (Freq_max - Freq_min) * rand;
            v(i, :) = v(i, :) + (Sol(i, :) - best) * Freq;
            S = Sol(i, :) + v(i, :);

            if rand < r
                S = best + 0.1 * randn(1, d) * A;
            end

            S = min(max(S, Lb), Ub);

            [Fnew, FE] = calculate_fitness(S', problem, FE);
            Fnew = Fnew(1);

            if Fnew <= Fitness(i) && rand > A
                Sol(i, :)  = S;
                Fitness(i) = Fnew;
            end

            if Fnew <= fmin
                best = S;
                fmin = Fnew;
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = fmin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Sol, Fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        t = t + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = fmin;

    best_fitness  = fmin;
    best_solution = best;
end

% ----------------------------------------------------------------------- %
% Life Choice-Based Optimizer (LCBO)
% Stored as lco, the name third-party ports use for LCBO
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N        = 30                % Population size (search agents)
%   r1       = 2.35              % Step coefficient of the best and next-better pulls
%   n        = ceil(sqrt(N))     % Size of the best group averaged in Eq. (1)
%   f1       = 1 -> 0 (linear)   % Weight of the pull to the best; f2 = 1 - f1 for the next better
%   z        = 0.875 / 0.7       % Branch thresholds: Eq. (1) 12.5 %, Eqs. (4)-(6) 70 %, Eq. (7) 17.5 %
%
% Algorithm Concept:
%   - The population is sorted by fitness after every iteration; each agent then
%     draws z ~ U(0, 1) and takes one of three moves
%   - z > 0.875, learning from the best group: sum(rand_k*x_k)/n over the n
%     best agents, one scalar rand_k per agent
%   - z < 0.7, knowing the very next best: x + rand*f2*r1*(x_{j-1} - x) +
%     rand*f1*r1*(x_1 - x), x_{j-1} the agent ranked just above, x_1 the best
%   - Otherwise, reviewing mistakes: ub - (x - lb)*rand, a jump toward the far
%     side of the box
%   - Strict greedy acceptance, agent by agent, in place
%
% Reference:
% Abhishek Khatri, Akash Gaba, K. P. S. Rana, Vineet Kumar,
% A novel life choice-based optimizer,
% Soft Computing 24(12) (2020) 9121-9141.
% https://doi.org/10.1007/s00500-019-04443-z
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper; no author code was found (File Exchange, GitHub). The
% paper is paywalled: its pseudo-code was read from the publisher's algorithm
% figure, Eqs. (1)-(7) from the restatement in Gupta, Thakur, Gupta, Multimed.
% Tools Appl. 81 (2022) 14475, which agrees with mealpy's OriginalLCO (n, r1 and
% the two thresholds). The population size is not in the accessible text; 30.
% f1 = 1 - (chance - 1)/(chances - 1) with chances - 1 = the iterations the
% budget buys, so f1 ends at 1/T as in the pseudo-code's loop. The best agent's
% x_{j-1} is itself (mealpy uses the best), so its Eq. (4)-(6) move is a null
% move that costs one evaluation, as in the pseudo-code. Candidates are clamped
% to the box: Eq. (1) contracts toward the origin and Eq. (6) can overshoot.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lco(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    r1 = 2.35;
    n = ceil(sqrt(N));
    numberOfChances = max(2, ceil((maxFE - N) / N) + 1);   % one evaluation per agent per chance

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    for e = 1:min(N, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    [fit, order] = sort(fit);
    X = X(order, :);
    best_fitness = fit(1);
    best_solution = X(1, :);

    currentChance = 1;
    while FE < maxFE
        f1 = 1 - (currentChance - 1) / (numberOfChances - 1);   % Eq. (2)
        f2 = 1 - f1;   % Eq. (3)

        for j = 1:N
            z = rand;
            if z > 0.875
                Xn = sum(rand(n, 1) .* X(1:n, :), 1) / n;   % Eq. (1)
            elseif z < 0.70
                prev = X(max(j - 1, 1), :);
                bestDiff = f1 * r1 * (X(1, :) - X(j, :));   % Eq. (4)
                betterDiff = f2 * r1 * (prev - X(j, :));   % Eq. (5)
                Xn = X(j, :) + rand * betterDiff + rand * bestDiff;   % Eq. (6)
            else
                Xn = ub - (X(j, :) - lb) * rand;   % Eq. (7)
            end
            Xn = min(max(Xn, lb), ub);
            [fn, FE] = calculate_fitness(Xn', problem, FE);
            if fn < fit(j)
                X(j, :) = Xn;
                fit(j) = fn;
                if fn < best_fitness
                    best_fitness = fn;
                    best_solution = Xn;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end

        [fit, order] = sort(fit);
        X = X(order, :);
        currentChance = currentChance + 1;
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

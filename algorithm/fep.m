% ----------------------------------------------------------------------- %
% Fast Evolutionary Programming (FEP)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   mu   = 100                  % Parents, each carrying its own step sizes
%   q    = 10                   % Opponents in the pairwise selection tournament
%   eta0 = 3.0                  % Initial step size, per coordinate
%   tau  = 1/sqrt(2*sqrt(D))    % Learning rate of the per-coordinate step size
%   tau' = 1/sqrt(2*D)          % Learning rate of the global step size
%
% Algorithm Concept:
%   - Each parent is a pair (x, eta): a point and its own vector of step sizes
%   - One offspring per parent, mutated by a Cauchy(0,1) step scaled by eta,
%     whose heavy tails make long jumps far more likely than the Gaussian step
%     of classical EP, which is the whole point of the method
%   - Step sizes are then updated log-normally with the two learning rates
%   - Selection is a tournament: every parent and offspring scores a win against
%     q random opponents it beats, and the mu highest scores form the next
%     generation, so a poor individual can still survive on a lucky draw
%
% Reference:
% Xin Yao, Yong Liu, Guangming Lin,
% Evolutionary programming made faster,
% IEEE Transactions on Evolutionary Computation 3 (1999) 82-102.
% https://doi.org/10.1109/4235.771163
% ----------------------------------------------------------------------- %
% Implementation Note:
% The paper's benchmarks are unconstrained outside their box, so it never
% clamps; here an offspring is clamped to the box, because a Cauchy tail
% otherwise leaves positions many orders of magnitude outside it, which is
% meaningless on the constrained suites and distorts every population metric the
% recorder takes. Step sizes are floored at 1e-30, without which a run that
% collapses stops moving and spends its remaining budget re-evaluating one point.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fep(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    mu = 100;
    q = 10;
    eta0 = 3.0;
    tau = 1 / sqrt(2 * sqrt(dim));
    tau_prime = 1 / sqrt(2 * dim);
    eta_min = 1e-30;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = repmat(lb, mu, 1) + rand(mu, dim) .* repmat(ub - lb, mu, 1);
    eta = eta0 * ones(mu, dim);

    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:mu
        if fitness(i) < best_fitness
            best_fitness = fitness(i);
            best_solution = X(i, :);
        end
        if i <= maxFE
            curve(i) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, fitness', population_history, fitness_history, history_index, maxFE);
        end
    end

    while FE < maxFE
        % Cauchy(0,1) step scaled by the parent's own step sizes, Eq. (5)
        delta = tan(pi * (rand(mu, dim) - 0.5));
        Xc = X + eta .* delta;
        Xc = min(max(Xc, repmat(lb, mu, 1)), repmat(ub, mu, 1));

        % Step sizes follow the offspring, Eq. (6)
        eta_c = eta .* exp(tau_prime * randn(mu, 1) * ones(1, dim) + tau * randn(mu, dim));
        eta_c = max(eta_c, eta_min);

        [fit_c, FE] = calculate_fitness(Xc', problem, FE);
        fit_c = fit_c(:);

        for i = 1:mu
            if fit_c(i) < best_fitness
                best_fitness = fit_c(i);
                best_solution = Xc(i, :);
            end
            eval_count = FE - mu + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Pairwise tournament over parents and offspring together
        pool_X = [X; Xc];
        pool_eta = [eta; eta_c];
        pool_fit = [fitness; fit_c];
        wins = zeros(2 * mu, 1);
        for i = 1:2 * mu
            opponents = randi(2 * mu, q, 1);
            wins(i) = sum(pool_fit(i) <= pool_fit(opponents));
        end
        [~, order] = sort(wins, 'descend');
        keep = order(1:mu);
        X = pool_X(keep, :);
        eta = pool_eta(keep, :);
        fitness = pool_fit(keep);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

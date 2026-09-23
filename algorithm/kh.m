% ----------------------------------------------------------------------- %
% Krill Herd (KH)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                      % Krill individuals
%   Nmax = 0.01, Vf = 0.02      % Maximum induced speed and foraging speed
%   Dmax = 0.01                 % Maximum diffusion speed
%   w = 0.9 -> 0.1              % Inertia of both motions, linear in the budget
%   Ct = 1                      % Scale of the position step
%   Cr = 0.2*Khat, Mu = 0.05*Khat  % Crossover and mutation rates, per individual
%
% Algorithm Concept:
%   - Three motions add up: induced motion from neighbours, foraging towards the
%     virtual food and the best, and a random physical diffusion that decays as
%     the budget is spent
%   - Neighbours are the krill inside a sensing distance, the mean distance to
%     the herd divided by five, so a crowded krill has more neighbours
%   - Every pull is weighted by the fitness gap normalised by the herd's spread,
%     so the attraction is towards better krill and away from worse ones
%   - The virtual food is the fitness-weighted centroid of the herd, costing one
%     evaluation per iteration
%   - A crossover and a mutation borrow coordinates from random krill and from
%     the best, each with a rate proportional to how far the krill is from best
%
% Reference:
% Amir Hossein Gandomi, Amir Hossein Alavi,
% Krill herd: A new bio-inspired optimization algorithm,
% Communications in Nonlinear Science and Numerical Simulation 17 (2012) 4831-4845.
% https://doi.org/10.1016/j.cnsns.2012.05.010
% ----------------------------------------------------------------------- %
% Implementation Note:
% The authors' code is behind a MathWorks login, so this follows the paper's
% equations, cross-checked line by line against the KH of the CRAN package
% metaheuristicOpt (KH.Algorithm.R), and takes that package's Cr = 0.2*Khat and
% Mu = 0.05*Khat, which the paper writes as 0.05/Khat -- a form that diverges for
% the best krill. Three deviations from the package: the sensing distance is the
% paper's mean neighbour distance over 5, not its numVar-scaled sum; the herd is
% evaluated once per iteration rather than three times, which is what the paper
% counts; and positions are clamped to the box every iteration, since it only
% clamps the final answer. The motion inertia falls 0.9 -> 0.1 over the budget,
% the paper's range, where the package holds it at 0.01.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = kh(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 40;
    Nmax = 0.01;
    Vf = 0.02;
    Dmax = 0.01;
    Ct = 1;
    mu_mut = 0.1;
    epsilon = 1e-5;
    dt = Ct * sum(ub - lb);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    Nmotion = zeros(N, dim);
    Fmotion = zeros(N, dim);

    best_fitness = inf;
    best_solution = zeros(1, dim);

    while FE < maxFE
        [K, FE] = calculate_fitness(X', problem, FE);
        K = K(:);

        for i = 1:N
            if K(i) < best_fitness
                best_fitness = K(i);
                best_solution = X(i, :);
            end
            eval_count = FE - N + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, K', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE
            break;
        end

        t = FE / maxFE;                      % spent budget, the schedule's clock
        w = 0.1 + 0.8 * (1 - t);             % inertia of both motions

        [Kbest, ibest] = min(K);
        Kworst = max(K);
        Xbest = X(ibest, :);
        spread = Kworst - Kbest;

        % Virtual food: the fitness-weighted centroid of the herd
        inv_K = 1 ./ K;
        if all(isfinite(inv_K)) && sum(inv_K) ~= 0
            Xfood = sum(X .* inv_K, 1) / sum(inv_K);
        else
            Xfood = mean(X, 1);
        end
        Xfood = min(max(Xfood, lb), ub);
        [Kfood, FE] = calculate_fitness(Xfood', problem, FE);
        Kfood = Kfood(1);
        if Kfood < best_fitness
            best_fitness = Kfood;
            best_solution = Xfood;
        end
        if FE >= 1 && FE <= maxFE
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, K', population_history, fitness_history, history_index, maxFE);
        end

        % Pairwise distances decide who senses whom
        dist = squareform_local(X);
        ds = mean(dist, 2) / 5;              % sensing distance, Eq. (4)

        Cbest = 2 * (rand + t);
        Cfood = 2 * (1 - t);
        alpha = zeros(N, dim);
        beta = zeros(N, dim);
        for i = 1:N
            neighbours = find(dist(i, :) < ds(i) & (1:N) ~= i);
            if ~isempty(neighbours)
                Khat = (K(i) - K(neighbours)) / max(spread, eps);
                diff = X(neighbours, :) - X(i, :);
                Xhat = diff ./ (sqrt(sum(diff .^ 2, 2)) + epsilon);
                alpha(i, :) = sum(Khat .* Xhat, 1);
            end
            alpha(i, :) = alpha(i, :) + Cbest * khat(K(i), Kbest, spread) ...
                          * xhat(X(i, :), Xbest, epsilon);

            beta(i, :) = Cfood * khat(K(i), Kfood, spread) * xhat(X(i, :), Xfood, epsilon) ...
                       + khat(K(i), Kbest, spread) * xhat(X(i, :), Xbest, epsilon);
        end

        Nmotion = Nmax * alpha + w * Nmotion;
        Fmotion = Vf * beta + w * Fmotion;
        D = Dmax * (1 - t) * (2 * rand(N, dim) - 1);

        X = X + dt * (Nmotion + Fmotion + D);

        % Genetic operators, rates set by each krill's gap to the best
        Kibest = zeros(N, 1);
        for i = 1:N
            Kibest(i) = khat(K(i), Kbest, spread);
        end
        cross_mask = rand(N, dim) < repmat(0.2 * Kibest, 1, dim);
        donors = X(randi(N, N, dim) + N * repmat(0:dim-1, N, 1));
        X(cross_mask) = donors(cross_mask);

        mut_mask = rand(N, dim) < repmat(0.05 * Kibest, 1, dim);
        P = X(randi(N, N, dim) + N * repmat(0:dim-1, N, 1));
        Q = X(randi(N, N, dim) + N * repmat(0:dim-1, N, 1));
        mutated = repmat(best_solution, N, 1) + mu_mut * (P - Q);
        X(mut_mask) = mutated(mut_mask);

        X = min(max(X, repmat(lb, N, 1)), repmat(ub, N, 1));
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

function k = khat(Ki, Kj, spread)
    if spread <= 0
        k = 0;
    else
        k = (Ki - Kj) / spread;
    end
end

function x = xhat(Xi, Xj, epsilon)
    d = Xj - Xi;
    x = d / (sqrt(sum(d .^ 2)) + epsilon);
end

% Pairwise Euclidean distances without the Statistics Toolbox
function D = squareform_local(X)
    sq = sum(X .^ 2, 2);
    D = sqrt(max(sq + sq' - 2 * (X * X'), 0));
end

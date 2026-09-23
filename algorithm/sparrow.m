% ----------------------------------------------------------------------- %
% Sparrow Search Algorithm (SSA)
% Stored as sparrow; the acronym SSA collides with the Salp Swarm Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 100                   % Population size
%   PD = 0.2                    % Fraction of the population that are producers
%   SD = 0.2                    % Fraction that detect danger each iteration
%   ST = 0.8                    % Safety threshold of the producer update
%
% Algorithm Concept:
%   - The population splits by rank: the best PD fraction are producers that
%     search widely, the rest are scroungers that follow them
%   - A producer shrinks its position geometrically while no predator is sensed
%     (a draw below ST), and takes a Gaussian step when one is
%   - A scrounger in the worse half flies away from the worst individual,
%     otherwise it steps towards the producers' best with a random sign pattern
%   - Each iteration a random SD fraction detect danger: those not already at
%     the best move towards it, the best ones step away from the worst, scaled
%     by their own fitness gap
%   - Each sparrow keeps its personal best, which is what the ranking reads
%
% Reference:
% Jiankai Xue, Bo Shen,
% A novel swarm intelligence optimization approach: sparrow search algorithm,
% Systems Science & Control Engineering 8 (2020) 22-34.
% https://doi.org/10.1080/21642583.2019.1708830
% ----------------------------------------------------------------------- %
% Implementation Note:
% Eq. (3) scales the producer step by the iteration cap, which the budget does
% not give, so the cap is counted from it: an iteration costs pop + round(SD*pop)
% evaluations, and iter_max is the budget divided by that. The released MATLAB
% code hard-codes 20 danger-aware individuals for its pop = 100; here it is
% round(SD * pop), which is the same 20 at that size and stays a fraction at any
% other. Everything else follows that code (Eq. 3 to Eq. 5 with the same
% pseudo-inverse step and the same 1e-50 guard on the fitness gap).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sparrow(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    pop = 100;
    PD = 0.2;
    SD = 0.2;
    ST = 0.8;

    n_prod = round(pop * PD);
    n_danger = max(1, round(pop * SD));
    iter_max = max(1, floor(maxFE / (pop + n_danger)));   % what Eq. (3) needs, counted from the budget

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = repmat(lb, pop, 1) + rand(pop, dim) .* repmat(ub - lb, pop, 1);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:);

    pX = X;              % personal bests
    pFit = fit;
    [best_fitness, ibest] = min(fit);
    best_solution = X(ibest, :);

    for i = 1:min(pop, maxFE)
        curve(i) = min(fit(1:i));
        [population_history, fitness_history, history_index] = record_history(...
            i, X, fit', population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        [~, sort_index] = sort(pFit);
        [fmax, iworst] = max(pFit);
        worse = X(iworst, :);

        r2 = rand;
        for i = 1:n_prod
            row = sort_index(i);
            if r2 < ST
                X(row, :) = pX(row, :) * exp(-i / (rand * iter_max));
            else
                X(row, :) = pX(row, :) + randn * ones(1, dim);
            end
            X(row, :) = bound_check(X(row, :), lb, ub);
        end

        [fit_prod, FE] = calculate_fitness(X(sort_index(1:n_prod), :)', problem, FE);
        fit(sort_index(1:n_prod)) = fit_prod(:);
        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            stage_done(X, fit, sort_index(1:n_prod), FE, best_fitness, best_solution, curve, ...
                       population_history, fitness_history, history_index, maxFE);

        % The scrounger step reads the best AFTER the producers have been scored
        [~, ibest_prod] = min(fit);
        best_prod = X(ibest_prod, :);

        for i = (n_prod + 1):pop
            row = sort_index(i);
            if i > pop / 2
                X(row, :) = randn * exp((worse - pX(row, :)) / i^2);
            else
                A = floor(rand(1, dim) * 2) * 2 - 1;          % random signs
                X(row, :) = best_prod + abs(pX(row, :) - best_prod) * (A' * (A * A')^(-1)) * ones(1, dim);
            end
            X(row, :) = bound_check(X(row, :), lb, ub);
        end

        [fit_scr, FE] = calculate_fitness(X(sort_index(n_prod+1:pop), :)', problem, FE);
        fit(sort_index(n_prod+1:pop)) = fit_scr(:);
        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            stage_done(X, fit, sort_index(n_prod+1:pop), FE, best_fitness, best_solution, curve, ...
                       population_history, fitness_history, history_index, maxFE);

        perm = randperm(pop);
        aware = sort_index(perm(1:n_danger));
        for j = 1:n_danger
            row = aware(j);
            if pFit(row) > best_fitness
                X(row, :) = best_solution + randn(1, dim) .* abs(pX(row, :) - best_solution);
            else
                X(row, :) = pX(row, :) + (2 * rand - 1) * abs(pX(row, :) - worse) ...
                            / (pFit(row) - fmax + 1e-50);
            end
            X(row, :) = bound_check(X(row, :), lb, ub);
        end

        [fit_aware, FE] = calculate_fitness(X(aware, :)', problem, FE);
        fit(aware) = fit_aware(:);
        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            stage_done(X, fit, aware, FE, best_fitness, best_solution, curve, ...
                       population_history, fitness_history, history_index, maxFE);

        % Personal bests, which the next iteration's ranking reads
        improved = fit < pFit;
        pFit(improved) = fit(improved);
        pX(improved, :) = X(improved, :);
        [pbest, ipbest] = min(pFit);
        if pbest < best_fitness
            best_fitness = pbest;
            best_solution = pX(ipbest, :);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Curve and history for the evaluations one stage of the iteration just spent
function [bf, bx, curve, ph, fh, hi] = stage_done(X, fit, rows, FE, bf, bx, curve, ph, fh, hi, maxFE)
    n = numel(rows);
    for i = 1:n
        if fit(rows(i)) < bf
            bf = fit(rows(i));
            bx = X(rows(i), :);
        end
        eval_count = FE - n + i;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bf;
            [ph, fh, hi] = record_history(eval_count, X, fit', ph, fh, hi, maxFE);
        end
    end
end

% A step out of the box is clamped; a non-finite coordinate has no side to clamp to
function x = bound_check(x, lb, ub)
    nf = ~isfinite(x);
    x(nf) = lb(nf) + rand(1, sum(nf)) .* (ub(nf) - lb(nf));
    x = min(max(x, lb), ub);
end

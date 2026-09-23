% ----------------------------------------------------------------------- %
% Univariate Marginal Distribution Algorithm, continuous (UMDA_c)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 100                  % Population size
%   trunc = 0.5                 % Fraction selected to build the model
%   sigma_min = 1e-12           % Floor on an estimated standard deviation
%
% Algorithm Concept:
%   - No crossover and no mutation: each generation fits a probability model to
%     the selected individuals and samples the next population from it
%   - The model is univariate, one independent Gaussian per coordinate, whose
%     mean and standard deviation are the maximum-likelihood estimates over the
%     truncation-selected half of the population
%   - Being univariate, it cannot represent dependence between coordinates, so
%     it converges fast on separable problems and stalls on rotated ones, which
%     is exactly why it is the baseline the Gaussian-network EDAs improve on
%   - Samples are clamped to the box
%
% Reference:
% Pedro Larranaga, Jose A. Lozano,
% Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation,
% Genetic Algorithms and Evolutionary Computation 2, Springer, 2002.
% https://doi.org/10.1007/978-1-4615-1539-5
% ----------------------------------------------------------------------- %
% Implementation Note:
% UMDA_c as defined in the book: truncation selection, per-coordinate maximum
% likelihood, no elitism. The standard deviation is floored at 1e-12 because a
% converged coordinate otherwise samples the same value for the rest of the run,
% and the best solution is tracked outside the population, which the definition
% does not keep.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = umda(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 100;
    trunc = 0.5;
    sigma_min = 1e-12;
    n_sel = max(2, round(trunc * N));

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:N
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
        [~, order] = sort(fitness);
        selected = X(order(1:n_sel), :);

        mu = mean(selected, 1);
        sigma = max(std(selected, 0, 1), sigma_min);

        X = repmat(mu, N, 1) + randn(N, dim) .* repmat(sigma, N, 1);
        X = min(max(X, repmat(lb, N, 1)), repmat(ub, N, 1));

        [fitness, FE] = calculate_fitness(X', problem, FE);
        fitness = fitness(:);

        for i = 1:N
            if fitness(i) < best_fitness
                best_fitness = fitness(i);
                best_solution = X(i, :);
            end
            eval_count = FE - N + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

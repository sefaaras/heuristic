% ----------------------------------------------------------------------- %
% Pareto-like Sequential Sampling (PSS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size        = 30                      % Samples per iteration (beta)
%   acceptance_rate = 0.95                    % Chance a coordinate is drawn near the best (alpha)
%   deviation       = 0.5*(1-alpha)*(1-i/gamma)*(ub-lb)   % Half-width of the prominent domain
%
% Algorithm Concept:
%   - No moves and no memory but the best: every iteration draws a fresh
%     population, coordinate by coordinate
%   - With probability alpha a coordinate is uniform in the prominent domain
%     [max(best - dev, lb), min(that + 2*dev, ub)], else uniform in [lb, ub]
%   - The half-width dev shrinks linearly to 0 over the run, but is recomputed
%     only after an iteration that matched or beat the best
%
% Reference:
% Mahmoud Shaqfa, Katrin Beyer,
% Pareto-like sequential sampling heuristic for global optimisation,
% Soft Computing 25(14) (2021) 9077-9096.
% https://doi.org/10.1007/s00500-021-05853-8
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' Octave/MATLAB main_algorithm.m in
% github.com/eesd-epfl/pareto-optimizer (last push 2021-10-22); seeded runs
% are bit-identical to it once its time index is aligned. That file runs
% i = 2..gamma+1, so dev is 0 one iteration early and negative in the last
% one, which can sample outside the box; i = 1..gamma is used, as in the
% authors' C++14 and Python versions and the paper's Eq. (6). Kept as
% released: dev is one scalar, so after a non-improving iteration every
% coordinate uses the last coordinate's width (unequal bounds only), and a
% tie with the best moves the prominent domain (<=; the C++ uses <). gamma =
% ceil((maxFe - beta)/beta), the last iteration truncated to the budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = pss(problem)

    dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    acceptance_rate = 0.95;
    pop_size = 30;
    iterations = max(1, ceil((maxFE - pop_size) / pop_size));   % gamma, one evaluation per sample

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    population = LB + rand(pop_size, dim) .* (UB - LB);
    nev = min(pop_size, maxFE);
    [funval, FE] = calculate_fitness(population(1:nev, :)', problem, FE);
    funval = funval(:)';

    best_fitness  = Inf;
    best_solution = population(1, :);
    for e = 1:nev
        if funval(e) < best_fitness
            best_fitness  = funval(e);
            best_solution = population(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, population(1:nev, :), funval, population_history, fitness_history, ...
            history_index, maxFE);
    end

    [best_val, best_index] = min(funval);   % funval_his(i - 1) of the release
    best_sol = population(best_index, :);
    new_solution = true;
    deviation = zeros(1, dim);

    i = 0;
    while FE < maxFE
        i = i + 1;
        random_pop = rand(pop_size, dim);
        accept = rand(dim, pop_size)';   % the release's per-feature rand, drawn feature by feature

        % The release recomputes the scalar deviation per feature only after an improving iteration
        if new_solution
            deviation = 0.5 * (1 - acceptance_rate) * (1 - (i / iterations)) * (UB - LB);   % Eq. (6)
        else
            deviation = repmat(deviation(end), 1, dim);
        end
        reduced_LB = max(best_sol - deviation, LB);       % Eq. (5)
        reduced_UB = min(reduced_LB + 2 * deviation, UB); % Eq. (4), anchored at the reduced lower bound

        near = accept <= acceptance_rate;
        newpop = LB + random_pop .* (UB - LB);   % Eq. (8), overall domain
        prominent = reduced_LB + random_pop .* (reduced_UB - reduced_LB);   % Eq. (7)
        newpop(near) = prominent(near);
        newpop = min(max(newpop, LB), UB);   % a no-op except for one-ulp rounding at reduced_UB

        nev = min(pop_size, maxFE - FE);
        [fnew, FE] = calculate_fitness(newpop(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for j = 1:nev
            population(j, :) = newpop(j, :);
            funval(j) = fnew(j);
            if fnew(j) < best_fitness
                best_fitness  = fnew(j);
                best_solution = newpop(j, :);
            end
            ec = FE - nev + j;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, population, funval, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        % Find the current best value; ties move the prominent domain, as released
        if min(funval) <= best_val
            [best_val, best_index] = min(funval);
            best_sol = population(best_index, :);
            new_solution = true;
        else
            new_solution = false;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

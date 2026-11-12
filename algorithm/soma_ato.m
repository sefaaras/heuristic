% ----------------------------------------------------------------------- %
% Self-Organizing Migrating Algorithm, All-To-One strategy (SOMA)
% Stored as soma_ato: MATLAB ships a demo function named soma
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize = 100               % Population size
%   PathLength = 3.0            % The path runs to three times the distance to the Leader
%   Step = 0.11                 % Path t = 0, 0.11, ..., 2.97 (28 points, t = 0 included)
%   PRT = 0.1                   % Probability that a coordinate moves on a jump
%
% Algorithm Concept:
%   - Each migration loop takes the current best individual as the Leader, fixed
%     for the whole loop
%   - Every other individual travels towards the Leader and past it, sampling
%     x + (Leader - x)*t.*PRTVector at t = 0, Step, 2*Step, ... <= PathLength
%   - PRTVector holds a 1 where a uniform draw is below PRT and a 0 elsewhere; a
%     new one is drawn for every jump
%   - The best point on the path replaces the individual if it is no worse
%   - Coordinates that leave the box are redrawn uniformly inside it
%
% Reference:
% Ivan Zelinka,
% SOMA -- Self-Organizing Migrating Algorithm,
% in: G. C. Onwubolu, B. V. Babu (Eds.), New Optimization Techniques in Engineering,
% Studies in Fuzziness and Soft Computing, vol. 141, Springer, 2004, pp. 167-217.
% https://doi.org/10.1007/978-3-540-39930-8_7
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from SOMA_ATO.m in github.com/diepquocbao/SOMA-Classical-MATLAB (Q. B. Diep,
% Zelinka's group) with its defaults. The chapter is paywalled; as quoted by Skanderova's
% review (Artif. Intell. Rev. 56 (2023) 101-172, doi:10.1007/s10462-022-10167-8) it gives
% PathLength in [1.1, 5] (3 sufficient), Step in [0.11, PathLength] and no divisor of it,
% PRT in [0, 1] and PopSize >= 10, all met here. Kept as released: the path starts at
% t = 0, so each path re-evaluates its own start (1 FE in 28), and the PRTVector is
% redrawn per jump (both variants are in use; which one the chapter prescribes could
% not be checked). The release tests the budget only between migration loops; here
% each path is cut at maxFe. Seeded runs match it bit for bit.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = soma_ato(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    PopSize    = 100;
    PathLength = 3.0;
    Step       = 0.11;
    PRT        = 0.1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    PopSize = max(2, min(PopSize, maxFE));
    % k*Step as the release's for-loop yields it; the vector 0:Step:PathLength differs in the last bit
    N_jump = floor(PathLength / Step) + 1;
    t_path = (0:N_jump - 1) * Step;

    % Drawn dimension-major per individual, the release's column order
    pop = lb + rand(dim, PopSize)' .* span;
    bsf = inf;
    bsf_solution = pop(1, :);
    [fit, FE, bsf, bsf_solution, curve] = evaluate_rows(pop, problem, FE, bsf, bsf_solution, curve);
    [population_history, fitness_history, history_index] = record_history( ...
        FE, pop, fit, population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        [~, id] = min(fit);
        Leader = pop(id, :);

        for j = 1:PopSize
            if FE >= maxFE
                break;
            end
            if j == id
                continue;
            end
            Migrant = pop(j, :);

            % Column c is jump c: Migrant + (Leader - Migrant)*t_c on the PRT-selected coordinates
            PRTVector = rand(dim, N_jump) < PRT;
            path = Migrant' + (Leader' - Migrant') .* t_path .* PRTVector;

            % find() walks the path column by column, the order the release redraws in
            out = path < lb' | path > ub' | ~isfinite(path);
            [rw, ~] = find(out);
            path(out) = lb(rw)' + rand(numel(rw), 1) .* span(rw)';

            n_eval = min(N_jump, maxFE - FE);
            offs = path(:, 1:n_eval)';
            [new_cost, FE, bsf, bsf_solution, curve] = evaluate_rows(offs, problem, FE, bsf, bsf_solution, curve);

            % A NaN incumbent could never be beaten, so any non-NaN path best replaces it
            [min_new_cost, idz] = min(new_cost);
            if min_new_cost <= fit(j) || (isnan(fit(j)) && ~isnan(min_new_cost))
                pop(j, :) = offs(idz, :);
                fit(j)    = min_new_cost;
            end
            [population_history, fitness_history, history_index] = record_history( ...
                FE, pop, fit, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Evaluates the rows of X (already cut to the budget) and tracks the best per evaluation
function [f, FE, bsf, bsf_solution, curve] = evaluate_rows(X, problem, FE, bsf, bsf_solution, curve)
    [fv, FE_new] = calculate_fitness(X', problem, FE);
    f = fv(:);
    for q = 1:numel(f)
        if f(q) < bsf
            bsf = f(q);
            bsf_solution = X(q, :);
        end
        curve(FE + q) = bsf;
    end
    FE = FE_new;
end

% ----------------------------------------------------------------------- %
% Self-Organizing Migrating Algorithm Team To Team Adaptive (SOMA T3A)
% CEC 2019 / GECCO 2019 100-Digit Challenge entry (score 93)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize = 100               % Population size
%   N_jump = 45                 % Points sampled on each migrant's path
%   m = 10, n = 5, k = 10       % Migrant pool, migrants moved per round, leader pool
%   PRT = 0.05 + 0.90*t         % Perturbation rate, t = FE/maxFe
%   Step = 0.15 - 0.08*t        % Distance between consecutive jumps
%
% Algorithm Concept:
%   - Organisation: m individuals are drawn at random and the best n of them
%     migrate; for each migrant k are drawn and the best of those leads it
%   - Migration: the migrant jumps N_jump times along the line to its leader, at
%     multiples of Step, each coordinate moving only if a uniform draw is below PRT
%   - Update: the best point on the path replaces the migrant if it is no worse
%   - PRT rises and Step shrinks linearly with the spent budget
%   - Coordinates that leave the box are redrawn uniformly inside it
%
% Reference:
% Quoc Bao Diep,
% Self-Organizing Migrating Algorithm Team To Team Adaptive -- SOMA T3A,
% 2019 IEEE Congress on Evolutionary Computation (CEC), 2019, pp. 1182-1187.
% https://doi.org/10.1109/CEC.2019.8790202
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's general release, SOMA_T3A_v1.m in
% github.com/diepquocbao/SOMA-T3A-MATLAB, with its default PopSize, N_jump, m, n
% and k and the paper's linear PRT and Step schedules (signs as released). The
% 93-point 100-Digit entry (Diep, Zelinka, Das, Senkerik, CCIS 1092 (2020) 155-165,
% https://doi.org/10.1007/978-3-030-37838-7_14) ran a tuned configuration --
% per-function m and k, PopSize 1500, N_jump 100, PRT = 0.08 + 0.5*t and a cosine
% Step clocked on 1e9 evaluations -- which is not ported. The release stops once
% fewer than N_jump+1 evaluations remain and lets a round's n paths overrun the
% budget; here each path is cut at maxFe, so FE >= maxFe is the only terminator.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = soma_t3a(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    PopSize = 100;
    N_jump  = 45;
    m       = 10;
    n       = 5;
    k       = 10;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    PopSize = min(PopSize, maxFE);
    m = min(m, PopSize);
    n = min(n, m);
    k = min(k, PopSize);

    % Drawn dimension-major per individual, the release's column order
    pop = lb + rand(dim, PopSize)' .* span;
    bsf = inf;
    bsf_solution = pop(1, :);
    [fit, FE, bsf, bsf_solution, curve] = evaluate_rows(pop, problem, FE, bsf, bsf_solution, curve);
    [population_history, fitness_history, history_index] = record_history( ...
        FE, pop, fit, population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        t    = FE / maxFE;
        PRT  = 0.05 + 0.90 * t;
        Step = 0.15 - 0.08 * t;

        % sortrows on [fitness, position, index] reproduces the release's tie order
        M = randperm(PopSize, m);
        [~, ordM] = sortrows([fit(M), pop(M, :), M(:)]);
        order_M = M(ordM);
        fit_M   = fit(order_M);
        pop_M   = pop(order_M, :);

        for j = 1:n
            if FE >= maxFE
                break;
            end
            Migrant = pop_M(j, :);

            K = randperm(PopSize, k);
            [~, ordK] = sortrows([fit(K), pop(K, :), K(:)]);
            leader_id = K(ordK(1));
            if order_M(j) == leader_id
                continue;
            end
            Leader = pop(leader_id, :);

            % Column c is jump c: Migrant + (Leader - Migrant)*c*Step on the PRT-selected coordinates
            PRTVector = rand(dim, N_jump) < PRT;
            path = Migrant' + (Leader' - Migrant') .* ((1:N_jump) * Step) .* PRTVector;

            % find() walks the path column by column, the order the release redraws in
            out = path < lb' | path > ub' | ~isfinite(path);
            [rw, ~] = find(out);
            path(out) = lb(rw)' + rand(numel(rw), 1) .* span(rw)';

            n_eval = min(N_jump, maxFE - FE);
            offs = path(:, 1:n_eval)';
            [new_cost, FE, bsf, bsf_solution, curve] = evaluate_rows(offs, problem, FE, bsf, bsf_solution, curve);

            [min_new_cost, idz] = min(new_cost);
            if min_new_cost <= fit_M(j)
                pop(order_M(j), :) = offs(idz, :);
                fit(order_M(j))    = min_new_cost;
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

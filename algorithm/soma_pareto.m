% ----------------------------------------------------------------------- %
% Self-Organizing Migrating Algorithm Pareto (SOMA Pareto)
% CEC 2019 100-Digit Challenge -- 6th place of 18 (score 85.04)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize = 100                        % Population size
%   N_jump = 10                          % Points sampled on the migrant's path
%   C, A, D = ceil([0.04 0.20 0.16]*PopSize)  % Leader pool, best group, migrant band
%   PRT = 0.50 + 0.45*cos(pi*t + pi)     % Perturbation rate 0.05 -> 0.95, t = FE/maxFe
%   Step = 0.35 + 0.15*cos(pi*t)         % Jump length 0.50 -> 0.20
%
% Algorithm Concept:
%   - Organisation by the Pareto principle: the population is sorted, the Leader is
%     drawn from the best C individuals and the Migrant from ranks A+1..A+D
%   - Migration: one Migrant per round jumps N_jump times towards the Leader at
%     multiples of Step, each coordinate moving only if a uniform draw is below PRT
%   - Update: the best point on the path replaces the Migrant if it is no worse
%   - PRT rises and Step shrinks along cosine schedules of the spent budget
%   - Coordinates that leave the box are redrawn uniformly inside it
%
% Reference:
% Thanh Cong Truong, Quoc Bao Diep, Ivan Zelinka, Roman Senkerik,
% Pareto-Based Self-organizing Migrating Algorithm Solving 100-Digit Challenge,
% Swarm, Evolutionary, and Memetic Computing and Fuzzy and Neural Computing (SEMCCO
% 2019), Communications in Computer and Information Science 1092 (2020) 13-20.
% https://doi.org/10.1007/978-3-030-37838-7_2
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's general release, SOMA_PARETO.m in
% github.com/diepquocbao/SOMA-Pareto-MATLAB (the FX 71724 code; method paper Diep,
% Zelinka, Das, MENDEL 25 (2019) 111-120, doi:10.13164/mendel.2019.1.111), with its
% defaults PopSize = 100 and N_jump = 10. The contest paper is paywalled and the
% organisers list a per-function tuned budget (1e5..1e12 FE), so no contest setting
% is ported. Kept as released: the soft PRT vector of MENDEL Eq. (6) (unselected
% coordinates move by FE/maxFe) is commented out there, so the vector is binary;
% Step falls as in MENDEL Eq. (5) (the Python release flips its sign). The release
% stops once fewer than N_jump evaluations remain; here the last path is cut at
% maxFe, so FE >= maxFe is the only terminator. Seeded runs match it bit for bit.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = soma_pareto(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    PopSize = 100;
    N_jump  = 10;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    PopSize = max(2, min(PopSize, maxFE));
    nC = ceil(PopSize * 0.04);
    nA = ceil(PopSize * 0.20);
    nD = min(ceil(PopSize * 0.16), PopSize - nA);

    % Drawn dimension-major per individual, the release's column order
    pop = lb + rand(dim, PopSize)' .* span;
    bsf = inf;
    bsf_solution = pop(1, :);
    [fit, FE, bsf, bsf_solution, curve] = evaluate_rows(pop, problem, FE, bsf, bsf_solution, curve);
    [population_history, fitness_history, history_index] = record_history( ...
        FE, pop, fit, population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        % (pi*FE)/maxFE, not pi*(FE/maxFE): the release's rounding, so its seeded runs match
        PRT  = 0.50 + 0.45 * cos(pi * FE / maxFE + pi);
        Step = 0.35 + 0.15 * cos(pi * FE / maxFE);

        % sortrows on [fitness, position] reproduces the release's tie order
        [~, ord] = sortrows([fit, pop]);
        fit = fit(ord);
        pop = pop(ord, :);

        mig     = randi([nA + 1, nA + nD]);
        Migrant = pop(mig, :);
        Leader  = pop(randi([1, nC]), :);

        % Column c is jump c: Migrant + (Leader - Migrant)*c*Step on the PRT-selected coordinates
        PRTVector = rand(dim, N_jump) < PRT;
        path = Migrant' + (Leader' - Migrant') .* ((1:N_jump) * Step) .* PRTVector;

        % find() walks the path column by column, the order the release redraws in
        out = path < lb' | path > ub' | ~isfinite(path);
        [rw, ~] = find(out);
        path(out) = lb(rw)' + rand(numel(rw), 1) .* span(rw)';

        n_eval = min(N_jump, maxFE - FE);
        offs = path(:, 1:n_eval)';
        FE0 = FE;
        [new_cost, FE, bsf, bsf_solution, curve] = evaluate_rows(offs, problem, FE, bsf, bsf_solution, curve);

        % The population only changes once the whole path has been evaluated
        for q = FE0 + 1:FE - 1
            [population_history, fitness_history, history_index] = record_history( ...
                q, pop, fit, population_history, fitness_history, history_index, maxFE);
        end
        [min_new_cost, idz] = min(new_cost);
        if min_new_cost <= fit(mig)
            pop(mig, :) = offs(idz, :);
            fit(mig)    = min_new_cost;
        end
        [population_history, fitness_history, history_index] = record_history( ...
            FE, pop, fit, population_history, fitness_history, history_index, maxFE);
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

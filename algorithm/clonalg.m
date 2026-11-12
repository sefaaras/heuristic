% ----------------------------------------------------------------------- %
% Clonal Selection Algorithm (CLONALG)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = n = 100                 % Antibodies, every one of them selected for cloning
%   beta = 0.1                  % round(beta*N) = 10 clones per antibody, Eq. (5)
%   d = 0                       % Lowest-affinity antibodies replaced per generation
%   L = 22                      % Bits per variable, plain binary decoded onto [lb, ub]
%   pm = 0.01 -> 0.008          % Per-bit hypermutation probability, x0.8 in the last generation
%
% Algorithm Concept:
%   - An antibody is a bitstring; each variable is an L-bit integer mapped
%     linearly onto [lb, ub]
%   - Every antibody is cloned round(beta*N) times and every clone bit flips with
%     probability pm; the last clone of each group stays an unmutated parent copy
%   - The best clone of each group replaces its parent, so each antibody
%     hill-climbs on its own and several optima survive in the population
%   - The d lowest-affinity antibodies are replaced by random bitstrings
%   - The population is re-evaluated at the start of every generation
%
% Reference:
% Leandro N. de Castro, Fernando J. Von Zuben,
% Learning and optimization using the clonal selection principle,
% IEEE Transactions on Evolutionary Computation 6 (3) (2002) 239-251.
% https://doi.org/10.1109/TEVC.2002.1011539
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, imalg3d.m in demo.zip (Unicamp FTP, linked from the
% page the paper's refs [5], [6] cite; Wayback copy of 2017-07-06), whose defaults are
% the paper's g3 run (Sec. VI-B): N = n = 100, beta = 0.1 by Eq. (5), d = 0, L = 22.
% The paper only asks for a mutation rate that falls with affinity, offering
% alpha = exp(-rho*f) (Eq. (6)) without a rho; the release flips bits with a fixed pm,
% used here. Kept as released: the population and each group's parent copy are
% re-evaluated (200 of 1100 FE per generation), and pm drops to 0.8*pm for the last
% generation (itpm = Ngen), Ngen here being the generations the budget allows.
% Minimisation maximises -f with the release's sort and tie order; a NaN fitness ranks
% worst. Seeded runs match the release bit for bit.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = clonalg(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N    = 100;
    n    = N;
    beta = 0.1;
    d    = 0;
    L    = 22;
    pm   = 0.01;
    pmr  = 0.8;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    nc  = round(beta * N);
    Nc  = n * nc;
    pcs = (1:n) * nc;
    % The release runs Ngen + 1 generations of N + Nc evaluations each
    itpm = max(1, ceil(maxFE / (N + Nc)) - 1);
    pma  = pm;
    nbit = L * dim;
    w    = reshape(2 .^ (L - 1:-1:0), 1, L);
    scl  = span / (2 ^ L - 1);

    % cadeia(N, nbit, 0, 0, 0): hardlim of U(-1, 1)
    P = (2 .* rand(N, nbit) - 1) >= 0;

    bsf = inf;
    bsf_solution = lb;
    it = 0;
    while FE < maxFE
        X = decode(P, w, lb, ub, scl, L, dim);
        m = min(N, maxFE - FE);
        [f, FE, bsf, bsf_solution, curve] = evaluate_rows(X(1:m, :), problem, FE, bsf, bsf_solution, curve);
        if m < N
            [population_history, fitness_history, history_index] = record_history( ...
                FE, X(1:m, :), f, population_history, fitness_history, history_index, maxFE);
            break;
        end
        [population_history, fitness_history, history_index] = record_history( ...
            FE, X, f, population_history, fitness_history, history_index, maxFE);

        % Affinity is -f; sort ascending puts the best last, ties in index order
        aff = -f;
        aff(isnan(aff)) = -inf;
        [~, ind] = sort(aff);
        sel = ind(end:-1:end - n + 1);

        T = P(repelem(sel, nc), :);
        M = rand(Nc, nbit) <= pm;
        T = xor(T, M);
        T(pcs, :) = P(sel, :);

        XT = decode(T, w, lb, ub, scl, L, dim);
        mt = min(Nc, maxFE - FE);
        [fT, FE, bsf, bsf_solution, curve] = evaluate_rows(XT(1:mt, :), problem, FE, bsf, bsf_solution, curve);
        if mt < Nc
            % The budget ends inside the clone batch: the population is still the one measured above
            [population_history, fitness_history, history_index] = record_history( ...
                FE, X, f, population_history, fitness_history, history_index, maxFE);
            break;
        end

        % max() takes the first best of each group, so a mutated clone wins a tie with the parent copy
        affT = -fT;
        affT(isnan(affT)) = -inf;
        [~, b] = max(reshape(affT, nc, n), [], 1);
        bcs = b(:) + (0:n - 1)' * nc;
        P(sel, :) = T(bcs, :);
        X(sel, :) = XT(bcs, :);
        f(sel)    = fT(bcs);
        [population_history, fitness_history, history_index] = record_history( ...
            FE, X, f, population_history, fitness_history, history_index, maxFE);

        % Receptor editing, Step 8 (d = 0 in the paper's optimisation runs)
        P(ind(1:d), :) = (2 .* rand(d, nbit) - 1) >= 0;

        it = it + 1;
        if rem(it, itpm) == 0
            pm = pm * pmr;
            if rem(it, 10 * itpm) == 0
                pm = pma;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Each L-bit block, most significant bit first, mapped linearly onto [lb, ub]
function X = decode(B, w, lb, ub, scl, L, dim)
    rows = size(B, 1);
    v = reshape(sum(reshape(double(B), rows, L, dim) .* w, 2), rows, dim);
    % The all-ones code can round one ulp past ub
    X = min(lb + v .* scl, ub);
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

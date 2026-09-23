% ----------------------------------------------------------------------- %
% Reconstructed Differential Evolution (RDE)
% CEC 2024 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 18*D -> 4              % Linear reduction over the budget
%   H = 5                       % Memory slots, initialised to Cr 0.8 and F 0.3
%   p = 0.25*(1 - 0.5*t)        % pbest rate, shrinking with the spent budget t
%   arc_rate = 1.0              % Archive size, shrinking linearly to 4
%   jumping_rate = 0.2          % Chance that a non-crossover coordinate is jittered
%   EB_rate = 0.5 (adaptive)    % Share of trials built by the ordered mutation
%
% Algorithm Concept:
%   - Built on LSHADE-RSP: donors are drawn by rank-based selective pressure,
%     with weights 3*(NP - rank), from the population and from the archive
%   - Half the trials, a share that adapts to which side is winning, use an
%     ordered mutation: pbest, r1 and r2 are re-sorted by fitness so the step
%     always goes towards the better of the three and along the medium-to-worst
%     difference
%   - A sixth memory slot that does not exist returns F = Cr = 0.9, mixing a
%     fixed aggressive setting in throughout the run; F is capped at 0.7 early
%     and Cr floored at 0.7 then 0.6, which keeps the first half's steps large
%   - With probability 0.2 the coordinates a trial did NOT take from the donor
%     are jittered by a Cauchy draw instead of copied from the parent
%
% Reference:
% Sichen Tao, Ruihan Zhao, Kaiyu Wang, Shangce Gao,
% An Efficient Reconstructed Differential Evolution Variant by Some of the
% Current State-of-the-art Strategies for Solving Single Objective Bound
% Constrained Problems,
% arXiv preprint arXiv:2404.16280 (2024).
% https://doi.org/10.48550/arXiv.2404.16280
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released RDE.cpp (CEC2024 BC-SOP submission) with
% every strategy flag on, as it ships. Its memory update is kept as written: the
% braces put only the Cr line under the else, so F's memory is rewritten on every
% successful generation while Cr's is frozen once the slot turns -1. Trials are
% evaluated one generation at a time rather than one at a time, which changes
% nothing, since a generation's trials are all built from the population as it
% stood at the start of that generation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rde(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    NP_max = 18 * dim;
    NP_min = 4;
    mem_size = 5;
    arc_rate = 1.0;
    psize_param = 0.25;
    jumping_rate = 0.2;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    NP = NP_max;
    pop = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(ub - lb, NP, 1);
    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
        track(pop, fitness, NP, FE, best_fitness, best_solution, curve, ...
              population_history, fitness_history, history_index, maxFE);

    memory_cr = 0.8 * ones(mem_size, 1);
    memory_f = 0.3 * ones(mem_size, 1);
    memory_iter = 1;
    archive = zeros(0, dim);
    archive_fit = zeros(0, 1);
    archive_size = round(arc_rate * (NP_max - NP_min));
    eb_rate = 0.5;

    while FE < maxFE
        NP = size(pop, 1);
        t = FE / maxFE;

        [~, order] = sort(fitness);          % rank 1 is the best
        rank_w = 3 * (NP - (0:NP-1)');       % selective pressure weights, Eq. RSP
        psize = psize_param * (1 - 0.5 * t);
        psizeval = max(2, floor(NP * psize));

        n_arch = size(archive, 1);
        if n_arch > 0
            [~, arch_order] = sort(archive_fit);
            arch_w = 3 * (n_arch - (0:n_arch-1)');
        end

        U = pop;
        F_used = zeros(NP, 1);
        CR_used = zeros(NP, 1);
        eb_flag = false(NP, 1);

        for i = 1:NP
            slot = randi(mem_size + 1);      % the extra slot is the fixed 0.9 pair

            prand = order(roulette(rank_w(1:psizeval)));
            while prand == i && t < 0.5
                prand = order(roulette(rank_w(1:psizeval)));
            end

            r1 = order(roulette(rank_w));
            while r1 == prand
                r1 = order(roulette(rank_w));
            end
            r2 = order(roulette(rank_w));
            while r2 == prand || r2 == r1
                r2 = order(roulette(rank_w));
            end

            F = -1;
            while F <= 0
                if slot <= mem_size
                    F = memory_f(slot) + 0.1 * tan(pi * (rand - 0.5));
                else
                    F = 0.9 + 0.1 * tan(pi * (rand - 0.5));
                end
            end
            F = min(F, 1);
            if t < 0.6 && F > 0.7
                F = 0.7;
            end

            use_archive = n_arch > 0 && rand < n_arch / (n_arch + NP);
            eb_flag(i) = rand < eb_rate;
            if use_archive
                r2a = arch_order(roulette(arch_w));
            end

            if eb_flag(i)
                if use_archive
                    [b, m, w] = eb_order(pop(prand, :), fitness(prand), pop(r1, :), fitness(r1), ...
                                         archive(r2a, :), archive_fit(r2a));
                else
                    [b, m, w] = eb_order(pop(prand, :), fitness(prand), pop(r1, :), fitness(r1), ...
                                         pop(r2, :), fitness(r2));
                end
                donor = pop(i, :) + F * (b - pop(i, :)) + F * (m - w);
            else
                if use_archive
                    donor = pop(i, :) + F * (pop(prand, :) - pop(i, :)) + F * (pop(r1, :) - archive(r2a, :));
                else
                    donor = pop(i, :) + F * (pop(prand, :) - pop(i, :)) + F * (pop(r1, :) - pop(r2, :));
                end
            end

            below = donor < lb;
            above = donor > ub;
            donor(below) = 0.5 * (lb(below) + pop(i, below));
            donor(above) = 0.5 * (ub(above) + pop(i, above));

            if slot <= mem_size
                if memory_cr(slot) < 0
                    CR = 0;
                else
                    CR = memory_cr(slot) + 0.1 * randn;
                end
            else
                CR = 0.9 + 0.1 * randn;
            end
            CR = min(max(CR, 0), 1);
            if t < 0.25
                CR = max(CR, 0.7);
            end
            if t < 0.5
                CR = max(CR, 0.6);
            end

            jrand = randi(dim);
            take = rand(1, dim) < CR;
            take(jrand) = true;
            perturb = rand < jumping_rate;
            if perturb
                jitter = pop(i, :) + 0.1 * tan(pi * (rand(1, dim) - 0.5));
                U(i, :) = jitter;
            else
                U(i, :) = pop(i, :);
            end
            U(i, take) = donor(take);

            F_used(i) = F;
            CR_used(i) = CR;
        end

        [fu, FE] = calculate_fitness(U', problem, FE);
        fu = fu(:);

        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            track(U, fu, NP, FE, best_fitness, best_solution, curve, ...
                  population_history, fitness_history, history_index, maxFE);

        improved = fu <= fitness;

        % Which side of the hybrid earned the improvement decides next generation's mix
        sum_eb = sum(fitness(improved & eb_flag) - fu(improved & eb_flag));
        sum_orig = sum(fitness(improved & ~eb_flag) - fu(improved & ~eb_flag));
        if sum_eb ~= 0 && sum_orig ~= 0
            eb_rate = min(max(sum_eb / (sum_eb + sum_orig), 0), 1);
        else
            eb_rate = 0.5;
        end

        s_cr = CR_used(improved);
        s_f = F_used(improved);
        s_df = abs(fitness(improved) - fu(improved));

        for i = find(improved)'
            if size(archive, 1) < archive_size
                archive(end+1, :) = pop(i, :); %#ok<AGROW>
                archive_fit(end+1, 1) = fitness(i); %#ok<AGROW>
            elseif archive_size > 0
                slot_a = randi(archive_size);
                archive(slot_a, :) = pop(i, :);
                archive_fit(slot_a) = fitness(i);
            end
        end
        pop(improved, :) = U(improved, :);
        fitness(improved) = fu(improved);

        if ~isempty(s_df) && sum(s_df) > 0
            old_cr = memory_cr(memory_iter);
            if memory_cr(memory_iter) == -1 || max(s_cr) == 0
                memory_cr(memory_iter) = -1;
            else
                memory_cr(memory_iter) = (mean_wl(s_cr, s_df) + old_cr) / 2;
            end
            memory_f(memory_iter) = mean_wl(s_f, s_df);   % outside the else, as in the source
            memory_iter = mod(memory_iter, mem_size) + 1;
        end

        NP_next = max(NP_min, round((NP_min - NP_max) / maxFE * FE + NP_max));
        if NP_next < NP
            [fitness, order2] = sort(fitness);
            pop = pop(order2, :);
            pop = pop(1:NP_next, :);
            fitness = fitness(1:NP_next);
        end
        archive_size = max(NP_min, round((maxFE - FE) / maxFE * arc_rate * (NP_max - NP_min)));
        if size(archive, 1) > archive_size
            archive = archive(1:archive_size, :);
            archive_fit = archive_fit(1:archive_size);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Rank-based selective pressure: a draw proportional to the given weights
function k = roulette(w)
    c = cumsum(w);
    k = find(rand * c(end) <= c, 1, 'first');
    if isempty(k)
        k = numel(w);
    end
end

% The three donors re-sorted so the step runs best <- current and medium -> worst
function [b, m, w] = eb_order(x1, f1, x2, f2, x3, f3)
    X = [x1; x2; x3];
    [~, idx] = sort([f1; f2; f3]);
    b = X(idx(1), :);
    m = X(idx(2), :);
    w = X(idx(3), :);
end

function m = mean_wl(vals, deltas)
    w = deltas / sum(deltas);
    m = sum(w .* vals .* vals) / sum(w .* vals);
end

function [bf, bx, curve, ph, fh, hi] = track(X, f, n, FE, bf, bx, curve, ph, fh, hi, maxFE)
    for i = 1:n
        if f(i) < bf
            bf = f(i);
            bx = X(i, :);
        end
        eval_count = FE - n + i;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bf;
            [ph, fh, hi] = record_history(eval_count, X, f', ph, fh, hi, maxFE);
        end
    end
end

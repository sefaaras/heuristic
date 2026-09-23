% ----------------------------------------------------------------------- %
% Multiple Local Search L-SHADE (MLS-LSHADE)
% CEC 2021 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18*D -> 4        % Linear reduction over the budget
%   H = round(0.06*D^2)         % Memory slots for F and CR
%   p = 0.11, arc_rate = 2.6    % pbest rate and archive size
%   stall = 50*floor(D/10)^2    % Generations without progress before a restart
%   s0 = 0.02 * box width       % Initial step of the local search
%   tol = 10^(-0.4*D)           % Step at which the local search stops
%
% Algorithm Concept:
%   - L-SHADE as the global search: current-to-pbest/1 with archive, F and CR
%     drawn from a memory of successful values, population shrinking linearly
%   - When the best has not improved for stall generations, a DSCG local search
%     (Davies-Swann-Campey with Gram-Schmidt) runs from the best individual
%   - That search walks one coordinate direction at a time with an expanding
%     step, fits a quadratic through four points to place the next iterate, and
%     re-orthogonalises its directions against the total progress vector
%   - After the local search the whole population is thrown away and drawn again,
%     with the best found so far kept aside, so the run is a sequence of
%     independent L-SHADE runs punctuated by local refinement
%
% Reference:
% Le Van Cuong, Nguyen Ngoc Bao, Huynh Thi Thanh Binh,
% Technical report: A Multi-start Local Search Algorithm with L-SHADE for
% Single Objective Bound Constrained Optimization,
% SoICT, Hanoi University of Science and Technology, CEC 2021 competition on
% single objective bound constrained optimization, technical report, 2021.
% https://github.com/P-N-Suganthan/2021-SO-BCO
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the released C++ (mls_lshade.cpp, search_algorithm.cpp). Three
% defects in that release are reproduced deliberately, because they are what the
% submitted algorithm does: the fourth interpolation point reads x(1) instead of
% x(j), so it is built from one coordinate of the incumbent; the interpolation
% divides the whole numerator, position included, by the curvature; and the local
% search returns its LAST iterate rather than the best it saw, so it can hand
% back a worse individual. Two guards are ours: the stall window is floored at
% one generation, since 50*floor(D/10)^2 is zero below D = 10 and the C++ then
% indexes before the start of its history, and the local search's working points
% start initialised, which that indexing left to chance.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mls_lshade(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    pop_size = 18 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;
    memory_size = max(1, round(0.06 * dim * dim));
    p_best_rate = 0.11;
    arc_rate = 2.6;
    stall = max(1, 50 * floor(dim / 10) ^ 2);
    s0 = 0.02 * mean(ub - lb);
    tol_ls = 10 ^ (-4.0 * (dim / 10.0));

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    best_fitness = inf;        % best over the whole run, across restarts
    best_solution = zeros(1, dim);
    pop = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
    [fitness, FE] = evaluate(pop, problem, FE);
    [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
        track(pop, fitness, pop_size, FE, best_fitness, best_solution, curve, ...
              population_history, fitness_history, history_index, maxFE);

    bsf_fit = min(fitness);    % best of the CURRENT restart, which the stall test reads
    memory_sf = 0.5 * ones(memory_size, 1);
    memory_cr = 0.5 * ones(memory_size, 1);
    memory_pos = 1;
    archive = zeros(0, dim);
    arc_size = round(arc_rate * pop_size);
    generation = 0;
    bsf_history = bsf_fit;

    while FE < maxFE
        generation = generation + 1;
        bsf_history(end+1) = bsf_fit; %#ok<AGROW>

        % Restart test: no progress over the stall window
        if generation > stall && numel(bsf_history) > stall
            if bsf_history(end - stall) - bsf_history(end) < 1e-8
                budget_ls = min(maxFE - FE, floor(maxFE / dim));
                [~, ibest] = min(fitness);
                [x_ls, f_ls, FE, curve, population_history, fitness_history, history_index, ...
                 best_fitness, best_solution] = dscg(pop(ibest, :), fitness(ibest), s0, tol_ls, ...
                    0.75, budget_ls, problem, FE, maxFE, pop, fitness, curve, ...
                    population_history, fitness_history, history_index, best_fitness, best_solution);
                pop(ibest, :) = x_ls;
                fitness(ibest) = f_ls;

                if FE >= maxFE
                    break;
                end

                % Renew: the population is drawn again and the restart best reset
                pop = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
                [fitness, FE] = evaluate(pop, problem, FE);
                [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                    track(pop, fitness, pop_size, FE, best_fitness, best_solution, curve, ...
                          population_history, fitness_history, history_index, maxFE);
                bsf_fit = min(fitness);
                generation = 0;
                bsf_history = bsf_fit;
                continue;
            end
        end

        [~, sorted_index] = sort(fitness);
        p_num = max(2, round(p_best_rate * pop_size));

        r = randi(memory_size, pop_size, 1);
        mu_sf = memory_sf(r);
        mu_cr = memory_cr(r);

        cr = mu_cr + 0.1 * randn(pop_size, 1);
        cr(mu_cr == -1) = 0;
        cr = min(max(cr, 0), 1);

        sf = zeros(pop_size, 1);
        for i = 1:pop_size
            while sf(i) <= 0
                sf(i) = mu_sf(i) + 0.1 * tan(pi * (rand - 0.5));
            end
        end
        sf = min(sf, 1);

        popAll = [pop; archive];
        r1 = zeros(pop_size, 1);
        r2 = zeros(pop_size, 1);
        for i = 1:pop_size
            r1(i) = randi(pop_size);
            while r1(i) == i
                r1(i) = randi(pop_size);
            end
            r2(i) = randi(size(popAll, 1));
            while r2(i) == i || r2(i) == r1(i)
                r2(i) = randi(size(popAll, 1));
            end
        end

        pbest = pop(sorted_index(randi(p_num, pop_size, 1)), :);
        vi = pop + sf .* (pbest - pop) + sf .* (pop(r1, :) - popAll(r2, :));
        vi = bound_mid(vi, pop, lb, ub);

        mask = rand(pop_size, dim) > cr(:, ones(1, dim));
        jrand = sub2ind([pop_size dim], (1:pop_size)', randi(dim, pop_size, 1));
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        [children_fitness, FE] = evaluate(ui, problem, FE);
        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            track(ui, children_fitness, pop_size, FE, best_fitness, best_solution, curve, ...
                  population_history, fitness_history, history_index, maxFE);

        S_sf = []; S_cr = []; S_df = [];
        for i = 1:pop_size
            if children_fitness(i) == fitness(i)
                pop(i, :) = ui(i, :);
                fitness(i) = children_fitness(i);
            elseif children_fitness(i) < fitness(i)
                if arc_size > 1
                    if size(archive, 1) < arc_size
                        archive(end+1, :) = pop(i, :); %#ok<AGROW>
                    else
                        archive(randi(arc_size), :) = pop(i, :);
                    end
                end
                S_df(end+1, 1) = abs(fitness(i) - children_fitness(i)); %#ok<AGROW>
                S_sf(end+1, 1) = sf(i); %#ok<AGROW>
                S_cr(end+1, 1) = cr(i); %#ok<AGROW>
                pop(i, :) = ui(i, :);
                fitness(i) = children_fitness(i);
            end
        end
        bsf_fit = min(bsf_fit, min(fitness));

        if ~isempty(S_df)
            w = S_df / sum(S_df);
            if max(S_cr) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = sum(w .* S_cr .* S_cr) / sum(w .* S_cr);
            end
            memory_sf(memory_pos) = sum(w .* S_sf .* S_sf) / sum(w .* S_sf);
            memory_pos = mod(memory_pos, memory_size) + 1;
        end

        plan_pop_size = round(((min_pop_size - max_pop_size) / maxFE) * FE + max_pop_size);
        plan_pop_size = max(plan_pop_size, min_pop_size);
        if pop_size > plan_pop_size
            [fitness, order] = sort(fitness);
            pop = pop(order, :);
            pop = pop(1:plan_pop_size, :);
            fitness = fitness(1:plan_pop_size);
            pop_size = plan_pop_size;
            arc_size = round(arc_rate * pop_size);
            if size(archive, 1) > arc_size
                archive = archive(1:arc_size, :);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

function [f, FE] = evaluate(X, problem, FE)
    [f, FE] = calculate_fitness(X', problem, FE);
    f = f(:);
end

% A coordinate outside the box moves to the midpoint with its parent
function vi = bound_mid(vi, pop, lb, ub)
    n = size(vi, 1);
    LB = repmat(lb, n, 1);
    UB = repmat(ub, n, 1);
    below = vi < LB;
    above = vi > UB;
    vi(below) = 0.5 * (LB(below) + pop(below));
    vi(above) = 0.5 * (UB(above) + pop(above));
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

% Davies-Swann-Campey search with Gram-Schmidt re-orthogonalisation
function [u, u_fit, FE, curve, ph, fh, hi, bf, bx] = dscg(u, u_fit, s0, tol, alpha, budget, ...
        problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx)
    dim = numel(u);
    x = repmat(u, dim + 2, 1);
    x_fit = repmat(u_fit, dim + 2, 1);
    direction = eye(dim + 1, dim);
    s = s0;
    spent = 0;

    while true
        i = 1;
        while i <= dim
            before = spent;
            [x(i + 1, :), x_fit(i + 1), used, FE, curve, ph, fh, hi, bf, bx] = line_search( ...
                x(i, :), x_fit(i), s, direction(i, :), problem, FE, maxFE, pop, popfit, ...
                curve, ph, fh, hi, bf, bx);
            spent = spent + used;
            if spent > budget || FE >= maxFE
                spent = before;
                break;
            end
            if x_fit(i + 1) < u_fit
                u_fit = x_fit(i + 1);
                u = x(i + 1, :);
            end
            i = i + 1;
        end
        if spent > budget || FE >= maxFE
            break;
        end

        z = x(dim + 1, :) - x(1, :);
        len = sqrt(sum(z .^ 2));

        if len <= 0
            x(dim + 2, :) = x(dim + 1, :);
            x_fit(dim + 2) = x_fit(dim + 1);
            s = s * 0.1;
            if s <= tol
                break;
            end
            x(1, :) = x(dim + 2, :);
            x_fit(1) = x_fit(dim + 2);
            continue;
        end

        direction(dim + 1, :) = z / len;
        [x(dim + 2, :), x_fit(dim + 2), used, FE, curve, ph, fh, hi, bf, bx] = line_search( ...
            x(dim + 1, :), x_fit(dim + 1), s, direction(dim + 1, :), problem, FE, maxFE, ...
            pop, popfit, curve, ph, fh, hi, bf, bx);
        spent = spent + used;
        if spent > budget || FE >= maxFE
            break;
        end
        if x_fit(dim + 2) < u_fit
            u_fit = x_fit(dim + 2);
            u = x(dim + 2, :);
        end

        if len < s
            s = s * alpha;
            if s <= tol
                break;
            end
            x(1, :) = x(dim + 2, :);
            x_fit(1) = x_fit(dim + 2);
        else
            direction = gram_schmidt(direction);
            x(1, :) = x(dim + 1, :);
            x_fit(1) = x_fit(dim + 1);
            x(2, :) = x(dim + 2, :);
            x_fit(2) = x_fit(dim + 2);
        end
    end

    % The reference hands back its last iterate, not the best it saw
    u = x(dim + 2, :);
    u_fit = x_fit(dim + 2);
end

function [result, result_fit, used, FE, curve, ph, fh, hi, bf, bx] = line_search( ...
        x00, x00_fit, s0, v, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx)
    lb = problem.lb;
    ub = problem.ub;
    s = s0;
    used = 0;
    x0 = x00;
    x0_fit = x00_fit;

    x = min(max(x0 + s * v, lb), ub);
    [x_fit, FE, curve, ph, fh, hi, bf, bx] = probe(x, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
    used = used + 1;

    ks = 1;
    if x_fit > x0_fit
        ks = 0;
        x = min(max(x - 2 * s * v, lb), ub);
        s = -s;
        [x_fit, FE, curve, ph, fh, hi, bf, bx] = probe(x, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
        used = used + 1;
        if x_fit <= x0_fit
            ks = 1;
        end
    end

    if ks == 1
        while used < 50 && FE < maxFE
            s = 2 * s;
            x0 = x;
            x0_fit = x_fit;
            x = min(max(x0 + s * v, lb), ub);
            [x_fit, FE, curve, ph, fh, hi, bf, bx] = probe(x, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
            used = used + 1;
            if x_fit > x0_fit
                s = s / 2;
                x = x0 + s * v;
                [x_fit, FE, curve, ph, fh, hi, bf, bx] = probe(x, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
                used = used + 1;
                break;
            end
        end
    end

    % Four interpolation points; x4 reads x(1), not x(j), exactly as released
    x1 = min(max(x0 - s * v, lb), ub);
    x2 = min(max(x0, lb), ub);
    x3 = min(max(x0 + 2 * s * v, lb), ub);
    x4 = min(max(x(1) + s * v, lb), ub);
    P = [x1; x2; x4; x3];
    Pf = zeros(4, 1);
    for q = 1:4
        [Pf(q), FE, curve, ph, fh, hi, bf, bx] = probe(P(q, :), problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
    end
    used = used + 4;

    [~, reject] = max(Pf);
    P(reject, :) = [];
    Pf(reject) = [];

    denom = Pf(1) - 2 * Pf(2) + Pf(3);
    xt = (P(2, :) + s * v * (Pf(1) - Pf(3)) / 2.0) / denom;
    xt = min(max(xt, lb), ub);
    xt(~isfinite(xt)) = P(2, ~isfinite(xt));
    [xt_fit, FE, curve, ph, fh, hi, bf, bx] = probe(xt, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx);
    used = used + 1;

    if xt_fit < Pf(2)
        result = xt;
        result_fit = xt_fit;
    else
        result = P(2, :);
        result_fit = Pf(2);
    end
    if x00_fit < result_fit
        result = x00;
        result_fit = x00_fit;
    end
end

function [f, FE, curve, ph, fh, hi, bf, bx] = probe(x, problem, FE, maxFE, pop, popfit, curve, ph, fh, hi, bf, bx)
    [fv, FE] = calculate_fitness(x', problem, FE);
    f = fv(1);
    if f < bf
        bf = f;
        bx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bf;
        [ph, fh, hi] = record_history(FE, pop, popfit', ph, fh, hi, maxFE);
    end
end

function D = gram_schmidt(D)
    for i = 1:size(D, 1)
        for j = 1:i-1
            p2 = sum(D(j, :) .^ 2);
            if p2 > 0
                D(i, :) = D(i, :) - (sum(D(i, :) .* D(j, :)) / p2) * D(j, :);
            end
        end
        len = sqrt(sum(D(i, :) .^ 2));
        if len > 0
            D(i, :) = D(i, :) / len;
        end
    end
end

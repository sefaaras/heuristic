% ----------------------------------------------------------------------- %
% L-SHADE with Neurodynamic Differential Evolution (LSHADE-ND)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18 * D -> 4       % Linear population size reduction
%   memory_size = 100 -> 5       % Historical memory, shrinks with the budget
%   p_best_rate = 0.11           % Greediness of the pbest term
%   arc_rate = 1.4               % Archive size relative to the population
%   nd_steps = 20                % Neurodynamic recurrence length per call
%   nd_delta = 1e-2              % Forward-difference step of the gradient
%   nd_budget = (2/3) * maxFe    % ND is switched off past this point
%
% Algorithm Concept:
%   - L-SHADE base: current-to-pbest/1 with an external archive, success-history
%     adaptation of F and CR, and linear population size reduction
%   - Each generation runs EITHER a neurodynamic step (with probability PND) or
%     an ordinary DE generation
%   - Neurodynamic step: starting from the current best, iterate 20 times
%     z <- mu*z + (1-mu)*clamp(z - grad f(z)) with mu = exp(-1); the gradient is
%     a forward difference and costs D evaluations per iteration
%   - The trajectory points compete with the population and the best five are merged in
%   - PND tracks the relative gain of the last two generations, floored at 0.01, and
%     collapses to 0.01 whenever the trajectory fails to beat its own start point
%   - CR is drawn from a normal distribution 80% of the time, a Cauchy the rest
%
% Reference:
% Karam M. Sallam, Ruhul A. Sarker, Daryl Essam, Saber M. Elsayed,
% Neurodynamic differential evolution algorithm and solving CEC2015
% competition problems,
% IEEE Congress on Evolutionary Computation (CEC), Sendai, Japan, 2015, pp. 1033-1040
% https://doi.org/10.1109/CEC.2015.7257003
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (LSHADEND/main_code.m + ND.m), not the paper.
% The release stops a run once |f_best - f_optimal| <= 1e-8; the optimum is not known to
% this harness and the rule would under-spend the budget, so FE >= maxFe is the sole
% terminator here. The neurodynamic phase costs 20*D + 22 evaluations and the release
% checks the budget only between generations, so guards were added inside the gradient
% loop and before the trajectory batch, and a trajectory cut short by the budget is
% truncated rather than padded. Reproduced deliberately: the gradient rescale divides by
% sum(abs(g)) recomputed while g is being overwritten, so it depends on component order.
% Scalar xmin/xmax became per-dimension lb/ub. The ND phase is a single-point local
% search, so it records the population it holds frozen, as ebocmar's LS2 does.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lshade_nd(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    lu = [lb; ub];

    % Algorithm parameters
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 100;
    min_memory_size = 5;
    pop_size = 18 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;
    nd_budget = (2 / 3) * maxFE;
    pnd = 1;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the main population
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, dim) .* repmat(lu(2, :) - lu(1, :), pop_size, 1);
    pop = popold;

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    bsf_fit_var = 1e+30;
    bsf_solution = zeros(1, dim);

    for i = 1:pop_size
        if fitness(i) < bsf_fit_var
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, history_index, maxFE);
        end
    end

    [~, loc] = min(fitness);

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    iter = 0;
    gen_best = zeros(1, 0);   % per-generation min(fitness); drives the PND update

    % Main loop
    while FE < maxFE
        iter = iter + 1;

        if rand < pnd && FE < nd_budget
            fe_from = FE + 1;
            [popold, fitness, FE, pnd] = neurodynamic( ...
                loc, popold, fitness, problem, lu, FE, maxFE, pnd, iter, gen_best);
            pop = popold;

            for i = 1:numel(fitness)
                if fitness(i) < bsf_fit_var
                    bsf_fit_var = fitness(i);
                    bsf_solution = popold(i, :);
                end
            end

            [curve, population_history, fitness_history, history_index] = record_span( ...
                fe_from, FE, bsf_fit_var, curve, pop, fitness, maxFE, ...
                population_history, fitness_history, history_index);
        else
            pop = popold;
            [~, sorted_index] = sort(fitness, 'ascend');

            mem_rand_index = ceil(memory_size * rand(pop_size, 1));
            mu_sf = memory_sf(mem_rand_index);
            mu_cr = memory_cr(mem_rand_index);

            % Generate crossover rate: normal 80% of the time, Cauchy otherwise
            if rand < 0.8
                cr = normrnd(mu_cr, 0.1);
                term_pos = find(mu_cr == -1);
                cr(term_pos) = 0;
            else
                cr = mu_cr + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
                pos = find(cr <= 0);
                while ~isempty(pos)
                    cr(pos) = mu_cr(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                    pos = find(cr <= 0);
                end
            end
            cr = min(cr, 1);
            cr = max(cr, 0);

            % Generate scaling factor
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
            sf = min(sf, 1);

            r0 = 1:pop_size;
            popAll = [pop; archive.pop];
            [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);

            pNP = max(round(p_best_rate * pop_size), 2);
            randindex = ceil(rand(1, pop_size) .* pNP);
            randindex = max(1, randindex);
            pbest = pop(sorted_index(randindex), :);

            vi = pop + sf(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
            vi = boundConstraint(vi, pop, lu);

            mask = rand(pop_size, dim) > cr(:, ones(1, dim));
            rows = (1:pop_size)';
            cols = floor(rand(pop_size, 1) * dim) + 1;
            jrand = sub2ind([pop_size dim], rows, cols);
            mask(jrand) = false;
            ui = vi;
            ui(mask) = pop(mask);

            % Evaluate offspring
            [children_fitness, FE] = calculate_fitness(ui', problem, FE);
            children_fitness = children_fitness(:);

            for i = 1:pop_size
                if children_fitness(i) < bsf_fit_var
                    bsf_fit_var = children_fitness(i);
                    bsf_solution = ui(i, :);
                end
            end

            for eval_idx = 1:pop_size
                eval_count = FE - pop_size + eval_idx;
                if eval_count >= 1 && eval_count <= maxFE
                    curve(eval_count) = bsf_fit_var;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, pop, fitness, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end

            dif = abs(fitness - children_fitness);

            I = (fitness > children_fitness);
            goodCR = cr(I == 1);
            goodF = sf(I == 1);
            dif_val = dif(I == 1);

            archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));

            [fitness, I] = min([fitness, children_fitness], [], 2);

            popold = pop;
            popold(I == 2, :) = ui(I == 2, :);

            if numel(goodCR) > 0
                dif_val = dif_val / sum(dif_val);

                memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);

                if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                    memory_cr(memory_pos) = -1;
                else
                    memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                end

                memory_pos = memory_pos + 1;
                if memory_pos > memory_size
                    memory_pos = 1;
                end
            end

            % Shrink the memory alongside the population, 100 -> 5
            plan_memory_size = round((((min_memory_size - 100) / maxFE) * FE) + 100);
            if memory_size > plan_memory_size
                memory_size = max(min_memory_size, plan_memory_size);
                if memory_pos > memory_size
                    memory_pos = 1;
                end
            end

            % Linear population size reduction
            plan_pop_size = round((((min_pop_size - max_pop_size) / maxFE) * FE) + max_pop_size);
            if pop_size > plan_pop_size
                reduction_ind_num = pop_size - plan_pop_size;
                if pop_size - reduction_ind_num < min_pop_size
                    reduction_ind_num = pop_size - min_pop_size;
                end

                pop_size = pop_size - reduction_ind_num;
                for r = 1:reduction_ind_num
                    [~, indBest] = sort(fitness, 'ascend');
                    worst_ind = indBest(end);
                    popold(worst_ind, :) = [];
                    pop(worst_ind, :) = [];
                    fitness(worst_ind, :) = [];
                end

                archive.NP = round(arc_rate * pop_size);
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1:archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
        end

        [~, indBest] = sort(fitness, 'ascend');
        loc = indBest(1);
        gen_best(iter) = min(fitness);
    end

    % Fill remaining curve values
    curve(FE:end) = bsf_fit_var;

    best_fitness = bsf_fit_var;
    best_solution = bsf_solution;
end

% Helper Functions

function [popold, fitness, FE, pnd] = neurodynamic( ...
        loc, popold, fitness, problem, lu, FE, maxFE, pnd, iter, gen_best)
% A 20-iteration gradient recurrence from the current best, whose trajectory then competes
    nd_steps = 20;
    mu = exp(-1);
    dim = size(popold, 2);

    mode_org = popold(loc, :);
    mode = mode_org;
    f_loc = fitness(loc);

    z = zeros(nd_steps, dim);
    taken = 0;
    for t = 1:nd_steps
        [g, FE] = nd_gradient(mode, f_loc, problem, lu, FE, maxFE);
        z(t, :) = mu .* mode + (1 - mu) .* g;
        mode = z(t, :);
        taken = t;
        if FE >= maxFE
            break;
        end
    end
    z = [z(1:taken, :); mode_org];

    if FE >= maxFE
        return;
    end

    [fod, FE] = calculate_fitness(z', problem, FE);
    fod = fod(:);

    [~, r] = min(fod);
    if iter <= 2
        pnd = 1;
    elseif r == size(z, 1)
        pnd = 0.01;                       % the trajectory never beat its start point
    else
        ref = gen_best(iter - 2);
        gain = ref - gen_best(iter - 1);
        if isfinite(ref) && ref ~= 0
            pnd = max(0.01, gain / ref);
        else
            pnd = 0.01;                   % the release divides by zero here
        end
    end

    popold(loc, :) = z(r, :);
    popold = boundConstraint(popold, popold, lu);

    if FE >= maxFE
        return;
    end
    [f_loc_new, FE] = calculate_fitness(popold(loc, :)', problem, FE);
    fitness(loc) = f_loc_new(1);

    [fitness, sorted_index] = sort(fitness, 'ascend');
    popold = popold(sorted_index, :);
    [fod, sorted_index1] = sort(fod, 'ascend');
    z = z(sorted_index1, :);

    for i = 1:min([5, numel(fitness), numel(fod)])
        if fitness(i) > fod(i)
            fitness(i) = fod(i);
            popold(i, :) = z(i, :);
        end
    end
end

function [g, FE] = nd_gradient(y, f, problem, lu, FE, maxFE)
% Forward-difference gradient at y, then one unit descent step clamped to the box.
    dim = numel(y);
    delta = 1e-2;
    x = min(max(y, lu(1, :)), lu(2, :));

    gy = zeros(1, dim);
    for j = 1:dim
        if FE >= maxFE
            break;
        end
        x_dum = x;
        x_dum(j) = x_dum(j) + delta;
        [f_dum, FE] = calculate_fitness(x_dum', problem, FE);
        gy(j) = (f_dum(1) - f) / delta;
    end

    % Out-of-box components are rescaled by the L1 norm of gy as it is overwritten
    out = gy > lu(2, :) | gy < lu(1, :);
    for j = 1:dim
        s = sum(abs(gy));
        if out(j) && s > 0
            gy(j) = (gy(j) / s) * lu(2, j);
        end
    end

    g = min(max(x - gy, lu(1, :)), lu(2, :));
end

function [curve, ph, fh, hidx] = record_span( ...
        fe_from, fe_to, bsf, curve, pop, fitness, maxFE, ph, fh, hidx)
% Fill the curve over the evaluations a phase consumed and sample the population it held
    lo = max(1, fe_from);
    up = min(fe_to, maxFE);
    if up < lo
        return;
    end
    curve(lo:up) = bsf;
    for ec = lo:up
        [ph, fh, hidx] = record_history(ec, pop, fitness, ph, fh, hidx, maxFE);
    end
end

function vi = boundConstraint(vi, pop, lu)
    [NP, ~] = size(pop);

    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000
            error('Cannot generate r1 in 1000 iterations');
        end
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else
            r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000
            error('Cannot generate r2 in 1000 iterations');
        end
    end
end

function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0, return; end
    if size(pop, 1) ~= size(funvalue, 1), error('check it'); end

    popAll = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end

    if size(popAll, 1) <= archive.NP
        archive.pop = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:archive.NP);
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

% ----------------------------------------------------------------------- %
% L-SHADE with Selective Mutation and Dynamic Perturbation (S-LSHADE-DP)
% CEC 2022 competition (GECCO track) entry
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size    = 100            % Reduced linearly to 4
%   memory_size = 6              % Historical memory H, one PER OPERATOR
%   arc_rate    = 2.6            % Archive size as a multiple of the population
%   p_best_rate = 0.11           % Greediness of the pbest term
%   M = 2, gamma = 0.3, NG = 20  % Operator pool, draw window, review period
%   stagnation threshold = 100   % Generations before an individual is perturbed
%
% Algorithm Concept:
%   - SELECTIVE MUTATION. Two operators compete, each with its OWN F memory:
%       op0  target/1  x_i + F*(x_r1 - x_r2)   (no attraction to a best)
%       op1  current-to-pbest/1 with archive   (ordinary L-SHADE)
%     Windows of width gamma = 0.3 pick op0 and op1 explicitly and the remaining
%     0.4 goes to the better recent IMPROVEMENT RATE, recomputed every NG = 20
%   - CR IS NOT ADAPTED but scheduled: 0 for the first half of the budget and
%     always for op0. CR = 0 plus the mandatory jrand takes ONE coordinate from
%     the donor, and axis-parallel steps cannot collapse the population
%   - DYNAMIC PERTURBATION. A per-individual stagnation counter starts after
%     half the budget; past 100 it drags half the coordinates towards a fresh
%     random point, x_j <- alpha*x_j + (1-alpha)*U(lb,ub) with alpha = FE/maxFe,
%     and RE-EVALUATES -- a restart weakening as the budget runs out
%
% Reference:
% Le Van Cuong, Nguyen Ngoc Bao, Nguyen Khanh Phuong, Huynh Thi Thanh Binh,
% Dynamic perturbation for population diversity management in differential
% evolution, Proceedings of the Genetic and Evolutionary Computation Conference
% Companion (GECCO '22), 2022, pp. 391-394.
% https://doi.org/10.1145/3520304.3529075
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' C++ competition submission (s_lshade_dp.cpp, de.h,
% main.cpp; Codes_of_Top_ranked_Algorithm in Suganthan's 2022-SO-BO repository).
% Parameters come from main.cpp (pop 100, H 6, arc 2.6, p 0.11) and de.h (M = 2,
% gamma = 0.3, NG = 20); the operators, memories, CR schedule, counters and
% dynamic perturbation are verbatim. Reference bug reproduced: the equal-fitness
% selection branch copies the child over the parent BEFORE comparing them, so
% its difference counter is always zero and the stagnation counter never resets,
% and those individuals reach the perturbation threshold sooner. Adaptations:
% the CEC 1e-8 optimum snap is dropped (no optimum here), the box comes from
% problem.lb/ub, trials are vectorised (exact), perturbation FEs count.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = slshade_dp(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters (main.cpp and de.h)
    pop_size     = 100;
    max_pop_size = pop_size;
    min_pop_size = 4;
    memory_size  = 6;
    arc_rate     = 2.6;
    p_best_rate  = 0.11;

    M     = 2;
    gamma = 0.3;
    NG    = 20;
    stag_threshold = 100;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    pop = repmat(lb, pop_size, 1) + rand(pop_size, D) .* repmat(ub - lb, pop_size, 1);

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    bsf  = inf;
    bsfx = pop(1, :);
    for i = 1:pop_size
        if fitness(i) < bsf
            bsf  = fitness(i);
            bsfx = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    stagnation = zeros(pop_size, 1);

    memory_sf  = 0.5 * ones(memory_size, M);
    memory_pos = ones(1, M);

    arc_size      = round(pop_size * arc_rate);
    archive_pop   = zeros(arc_size, D);
    arc_ind_count = 0;

    improve_fitness = zeros(1, M);
    consumed_fes    = zeros(1, M);
    best_op         = 1;                 % 1-based; the reference starts at op0
    p_num           = max(round(pop_size * p_best_rate), 2);

    generation = 0;

    % Main loop
    while FE < maxFE
        generation = generation + 1;

        [~, sorted_array] = sort(fitness, 'ascend');

        % Operator selection
        rnd    = rand(pop_size, 1);
        mut_op = zeros(pop_size, 1);
        for op = 1:M
            band = (rnd >= (op - 1) * gamma) & (rnd < op * gamma);
            mut_op(band) = op;
        end
        mut_op(mut_op == 0) = best_op;

        for op = 1:M
            consumed_fes(op) = consumed_fes(op) + sum(mut_op == op);
        end

        % Scaling factors from the chosen operator's memory
        mem_idx = randi(memory_size, pop_size, 1);
        mu_sf   = memory_sf(sub2ind([memory_size M], mem_idx, mut_op));

        sf  = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        bad = sf <= 0;
        guard = 0;
        while any(bad)
            sf(bad) = mu_sf(bad) + 0.1 * tan(pi * (rand(sum(bad), 1) - 0.5));
            bad = sf <= 0;
            guard = guard + 1;
            if guard > 1000
                sf(bad) = 0.5;
                break;
            end
        end
        sf = min(sf, 1);

        % Crossover rates: zero for op0 and for the whole first half
        cr = zeros(pop_size, 1);
        if FE > 0.5 * maxFE
            second = (mut_op == 2);
            cr(second) = rand(sum(second), 1);
        end

        % Donor indices
        popAll   = [pop; archive_pop(1:arc_ind_count, :)];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), 1:pop_size);

        vi = zeros(pop_size, D);

        is1 = (mut_op == 1);
        if any(is1)
            % op0: target/1 with archive
            vi(is1, :) = pop(is1, :) + sf(is1, ones(1, D)) .* (pop(r1(is1), :) - popAll(r2(is1), :));
        end
        is2 = (mut_op == 2);
        if any(is2)
            % op1: current-to-pbest/1 with archive
            pb = sorted_array(randi(p_num, sum(is2), 1));
            vi(is2, :) = pop(is2, :) ...
                       + sf(is2, ones(1, D)) .* (pop(pb, :) - pop(is2, :)) ...
                       + sf(is2, ones(1, D)) .* (pop(r1(is2), :) - popAll(r2(is2), :));
        end

        % Binomial crossover with a mandatory coordinate
        mask = rand(pop_size, D) < cr(:, ones(1, D));
        mask(sub2ind([pop_size D], (1:pop_size)', randi(D, pop_size, 1))) = true;
        ui = pop;
        ui(mask) = vi(mask);

        ui = boundConstraint(ui, pop, lu);

        % Evaluate
        nEval = min(pop_size, maxFE - FE);
        [children_fitness, FE] = calculate_fitness(ui(1:nEval, :)', problem, FE);
        children_fitness = children_fitness(:);

        for i = 1:nEval
            if children_fitness(i) < bsf
                bsf  = children_fitness(i);
                bsfx = ui(i, :);
            end
            ec = FE - nEval + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Generation alternation
        late  = (FE >= 0.5 * maxFE);
        equal = false(pop_size, 1);   equal(1:nEval) = children_fitness == fitness(1:nEval);
        bettr = false(pop_size, 1);   bettr(1:nEval) = children_fitness <  fitness(1:nEval);
        worse = false(pop_size, 1);   worse(1:nEval) = children_fitness >  fitness(1:nEval);

        % Equal fitness: the reference compares AFTER copying, so the stagnation counter always increments
        if any(equal)
            pop(equal, :)  = ui(equal, :);
            fitness(equal) = children_fitness(equal(1:nEval));
            if late
                stagnation(equal) = stagnation(equal) + 1;
            end
        end

        % Strict improvement: archive the parent, record the success
        if any(bettr)
            [archive_pop, arc_ind_count] = updateArchive(archive_pop, arc_ind_count, ...
                                                         arc_size, pop(bettr, :));
            dif = abs(fitness(bettr) - children_fitness(bettr(1:nEval)));
            ops = mut_op(bettr);
            sfs = sf(bettr);
            for op = 1:M
                sel = (ops == op);
                if any(sel)
                    improve_fitness(op) = improve_fitness(op) + sum(dif(sel));
                    memory_sf(memory_pos(op), op) = lehmer(sfs(sel), dif(sel));
                    memory_pos(op) = memory_pos(op) + 1;
                    if memory_pos(op) > memory_size
                        memory_pos(op) = 1;
                    end
                end
            end

            pop(bettr, :)  = ui(bettr, :);
            fitness(bettr) = children_fitness(bettr(1:nEval));
            stagnation(bettr) = 0;
        end

        if late && any(worse)
            stagnation(worse) = stagnation(worse) + 1;
        end

        % Review the best operator every NG generations
        if mod(generation, NG) == 0
            new_best = -1;
            best_rate = 0;
            for op = 1:M
                if consumed_fes(op) > 0
                    rate = improve_fitness(op) / consumed_fes(op);
                    if rate > best_rate
                        best_rate = rate;
                        new_best  = op;
                    end
                end
                consumed_fes(op)    = 0;
                improve_fitness(op) = 0;
            end
            if new_best == -1
                best_op = 1;
            else
                best_op = new_best;
            end
        end

        % Linear population size reduction
        plan_pop_size = round(((min_pop_size - max_pop_size) / maxFE) * FE + max_pop_size);
        if pop_size > plan_pop_size
            reduction = pop_size - plan_pop_size;
            if pop_size - reduction < min_pop_size
                reduction = pop_size - min_pop_size;
            end
            for r = 1:reduction
                [~, worst] = max(fitness);
                pop(worst, :)    = [];
                fitness(worst)   = [];
                stagnation(worst) = [];
                pop_size = pop_size - 1;
            end

            arc_size = round(pop_size * arc_rate);
            if arc_ind_count > arc_size
                arc_ind_count = arc_size;
            end
            archive_pop = archive_pop(1:max(arc_size, 1), :);

            p_num = max(round(pop_size * p_best_rate), 2);
        end

        % Dynamic perturbation
        stale = find(stagnation > stag_threshold);
        for t = 1:numel(stale)
            if FE >= maxFE
                break;
            end
            i     = stale(t);
            alpha = FE / maxFE;
            hit   = rand(1, D) <= 0.5;
            if any(hit)
                rand_x = lb(hit) + rand(1, sum(hit)) .* (ub(hit) - lb(hit));
                pop(i, hit) = alpha * pop(i, hit) + (1 - alpha) * rand_x;
            end

            [fi, FE] = calculate_fitness(pop(i, :)', problem, FE);
            fitness(i) = fi(1);
            stagnation(i) = 0;

            if fitness(i) < bsf
                bsf  = fitness(i);
                bsfx = pop(i, :);
            end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, pop, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function m = lehmer(sfs, dif)
% Weighted Lehmer mean of the successful scaling factors.
    s = sum(dif);
    if s <= 0
        w = ones(numel(dif), 1) / numel(dif);
    else
        w = dif / s;
    end
    den = w' * sfs;
    if den == 0
        m = 0.5;
    else
        m = (w' * (sfs .^ 2)) / den;
    end
end

function vi = boundConstraint(vi, pop, lu)
% L-SHADE bound handling: midpoint between the parent and the violated bound.
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
% r1 in [1,NP1] with r1 ~= r0; r2 in [1,NP2] with r2 ~= r1 and r2 ~= r0.
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = (r1 == r0);
        if sum(pos) == 0, break; end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:1000
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    r1 = r1(:);
    r2 = r2(:);
end

function [archive_pop, arc_ind_count] = updateArchive(archive_pop, arc_ind_count, arc_size, parents)
% Append while there is room, then overwrite a random slot for each further parent
    if arc_size <= 1
        return;
    end
    for i = 1:size(parents, 1)
        if arc_ind_count < arc_size
            arc_ind_count = arc_ind_count + 1;
            archive_pop(arc_ind_count, :) = parents(i, :);
        else
            archive_pop(randi(arc_size), :) = parents(i, :);
        end
    end
end

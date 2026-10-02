% ----------------------------------------------------------------------- %
% Improved L-SHADE (iL-SHADE)
% CEC 2016 competition -- 3rd place (Friedman ranking, CEC 2014 suite)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 12*D -> 4        % Linear reduction over the budget
%   H = 6                       % Memory slots, F starting at 0.5 and CR at 0.8
%   p = 0.2 -> 0.1              % pbest rate, halved over the budget
%   arc_rate = 2.6              % Archive size, a multiple of the population
%   F <= 0.7 / 0.8 / 0.9        % Caps in the first quarter, half and three quarters
%   CR >= 0.5 / 0.25            % Floors in the first quarter and half
%
% Algorithm Concept:
%   - L-SHADE's frame: current-to-pbest/1 with an external archive, F and CR
%     drawn per individual from a memory of what succeeded, population shrinking
%     linearly to four
%   - The last memory slot is not adapted but fixed at F = CR = 0.9, so an
%     aggressive setting stays available however the memory drifts
%   - A memory slot is updated to the mean of its old value and the new weighted
%     Lehmer mean, which halves how fast the memory can move
%   - Early on the schedule bounds the operators directly: CR is floored and F is
%     capped, so the first half of the run cannot take a large, disruptive step
%   - While less than half the budget is spent the pbest donor may not be the
%     individual being mutated
%
% Reference:
% Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic,
% iL-SHADE: Improved L-SHADE algorithm for single objective real-parameter
% optimization,
% 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 1188-1195.
% https://doi.org/10.1109/CEC.2016.7743922
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released C++ (lshade.cc of the CEC2016 submission).
% One quirk is kept: the pbest rate only decays inside the branch that shrinks
% the population, so in a generation where the population does not shrink the
% rate and p_num stay where they were.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ilshade(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    lu = [lb; ub];

    p_best_rate0 = 0.2;
    arc_rate = 2.6;
    memory_size = 6;
    pop_size = 12 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    pop = repmat(lu(1, :), pop_size, 1) + rand(pop_size, dim) .* repmat(lu(2, :) - lu(1, :), pop_size, 1);
    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:pop_size
        if fitness(i) < best_fitness
            best_fitness = fitness(i);
            best_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness', population_history, fitness_history, history_index, maxFE);
        end
    end

    memory_sf = 0.5 * ones(memory_size, 1);
    memory_cr = 0.8 * ones(memory_size, 1);
    memory_pos = 1;
    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, dim);
    archive.funvalues = zeros(0, 1);
    p_best_rate = p_best_rate0;
    p_num = max(2, round(pop_size * p_best_rate));

    while FE < maxFE
        t = FE / maxFE;
        [~, sorted_index] = sort(fitness);

        r = randi(memory_size, pop_size, 1);
        mu_sf = memory_sf(r);
        mu_cr = memory_cr(r);
        fixed = (r == memory_size);      % the last slot is not adapted
        mu_sf(fixed) = 0.9;
        mu_cr(fixed) = 0.9;

        cr = mu_cr + 0.1 * randn(pop_size, 1);
        cr(mu_cr < 0) = 0;
        cr = min(max(cr, 0), 1);
        if t < 0.25
            cr = max(cr, 0.5);
        end
        if t < 0.5
            cr = max(cr, 0.25);
        end

        sf = zeros(pop_size, 1);
        for i = 1:pop_size
            while sf(i) <= 0
                sf(i) = mu_sf(i) + 0.1 * tan(pi * (rand - 0.5));
            end
        end
        sf = min(sf, 1);
        if t < 0.25
            sf = min(sf, 0.7);
        end
        if t < 0.5
            sf = min(sf, 0.8);
        end
        if t < 0.75
            sf = min(sf, 0.9);
        end

        pbest_idx = zeros(pop_size, 1);
        for i = 1:pop_size
            pbest_idx(i) = sorted_index(randi(p_num));
            while t < 0.5 && pbest_idx(i) == i
                pbest_idx(i) = sorted_index(randi(p_num));
            end
        end
        pbest = pop(pbest_idx, :);

        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), 1:pop_size);

        vi = pop + sf(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
        vi = boundConstraint(vi, pop, lu);

        mask = rand(pop_size, dim) > cr(:, ones(1, dim));
        jrand = sub2ind([pop_size dim], (1:pop_size)', floor(rand(pop_size, 1) * dim) + 1);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        [children_fitness, FE] = calculate_fitness(ui', problem, FE);
        children_fitness = children_fitness(:);

        for i = 1:pop_size
            if children_fitness(i) < best_fitness
                best_fitness = children_fitness(i);
                best_solution = ui(i, :);
            end
            eval_count = FE - pop_size + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        S_cr = []; S_sf = []; S_df = [];
        for i = 1:pop_size
            if children_fitness(i) == fitness(i)
                pop(i, :) = ui(i, :);
                fitness(i) = children_fitness(i);
            elseif children_fitness(i) < fitness(i)
                archive = updateArchive(archive, pop(i, :), fitness(i));
                S_df(end+1, 1) = abs(fitness(i) - children_fitness(i)); %#ok<AGROW>
                S_sf(end+1, 1) = sf(i); %#ok<AGROW>
                S_cr(end+1, 1) = cr(i); %#ok<AGROW>
                pop(i, :) = ui(i, :);
                fitness(i) = children_fitness(i);
            end
        end

        if ~isempty(S_df)
            old_sf = memory_sf(memory_pos);
            old_cr = memory_cr(memory_pos);
            w = S_df / sum(S_df);
            new_sf = sum(w .* S_sf .* S_sf) / sum(w .* S_sf);
            if sum(w .* S_cr) == 0 || memory_cr(memory_pos) == -1
                new_cr = -1;
            else
                new_cr = sum(w .* S_cr .* S_cr) / sum(w .* S_cr);
            end
            % A slot moves only half way towards the new mean
            memory_sf(memory_pos) = (new_sf + old_sf) / 2;
            memory_cr(memory_pos) = (new_cr + old_cr) / 2;
            memory_pos = mod(memory_pos, memory_size) + 1;
        end

        plan_pop_size = max(min_pop_size, ...
            round(((min_pop_size - max_pop_size) / maxFE) * FE + max_pop_size));
        if pop_size > plan_pop_size
            [fitness, order] = sort(fitness);
            pop = pop(order, :);
            pop = pop(1:plan_pop_size, :);
            fitness = fitness(1:plan_pop_size);
            pop_size = plan_pop_size;

            archive.NP = round(arc_rate * pop_size);
            if size(archive.pop, 1) > archive.NP
                keep = randperm(size(archive.pop, 1), archive.NP);
                archive.pop = archive.pop(keep, :);
                archive.funvalues = archive.funvalues(keep);
            end
            % The reference only refreshes the pbest rate here
            p_best_rate = p_best_rate0 * (1 - 0.5 * FE / maxFE);
            p_num = max(2, round(pop_size * p_best_rate));
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

% A coordinate outside the box moves to the midpoint with its parent
function vi = boundConstraint(vi, pop, lu)
    [NP, D] = size(pop);
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
    NP0 = numel(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    r1 = r1(:);
    r2 = r2(:);
end

function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0
        return;
    end
    if size(archive.pop, 1) < archive.NP
        archive.pop(end+1, :) = pop;
        archive.funvalues(end+1, 1) = funvalue;
    else
        slot = randi(archive.NP);
        archive.pop(slot, :) = pop;
        archive.funvalues(slot) = funvalue;
    end
end

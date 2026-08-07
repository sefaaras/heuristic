% ----------------------------------------------------------------------- %
% Weighted current-to-pBest/1 SHADE (jSO)
% CEC 2017 competition runner-up
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size     = round(sqrt(D) * log(D) * 25)   % Initial population size
%   min_pop_size = 4                              % Final size (linear reduction)
%   memory_size  = 5                              % Historical memory H
%   M_F   = 0.3, M_CR = 0.8                       % Initial memory contents
%   arc_rate     = 1                              % Archive size factor
%   p_best_rate  = 0.25                           % Decays to 0.125 over the budget
%
% Algorithm Concept:
%   - iL-SHADE with a weighted mutation strategy, current-to-pBest-w/1:
%       v_i = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)
%     where Fw is a budget-dependent multiple of F (0.7*F, 0.8*F, 1.2*F in the
%     first 20 %, the first 40 %, and the remainder of the evaluation budget)
%   - The H-th memory slot is virtual: drawing it yields mu_F = mu_CR = 0.9
%     regardless of what the update wrote there (iL-SHADE)
%   - Parameter clamps tied to the consumed budget: F <= 0.7 while FE < 0.6*maxFE,
%     CR >= 0.7 while FE < 0.25*maxFE and CR >= 0.6 while FE < 0.5*maxFE
%   - The pbest donor may not be the target while FE < 0.5*maxFE
%   - Memory update is the weighted Lehmer mean averaged with the previous
%     memory content (iL-SHADE half-update), plus linear population reduction
%
% Reference:
% Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic,
% Single Objective Real-Parameter Optimization: Algorithm jSO,
% 2017 IEEE Congress on Evolutionary Computation (CEC), 2017, pp. 1311-1318.
% https://doi.org/10.1109/CEC.2017.7969456
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' reference C++ release (jSO/lshade.cc), including two
% quirks of the published code that are deliberately NOT "fixed": the terminal
% CR marker -1 is written and then half-averaged with the previous memory
% content, so the stored value is (old_CR - 1)/2 and a negative mu_CR later
% yields CR = 0 -- the behaviour jSO actually competed with; and the CR lower
% bounds are applied after that CR = 0 assignment, so an early-budget terminal
% slot still produces CR = 0.7.
% pop_size is floored at min_pop_size so the D = 1 edge case (log(1) = 0)
% cannot produce an empty population.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jso(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % jSO control parameters (main.cc of the reference release)
    memory_size    = 5;
    arc_rate       = 1;
    g_p_best_rate  = 0.25;
    min_pop_size   = 4;
    pop_size       = max(round(sqrt(dim) * log(dim) * 25), min_pop_size);
    max_pop_size   = pop_size;

    p_best_rate = g_p_best_rate;
    p_num       = max(round(pop_size * p_best_rate), 2);

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial population
    pop = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    bsf          = inf;
    bsf_solution = zeros(1, dim);
    for i = 1:pop_size
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Historical memories and external archive
    memory_sf  = 0.3 .* ones(memory_size, 1);
    memory_cr  = 0.8 .* ones(memory_size, 1);
    memory_pos = 1;

    arc_size      = round(pop_size * arc_rate);
    archive_pop   = zeros(arc_size, dim);
    arc_ind_count = 0;

    % Main loop
    while FE < maxFE
        [~, sorted_index] = sort(fitness, 'ascend');

        % Sample mu_F / mu_CR; drawing the H-th slot yields the fixed 0.9 pair
        mem_idx = randi(memory_size, pop_size, 1);
        mu_sf   = memory_sf(mem_idx);
        mu_cr   = memory_cr(mem_idx);
        last    = (mem_idx == memory_size);
        mu_sf(last) = 0.9;
        mu_cr(last) = 0.9;

        % Crossover rates
        cr = mu_cr + 0.1 * randn(pop_size, 1);
        cr = min(max(cr, 0), 1);
        cr(mu_cr < 0) = 0;
        if FE < 0.25 * maxFE
            cr = max(cr, 0.7);
        end
        if FE < 0.50 * maxFE
            cr = max(cr, 0.6);
        end

        % Scaling factors (Cauchy, resampled until strictly positive)
        sf  = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos     = find(sf <= 0);
        end
        sf = min(sf, 1);
        if FE < 0.6 * maxFE
            sf(sf > 0.7) = 0.7;
        end

        % pbest donors; may not equal the target during the first half
        randindex = max(1, ceil(rand(pop_size, 1) .* p_num));
        pbest_ind = sorted_index(randindex);
        if FE < 0.50 * maxFE
            target  = (1:pop_size)';
            clash   = find(pbest_ind == target);
            guard   = 0;
            while ~isempty(clash)
                randindex(clash) = max(1, ceil(rand(length(clash), 1) .* p_num));
                pbest_ind(clash) = sorted_index(randindex(clash));
                clash = clash(pbest_ind(clash) == clash);
                guard = guard + 1;
                if guard > 1000
                    error('jso:pbestResample', 'Cannot draw a pbest donor in 1000 iterations');
                end
            end
        end

        % Donor indices: r1 from the population, r2 from population + archive
        popAll   = [pop; archive_pop(1:arc_ind_count, :)];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), 1:pop_size);

        % Weighted mutation: Fw is a budget-dependent multiple of F
        if FE < 0.2 * maxFE
            jF = sf * 0.7;
        elseif FE < 0.4 * maxFE
            jF = sf * 0.8;
        else
            jF = sf * 1.2;
        end

        vi = pop + jF(:, ones(1, dim)) .* (pop(pbest_ind, :) - pop) ...
                 + sf(:, ones(1, dim)) .* (pop(r1, :) - popAll(r2, :));
        vi = boundConstraint(vi, pop, lu);

        % Binomial crossover
        mask       = rand(pop_size, dim) > cr(:, ones(1, dim));
        rows       = (1:pop_size)';
        cols       = floor(rand(pop_size, 1) * dim) + 1;
        mask(sub2ind([pop_size dim], rows, cols)) = false;
        ui         = vi;
        ui(mask)   = pop(mask);

        [children_fitness, FE] = calculate_fitness(ui', problem, FE);
        children_fitness = children_fitness(:);

        % Best-so-far tracking, curve and history
        for i = 1:pop_size
            if children_fitness(i) < bsf
                bsf          = children_fitness(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - pop_size + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection: ties replace the parent but do not count as a success
        equal  = (children_fitness == fitness);
        better = (children_fitness < fitness);

        goodF   = sf(better);
        goodCR  = cr(better);
        dif_val = abs(fitness(better) - children_fitness(better));

        [archive_pop, arc_ind_count] = updateArchive(archive_pop, arc_ind_count, ...
                                                     arc_size, pop(better, :));

        replaced             = equal | better;
        pop(replaced, :)     = ui(replaced, :);
        fitness(replaced)    = children_fitness(replaced);

        % Memory update: weighted Lehmer mean averaged with the previous content
        if ~isempty(goodF)
            old_sf = memory_sf(memory_pos);
            old_cr = memory_cr(memory_pos);

            weight      = dif_val / sum(dif_val);
            temp_sum_sf = weight' * goodF;
            temp_sum_cr = weight' * goodCR;

            memory_sf(memory_pos) = (weight' * (goodF .^ 2)) / temp_sum_sf;
            if temp_sum_cr == 0
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (weight' * (goodCR .^ 2)) / temp_sum_cr;
            end

            memory_sf(memory_pos) = (memory_sf(memory_pos) + old_sf) / 2;
            memory_cr(memory_pos) = (memory_cr(memory_pos) + old_cr) / 2;

            memory_pos = memory_pos + 1;
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

            for r = 1:reduction_ind_num
                [~, worst_ind]      = max(fitness);
                pop(worst_ind, :)   = [];
                fitness(worst_ind)  = [];
                pop_size            = pop_size - 1;
            end

            arc_size = round(pop_size * arc_rate);
            if arc_ind_count > arc_size
                arc_ind_count = arc_size;
            end
            archive_pop = archive_pop(1:max(arc_size, 1), :);

            p_best_rate = g_p_best_rate * (1.0 - 0.5 * FE / maxFE);
            p_num       = max(round(pop_size * p_best_rate), 2);
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function vi = boundConstraint(vi, pop, lu)
% L-SHADE bound handling: violating component moved to the parent/bound midpoint
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
        if sum(pos) == 0
            break;
        end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        if i == 1000
            error('gnR1R2:r1', 'Cannot generate r1 in 1000 iterations');
        end
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:1000
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        if i == 1000
            error('gnR1R2:r2', 'Cannot generate r2 in 1000 iterations');
        end
    end
end

function [archive_pop, arc_ind_count] = updateArchive(archive_pop, arc_ind_count, arc_size, parents)
% Reference archive semantics: append while there is room, then overwrite a random slot
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

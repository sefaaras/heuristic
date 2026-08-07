% ----------------------------------------------------------------------- %
% Multiple Adaptation based Differential Evolution (MadDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size    = min(2*D^2, 40*D)  % Initial population, capped above D = 20
%   memory_size = 10 * D        % Success-history memory size
%   p_best_rate = 0.18, arc_rate = 2.3, q_cr_rate = 0.01
%
% Algorithm Concept:
%   - Success-history parameter adaptation with linear population reduction
%   - Three mutation operators (current-to-pbest/1 with archive,
%     current-to-rand/1 with archive, weighted-rand-to-qbest/1) with
%     probability adaptation, plus a q-best binomial crossover
%
% Reference:
% S. Biswas, D. Saha, S. De, A. D. Cobb, S. Das, B. A. Jalaian,
% Improving Differential Evolution through Bayesian Hyperparameter
% Optimization,
% 2021 IEEE Congress on Evolutionary Computation (CEC), Krakow, Poland,
% 2021, pp. 832-840.
% https://doi.org/10.1109/CEC45853.2021.9504792
% ----------------------------------------------------------------------- %
% Implementation Note:
% pop_size is capped at 40*D. The reference's 2*D^2 was tuned on CEC2021, which
% runs D = 10 and 20 only, and the two laws meet exactly at D = 20, so every
% dimension the authors tested is untouched and only the extrapolation above it
% changes. Uncapped, 2*D^2 reaches 49928 at D = 158 (CEC2020RW) and leaves 20
% generations of the 1e6 budget, so the linear population reduction the
% algorithm is built around never runs. 40*D is the top of the linear band its
% own family uses (L-SHADE 18*D, NL-SHADE-RSP 30*D, AGSK 40*D).
%     D       2*D^2     40*D
%     20        800      800
%     100     20000     4000
%     158     49928     6320
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = madde(problem)

    % Extract problem parameters
    problem_size = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    max_nfes = problem.maxFe;
    lu = [lb; ub];

    % Hyper-parameters
    q_cr_rate = 0.01;
    p_best_rate = 0.18;
    arc_rate = 2.3;
    memory_size = 10 * problem_size;
    pop_size = min(2 * problem_size ^ 2, 40 * problem_size);

    max_pop_size = pop_size;
    min_pop_size = 4.0;

    FE = 0;
    curve = zeros(1, max_nfes);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the main population (Sobol sequence)
    p = sobolset(problem_size, 'Skip', 1e4, 'Leap', 1e3);
    p = scramble(p, 'MatousekAffineOwen');
    rand0 = net(p, pop_size);
    popold = repmat(lu(1, :), pop_size, 1) + rand0 .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    pop = popold;

    bsf_fit_var = 1e+30;
    bsf_solution = zeros(1, problem_size);

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    for i = 1:pop_size
        if fitness(i) < bsf_fit_var
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= max_nfes
            curve(i) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_top_k(...
                i, pop, fitness', ...
                population_history, fitness_history, history_index, max_nfes);
        end
    end

    % Probabilities of DE operators
    num_de = 3;
    count_S = zeros(1, num_de);
    probDE = 1 ./ num_de .* ones(1, num_de);

    memory_sf = 0.2 .* ones(memory_size, 1);
    memory_cr = 0.2 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, problem_size);
    archive.funvalues = zeros(0, 1);

    % Main loop
    while FE < max_nfes
        pop = popold;
        [fitness, sorted_index] = sort(fitness, 'ascend');
        pop = pop(sorted_index, :);

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);

        % Crossover rate
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        % Scaling factor
        sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end
        sf = min(sf, 1);

        r0 = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2, r3] = gnR1R2(pop_size, size(popAll, 1), r0);
        vi = zeros(pop_size, problem_size);

        % Operator selection
        bb = rand(pop_size, 1);
        probiter = probDE(1, :);
        l2 = sum(probDE(1:2));

        de_1 = bb <= probiter(1) * ones(pop_size, 1);
        de_2 = bb > probiter(1) * ones(pop_size, 1) & bb <= (l2 * ones(pop_size, 1));
        de_3 = bb > l2 * ones(pop_size, 1) & bb <= (ones(pop_size, 1));

        pNP = max(round(p_best_rate * pop_size), 2);
        randindex = floor(rand(1, pop_size) .* pNP) + 1;
        pbest = pop(randindex, :);

        % DE/current-to-p-best/1 with archive
        vi(de_1 == 1, :) = pop(de_1 == 1, :) + ...
            sf(de_1 == 1, ones(1, problem_size)) .* ...
            (pbest(de_1 == 1, :) - pop(de_1 == 1, :) + pop(r1(de_1 == 1), :) - popAll(r2(de_1 == 1), :));

        % DE/current-to-rand/1 with archive
        vi(de_2 == 1, :) = pop(de_2 == 1, :) + ...
            sf(de_2 == 1, ones(1, problem_size)) .* ...
            (pop(r1(de_2 == 1), :) - popAll(r2(de_2 == 1), :));

        % DE/weighted-rand-to-q-best/1 with attraction
        q_best_rate = 2 * p_best_rate - p_best_rate * (FE / max_nfes);
        qNP = max(round(q_best_rate * pop_size), 2);
        randindex = floor(rand(1, pop_size) .* qNP) + 1;
        qbest = pop(randindex, :);

        attraction = repmat(0.5 + 0.5 * (FE / max_nfes), pop_size, problem_size);

        vi(de_3 == 1, :) = sf(de_3 == 1, ones(1, problem_size)) .* ...
            (pop(r1(de_3 == 1), :) + ...
            attraction(de_3 == 1, :) .* (qbest(de_3 == 1, :) - pop(r3(de_3 == 1), :)));

        vi = boundConstraint(vi, pop, lu);

        % q-best binomial crossover
        mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size));
        rows = (1:pop_size)'; cols = floor(rand(pop_size, 1) * problem_size) + 1;
        jrand = sub2ind([pop_size problem_size], rows, cols);
        mask(jrand) = false;

        qNP = max(round(q_best_rate * size(popAll, 1)), 2);
        randindex = floor(rand(1, size(popAll, 1)) .* qNP) + 1;
        popAllbest = popAll(randindex, :);
        popAllbest = popAllbest(1:pop_size, :);

        bb = rand(pop_size, 1) <= repmat(q_cr_rate, pop_size, 1);
        qbest = pop; qbest(bb, :) = popAllbest(bb, :);

        ui = vi; ui(mask) = qbest(mask);

        % Evaluate offspring
        cur_np = size(ui, 1);
        [children_fitness, FE] = calculate_fitness(ui', problem, FE);
        children_fitness = children_fitness(:);

        for i = 1:cur_np
            if children_fitness(i) < bsf_fit_var
                bsf_fit_var = children_fitness(i);
                bsf_solution = ui(i, :);
            end
        end

        for eval_idx = 1:cur_np
            eval_count = FE - cur_np + eval_idx;
            if eval_count >= 1 && eval_count <= max_nfes
                curve(eval_count) = bsf_fit_var;
                [population_history, fitness_history, history_index] = record_top_k(...
                    eval_count, pop, fitness', ...
                    population_history, fitness_history, history_index, max_nfes);
            end
        end

        dif = abs(fitness - children_fitness);

        % Selection
        I = (fitness > children_fitness);
        goodCR = cr(I == 1);
        goodF = sf(I == 1);
        dif_val = dif(I == 1);

        archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));

        % Update operator probabilities
        diff2 = max(0, (fitness - children_fitness)) ./ abs(fitness);
        count_S(1) = max(0, mean(diff2(de_1 == 1)));
        count_S(2) = max(0, mean(diff2(de_2 == 1)));
        count_S(3) = max(0, mean(diff2(de_3 == 1)));

        if count_S ~= 0
            probDE = max(0.1, min(0.9, count_S ./ (sum(count_S))));
        else
            probDE = 1.0 / 3 * ones(1, 3);
        end

        % Update population and fitness
        [fitness, I] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(I == 2, :) = ui(I == 2, :);

        num_success_params = numel(goodCR);
        if num_success_params > 0
            sum_dif = sum(dif_val);
            dif_val = dif_val / sum_dif;

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
        else
            memory_cr(memory_pos) = 0.5;
            memory_sf(memory_pos) = 0.5;
        end

        % Linear population size reduction
        plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * FE) + max_pop_size);

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
                archive.funvalues = archive.funvalues(rndpos, :);
            end
        end
    end

    curve(min(FE, max_nfes):end) = bsf_fit_var;

    best_solution = bsf_solution;
    best_fitness = bsf_fit_var;
end

% Helper functions
function [pop_hist, fit_hist, hist_idx] = record_top_k(current_fe, population, fitness, pop_hist, fit_hist, hist_idx, maxFE)
% Kept for existing call sites; record_history stores population metrics, not raw positions
    [pop_hist, fit_hist, hist_idx] = record_history(current_fe, population, fitness, ...
        pop_hist, fit_hist, hist_idx, maxFE);
end

function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
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
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0
            break;
        else
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000
            error('Cannot generate r3 in 1000 iterations');
        end
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

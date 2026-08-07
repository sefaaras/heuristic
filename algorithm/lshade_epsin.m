% ----------------------------------------------------------------------- %
% L-SHADE with Ensemble Sinusoidal Parameter Adaptation (L-SHADE-EpSin)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18 * D            % Initial population size
%   min_pop_size = 4             % Minimum population size
%   p_best_rate = 0.11           % Percentage of top solutions for pbest
%   arc_rate = 1.4               % Archive rate
%   memory_size = 5              % Historical memory size for CR and F
%   freq_inti = 0.5              % Initial frequency for sinusoidal adaptation
%   popsize_LS = 10              % Local search population size
%   GenMaxSelected = 250         % Max generations for local search
%
% Algorithm Concept:
%   - L-SHADE with ensemble sinusoidal scaling factor adaptation
%   - Two sinusoidal forms compete based on success history
%   - Gaussian-based local search activated when population is small
%   - Linear population size reduction
%
% Reference:
% Noor Awad, Mostafa Ali, Ponnuthurai Suganthan, Robert G. Reynolds,
% An ensemble sinusoidal parameter adaptation incorporated with
% L-SHADE for solving CEC2014 benchmark problems,
% IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 2958-2965
% https://doi.org/10.1109/CEC.2016.7744163
% ----------------------------------------------------------------------- %
% Implementation Note:
% G_Max, the horizon of the sinusoidal F schedule, is counted from the LPSR
% population plan instead of the reference's per-dimension lookup. Counting
% reproduces all four of its values exactly at the maxFE = 10000*D they were
% written for, and stays right elsewhere: the table overruns by 10x on
% cec2020_10 and 37x on cec2020_20, where gg/G_Max drives the second sinusoidal
% form to F in [-2.3, 3.3] instead of [0, 1].
% The Gaussian local search fires once, late under LPSR. Each popLS point is
% evaluated exactly once and its value carried in fitness_LS, so recording that
% subpopulation costs no evaluations of its own; the popLS initialisation is
% folded into the curve at the FE it actually consumed.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lshade_epsin(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    lu = [lb; ub];

    % Algorithm parameters
    freq_inti = 0.5;
    GenMaxSelected = 250;

    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    pop_size = 18 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;

    G_Max = lpsr_generations(maxFE, max_pop_size, min_pop_size);

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
            [population_history, fitness_history, history_index] = record_top_k(...
                i, pop, fitness', ...
                population_history, fitness_history, history_index, maxFE);
        end
    end

    % Initialize LS population
    counter = 0;
    popsize_LS = 10;
    popLS = repmat(lu(1, :), popsize_LS, 1) + rand(popsize_LS, dim) .* repmat(lu(2, :) - lu(1, :), popsize_LS, 1);
    FE_before = FE;
    [fitness_LS, FE] = calculate_fitness(popLS', problem, FE);
    fitness_LS = fitness_LS(:);

    % These land after the main population, so the curve carries on from there
    for i = 1:popsize_LS
        if fitness_LS(i) < bsf_fit_var
            bsf_fit_var = fitness_LS(i);
            bsf_solution = popLS(i, :);
        end
        if FE_before + i <= maxFE
            curve(FE_before + i) = bsf_fit_var;
        end
    end

    [fitness_LS, Indecis] = sort(fitness_LS);
    popLS = popLS(Indecis, :);
    BestPoint = popLS(1, :);

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_freq = freq_inti * ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    % Main loop
    gg = 0;

    while FE < maxFE
        gg = gg + 1;

        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);
        mu_freq = memory_freq(mem_rand_index);

        % Generate crossover rate
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        % Generate scaling factor
        sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end

        freq = mu_freq + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos_f = find(freq <= 0);
        while ~isempty(pos_f)
            freq(pos_f) = mu_freq(pos_f) + 0.1 * tan(pi * (rand(length(pos_f), 1) - 0.5));
            pos_f = find(freq <= 0);
        end

        sf = min(sf, 1);
        freq = min(freq, 1);

        if FE <= maxFE / 2
            c = rand;
            if c < 0.5
                sf = 0.5 .* (sin(2 .* pi .* freq_inti .* gg + pi) .* ((G_Max - gg) / G_Max) + 1) .* ones(pop_size, dim);
            else
                sf = 0.5 * (sin(2 * pi .* freq(:, ones(1, dim)) .* gg) .* (gg / G_Max) + 1) .* ones(pop_size, dim);
            end
        end

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

        % Update best
        for i = 1:pop_size
            if children_fitness(i) < bsf_fit_var
                bsf_fit_var = children_fitness(i);
                bsf_solution = ui(i, :);
            end
        end

        % Record curve and history
        for eval_idx = 1:pop_size
            eval_count = FE - pop_size + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = bsf_fit_var;
                [population_history, fitness_history, history_index] = record_top_k(...
                    eval_count, pop, fitness', ...
                    population_history, fitness_history, history_index, maxFE);
            end
        end

        dif = abs(fitness - children_fitness);

        I = (fitness > children_fitness);
        goodCR = cr(I == 1);
        goodF = sf(I == 1);
        goodFreq = freq(I == 1);
        dif_val = dif(I == 1);

        archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));

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

            if max(goodFreq) == 0 || memory_freq(memory_pos) == -1
                memory_freq(memory_pos) = -1;
            else
                memory_freq(memory_pos) = (dif_val' * (goodFreq .^ 2)) / (dif_val' * goodFreq);
            end

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

        % Gaussian Local Search (activated when NP <= 20 for the first time)
        if pop_size <= 20
            counter = counter + 1;
        end

        flag_LS = (counter == 1);

        if flag_LS && FE < maxFE
            r_index = randi([1 pop_size], 1, popsize_LS);
            for gen_LS = 0:GenMaxSelected
                if FE >= maxFE, break; end

                for i = 1:popsize_LS
                    if FE >= maxFE, break; end

                    GeneratePoint = normrnd(BestPoint, (log(gg) / gg) * (abs((popLS(i, :) - BestPoint))), [1 dim]) + ...
                        (randn * BestPoint - randn * popLS(i, :));
                    GeneratePoint = Bound_Checking(GeneratePoint, lb, ub);

                    [fit_val, FE] = calculate_fitness(GeneratePoint', problem, FE);

                    if fit_val < fitness(r_index(i))
                        fitness(r_index(i)) = fit_val;
                        pop(r_index(i), :) = GeneratePoint;
                    end

                    if fit_val < bsf_fit_var
                        bsf_fit_var = fit_val;
                        bsf_solution = GeneratePoint;
                    end

                    popLS(i, :) = GeneratePoint;
                    fitness_LS(i) = fit_val;

                    if FE <= maxFE
                        curve(FE) = bsf_fit_var;
                    end
                end

                [population_history, fitness_history, history_index] = record_top_k(...
                    FE, popLS, fitness_LS, ...
                    population_history, fitness_history, history_index, maxFE);

                [fitness_LS, SortedIndex] = sort(fitness_LS);
                popLS = popLS(SortedIndex, :);
                BestPoint = popLS(1, :);
            end
        end

    end

    % Fill remaining curve values
    curve(FE:end) = bsf_fit_var;

    best_fitness = bsf_fit_var;
    best_solution = bsf_solution;
end

% Helper Functions

function [pop_hist, fit_hist, hist_idx] = record_top_k(...
    current_fe, population, fitness, ...
    pop_hist, fit_hist, hist_idx, maxFE)
% Kept for existing call sites; record_history stores population metrics, not raw positions
    [pop_hist, fit_hist, hist_idx] = record_history(current_fe, population, fitness, ...
        pop_hist, fit_hist, hist_idx, maxFE);
end

function g = lpsr_generations(maxFE, max_pop_size, min_pop_size)
% Generations the linear population size reduction plan allows on this budget
    g = 0;
    fe = 0;
    while fe < maxFE
        fe = fe + max(min_pop_size, ...
                      round(((min_pop_size - max_pop_size) / maxFE) * fe + max_pop_size));
        g = g + 1;
    end
    g = max(1, g - 1);
end

function p = Bound_Checking(p, lowB, upB)
    for i = 1:size(p, 1)
        upper = double(gt(p(i, :), upB));
        lower = double(lt(p(i, :), lowB));
        up = find(upper == 1);
        lo = find(lower == 1);
        if (size(up, 2) + size(lo, 2) > 0)
            for j = 1:size(up, 2)
                p(i, up(j)) = (upB(up(j)) - lowB(up(j))) * rand() + lowB(up(j));
            end
            for j = 1:size(lo, 2)
                p(i, lo(j)) = (upB(lo(j)) - lowB(lo(j))) * rand() + lowB(lo(j));
            end
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

function [r1, r2] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0, break;
        else, r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000, error('Cannot generate r1 in 1000 iterations'); end
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break;
        else, r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000, error('Cannot generate r2 in 1000 iterations'); end
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

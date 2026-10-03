% ----------------------------------------------------------------------- %
% Dynamic Fitness-Distance Balance based L-SHADE (dFDB-LSHADE)
% Variant of lshade: first-quarter dFDB pick replaces r1 and, half the time, resizes the pbest pool
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18*D -> 4 (linear)  % LPSR as in lshade
%   p_best_rate = 0.11             % Greediness of the pbest term
%   arc_rate = 1.4                 % Archive size relative to the population
%   memory_size = 5                % Historical memory H for F and CR
%   frequency = 10                 % dFDB weight cycles over the budget
%   w = 0.6 -> 0 (sawtooth)        % Distance weight in the dFDB score
%   dfdb_window = 0.25 * maxFE     % dFDB is used only while FE is below this
%   fdb_threshold = 0.5            % Per-generation draw above it switches the pbest pool
%
% Algorithm Concept:
%   - SHADE adaptation of F and CR from a success memory, Lehmer-mean update
%   - Linear population size reduction 18*D -> 4 over the budget
%   - current-to-pbest/1 with archive; while FE < 0.25*maxFE the r1 donor is
%     the single dFDB-selected individual (Eqs. 17-18 of the paper)
%   - In the same window, a generation-wide draw > 0.5 draws pbest from the
%     top fdb_index ranks instead of the top p*NP (Case-1)
%   - dFDB pick: argmax of (1-w)*normFitness + w*normDistance to the best
%
% Reference:
% Ibrahim Yildirim, Mustafa Hakan Bozkurt, Hamdi Tolga Kahraman, Sefa Aras,
% Dental X-Ray image enhancement using a novel evolutionary optimization
% algorithm, Engineering Applications of Artificial Intelligence 142 (2025)
% 109879. https://doi.org/10.1016/j.engappai.2024.109879
% Components:
%   L-SHADE - Ryoji Tanabe, Alex S. Fukunaga, Improving the search performance
%     of SHADE using linear population size reduction, IEEE CEC 2014,
%     1658-1665, https://doi.org/10.1109/CEC.2014.6900380
%   dFDB - Hamdi Tolga Kahraman, Huseyin Bakir, Serhat Duman, Mehmet Kati,
%     Sefa Aras, Ugur Guvenc, Dynamic FDB selection method and its application:
%     modeling and optimizing of directional overcurrent relays coordination,
%     Applied Intelligence 52(5) (2022) 4873-4908,
%     https://doi.org/10.1007/s10489-021-02629-3
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' package (FX 177469, dfdb_lshade-1.0.1,
% dfdb_lshade_case_1.m + dFDB.m) as a delta on this repository's lshade.m.
% Case: Sec. 5.5 (p. 12) "Case-1 will be referred to as dFDB-LSHADE"; Table 3
% (p. 8) defines Case-1 as Eq. (17) 50 % / Eq. (18) 50 % in the first quarter.
% As released, Eq. (17)'s pbest is not the dFDB individual: the pbest pool size
% becomes fdb_index, a population index used as a rank count. Kept, as is the
% release's w on the distance term (the paper's Eq. (10) puts it on fitness).
% pop_size 18*D follows lshade.m and the package's image-enhancement copy (its
% benchmark stub sets 50); the dFDB period round(maxFE/10) is floored at 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dfdb_lshade(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    lu = [lb; ub];

    % Algorithm parameters
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    pop_size = 18 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;
    frequency = 10;
    dfdb_window = 0.25;
    fdb_threshold = 0.5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the main population
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, dim) .* repmat(lu(2, :) - lu(1, :), pop_size, 1);
    pop = popold;

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);  % Ensure column vector

    bsf_fit_var = inf;
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

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    % Main loop
    while FE < maxFE
        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);

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
        sf = min(sf, 1);

        r0 = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);

        % One dFDB pick and one switch draw per generation, where the release takes them
        fdb_index = dynamicFitnessDistanceBalance(pop, fitness, frequency, FE, maxFE);
        fdb_percent = rand();
        in_window = FE < dfdb_window * maxFE;

        pNP = max(round(p_best_rate * pop_size), 2);
        if fdb_percent > fdb_threshold && in_window
            randindex = ceil(rand(1, pop_size) .* fdb_index);  % Eq. (17) as released
        else
            randindex = ceil(rand(1, pop_size) .* pNP);
        end
        randindex = max(1, randindex);
        pbest = pop(sorted_index(randindex), :);

        if in_window
            vi = pop + sf(:, ones(1, dim)) .* (pbest - pop + pop(fdb_index, :) - popAll(r2, :));
        else
            vi = pop + sf(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
        end
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
        children_fitness = children_fitness(:);  % Ensure column vector

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

% Released dFDB.m: distance weight w falls 0.6 -> 0 and resets every round(Gen/frequency) FEs
function index = dynamicFitnessDistanceBalance(population, fitness, frequency, iter, Gen)
    fx = max(1, round(Gen / frequency));
    y = mod(iter, fx);
    w = (y/fx * -0.6) + 0.6;
    fitness = fitness(:)';
    populationSize = numel(fitness);
    if min(fitness) == max(fitness)
        index = randi(populationSize);
        return;
    end
    [~, bestIndex] = min(fitness);
    best = population(bestIndex, :);
    distances = sum(abs(best - population), 2)';
    minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
    minDistance = min(distances); maxMinDistance = max(distances) - minDistance;
    normFitness = 1 - ((fitness - minFitness) / maxMinFitness);
    normDistances = (distances - minDistance) / maxMinDistance;
    [~, index] = max((1-w)*normFitness + w*normDistances);
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

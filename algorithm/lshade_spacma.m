% ----------------------------------------------------------------------- %
% L-SHADE-SPACMA (L-SHADE with Semi-Parameter Adaptation and CMA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18 * D            % Initial population size
%   min_pop_size = 4             % Minimum population size
%   p_best_rate = 0.11           % Percentage of top solutions for pbest
%   arc_rate = 1.4               % Archive rate
%   memory_size = 5              % Historical memory size for CR and F
%   L_Rate = 0.80                % Learning rate for hybridization
%   First_class_percentage = 0.5 % Initial probability for DE class
%
% Algorithm Concept:
%   - L-SHADE hybridized with CMA-ES
%   - Adaptive class selection between DE and CMA-ES offspring
%   - Semi-parameter adaptation for scaling factor
%   - Linear population size reduction
%
% Reference:
% Noor Awad, Mostafa Ali, Ponnuthurai Suganthan,
% Semi-parameter adaptation for differential evolution,
% ACM Genetic and Evolutionary Computation Conference (GECCO), 2017
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lshade_spacma(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    lu = [lb; ub];

    % Algorithm parameters
    L_Rate = 0.80;
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    pop_size = 18 * dim;
    max_pop_size = pop_size;
    min_pop_size = 4;
    First_class_percentage = 0.5;

    FE = 0;
    curve = zeros(1, maxFE);

    % History recording (store only top 100)
    history_pop_size = 100;
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, history_pop_size, dim);
    fitness_history = zeros(history_size, history_pop_size);
    history_index = 1;

    %% Initialize the main population
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, dim) .* repmat(lu(2, :) - lu(1, :), pop_size, 1);
    pop = popold;

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    bsf_fit_var = 1e+30;
    bsf_solution = zeros(1, dim);

    for i = 1:pop_size
        if fitness(i) < bsf_fit_var && isreal(pop(i, :)) && sum(isnan(pop(i, :))) == 0 && min(pop(i, :)) >= -100 && max(pop(i, :)) <= 100
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_top_k(...
                i, pop, fitness', history_pop_size, dim, ...
                population_history, fitness_history, history_index, sampling_interval, history_size);
        end
    end

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = round(arc_rate * pop_size);
    archive.pop = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    memory_1st_class_percentage = First_class_percentage .* ones(memory_size, 1);

    %% Initialize CMA-ES parameters
    sigma = 0.5;
    xmean = rand(dim, 1);
    mu_cma = pop_size / 2;
    weights_cma = log(mu_cma + 1/2) - log(1:mu_cma)';
    mu_cma = floor(mu_cma);
    weights_cma = weights_cma / sum(weights_cma);
    mueff = sum(weights_cma)^2 / sum(weights_cma .^ 2);

    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim);
    cs = (mueff + 2) / (dim + mueff + 5);
    c1 = 2 / ((dim + 1.3)^2 + mueff);
    cmu_cma = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)^2 + mueff));
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs;

    pc = zeros(dim, 1);
    ps_cma = zeros(dim, 1);
    B = eye(dim, dim);
    D = ones(dim, 1);
    C = B * diag(D .^ 2) * B';
    invsqrtC = B * diag(D .^ -1) * B';
    eigeneval = 0;
    chiN = dim^0.5 * (1 - 1/(4*dim) + 1/(21*dim^2));

    Hybridization_flag = 1;

    %% Main loop
    while FE < maxFE
        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);
        mem_rand_ratio = rand(pop_size, 1);

        % Generate crossover rate
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        % Generate scaling factor (semi-parameter adaptation)
        if FE <= maxFE / 2
            sf = 0.45 + 0.1 * rand(pop_size, 1);
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = 0.45 + 0.1 * rand(length(pos), 1);
                pos = find(sf <= 0);
            end
        else
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
        end
        sf = min(sf, 1);

        % Hybridization class selection
        Class_Select_Index = (memory_1st_class_percentage(mem_rand_index) >= mem_rand_ratio);
        if Hybridization_flag == 0
            Class_Select_Index = or(Class_Select_Index, ~Class_Select_Index);
        end

        r0 = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);

        pNP = max(round(p_best_rate * pop_size), 2);
        randindex = ceil(rand(1, pop_size) .* pNP);
        randindex = max(1, randindex);
        pbest = pop(sorted_index(randindex), :);

        vi = [];
        temp = [];
        if sum(Class_Select_Index) ~= 0
            vi(Class_Select_Index, :) = pop(Class_Select_Index, :) + sf(Class_Select_Index, ones(1, dim)) .* ...
                (pbest(Class_Select_Index, :) - pop(Class_Select_Index, :) + pop(r1(Class_Select_Index), :) - popAll(r2(Class_Select_Index), :));
        end

        if sum(~Class_Select_Index) ~= 0
            for k = 1:sum(~Class_Select_Index)
                temp(:, k) = xmean + sigma * B * (D .* randn(dim, 1));
            end
            vi(~Class_Select_Index, :) = temp';
        end

        if ~isreal(vi)
            Hybridization_flag = 0;
            continue;
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
        children_fitness = children_fitness(:);

        % Update best
        for i = 1:pop_size
            if children_fitness(i) < bsf_fit_var && isreal(ui(i, :)) && sum(isnan(ui(i, :))) == 0 && min(ui(i, :)) >= -100 && max(ui(i, :)) <= 100
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
                    eval_count, pop, fitness', history_pop_size, dim, ...
                    population_history, fitness_history, history_index, sampling_interval, history_size);
            end
        end

        dif = abs(fitness - children_fitness);

        Child_is_better_index = (fitness > children_fitness);
        goodCR = cr(Child_is_better_index == 1);
        goodF = sf(Child_is_better_index == 1);
        dif_val = dif(Child_is_better_index == 1);
        dif_val_Class_1 = dif(and(Child_is_better_index, Class_Select_Index) == 1);
        dif_val_Class_2 = dif(and(Child_is_better_index, ~Class_Select_Index) == 1);

        archive = updateArchive(archive, popold(Child_is_better_index == 1, :), fitness(Child_is_better_index == 1));

        [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);

        popold = pop;
        popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);

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

            if Hybridization_flag == 1
                memory_1st_class_percentage(memory_pos) = memory_1st_class_percentage(memory_pos) * L_Rate + ...
                    (1 - L_Rate) * (sum(dif_val_Class_1) / (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
                memory_1st_class_percentage(memory_pos) = min(memory_1st_class_percentage(memory_pos), 0.8);
                memory_1st_class_percentage(memory_pos) = max(memory_1st_class_percentage(memory_pos), 0.2);
            end

            memory_pos = memory_pos + 1;
            if memory_pos > memory_size
                memory_pos = 1;
            end
        end

        %% Linear population size reduction
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
                Child_is_better_index(worst_ind, :) = [];
            end

            archive.NP = round(arc_rate * pop_size);
            if size(archive.pop, 1) > archive.NP
                rndpos = randperm(size(archive.pop, 1));
                rndpos = rndpos(1:archive.NP);
                archive.pop = archive.pop(rndpos, :);
            end

            % Update CMA parameters
            mu_cma = pop_size / 2;
            weights_cma = log(mu_cma + 1/2) - log(1:mu_cma)';
            mu_cma = floor(mu_cma);
            weights_cma = weights_cma / sum(weights_cma);
            mueff = sum(weights_cma)^2 / sum(weights_cma .^ 2);
        end

        %% CMA-ES Adaptation
        if Hybridization_flag == 1
            [~, popindex] = sort(fitness);
            xold = xmean;
            xmean = popold(popindex(1:mu_cma), :)' * weights_cma;

            ps_cma = (1 - cs) * ps_cma + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
            hsig = sum(ps_cma .^ 2) / (1 - (1 - cs)^(2 * FE / pop_size)) / dim < 2 + 4/(dim + 1);
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

            artmp = (1/sigma) * (popold(popindex(1:mu_cma), :)' - repmat(xold, 1, mu_cma));
            C = (1 - c1 - cmu_cma) * C + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) + cmu_cma * artmp * diag(weights_cma) * artmp';

            sigma = sigma * exp((cs / damps) * (norm(ps_cma) / chiN - 1));

            if FE - eigeneval > pop_size / (c1 + cmu_cma) / dim / 10
                eigeneval = FE;
                C = triu(C) + triu(C, 1)';
                if sum(sum(isnan(C))) > 0 || sum(sum(~isfinite(C))) > 0 || ~isreal(C)
                    Hybridization_flag = 0;
                    continue;
                end
                [B, D_mat] = eig(C);
                D = sqrt(diag(D_mat));
                invsqrtC = B * diag(D .^ -1) * B';
            end
        end

    end

    % Fill remaining curve values
    curve(FE:end) = bsf_fit_var;

    best_fitness = bsf_fit_var;
    best_solution = bsf_solution;
end

%% --- Helper Functions ---

function [pop_hist, fit_hist, hist_idx] = record_top_k(...
    current_fe, population, fitness, history_pop_size, dim, ...
    pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    if mod(current_fe, sampling_interval) == 0 || hist_idx <= history_size
        if hist_idx <= history_size
            current_size = size(population, 1);
            [sorted_fit, sorted_idx] = sort(fitness);
            top_k = min(history_pop_size, current_size);
            rec_pop = NaN(history_pop_size, dim);
            rec_fit = NaN(1, history_pop_size);
            rec_pop(1:top_k, :) = population(sorted_idx(1:top_k), :);
            rec_fit(1:top_k) = sorted_fit(1:top_k);
            pop_hist(hist_idx, :, :) = rec_pop;
            fit_hist(hist_idx, :) = rec_fit;
            hist_idx = hist_idx + 1;
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

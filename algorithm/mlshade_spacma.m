% ----------------------------------------------------------------------- %
% Modified L-SHADE with Semi-Parameter Adaptation and CMA-ES (mLSHADE-SPACMA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size    = 18*D    % Initial population (linear reduction to 4)
%   arc_rate    = 1.4     % Archive size factor
%   memory_size = 5       % SHADE memory length
%   p_best_rate = 0.11    % pbest fraction
%   L_Rate      = 0.80    % Learning rate of the class-1 percentage
%   k1          = 3       % Rank-based selective pressure
%   First_class_percentage = 0.5
%
% Algorithm Concept:
%   - Hybrid of L-SHADE (class 1) and CMA-ES sampling (class 2); the share of
%     each class is adapted per memory slot from their relative improvement
%   - Modification 1 (precise elimination and generation): the worst 1 % of
%     the population is regenerated around the two best individuals during
%     the first half of the run
%   - Modification 2 (rank-based semi-parametric mutation): the second
%     difference vector is drawn by rank-proportional roulette (Rank = k1*(NP-i)+1)
%     instead of uniformly
%   - Modification 3 (elite external archive): the archive keeps the best
%     solutions instead of a random subset
%
% Reference:
% Shengwei Fu, Chi Ma, Ke Li, Cankun Xie, Qingsong Fan, Haisong Huang,
% Jiangxue Xie, Guozhang Zhang, Mingyang Yu,
% Modified LSHADE-SPACMA with new mutation strategy and external archive
% mechanism for numerical optimization and point cloud registration,
% Artificial Intelligence Review 58, 72 (2025).
% https://doi.org/10.1007/s10462-024-11053-1
% ----------------------------------------------------------------------- %
% Implementation Note:
% Two defects inherited from the LSHADE-SPACMA base, treated the same way there.
% The CMA-ES step size and mean are scaled to the box: sigma = 0.5 and
% xmean = rand(D,1) are written for the CEC box and are unchanged there, but on
% CEC2020RW RC44, whose box is 1920 wide and not centred at zero, the reference's
% mean sits outside the population and the first (xmean - xold)/sigma drives sigma
% to Inf, turning every CMA-ES offspring NaN by generation two. The eigen update
% also floors the spectrum before the inverse square root, which the reference's
% NaN/Inf/complex test on C does not cover for a real symmetric C that has drifted
% indefinite. The eigenvalue matrix does not shadow D, which sampling reads as a
% vector.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mlshade_spacma(problem)

    problem_size = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    max_nfes = problem.maxFe;

    L_Rate = 0.80;
    k1 = 3;

    lu = [lb; ub];

    curve = zeros(1, max_nfes);
    pop_size = 18 * problem_size;
    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Parameter settings for L-SHADE
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    max_pop_size = pop_size;
    min_pop_size = 4.0;

    First_calss_percentage = 0.5;

    % Initialise the main population
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* ...
             (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    pop = popold;

    nfes = 0;
    [fitness, nfes] = calculate_fitness(pop', problem, nfes);
    fitness = fitness(:);

    bsf_fit_var = 1e+30;
    bsf_solution = zeros(1, problem_size);

    for i = 1:pop_size
        if (fitness(i) < bsf_fit_var && isreal(pop(i, :)) && sum(isnan(pop(i, :))) == 0 ...
                && all(pop(i, :) >= lb) && all(pop(i, :) <= ub))
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= max_nfes
            curve(i) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, ...
                history_index, max_nfes);
        end
    end

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = arc_rate * pop_size;
    archive.pop = zeros(0, problem_size);
    archive.funvalues = zeros(0, 1);

    memory_1st_class_percentage = First_calss_percentage .* ones(memory_size, 1);

    % Initialise the CMA-ES parameters; sigma and xmean are scaled to the box (see note)
    sigma = 0.5 * mean(ub - lb) / 200;
    xmean = ((lb + ub) / 2)' + rand(problem_size, 1) .* ((ub - lb)' / 200);
    mu = pop_size / 2;
    weights = log(mu + 1/2) - log(1:mu)';
    mu = floor(mu);
    weights = weights / sum(weights);
    mueff = sum(weights) ^ 2 / sum(weights .^ 2);

    cc = (4 + mueff / problem_size) / (problem_size + 4 + 2 * mueff / problem_size);
    cs = (mueff + 2) / (problem_size + mueff + 5);
    c1 = 2 / ((problem_size + 1.3) ^ 2 + mueff);
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((problem_size + 2) ^ 2 + mueff));
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (problem_size + 1)) - 1) + cs;

    pc = zeros(problem_size, 1);
    ps = zeros(problem_size, 1);
    B = eye(problem_size, problem_size);
    D = ones(problem_size, 1);
    C = B * diag(D .^ 2) * B';
    invsqrtC = B * diag(D .^ -1) * B';
    eigeneval = 0;
    chiN = problem_size ^ 0.5 * (1 - 1 / (4 * problem_size) + 1 / (21 * problem_size ^ 2));

    % Main loop
    Hybridization_flag = 1;

    while nfes < max_nfes
        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        if nfes < max_nfes / 2
            num = floor(0.99 * pop_size);
            sorted_index1 = sorted_index(num:end);
            pop(sorted_index1, :) = repmat(pop(sorted_index(1)), pop_size - num + 1, 1) + ...
                rand(pop_size - num + 1, problem_size) .* ...
                repmat(pop(sorted_index(1)) - pop(sorted_index(2)), pop_size - num + 1, 1);
        end
        pop = boundConstraint(pop, popold, lb, ub);

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);
        mem_rand_ratio = rand(pop_size, 1);

        % Crossover rate
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        % Scaling factor
        if (nfes <= 0.5 * max_nfes)
            sf = 0.5 + .1 * rand(pop_size, 1);
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = 0.5 + 0.1 * rand(length(pos), 1);
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

        % Hybridization class probability
        Class_Select_Index = (memory_1st_class_percentage(mem_rand_index) >= mem_rand_ratio);
        if (Hybridization_flag == 0)
            Class_Select_Index = or(Class_Select_Index, ~Class_Select_Index);
        end

        r0 = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0); %#ok<ASGLU>

        pNP = max(round(p_best_rate * pop_size), 2);
        randindex = ceil(rand(1, pop_size) .* pNP);
        randindex = max(1, randindex);
        pbest = pop(sorted_index(randindex), :);

        Ri1 = 1:pop_size;
        Rank1 = k1 * (pop_size - Ri1) + 1;
        Pr1 = Rank1 ./ sum(Rank1);
        pop1 = zeros(pop_size, problem_size);
        r11 = randsample(pop_size, pop_size, true, Pr1);
        for j = 1:pop_size
            jj = r11(j);
            pop1(j, :) = pop(sorted_index(jj), :);
        end

        vi = zeros(pop_size, problem_size);
        if (sum(Class_Select_Index) ~= 0)
            vi(Class_Select_Index, :) = pop(Class_Select_Index, :) + ...
                sf(Class_Select_Index, ones(1, problem_size)) .* ...
                (pbest(Class_Select_Index, :) - pop(Class_Select_Index, :) + ...
                 pop1((Class_Select_Index), :) - popAll(r2(Class_Select_Index), :));
        end

        if (sum(~Class_Select_Index) ~= 0)
            temp = zeros(problem_size, sum(~Class_Select_Index));
            for k = 1:sum(~Class_Select_Index)
                temp(:, k) = xmean + sigma * B * (D .* randn(problem_size, 1));
            end
            vi(~Class_Select_Index, :) = temp';
        end

        if (~isreal(vi))
            Hybridization_flag = 0;
            continue;
        end

        vi = boundConstraint(vi, pop, lb, ub);

        mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size));
        rows = (1:pop_size)'; cols = floor(rand(pop_size, 1) * problem_size) + 1;
        jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
        ui = vi; ui(mask) = pop(mask);

        nfes1 = nfes;
        [children_fitness, nfes] = calculate_fitness(ui', problem, nfes);
        children_fitness = children_fitness(:);

        for i = 1:pop_size
            if (children_fitness(i) < bsf_fit_var && isreal(ui(i, :)) && sum(isnan(ui(i, :))) == 0 ...
                    && all(ui(i, :) >= lb) && all(ui(i, :) <= ub))
                bsf_fit_var = children_fitness(i);
                bsf_solution = ui(i, :);
            end
        end
        for k = 1:pop_size
            ec = nfes1 + k;
            if ec >= 1 && ec <= max_nfes
                curve(ec) = bsf_fit_var;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fitness, population_history, fitness_history, ...
                    history_index, max_nfes);
            end
        end

        dif = abs(fitness - children_fitness);

        Child_is_better_index = (fitness > children_fitness);
        goodCR = cr(Child_is_better_index == 1);
        goodF  = sf(Child_is_better_index == 1);
        dif_val = dif(Child_is_better_index == 1);
        dif_val_Class_1 = dif(and(Child_is_better_index, Class_Select_Index) == 1);
        dif_val_Class_2 = dif(and(Child_is_better_index, ~Class_Select_Index) == 1);

        archive = updateArchive(archive, popold(Child_is_better_index == 1, :), ...
                                fitness(Child_is_better_index == 1));

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

            if (Hybridization_flag == 1)
                memory_1st_class_percentage(memory_pos) = ...
                    memory_1st_class_percentage(memory_pos) * L_Rate + (1 - L_Rate) * ...
                    (sum(dif_val_Class_1) / (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
                memory_1st_class_percentage(memory_pos) = min(memory_1st_class_percentage(memory_pos), 0.8);
                memory_1st_class_percentage(memory_pos) = max(memory_1st_class_percentage(memory_pos), 0.2);
            end

            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end

        % Population size reduction
        plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);

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
                [~, sortIdx] = sort(archive.funvalues);
                bestIdx = sortIdx(1:archive.NP);
                archive.pop = archive.pop(bestIdx, :);
                archive.funvalues = archive.funvalues(bestIdx, :);
            end

            mu = pop_size / 2;
            weights = log(mu + 1/2) - log(1:mu)';
            mu = floor(mu);
            weights = weights / sum(weights);
            mueff = sum(weights) ^ 2 / sum(weights .^ 2);
        end

        % CMA-ES adaptation
        if (Hybridization_flag == 1)
            [~, popindex] = sort(fitness);
            xold = xmean;
            xmean = popold(popindex(1:mu), :)' * weights;

            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
            hsig = sum(ps .^ 2) / (1 - (1 - cs) ^ (2 * nfes / pop_size)) / problem_size < 2 + 4 / (problem_size + 1);
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

            artmp = (1 / sigma) * (popold(popindex(1:mu), :)' - repmat(xold, 1, mu));
            C = (1 - c1 - cmu) * C ...
                + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
                + cmu * artmp * diag(weights) * artmp';

            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));

            if nfes - eigeneval > pop_size / (c1 + cmu) / problem_size / 10
                eigeneval = nfes;
                C = triu(C) + triu(C, 1)';
                if (sum(sum(isnan(C))) > 0 || sum(sum(~isfinite(C))) > 0 || ~isreal(C))
                    Hybridization_flag = 0;
                    continue;
                end
                [B, D_mat] = eig(C);
                e = real(diag(D_mat));
                emax = max(e);
                if ~isfinite(emax) || emax <= 0
                    C = eye(problem_size); B = eye(problem_size); e = ones(problem_size, 1);   % fully degenerate -> reset
                else
                    e = max(e, emax * 1e-14);   % floor the spectrum so the inverse root stays real
                end
                B = real(B);
                D = sqrt(e);
                invsqrtC = B * diag(D .^ -1) * B';
            end
        end
    end

    curve(min(nfes, max_nfes):end) = bsf_fit_var;

    best_fitness  = bsf_fit_var;
    best_solution = bsf_solution;
end

% Random index generation
function [r1, r2] = gnR1R2(NP1, NP2, r0)
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
end

% Elite external archive
function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0, return; end
    if size(pop, 1) ~= size(funvalue, 1), error('mlshade_spacma:archive', 'check it'); end

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
        [~, sortIdx] = sort(funvalues);
        bestIdx = sortIdx(1:floor(archive.NP));
        archive.pop = popAll(bestIdx, :);
        archive.funvalues = funvalues(bestIdx, :);
    end
end

% Midpoint bound repair
function vi = boundConstraint(vi, pop, lb, ub)
    NP = size(pop, 1);
    xl = repmat(lb, NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu = repmat(ub, NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

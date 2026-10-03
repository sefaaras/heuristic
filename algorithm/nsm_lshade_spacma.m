% ----------------------------------------------------------------------- %
% Natural Survivor Method based L-SHADE-SPACMA (NSM-LSHADE-SPACMA)
% Variant of lshade_spacma: NSM survivor selection
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18*D -> 4 (linear)         % As lshade_spacma
%   p_best_rate = 0.11, arc_rate = 1.4    % pbest share and archive size (lshade_spacma)
%   memory_size = 5, L_Rate = 0.80        % F/CR memory, DE vs CMA-ES class learning rate
%   c = 1-Singer | Piecewise | 1-Logistic % Chaotic map over FE for D <= 30 | < 100 | >= 100
%   p_NSM = c(FE)                         % NSM vs greedy test, drawn per individual
%   n_centre = NP | ceil(c*NP) | 3        % Best individuals averaged into the centre
%   w = [0.9388 0.8586 0.2677]            % NSM weights for D <= 30: fitness, best, centre
%   w = [0.8271 0.3483 0.2723]            % NSM weights for D > 30
%
% Algorithm Concept:
%   - current-to-pbest/1 and CMA-ES offspring classes, semi-parameter adaptation
%     and linear population reduction, as in lshade_spacma
%   - Survivor selection: per individual, with probability c(FE), the greedy
%     test is replaced by comparing the NSM scores of the parent and the trial
%   - NSM score = w1*fitness + w2*distance from the best + w3*distance from the
%     centre of the n_centre best, each min-max normalised, L1 distances
%   - A trial worse than the population's worst is rejected and one better than
%     its best is accepted, so the best is never replaced by a worse point
%   - An NSM survivor counts as a success for the F/CR memories, the class
%     probability and the archive, even when it is worse than its parent
%
% Reference:
% Hamdi Tolga Kahraman, Mehmet Kati, Sefa Aras, Durdane Ayse Tasci,
% Development of the Natural Survivor Method (NSM) for designing an updating
% mechanism in metaheuristic search algorithms,
% Engineering Applications of Artificial Intelligence 122 (2023) 106121.
% https://doi.org/10.1016/j.engappai.2023.106121
% Components: base algorithm L-SHADE-SPACMA --
% Ali W. Mohamed, Anas A. Hadi, Anas M. Fattouh, Kamal M. Jambi,
% LSHADE with semi-parameter adaptation hybrid with CMA-ES for solving
% CEC 2017 benchmark problems,
% 2017 IEEE Congress on Evolutionary Computation (CEC), 2017, pp. 145-152
% https://doi.org/10.1109/CEC.2017.7969307
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' group working copy (lshadeNSM.m, which despite its
% name is L-SHADE-SPACMA, its NSM score file and the group's chaotic maps) as a
% delta onto the pool's lshade_spacma, whose box-scaled CMA-ES start, eigenvalue
% floor and Inf-initialised bsf (see its note) are inherited. NSM costs no extra
% evaluation; the maps are indexed by FE after each batch, as in the copy.
% Kept as written: when every survivor of a generation has its parent's fitness
% (possible only through NSM), the Lehmer means are 0/0 and the memory slot turns
% NaN, which the clamps read as F = CR = 1 and a 0.8 class share until the slot is
% overwritten; 7 of 443 and 14 of 679 success generations on CEC2014 F17 at D = 10
% and 30 (10000*D FE).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nsm_lshade_spacma(problem)

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

    % NSM state of the group's nsmInitialization: one chaotic value per FE
    nsm_map = nsm_chaos_map(dim, maxFE + 1000);

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

    bsf_fit_var = inf;
    bsf_solution = zeros(1, dim);

    for i = 1:pop_size
        if fitness(i) < bsf_fit_var && is_valid_solution(pop(i, :), lu)
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

    memory_1st_class_percentage = First_class_percentage .* ones(memory_size, 1);

    % Initialize CMA-ES parameters; sigma and xmean are scaled to the box (see note)
    sigma = 0.5 * mean(ub - lb) / 200;
    xmean = ((lb + ub) / 2)' + rand(dim, 1) .* ((ub - lb)' / 200);
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

    % Main loop
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
            if children_fitness(i) < bsf_fit_var && is_valid_solution(ui(i, :), lu)
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

        % Survivor selection in sequence, so each NSM test sees the replacements before it
        c_fe = nsm_map(min(FE, end));
        [n_centre, nsm_w] = nsm_band(dim, pop_size, c_fe);
        fitOld = fitness;
        Child_is_better_index = false(pop_size, 1);
        for k = 1:pop_size
            if rand < c_fe
                [f_worst, i_worst] = max(fitness);
                [f_best, i_best] = min(fitness);
                if f_worst > children_fitness(k) && nsm_accept(pop(k, :), fitness(k), ...
                        ui(k, :), children_fitness(k), pop(i_worst, :), f_worst, ...
                        pop(i_best, :), f_best, pop, fitness, n_centre, nsm_w)
                    fitness(k) = children_fitness(k);
                    pop(k, :) = ui(k, :);
                    Child_is_better_index(k) = true;
                end
            elseif children_fitness(k) < fitness(k)
                fitness(k) = children_fitness(k);
                pop(k, :) = ui(k, :);
                Child_is_better_index(k) = true;
            end
        end

        goodCR = cr(Child_is_better_index);
        goodF = sf(Child_is_better_index);
        dif_val = dif(Child_is_better_index);
        dif_val_Class_1 = dif(Child_is_better_index & Class_Select_Index);
        dif_val_Class_2 = dif(Child_is_better_index & ~Class_Select_Index);

        archive = updateArchive(archive, popold(Child_is_better_index, :), fitOld(Child_is_better_index));

        popold = pop;

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

            % Update CMA parameters
            mu_cma = pop_size / 2;
            weights_cma = log(mu_cma + 1/2) - log(1:mu_cma)';
            mu_cma = floor(mu_cma);
            weights_cma = weights_cma / sum(weights_cma);
            mueff = sum(weights_cma)^2 / sum(weights_cma .^ 2);
        end

        % CMA-ES Adaptation
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
                e = real(diag(D_mat));
                emax = max(e);
                if ~isfinite(emax) || emax <= 0
                    C = eye(dim); B = eye(dim); e = ones(dim, 1);   % fully degenerate -> reset
                else
                    e = max(e, emax * 1e-14);   % floor the spectrum so the inverse root stays real
                end
                B = real(B);
                D = sqrt(e);
                invsqrtC = B * diag(D .^ -1) * B';
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

function ok = is_valid_solution(x, lu)
    ok = isreal(x) && all(isfinite(x)) && all(x >= lu(1, :)) && all(x <= lu(2, :));
end

% The group's nsmInitialization: x1 = 0.7, Singer (reversed) / Piecewise / Logistic (reversed)
function c = nsm_chaos_map(dim, n)
    c = zeros(n, 1);
    x = 0.7;
    P = 0.4;
    for k = 1:n
        c(k) = x;
        if dim <= 30
            x = 1.07 * (7.86 * x - 23.31 * (x^2) + 28.75 * (x^3) - 13.302875 * (x^4));
        elseif dim < 100
            if x >= 0 && x < P
                x = x / P;
            elseif x >= P && x < 0.5
                x = (x - P) / (0.5 - P);
            elseif x >= 0.5 && x < 1 - P
                x = (1 - P - x) / (0.5 - P);
            elseif x >= 1 - P && x < 1
                x = (1 - x) / P;
            else
                x = 0;   % the copy's preallocated zero when no branch applies
            end
        else
            x = 4 * x * (1 - x);
        end
    end
    if dim <= 30 || dim >= 100
        c = 1 - c;
    end
end

% The group's nsmUpdatePopulation: centre size and NSM weights by dimension band
function [n_centre, w] = nsm_band(dim, pop_size, c)
    if dim <= 30
        n_centre = pop_size;
        w = [0.9388 0.8586 0.2677];
    elseif dim < 100
        n_centre = ceil(c * pop_size);
        w = [0.8271 0.3483 0.2723];
    else
        n_centre = 3;
        w = [0.8271 0.3483 0.2723];
    end
end

% NSM survivor test (FDDBPopulasyonGuncelleTest of the group's copy): true keeps xn over xo
function accept = nsm_accept(xo, fo, xn, fn, xw, fw, xb, fb, X, fit, n_centre, w)
    if fb == fw || fb > fn
        accept = true;
    elseif fb == fo
        accept = false;
    else
        [~, order] = sort(fit);
        n_centre = max(n_centre, 2);
        centre = sum(X(order(1:n_centre), :), 1) / n_centre;
        cand = [xo; xn; xw; xb];
        % Distance from the centre is min-max normalised over old, new, worst and best
        dc = sum(abs(cand - centre), 2)';
        dc = (dc - min(dc)) / (max(dc) - min(dc));
        % Distance from the best is divided by its maximum over old, new and worst
        db = sum(abs(cand(1:3, :) - xb), 2)';
        db = db / max(db);
        nf = 1 - ([fo, fn] - fb) / (fw - fb);
        score = w(1) * nf + w(2) * db(1:2) + w(3) * dc(1:2);
        accept = score(2) > score(1);
    end
end

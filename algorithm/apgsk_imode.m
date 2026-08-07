% ----------------------------------------------------------------------- %
% Adaptive-Parameter GSK hybridized with IMODE (APGSK-IMODE)
% CEC 2021 competition entry
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP  = 30*D          % Total population, split PS2 = round(NP/4) for APGSK
%   CS  = 50            % Cycle length between sub-population exchanges
%   n_opr = 3           % IMODE mutation operators
%   memory_size = 15*D  % IMODE historical memory for F and CR
%   arch_rate = 1.4     % IMODE archive size as a multiple of PS1
%   KF_pool = [0.1 1.0 0.5 1.0], KR_pool = [0.2 0.1 0.9 0.9]   % APGSK pools
%
% Algorithm Concept:
%   - Two co-evolving sub-populations: EA_1 driven by IMODE (3 DE operators,
%     linear population reduction, archive) and EA_2 driven by APGSK
%     (junior/senior gaining-sharing knowledge with adaptive KF/KR).
%   - A cycle (CS) periodically shares the best solution between them and
%     re-weights how much each sub-population runs.
%
% Reference:
% A. W. Mohamed, A. A. Hadi, P. Agrawal, K. M. Sallam, A. K. Mohamed,
% Gaining-Sharing Knowledge Based Algorithm with Adaptive Parameters Hybrid
% with IMODE Algorithm for Solving CEC 2021 Benchmark Problems,
% 2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. 841-848.
% https://doi.org/10.1109/CEC45853.2021.9504814
% Components:
%   IMODE - K. M. Sallam et al., "Improved Multi-operator Differential
%     Evolution Algorithm for Solving Unconstrained Problems," IEEE CEC 2020.
%     https://doi.org/10.1109/CEC48606.2020.9185577
%   GSK/APGSK - A. W. Mohamed et al., "Gaining-sharing knowledge based
%     algorithm...," Int. J. Mach. Learn. Cybern. 11 (2020) 1501-1529.
%     https://doi.org/10.1007/s13042-019-01053-x
% ----------------------------------------------------------------------- %
% Implementation Note:
% Reference hardcodes the [-100,100] CEC box and a length-10 optima
% array; here the bounds are generalized to the actual problem bounds
% (identical on the CEC suites), f_optimal is set to -inf (disables the
% optimum-based early stop, since optima are unknown/nonzero here), and the
% EA_2 split uses round(NP/4) so it stays integer for every dimension.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = apgsk_imode(problem)

    % Extract problem parameters
    D = problem.dimension;
    max_nfes = problem.maxFe;
    lb = problem.lb;
    ub = problem.ub;

    Par = Introd_Par(D, max_nfes, lb, ub);
    D = Par.n;
    lu = [lb; ub];
    NP = Par.PopSize;
    min_NP = 12.0;
    min_NP1 = 4;

    FE = 0;
    curve = zeros(1, max_nfes);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Pop = repmat(lu(1, :), NP, 1) + rand(NP, D) .* (repmat(lu(2, :) - lu(1, :), NP, 1));
    [Fit, FE] = calculate_fitness(Pop', problem, FE);
    Fit = Fit(:);

    PS2 = round(NP / 4);
    PS1 = NP - PS2;
    max_NP1 = PS1;
    max_NP2 = PS2;

    % IMODE sub-population
    EA_1 = Pop(1:PS1, :); EA_obj1 = Fit(1:PS1);
    % APGSK sub-population
    EA_2 = Pop(PS1 + 1:size(Pop, 1), :); EA_obj2 = Fit(PS1 + 1:size(Pop, 1));

    arch_rate = 1.4;
    archive.NP = arch_rate * PS1;
    archive.pop = zeros(0, Par.n);
    archive.funvalues = zeros(0, 1);

    EA_1old = EA_1;
    hist_pos = 1;
    memory_size = 15 * Par.n;
    archive_f = ones(1, memory_size) .* 0.5;
    archive_Cr = ones(1, memory_size) .* 0.5;
    archive_T = ones(1, memory_size) .* 0.1;
    archive_freq = ones(1, memory_size) .* 0.5;
    F = normrnd(0.5, 0.15, 1, NP);
    cr = normrnd(0.5, 0.15, 1, NP);
    probDE1 = 1 ./ Par.n_opr .* ones(1, Par.n_opr);

    [bestold, bes_l] = min(Fit); bestx = Pop(bes_l, :);
    bsf = bestold; bsf_sol = bestx;

    [curve, population_history, fitness_history, history_index] = rec_block(...
        Pop, Fit, 1, FE, bsf, curve, population_history, fitness_history, ...
        history_index, max_nfes);

    Probs = [1 1];
    it = 0;
    cy = 0;
    indx = 0;
    stop_con = 0;

    while stop_con == 0
        it = it + 1;
        cy = cy + 1;

        % Determine the best phase / share information
        if (cy == ceil(Par.CS + 1))
            qual(1) = min(EA_obj1);
            qual(2) = min(EA_obj2);
            norm_qual = qual ./ sum(qual);
            norm_qual = 1 - norm_qual;
            Probs = norm_qual;
            Probs = max(0.1, min(0.9, Probs ./ sum(Probs)));
            [~, indx] = max(Probs);
            if Probs(1) == Probs(2)
                indx = 0;
            end
            if indx > 0
                Probs = [0 0];
                Probs(indx) = 1;
            end
        elseif cy == 2 * ceil(Par.CS)
            if indx == 1
                EA_2(PS2, :) = EA_1(1, :);
                EA_obj2(PS2) = EA_obj1(1);
                [EA_obj2, ind] = sort(EA_obj2);
                EA_2 = EA_2(ind, :);
            else
                if all(EA_2(1, :) >= lb) && all(EA_2(1, :) <= ub)
                    EA_1(PS1, :) = EA_2(1, :);
                    EA_obj1(PS1) = EA_obj2(1);
                    [EA_obj1, ind] = sort(EA_obj1);
                    EA_1 = EA_1(ind, :);
                end
            end
            cy = 1; Probs = ones(1, 2);
        end

        if (rand <= Probs(1))
            % IMODE
            ev0 = FE;
            [EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T, archive_freq, FE, F, cr] = ...
                IMODE(EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T, ...
                archive_freq, Par.xmin, Par.xmax, Par.n, PS1, FE, Par.Max_FES, Par.Gmax, F, cr, problem, lb, ub);
            if bestold < bsf, bsf = bestold; bsf_sol = bestx; end
            [curve, population_history, fitness_history, history_index] = rec_block(...
                EA_1, EA_obj1, ev0 + 1, FE, bsf, curve, population_history, fitness_history, ...
                history_index, max_nfes);

            if FE >= max_nfes
                stop_con = 1;
            end

            plan_NP1 = round((((min_NP1 - max_NP1) / (max_nfes)) * FE) + max_NP1);
            Par.p_best_rate = (((Par.p_best_rate_min - Par.p_best_rate_max) / (max_nfes)) * FE) + Par.p_best_rate_max;

            if PS1 > plan_NP1
                reduction_ind_num = PS1 - plan_NP1;
                if PS1 - reduction_ind_num < min_NP1
                    reduction_ind_num = PS1 - min_NP1;
                end
                PS1 = PS1 - reduction_ind_num;
                for r = 1:reduction_ind_num
                    [~, indBest] = sort(EA_obj1, 'ascend');
                    worst_ind = indBest(end);
                    EA_1(worst_ind, :) = [];
                    EA_obj1(worst_ind, :) = [];
                end
                archive.NP = PS1;
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1:archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
        end

        if stop_con == 0 && (rand < Probs(2))
            % APGSK
            ev0 = FE;
            [EA_2, EA_obj2, Par, FE] = APGSK_fun(EA_2, EA_obj2, lu, Par, FE, problem);
            [m2, i2] = min(EA_obj2);
            if m2 < bsf, bsf = m2; bsf_sol = EA_2(i2, :); end
            [curve, population_history, fitness_history, history_index] = rec_block(...
                EA_2, EA_obj2, ev0 + 1, FE, bsf, curve, population_history, fitness_history, ...
                history_index, max_nfes);

            if FE >= max_nfes
                stop_con = 1;
            end

            plan_NP2 = round((((min_NP - max_NP2) / (max_nfes)) * FE) + max_NP2);
            Par.p_best_rate = (((Par.p_best_rate_min - Par.p_best_rate_max) / (max_nfes)) * FE) + Par.p_best_rate_max;

            if PS2 > plan_NP2
                reduction_ind_num = PS2 - plan_NP2;
                if PS2 - reduction_ind_num < min_NP
                    reduction_ind_num = PS2 - min_NP;
                end
                PS2 = PS2 - reduction_ind_num;
                for r = 1:reduction_ind_num
                    [~, indBest] = sort(EA_obj2, 'ascend');
                    worst_ind = indBest(end);
                    EA_2(worst_ind, :) = [];
                    EA_obj2(worst_ind, :) = [];
                    Par.K(worst_ind, :) = [];
                end
            end
        end

        if (FE >= Par.max_nfes)
            stop_con = 1;
        end
    end

    curve(min(FE, max_nfes):end) = bsf;

    best_solution = bsf_sol;
    best_fitness = bsf;
end

% Record best-so-far curve + top-k history over an FE block
function [curve, ph, fh, hidx] = rec_block(pop, costs, fe_from, fe_to, bestval, curve, ph, fh, hidx, maxFE)
    fe_to = min(fe_to, maxFE);
    if fe_from < 1, fe_from = 1; end
    if fe_to < fe_from, return; end
    costs = costs(:)';
    for ec = fe_from:fe_to
        curve(ec) = bestval;
        [ph, fh, hidx] = record_history(ec, pop, costs, ph, fh, hidx, maxFE);
    end
end

% Parameters
function [Par] = Introd_Par(n, maxfes, lb, ub)
    Par.n_opr = 3;
    Par.n = n;
    Par.CS = 50;
    Par.max_nfes = maxfes;
    Par.Max_FES = maxfes;
    Par.Gmax = 1000;

    Par.xmin = lb;
    Par.xmax = ub;
    Par.f_optimal = -inf;
    Par.PopSize = 30 * Par.n;

    Par.p_best_rate = 0.5;
    Par.p_best_rate_max = Par.p_best_rate;
    Par.p_best_rate_min = 0.15;

    PS2 = round(Par.PopSize / 4);
    Kind = rand(PS2, 1);
    K = zeros(PS2, 1);
    K(Kind < 0.5, :) = rand(sum(Kind < 0.5), 1);
    K(Kind >= 0.5, :) = ceil(20 * rand(sum(Kind >= 0.5), 1));

    Par.All_Imp = zeros(1, 4);
    Par.K = K;
    % Same weights APGSK_fun sets early on, so KW_ind is never empty on a very small maxFe
    Par.KW_ind = [0.85 0.05 0.05 0.05];
    Par.Printing = 0;
end

% IMODE stage
function [x, xold, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T, archive_freq, current_eval, F, cr] = ...
    IMODE(x, xold, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, archive_T, archive_freq, xmin, xmax, n, ...
    PopSize, current_eval, Max_FES, G_Max, F, cr, problem, lb, ub) %#ok<INUSD>

    vi = zeros(PopSize, n);

    mem_rand_index = ceil(memory_size * rand(PopSize, 1));
    mu_sf = archive_f(mem_rand_index);
    mu_cr = archive_Cr(mem_rand_index);

    cr = normrnd(mu_cr, 0.1);
    term_pos = find(mu_cr == -1);
    cr(term_pos) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);

    F = mu_sf + 0.1 * tan(pi * (rand(1, PopSize) - 0.5));
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1, length(pos)) - 0.5));
        pos = find(F <= 0);
    end
    F = min(F, 1);
    F = F';
    [fitx, inddd] = sort(fitx);
    x = x(inddd, :);
    [cr, ~] = sort(cr);

    popAll = [x; archive.pop];
    r0 = 1:PopSize;
    [r1, r2, r3] = gnR1R2(PopSize, size(popAll, 1), r0);

    bb = rand(PopSize, 1);
    probiter = prob(1, :);
    l2 = sum(prob(1:2));
    op_1 = bb <= probiter(1) * ones(PopSize, 1);
    op_2 = bb > probiter(1) * ones(PopSize, 1) & bb <= (l2 * ones(PopSize, 1));
    op_3 = bb > l2 * ones(PopSize, 1) & bb <= (ones(PopSize, 1));

    pNP = max(round(0.25 * PopSize), 1);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_1 == 1, :) = x(op_1 == 1, :) + F(op_1 == 1, ones(1, n)) .* (phix(op_1 == 1, :) - x(op_1 == 1, :) + x(r1(op_1 == 1), :) - popAll(r2(op_1 == 1), :));
    vi(op_2 == 1, :) = x(op_2 == 1, :) + F(op_2 == 1, ones(1, n)) .* (phix(op_2 == 1, :) - x(op_2 == 1, :) + x(r1(op_2 == 1), :) - x(r3(op_2 == 1), :));
    pNP = max(round(0.5 * PopSize), 2);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_3 == 1, :) = F(op_3 == 1, ones(1, n)) .* x(r1(op_3 == 1), :) + F(op_3 == 1, ones(1, n)) .* (phix(op_3 == 1, :) - x(r3(op_3 == 1), :));

    vi = han_boun(vi, xmax, xmin, x, PopSize, 2);

    if rand < 0.3
        mask = rand(PopSize, n) > cr(:, ones(1, n));
        rows = (1:PopSize)'; cols = floor(rand(PopSize, 1) * n) + 1;
        jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
        ui = vi; ui(mask) = x(mask);
    else
        ui = x;
        startLoc = randi(n, PopSize, 1);
        for i = 1:PopSize
            l = startLoc(i);
            while (rand < cr(i) && l < n)
                l = l + 1;
            end
            for j = startLoc(i):l
                ui(i, j) = vi(i, j);
            end
        end
    end

    [fitx_new, current_eval] = calculate_fitness(ui', problem, current_eval);
    fitx_new = fitx_new(:);

    for i = 1:PopSize
        if fitx_new(i) < bestold
            bestold = fitx_new(i);
            bestx = ui(i, :);
        end
    end

    dif = abs(fitx - fitx_new);
    I = (fitx_new < fitx);
    goodCR = cr(I == 1);
    goodF = F(I == 1);

    archive = updateArchive(archive, x(I == 1, :), fitx(I == 1));

    diff2 = max(0, (fitx - fitx_new)) ./ abs(fitx);
    count_S(1) = max(0, mean(diff2(op_1 == 1)));
    count_S(2) = max(0, mean(diff2(op_2 == 1)));
    count_S(3) = max(0, mean(diff2(op_3 == 1)));
    if count_S ~= 0
        prob = max(0.1, min(0.9, count_S ./ (sum(count_S))));
    else
        prob = 1 / 3 * ones(1, 3);
    end

    fitx(I == 1) = fitx_new(I == 1);
    xold(I == 1, :) = x(I == 1, :);
    x(I == 1, :) = ui(I == 1, :);

    if size(goodF, 1) == 1, goodF = goodF'; end
    if size(goodCR, 1) == 1, goodCR = goodCR'; end
    num_success_params = numel(goodCR);
    if num_success_params > 0
        weightsDE = dif(I == 1) ./ sum(dif(I == 1));
        archive_f(hist_pos) = (weightsDE' * (goodF .^ 2)) ./ (weightsDE' * goodF);
        if max(goodCR) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (weightsDE' * (goodCR .^ 2)) / (weightsDE' * goodCR);
        end
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size, hist_pos = 1; end
    else
        archive_Cr(hist_pos) = 0.2;
        archive_f(hist_pos) = 0.2;
    end

    [fitx, ind] = sort(fitx);
    x = x(ind, :);
    xold = xold(ind, :);

    if fitx(1) < bestold && all(x(ind(1), :) >= xmin) && all(x(ind(1), :) <= xmax)
        bestold = fitx(1);
        bestx = x(1, :);
    end
end

% APGSK stage
function [pop, fitness, Par, nfes] = APGSK_fun(pop, fitness, lu, Par, nfes, problem)

    [pop_size, problem_size] = size(pop);

    KF_pool = [0.1 1.0 0.5 1.0];
    KF_poool = [-0.1 -0.1 -0.1 -0.1];
    KR_pool = [0.2 0.1 0.9 0.9];

    max_nfes = Par.max_nfes;
    All_Imp = Par.All_Imp;
    KW_ind = Par.KW_ind;

    if (nfes < 0.1 * max_nfes)
        KW_ind = [0.85 0.05 0.05 0.05];
        K_rand_ind = rand(pop_size, 1);
        K_rand_ind(K_rand_ind > sum(KW_ind(1:3)) & K_rand_ind <= sum(KW_ind(1:4))) = 4;
        K_rand_ind(K_rand_ind > sum(KW_ind(1:2)) & K_rand_ind <= sum(KW_ind(1:3))) = 3;
        K_rand_ind(K_rand_ind > KW_ind(1) & K_rand_ind <= sum(KW_ind(1:2))) = 2;
        K_rand_ind(K_rand_ind > 0 & K_rand_ind <= KW_ind(1)) = 1;
        KF = KF_pool(K_rand_ind)';
        KR = KR_pool(K_rand_ind)';
    else
        KW_ind = 0.95 * KW_ind + 0.05 * All_Imp;
        KW_ind = KW_ind ./ sum(KW_ind);
        K_rand_ind = rand(pop_size, 1);
        K_rand_ind(K_rand_ind > sum(KW_ind(1:3)) & K_rand_ind <= sum(KW_ind(1:4))) = 4;
        K_rand_ind(K_rand_ind > sum(KW_ind(1:2)) & K_rand_ind <= sum(KW_ind(1:3))) = 3;
        K_rand_ind(K_rand_ind > KW_ind(1) & K_rand_ind <= sum(KW_ind(1:2))) = 2;
        K_rand_ind(K_rand_ind > 0 & K_rand_ind <= KW_ind(1)) = 1;
        KR = KR_pool(K_rand_ind)';
        if rand >= 0.1 && nfes > 0.5 * max_nfes
            KF = KF_pool(K_rand_ind)';
        else
            KF = KF_poool(K_rand_ind)';
        end
    end

    if rand > (nfes / max_nfes)
        D_Gained_Shared_Junior = ceil((1) * round((problem_size) * ((1 - nfes / max_nfes).^((0.5)))));
    else
        D_Gained_Shared_Junior = ceil((1) * round((problem_size) * ((1 - nfes / max_nfes).^((2)))));
    end
    D_Gained_Shared_Senior = problem_size - D_Gained_Shared_Junior; %#ok<NASGU>

    [valBest, indBest] = sort(fitness, 'ascend');
    [Rg1, Rg2, Rg3] = Gained_Shared_Junior_R1R2R3(indBest);
    [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest);
    R01 = 1:pop_size;
    Gained_Shared_Junior = zeros(pop_size, problem_size);
    ind1 = fitness(R01) > fitness(Rg3);
    if (sum(ind1) > 0)
        Gained_Shared_Junior(ind1, :) = pop(ind1, :) + KF(ind1, ones(1, problem_size)) .* (pop(Rg1(ind1), :) - pop(Rg2(ind1), :) + pop(Rg3(ind1), :) - pop(ind1, :));
    end
    ind1 = ~ind1;
    if (sum(ind1) > 0)
        Gained_Shared_Junior(ind1, :) = pop(ind1, :) + KF(ind1, ones(1, problem_size)) .* (pop(Rg1(ind1), :) - pop(Rg2(ind1), :) + pop(ind1, :) - pop(Rg3(ind1), :));
    end
    R0 = 1:pop_size;
    Gained_Shared_Senior = zeros(pop_size, problem_size);
    ind = fitness(R0) > fitness(R2);
    if (sum(ind) > 0)
        Gained_Shared_Senior(ind, :) = pop(ind, :) + KF(ind, ones(1, problem_size)) .* (pop(R1(ind), :) - pop(ind, :) + pop(R2(ind), :) - pop(R3(ind), :));
    end
    ind = ~ind;
    if (sum(ind) > 0)
        Gained_Shared_Senior(ind, :) = pop(ind, :) + KF(ind, ones(1, problem_size)) .* (pop(R1(ind), :) - pop(R2(ind), :) + pop(ind, :) - pop(R3(ind), :));
    end
    Gained_Shared_Junior = boundConstraint(Gained_Shared_Junior, pop, lu);
    Gained_Shared_Senior = boundConstraint(Gained_Shared_Senior, pop, lu);

    D_Gained_Shared_Junior_mask = rand(pop_size, problem_size) <= (D_Gained_Shared_Junior(:, ones(1, problem_size)) ./ problem_size);
    D_Gained_Shared_Senior_mask = ~D_Gained_Shared_Junior_mask;
    D_Gained_Shared_Junior_rand_mask = rand(pop_size, problem_size) <= KR(:, ones(1, problem_size));
    D_Gained_Shared_Junior_mask = and(D_Gained_Shared_Junior_mask, D_Gained_Shared_Junior_rand_mask);
    D_Gained_Shared_Senior_rand_mask = rand(pop_size, problem_size) <= KR(:, ones(1, problem_size));
    D_Gained_Shared_Senior_mask = and(D_Gained_Shared_Senior_mask, D_Gained_Shared_Senior_rand_mask);
    ui = pop;
    ui(D_Gained_Shared_Junior_mask) = Gained_Shared_Junior(D_Gained_Shared_Junior_mask);
    ui(D_Gained_Shared_Senior_mask) = Gained_Shared_Senior(D_Gained_Shared_Senior_mask);

    [children_fitness, nfes] = calculate_fitness(ui', problem, nfes);
    children_fitness = children_fitness(:);
    for i = 1:pop_size
        if children_fitness(i) < valBest(1)
            valBest(1) = children_fitness(i);
        end
    end

    dif = abs(fitness - children_fitness);
    Child_is_better_index = (fitness > children_fitness);
    All_Imp = zeros(1, 4);
    for i = 1:4
        if (sum(and(Child_is_better_index, K_rand_ind == i)) > 0)
            All_Imp(i) = sum(dif(and(Child_is_better_index, K_rand_ind == i)));
        else
            All_Imp(i) = 0;
        end
    end

    if (sum(All_Imp) ~= 0)
        All_Imp = All_Imp ./ sum(All_Imp);
        [~, Imp_Ind] = sort(All_Imp);
        for imp_i = 1:length(All_Imp) - 1
            All_Imp(Imp_Ind(imp_i)) = max(All_Imp(Imp_Ind(imp_i)), 0.05);
        end
        All_Imp(Imp_Ind(end)) = 1 - sum(All_Imp(Imp_Ind(1:end - 1)));
    else
        Imp_Ind = 1:length(All_Imp); %#ok<NASGU>
        All_Imp(:) = 1 / length(All_Imp);
    end
    [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);
    pop(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);

    Par.All_Imp = All_Imp;
    Par.KW_ind = KW_ind;
end

% Gaining-Sharing R indices
function [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest)
    pop_size = length(indBest);
    R1 = indBest(1:round(pop_size * 0.1));
    R1rand = ceil(length(R1) * rand(pop_size, 1));
    R1 = R1(R1rand);
    R2 = indBest(round(pop_size * 0.1) + 1:round(pop_size * 0.9));
    R2rand = ceil(length(R2) * rand(pop_size, 1));
    R2 = R2(R2rand);
    R3 = indBest(round(pop_size * 0.9) + 1:end);
    R3rand = ceil(length(R3) * rand(pop_size, 1));
    R3 = R3(R3rand);
end

function [R1, R2, R3] = Gained_Shared_Junior_R1R2R3(indBest)
    pop_size = length(indBest);
    R0 = 1:pop_size;
    R1 = zeros(1, pop_size);
    R2 = zeros(1, pop_size);
    for i = 1:pop_size
        ind = find(indBest == i);
        if (ind == 1)
            R1(i) = indBest(2);
            R2(i) = indBest(3);
        elseif (ind == pop_size)
            R1(i) = indBest(pop_size - 2);
            R2(i) = indBest(pop_size - 1);
        else
            R1(i) = indBest(ind - 1);
            R2(i) = indBest(ind + 1);
        end
    end
    R3 = floor(rand(1, pop_size) * pop_size) + 1;
    for i = 1:99999999
        pos = ((R3 == R2) | (R3 == R1) | (R3 == R0));
        if sum(pos) == 0
            break;
        else
            R3(pos) = floor(rand(1, sum(pos)) * pop_size) + 1;
        end
        if i > 1000
            error('Cannot generate R3 in 1000 iterations');
        end
    end
end

function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0, break; else, r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1; end
        if i > 1000, error('Cannot generate r1 in 1000 iterations'); end
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; else, r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1; end
        if i > 1000, error('Cannot generate r2 in 1000 iterations'); end
    end
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0, break; else, r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1; end
        if i > 1000, error('Cannot generate r3 in 1000 iterations'); end
    end
end

function x = han_boun(x, xmax, xmin, x2, PopSize, ~)
    hb = randi(3);
    switch hb
        case 1
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x(pos) = (x2(pos) + x_L(pos)) / 2;
            x_U = repmat(xmax, PopSize, 1);
            pos = x > x_U;
            x(pos) = (x2(pos) + x_U(pos)) / 2;
        case 2
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x_U = repmat(xmax, PopSize, 1);
            x(pos) = min(x_U(pos), max(x_L(pos), 2 * x_L(pos) - x2(pos)));
            pos = x > x_U;
            x(pos) = max(x_L(pos), min(x_U(pos), 2 * x_L(pos) - x2(pos)));
        case 3
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x_U = repmat(xmax, PopSize, 1);
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
            pos = x > x_U;
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
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
        temp_NP = floor(archive.NP);
        rndpos = rndpos(1:temp_NP);
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

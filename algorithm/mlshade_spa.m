% ----------------------------------------------------------------------- %
% Multi-population L-SHADE with Semi-Parameter Adaptation (MLSHADE-SPA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP     = 250     % Initial population (linear reduction to 20)
%   min_NP = 20
%   CC_nfes = max_nfes/50    % Evaluation budget of one cooperative cycle
%   memory_size = 5, p_best_rate = 0.1
%
% Algorithm Concept:
%   - A cooperative-coevolution framework alternating a global full-dimensional
%     LSHADE-SPA phase with a divide-and-conquer phase that splits the variables
%     randomly into three groups, one per DE variant:
%       LSHADE-SPA : current-to-pbest/1 with archive and semi-parameter
%                    adaptation (fixed F in the first half of the cycle)
%       ANDE       : triangular mutation over the best/middle/worst of three
%                    random individuals
%       EADE       : best/middle/worst directed mutation with sign-flipped F
%   - The three variants' budget is re-allocated every cycle in proportion to
%     their measured improvement (90 % inertia, 10 % news)
%   - MMTS refines the incumbent by modified multiple-trajectory local search,
%     and Cr_Adaptation runs an 11-value CR pool with a 21-generation counter
%
% Reference:
% Anas A. Hadi, Ali W. Mohamed, Kamal M. Jambi,
% LSHADE-SPA memetic framework for solving large-scale optimization problems,
% Complex & Intelligent Systems 5 (2019) 25-40.
% https://doi.org/10.1007/s40747-018-0086-8
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mlshade_spa(problem)

    D = problem.dimension;
    max_nfes = problem.maxFe;

    lu = [problem.lb; problem.ub];

    NP = 250;
    max_NP = NP;
    min_NP = 20.0;

    FE    = 0;
    curve = zeros(1, max_nfes);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    ctx = struct('problem', problem, 'maxFE', max_nfes);

    % Initialise
    Pop = repmat(lu(1, :), NP, 1) + rand(NP, D) .* (repmat(lu(2, :) - lu(1, :), NP, 1));
    [Fit, FE] = calculate_fitness(Pop', problem, FE);
    Fit = Fit(:);

    bsf = min(Fit);
    for eval_count = 1:min(NP, max_nfes)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Pop, Fit, population_history, fitness_history, ...
            history_index, max_nfes);
    end

    CC_nfes = max_nfes / 50;

    Par = struct();
    Par.stgCount   = zeros(NP, 1);
    Par.CRNewFlags = zeros(NP, 1);
    Par.CR         = zeros(NP, 1);
    Par.CRRatio    = zeros(11, 1);
    Par.memory_size = 5;
    Par.memory_sf   = 0.5 .* ones(Par.memory_size, 1);
    Par.memory_pos  = 1;

    archive.NP = NP;
    archive.Pop = [];
    archive.funvalues = [];

    flag = 0;
    All_Imp = [1 1 1] / 3;
    LSHADESPA_CC_nfes = 0; ANDE_CC_nfes = 0; EADE_CC_nfes = 0;

    % Main loop
    while FE < max_nfes
        FE_cycle_start = FE;

        Par.GenRatio = FE / max_nfes;

        EA_nfes = round(CC_nfes / 2);
        EA_nfes = EA_nfes - mod(EA_nfes, NP);

        MMTS_nfes = round(CC_nfes / 2);
        MMTS_nfes = MMTS_nfes - mod(MMTS_nfes, NP);

        % Global (full-dimensional) LSHADE-SPA phase
        CC_Group_Ind = ones(1, D);
        Alg_fit = round(0.5 * EA_nfes) - mod(round(0.5 * EA_nfes), NP);
        if Alg_fit + FE > max_nfes
            Alg_fit = max_nfes - FE;
        end
        [Pop, Fit, archive, Par, FE, bsf, curve, population_history, fitness_history, history_index] = ...
            CC_LSHADESPA(Alg_fit, Pop, Fit, lu, CC_Group_Ind == 1, archive, Par, ctx, FE, bsf, ...
                         curve, population_history, fitness_history, history_index);
        if FE >= max_nfes, break; end

        % Budget allocation over the three CC variants
        if flag == 0
            Alg_CC_nfes = (0.5 * EA_nfes / 3);
            LSHADESPA_CC_nfes = round(Alg_CC_nfes);
            ANDE_CC_nfes = round(Alg_CC_nfes);
            EADE_CC_nfes = 0.5 * EA_nfes - (LSHADESPA_CC_nfes + ANDE_CC_nfes);
            flag = 1;
        else
            LSHADESPA_CC_nfes = round(0.9 * LSHADESPA_CC_nfes + 0.1 * 0.5 * EA_nfes * All_Imp(1));
            ANDE_CC_nfes      = round(0.9 * ANDE_CC_nfes      + 0.1 * 0.5 * EA_nfes * All_Imp(2));
            EADE_CC_nfes      = 0.5 * EA_nfes - (LSHADESPA_CC_nfes + ANDE_CC_nfes);
        end

        Group_No = min(3, D);   % D dimensions cannot fill more than D non-empty groups
        CC_Group_Ind = ceil(Group_No * rand(1, D));
        tries = 0;
        while (length(unique(CC_Group_Ind)) ~= Group_No) && tries < 1000
            CC_Group_Ind = ceil(Group_No * rand(1, D));
            tries = tries + 1;
        end

        Fit0 = Fit;

        % group 1: LSHADE-SPA
        Alg_Group_Ind = (CC_Group_Ind == 1);
        LSHADESPA_Fit = Fit;   % a group left empty by the split contributes no improvement
        if any(Alg_Group_Ind)
            LSHADESPA_CC_nfes = LSHADESPA_CC_nfes - mod(LSHADESPA_CC_nfes, NP);
            if (LSHADESPA_CC_nfes + FE) > max_nfes
                LSHADESPA_CC_nfes = max_nfes - FE;
            end
            [Pop, LSHADESPA_Fit, archive, Par, FE, bsf, curve, population_history, fitness_history, history_index] = ...
                CC_LSHADESPA(LSHADESPA_CC_nfes, Pop, Fit, lu, Alg_Group_Ind, archive, Par, ctx, FE, bsf, ...
                             curve, population_history, fitness_history, history_index);
        end
        if FE >= max_nfes, Fit = LSHADESPA_Fit; break; end

        % group 2: ANDE
        Alg_Group_Ind = (CC_Group_Ind == 2);
        ANDE_Fit = LSHADESPA_Fit;
        if any(Alg_Group_Ind)
            ANDE_CC_nfes = ANDE_CC_nfes - mod(ANDE_CC_nfes, NP);
            if (ANDE_CC_nfes + FE) > max_nfes
                ANDE_CC_nfes = max_nfes - FE;
            end
            [Pop, ANDE_Fit, Par, FE, bsf, curve, population_history, fitness_history, history_index] = ...
                CC_ANDE(ANDE_CC_nfes, Pop, LSHADESPA_Fit, lu, Alg_Group_Ind, Par, ctx, FE, bsf, ...
                        curve, population_history, fitness_history, history_index);
        end
        if FE >= max_nfes, Fit = ANDE_Fit; break; end

        % group 3: EADE
        Alg_Group_Ind = (CC_Group_Ind == 3);
        EADE_Fit = ANDE_Fit;
        if any(Alg_Group_Ind)
            EADE_CC_nfes = EADE_CC_nfes - mod(EADE_CC_nfes, NP);
            if (EADE_CC_nfes + FE) > max_nfes
                EADE_CC_nfes = max_nfes - FE;
            end
            [Pop, EADE_Fit, Par, FE, bsf, curve, population_history, fitness_history, history_index] = ...
                CC_EADE(EADE_CC_nfes, Pop, ANDE_Fit, lu, Alg_Group_Ind, Par, ctx, FE, bsf, ...
                        curve, population_history, fitness_history, history_index);
        end
        if FE >= max_nfes, Fit = EADE_Fit; break; end

        % Improvement ratio of each variant
        Imp = zeros(NP, 3);
        Imp(:, 1) = Fit0 - LSHADESPA_Fit;
        Imp(:, 2) = LSHADESPA_Fit - ANDE_Fit;
        Imp(:, 3) = ANDE_Fit - EADE_Fit;
        All_Imp = sum(Imp) ./ NP;
        if (max(All_Imp) ~= 0)
            denom = [max(LSHADESPA_CC_nfes, 1), max(ANDE_CC_nfes, 1), max(EADE_CC_nfes, 1)];
            All_Imp = All_Imp ./ denom;
            All_Imp = All_Imp ./ sum(All_Imp);
            [~, Imp_Ind] = sort(All_Imp);
            for imp_i = 1:length(All_Imp) - 1
                All_Imp(Imp_Ind(imp_i)) = max(All_Imp(Imp_Ind(imp_i)), 0.1);
            end
            All_Imp(Imp_Ind(end)) = 1 - sum(All_Imp(Imp_Ind(1:end-1)));
        else
            All_Imp = ones(1, 3) / 3;
        end
        Fit = EADE_Fit;

        % MMTS local search
        MMTS_Group_Ind = true(1, D);
        if MMTS_nfes + FE > max_nfes
            MMTS_nfes = max_nfes - FE;
        end
        [Pop, Fit, FE, bsf, curve, population_history, fitness_history, history_index] = ...
            MMTS(MMTS_nfes, Pop, Fit, lu, MMTS_Group_Ind, ctx, FE, bsf, ...
                 curve, population_history, fitness_history, history_index);
        if FE >= max_nfes, break; end

        % Population size reduction
        plan_NP = round((((min_NP - max_NP) / (0.5 * max_nfes)) * FE) + max_NP);
        if NP > plan_NP
            reduction_ind_num = NP - plan_NP;
            if NP - reduction_ind_num < min_NP
                reduction_ind_num = NP - min_NP;
            end
            NP = NP - reduction_ind_num;
            for r = 1:reduction_ind_num
                [~, indBest] = sort(Fit, 'ascend');
                worst_ind = indBest(end);
                Pop(worst_ind, :) = [];
                Fit(worst_ind, :) = [];
                Par.stgCount(worst_ind)   = [];
                Par.CRNewFlags(worst_ind) = [];
                Par.CR(worst_ind)         = [];
            end
            archive.NP = NP;
            if size(archive.Pop, 1) > archive.NP
                rndpos = randperm(size(archive.Pop, 1));
                rndpos = rndpos(1:archive.NP);
                archive.Pop = archive.Pop(rndpos, :);
                archive.funvalues = archive.funvalues(rndpos, :);
            end
        end

        % Safety valve against an all-zero cycle split; never fires at the framework minimum of 1e5 FEs
        if FE == FE_cycle_start
            break;
        end
    end

    curve(min(FE, max_nfes):end) = bsf;

    [Best_fit, I_best] = min(Fit);
    best_fitness  = Best_fit;
    best_solution = Pop(I_best, :);
    if bsf < Best_fit
        best_fitness = bsf;
        best_solution = Pop(I_best, :);
        curve(min(FE, max_nfes):end) = bsf;
    end
end

% CC LSHADE-SPA
function [PopOld, Fit, archive, Par, FE, bsf, curve, ph, fh, hi] = ...
        CC_LSHADESPA(CC_nfes, Pop, Fit, lu, CC_Alg_Ind, archive, Par, ctx, FE, bsf, curve, ph, fh, hi)

    PopOld = Pop;
    Eval_Pop = Pop;
    luG = lu(:, CC_Alg_Ind);
    Pop = Pop(:, CC_Alg_Ind);
    [NP, D] = size(Pop);

    p_best_rate = 0.1;
    stgCount   = Par.stgCount;
    CRNewFlags = Par.CRNewFlags;
    CR         = Par.CR;
    CRRatio    = Par.CRRatio;
    memory_size = Par.memory_size;
    memory_sf   = Par.memory_sf;
    memory_pos  = Par.memory_pos;

    nfes = 0;
    while nfes < CC_nfes && FE < ctx.maxFE
        mem_rand_index = ceil(memory_size * rand(NP, 1));
        mu_sf = memory_sf(mem_rand_index);

        [A, CR, stgCount] = Cr_Adaptation(CRNewFlags, Par.GenRatio, stgCount, CRRatio, CR);

        [~, sorted_index] = sort(Fit, 'ascend');

        if (nfes <= CC_nfes / 2)
            sf = 0.45 + .1 * rand(NP, 1);
        else
            sf = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
        end
        sf = min(sf, 1);

        r0 = 1:NP;
        if (size(archive.Pop, 1) ~= 0)
            popAll = [Pop; archive.Pop(:, CC_Alg_Ind)];
        else
            popAll = Pop;
        end
        [r1, r2] = gnR1R2(NP, size(popAll, 1), r0);

        pNP = max(round(p_best_rate * NP), 2);
        randindex = ceil(rand(1, NP) .* pNP);
        randindex = max(1, randindex);
        pbest = Pop(sorted_index(randindex), :);

        X = Pop + sf(:, ones(1, D)) .* (pbest - Pop + Pop(r1, :) - popAll(r2, :));
        X = boundConstraint(X, Pop, luG);

        mask = rand(NP, D) > CR(:, ones(1, D));
        Rnd = ceil(D * rand(NP, 1));
        jrand = sub2ind([NP D], (1:NP)', Rnd);
        mask(jrand) = false;
        X(mask) = Pop(mask);

        Eval_Pop(:, CC_Alg_Ind) = X;
        [Child_Fit, FE] = calculate_fitness(Eval_Pop', ctx.problem, FE);
        Child_Fit = Child_Fit(:);
        nfes = nfes + NP;

        [bsf, curve, ph, fh, hi] = stampN(FE, ctx.maxFE, NP, min(bsf, min(Child_Fit)), ...
                                          curve, Eval_Pop, Child_Fit, ph, fh, hi);

        Fit_imp_inf = (Child_Fit <= Fit);
        goodF = sf(Fit_imp_inf);
        dif = abs(Fit - Child_Fit);
        dif_val = dif(Fit_imp_inf);

        Eval_Pop(:, CC_Alg_Ind) = Pop;
        archive = updateArchive(archive, Eval_Pop(Fit_imp_inf, :), Fit(Fit_imp_inf));

        if numel(goodF) > 0
            dif_val = dif_val / sum(dif_val);
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end

        CRNewFlags(Fit_imp_inf)  = 1;
        CRNewFlags(~Fit_imp_inf) = 0;

        val = 1 - Child_Fit ./ Fit;

        Pop(Fit_imp_inf, :) = X(Fit_imp_inf, :);
        Fit(Fit_imp_inf)    = Child_Fit(Fit_imp_inf);

        for j = 1:length(A)
            A_ind = A(j) == CR;
            CRRatio(j) = CRRatio(j) + sum(val(and(A_ind, Fit_imp_inf)));
        end
    end

    PopOld(:, CC_Alg_Ind) = Pop;
    Par.stgCount = stgCount; Par.CRNewFlags = CRNewFlags; Par.CR = CR; Par.CRRatio = CRRatio;
    Par.memory_size = memory_size; Par.memory_sf = memory_sf; Par.memory_pos = memory_pos;
end

% CC ANDE
function [PopOld, Fit, Par, FE, bsf, curve, ph, fh, hi] = ...
        CC_ANDE(CC_nfes, Pop, Fit, lu, CC_Alg_Ind, Par, ctx, FE, bsf, curve, ph, fh, hi)

    PopOld = Pop;
    Eval_Pop = Pop;
    luG = lu(:, CC_Alg_Ind);
    Pop = Pop(:, CC_Alg_Ind);
    [NP, D] = size(Pop);

    stgCount   = Par.stgCount;
    CRNewFlags = Par.CRNewFlags;
    CR         = Par.CR;
    CRRatio    = Par.CRRatio;

    nfes = 0;
    while nfes < CC_nfes && FE < ctx.maxFE
        [A, CR, stgCount] = Cr_Adaptation(CRNewFlags, Par.GenRatio, stgCount, CRRatio, CR);

        R = Gen_R(NP, 3);
        R(:, 1) = [];
        fr = Fit(R);
        [~, I] = sort(fr, 2);
        R_S = zeros(NP, 3);
        for i = 1:NP
            R_S(i, :) = R(i, I(i, :));
        end
        rb = R_S(:, 1); rm = R_S(:, 2); rw = R_S(:, 3);

        F = 0.20 + 0.6 * rand(NP, D);

        p1 = ones(NP, 1);
        p2 = 0.75 + 0.25 * rand(NP, 1);
        p3 = 0.50 + 0.25 * rand(NP, 1);
        p = [p1 p2 p3];
        w = p ./ repmat(sum(p, 2), 1, 3);

        w1 = repmat(w(:, 1), 1, D);
        w2 = repmat(w(:, 2), 1, D);
        w3 = repmat(w(:, 3), 1, D);

        X = w1 .* Pop(rb, :) + w2 .* Pop(rm, :) + w3 .* Pop(rw, :) + 2 * F .* (Pop(rb, :) - Pop(rw, :));
        X = boundConstraint(X, Pop, luG);

        mask = rand(NP, D) > CR(:, ones(1, D));
        Rnd = ceil(D * rand(NP, 1));
        jrand = sub2ind([NP D], (1:NP)', Rnd);
        mask(jrand) = false;
        X(mask) = Pop(mask);

        Eval_Pop(:, CC_Alg_Ind) = X;
        [Child_Fit, FE] = calculate_fitness(Eval_Pop', ctx.problem, FE);
        Child_Fit = Child_Fit(:);
        nfes = nfes + NP;

        [bsf, curve, ph, fh, hi] = stampN(FE, ctx.maxFE, NP, min(bsf, min(Child_Fit)), ...
                                          curve, Eval_Pop, Child_Fit, ph, fh, hi);

        Fit_imp_inf = (Child_Fit <= Fit);
        CRNewFlags(Fit_imp_inf)  = 1;
        CRNewFlags(~Fit_imp_inf) = 0;
        val = 1 - Child_Fit ./ Fit;

        Pop(Fit_imp_inf, :) = X(Fit_imp_inf, :);
        Fit(Fit_imp_inf)    = Child_Fit(Fit_imp_inf);

        for j = 1:length(A)
            A_ind = A(j) == CR;
            CRRatio(j) = CRRatio(j) + sum(val(and(A_ind, Fit_imp_inf)));
        end
    end

    PopOld(:, CC_Alg_Ind) = Pop;
    Par.stgCount = stgCount; Par.CRNewFlags = CRNewFlags; Par.CR = CR; Par.CRRatio = CRRatio;
end

% CC EADE
function [PopOld, Fit, Par, FE, bsf, curve, ph, fh, hi] = ...
        CC_EADE(CC_nfes, Pop, Fit, lu, CC_Alg_Ind, Par, ctx, FE, bsf, curve, ph, fh, hi)

    PopOld = Pop;
    Eval_Pop = Pop;
    luG = lu(:, CC_Alg_Ind);
    Pop = Pop(:, CC_Alg_Ind);
    [NP, D] = size(Pop);

    stgCount   = Par.stgCount;
    CRNewFlags = Par.CRNewFlags;
    CR         = Par.CR;
    CRRatio    = Par.CRRatio;

    nfes = 0;
    while nfes < CC_nfes && FE < ctx.maxFE
        [A, CR, stgCount] = Cr_Adaptation(CRNewFlags, Par.GenRatio, stgCount, CRRatio, CR);

        X = zeros(NP, D);
        mut_prop = rand(NP, 1) <= 0.5;

        r = genR_EADE(Fit);
        F1 = rand(NP, 1);
        F2 = rand(NP, 1);

        X(r(mut_prop, 1), :) = Pop(r(mut_prop, 4), :) + ...
            F1(mut_prop, ones(1, D)) .* (Pop(r(mut_prop, 2), :) - (Pop(r(mut_prop, 4), :))) + ...
            F2(mut_prop, ones(1, D)) .* ((Pop(r(mut_prop, 4), :)) - (Pop(r(mut_prop, 3), :)));

        r = Gen_R(NP, 4);
        temp = sum(~mut_prop);
        Fs = rand(temp, 1);
        Fs(Fs > 0.5) = -1 .* Fs(Fs > 0.5);
        F1(~mut_prop) = Fs;

        X(r(~mut_prop, 1), :) = Pop(r(~mut_prop, 4), :) + ...
            F1(~mut_prop, ones(1, D)) .* (Pop(r(~mut_prop, 2), :) - Pop(r(~mut_prop, 3), :));

        mask = rand(NP, D) > CR(:, ones(1, D));
        Rnd = ceil(D * rand(NP, 1));
        jrand = sub2ind([NP D], (1:NP)', Rnd);
        mask(jrand) = false;
        X(mask) = Pop(mask);

        Temp_Pop = repmat(luG(1, :), NP, 1) + rand(NP, D) .* (repmat(luG(2, :) - luG(1, :), NP, 1));
        xl = repmat(luG(1, :), NP, 1);
        pos = X < xl;
        X(pos) = Temp_Pop(pos);
        xu = repmat(luG(2, :), NP, 1);
        pos = X > xu;
        X(pos) = Temp_Pop(pos);

        Eval_Pop(:, CC_Alg_Ind) = X;
        [Child_Fit, FE] = calculate_fitness(Eval_Pop', ctx.problem, FE);
        Child_Fit = Child_Fit(:);
        nfes = nfes + NP;

        [bsf, curve, ph, fh, hi] = stampN(FE, ctx.maxFE, NP, min(bsf, min(Child_Fit)), ...
                                          curve, Eval_Pop, Child_Fit, ph, fh, hi);

        Fit_imp_inf = (Child_Fit <= Fit);
        CRNewFlags(Fit_imp_inf)  = 1;
        CRNewFlags(~Fit_imp_inf) = 0;
        val = 1 - Child_Fit ./ Fit;

        Pop(Fit_imp_inf, :) = X(Fit_imp_inf, :);
        Fit(Fit_imp_inf)    = Child_Fit(Fit_imp_inf);

        for j = 1:length(A)
            A_ind = A(j) == CR;
            CRRatio(j) = CRRatio(j) + sum(val(and(A_ind, Fit_imp_inf)));
        end
    end

    PopOld(:, CC_Alg_Ind) = Pop;
    Par.stgCount = stgCount; Par.CRNewFlags = CRNewFlags; Par.CR = CR; Par.CRRatio = CRRatio;
end

% MMTS local search
function [PopOld, Fit, FE, bsf, curve, ph, fh, hi] = ...
        MMTS(CC_nfes, Pop, Fit, lu, CC_Alg_Ind, ctx, FE, bsf, curve, ph, fh, hi)

    Lbound = lu(1, CC_Alg_Ind);
    Ubound = lu(2, CC_Alg_Ind);

    PopOld = Pop;
    Eval_Pop = Pop;
    Pop = Pop(:, CC_Alg_Ind);
    [~, D] = size(Pop);

    [~, in] = sort(Fit);
    LS_ind = in(1);
    Eval_Pop = Eval_Pop(LS_ind, :);
    LS_Pop = Pop(LS_ind, :);
    LS_Fit = Fit(LS_ind);
    LS_SR = (max(Pop, [], 1) - min(Pop, [], 1)) .* rand(1, D);
    LS_SR = min(LS_SR, 0.2 * (Ubound(1, 1:D) - Lbound(1, 1:D)));

    dimp = randperm(D);
    nfes = 0;
    LS_Imp_Flag = 1;

    while (nfes <= CC_nfes) && FE < ctx.maxFE
        LS_Last_Fit = LS_Fit;
        if LS_Imp_Flag == 0
            LS_SR = LS_SR .* rand(1, D);
        end
        for i = 1:D
            k = 0;
            LS_Flag = 1;
            while LS_Flag
                k = k + 1;
                LS_Child_pos = LS_Pop;
                LS_Child_pos(dimp(i)) = LS_Child_pos(dimp(i)) + k * LS_SR(dimp(i));
                if LS_Child_pos(dimp(i)) > Ubound(dimp(i)), break; end
                Eval_Pop(CC_Alg_Ind) = LS_Child_pos;
                [LS_Child_fit, FE] = calculate_fitness(Eval_Pop', ctx.problem, FE);
                nfes = 1 + nfes;
                % MMTS is single-point, so record PopOld, the population held frozen during the line search
                [bsf, curve, ph, fh, hi] = stampN(FE, ctx.maxFE, 1, min(bsf, LS_Child_fit), ...
                                                  curve, PopOld, Fit, ph, fh, hi);
                if nfes > CC_nfes || FE >= ctx.maxFE
                    Fit(LS_ind) = LS_Fit;
                    Pop(LS_ind, :) = LS_Pop;
                    PopOld(:, CC_Alg_Ind) = Pop;
                    return;
                end
                if LS_Child_fit <= LS_Fit
                    LS_Fit = LS_Child_fit;
                    LS_Pop = LS_Child_pos;
                else
                    LS_Flag = 0;
                end
            end
            if k <= 1
                k = 0;
                LS_Flag = 1;
                while LS_Flag
                    k = k + 1;
                    LS_Child_pos = LS_Pop;
                    LS_Child_pos(dimp(i)) = LS_Child_pos(dimp(i)) - k * LS_SR(dimp(i));
                    if LS_Child_pos(dimp(i)) < Lbound(dimp(i)), break; end
                    Eval_Pop(CC_Alg_Ind) = LS_Child_pos;
                    [LS_Child_fit, FE] = calculate_fitness(Eval_Pop', ctx.problem, FE);
                    nfes = 1 + nfes;
                    % Same as the forward step: record the held population, not the single LS probe
                    [bsf, curve, ph, fh, hi] = stampN(FE, ctx.maxFE, 1, min(bsf, LS_Child_fit), ...
                                                      curve, PopOld, Fit, ph, fh, hi);
                    if nfes > CC_nfes || FE >= ctx.maxFE
                        Fit(LS_ind) = LS_Fit;
                        Pop(LS_ind, :) = LS_Pop;
                        PopOld(:, CC_Alg_Ind) = Pop;
                        return;
                    end
                    if LS_Child_fit <= LS_Fit
                        LS_Fit = LS_Child_fit;
                        LS_Pop = LS_Child_pos;
                    else
                        LS_Flag = 0;
                    end
                end
            end
        end
        if LS_Last_Fit <= LS_Fit
            LS_Imp_Flag = 0;
        else
            LS_Imp_Flag = 1;
        end
    end

    Fit(LS_ind) = LS_Fit;
    Pop(LS_ind, :) = LS_Pop;
    PopOld(:, CC_Alg_Ind) = Pop;
end

% Helpers
function [A, CR, stgCount] = Cr_Adaptation(CRNewFlags, GenRatio, stgCount, CRRatio, CRs)
    if (GenRatio <= (1/10))
        if (GenRatio <= (1/60))
            A = [0.05 0.1];
        elseif (GenRatio <= (1/40))
            A = [0.05 0.1 0.2 0.3];
        elseif (GenRatio <= (1/30))
            A = [0.05 0.1 0.2 0.3 0.4 0.5];
        elseif (GenRatio <= (1/24))
            A = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7];
        elseif (GenRatio <= (1/20))
            A = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
        else
            A = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95];
        end
    else
        A = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95];
    end

    CR = CRs;
    CR_New_ind = (CRNewFlags == 0);

    if (sum(CR_New_ind) > 0)
        if (GenRatio <= (1/10))
            paraIndex = ceil(length(A) * rand(sum(CR_New_ind), 1));
            CR(CR_New_ind) = A(paraIndex);
        else
            stgCount(CR_New_ind) = stgCount(CR_New_ind) + 1;
            stgCount_ind = (stgCount == 21);
            paraIndex = ceil(length(A) * rand(sum(stgCount_ind), 1));
            stgCount(stgCount_ind) = 0;
            CR(stgCount_ind) = A(paraIndex);
        end
    end

    CR_Imp_ind = (CRNewFlags == 1);
    if (sum(CR_Imp_ind) > 0)
        % Guard: restrict the argmax to the active pool and force one index.
        nA = numel(A);
        [~, bestA] = max(abs(CRRatio(1:nA)));
        CR(CR_Imp_ind) = A(bestA(1));
    end
end

function R = Gen_R(NP_Size, N)
    R = zeros(N + 1, NP_Size);
    R(1, :) = 1:NP_Size;
    for i = 2:N + 1
        R(i, :) = ceil(rand(NP_Size, 1) * NP_Size);
        flag = 0;
        guard = 0;
        while flag ~= 1 && guard < 1000
            pos = (R(i, :) == R(1, :));
            for w = 2:i - 1
                pos = or(pos, (R(i, :) == R(w, :)));
            end
            if sum(pos) == 0
                flag = 1;
            else
                R(i, pos) = floor(rand(sum(pos), 1) * NP_Size) + 1;
            end
            guard = guard + 1;
        end
    end
    R = R';
end

function r = genR_EADE(Fit)
    NP = length(Fit);
    r = zeros(NP, 4);
    r(:, 1) = 1:NP;

    [~, Fit_index] = sort(Fit, 'ascend');
    T = ceil(length(Fit_index) / 10);
    Best   = Fit_index(1:T);
    Mid    = Fit_index(T + 1:end - T);
    Worest = Fit_index(end - T + 1:end);
    if isempty(Mid), Mid = Fit_index; end

    r(:, 2) = Best(ceil(length(Best) * rand(NP, 1)));
    r(:, 3) = Worest(ceil(length(Worest) * rand(NP, 1)));
    r(:, 4) = Mid(ceil(length(Mid) * rand(NP, 1)));

    pos = r(:, 2) == r(:, 3);
    guard = 0;
    while (sum(pos) ~= 0) && guard < 1000
        r(pos, 3) = Worest(ceil(length(Worest) * rand(sum(pos), 1)));
        pos = r(:, 2) == r(:, 3);
        guard = guard + 1;
    end

    pos = r(:, 3) == r(:, 4);
    guard = 0;
    while (sum(pos) ~= 0) && guard < 1000
        r(pos, 4) = Mid(ceil(length(Mid) * rand(sum(pos), 1)));
        pos = r(:, 3) == r(:, 4);
        guard = guard + 1;
    end
end

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

function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0, return; end
    popAll = [archive.Pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end
    if size(popAll, 1) <= archive.NP
        archive.Pop = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:ceil(archive.NP));
        archive.Pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

function vi = boundConstraint(vi, pop, lu)
    NP = size(pop, 1);
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [bsf, curve, ph, fh, hi] = stampN(FE, maxFE, n, bsf, curve, X, Fit, ph, fh, hi)
    for k = 1:n
        ec = FE - n + k;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

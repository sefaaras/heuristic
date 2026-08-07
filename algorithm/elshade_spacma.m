% ----------------------------------------------------------------------- %
% Enhanced L-SHADE with Semi-Parameter Adaptation and CMA-ES (ELSHADE-SPACMA)
% CEC 2018 competition -- 3rd place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP           = 18 * D          % Initial population (linear reduction to 4)
%   min_NP       = 4
%   arc_rate     = 1.4             % Archive size factor
%   memory_size  = 5, M_F = M_CR = 0.5
%   p_best_rate  = 0.3 -> 0.15     % Decays linearly over the budget
%   First_class_percentage = 0.5   % Initial DE-vs-CMA-ES sampling split
%   L_Rate       = 0.80            % Learning rate of the class probability
%
% Algorithm Concept:
%   - Each cycle runs TWO generations with different operators:
%     1. LSHADE-SPACMA: an adapted per-memory-slot probability assigns each
%        individual to class 1 (current-to-pbest/1 with archive) or class 2
%        (drawn straight from the CMA-ES model N(xmean, sigma^2 C)); the
%        probability follows each class's improvement, clamped to [0.2, 0.8]
%     2. EADE-SPA: triangular mutation x_mid + F*(x_best - x_worst) with donors
%        from the best 10%, middle 80% and worst 10%, and F ~ U(0,1)
%   - Semi-parameter adaptation: for the first half of the budget F is drawn
%     from the fixed band U(0.45, 0.55); only afterwards does it switch to the
%     success-history Cauchy sampling. Cr is success-history adapted throughout
%   - The CMA-ES model is updated from the population after every LSHADE-SPACMA
%     generation, and hybridisation switches itself off if it ever degenerates
%
% Reference:
% Anas A. Hadi, Ali W. Mohamed, Kamal M. Jambi,
% Single-Objective Real-Parameter Optimization: Enhanced LSHADE-SPACMA
% Algorithm, Technical Report / CEC 2018 competition entry; published as
% Heuristics for Optimization and Learning, Studies in Computational
% Intelligence 906, Springer, 2021, pp. 103-121.
% https://doi.org/10.1007/978-3-030-58930-1_7
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (ELSHADE_SPACMA.m and its seven
% helpers, "Codes for best 3" in Suganthan's CEC2018 repository). Three
% reference properties are kept: EADE-SPA builds a scaling factor through the
% full semi-parameter machinery but mutates with an independent F ~ U(0,1);
% a failed CMA-ES update sets Hybridization_flag = 1 where 0 was evidently
% meant; and selection uses "<=", so equal-fitness trials replace and count.
% Adaptations: an aborted generation does not advance the FE counter; evalPop
% clamps its block to the box; sigma and xmean are scaled to the box, which
% leaves the CEC value untouched but stops a wide off-centre box such as RC44
% sending sigma to Inf; and the eigen step floors the spectrum before inverting.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = elshade_spacma(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';      % 1 x D
    ub    = problem.ub(:)';      % 1 x D
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters
    NP       = 18 * D;
    max_NP   = NP;
    min_NP   = 4;
    arc_rate = 1.4;
    L_Rate   = 0.80;

    memory_size = 5;
    memory_sf   = 0.5 .* ones(memory_size, 1);
    memory_cr   = 0.5 .* ones(memory_size, 1);
    memory_pos  = 1;
    memory_1st  = 0.5 .* ones(memory_size, 1);

    p_best_rate_max = 0.3;
    p_best_rate_min = 0.15;
    p_best_rate     = p_best_rate_max;

    Hybridization_flag = 1;

    archive.NP        = arc_rate * NP;
    archive.Pop       = zeros(0, D);
    archive.funvalues = zeros(0, 1);

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    ph = [];  % record_history allocates the metric buffers on its first sample
    fh = [];
    hi = 1;

    % Initial population
    Pop = repmat(lb, NP, 1) + rand(NP, D) .* repmat(ub - lb, NP, 1);

    bsf  = inf;
    bsfx = Pop(1, :);
    [Fit, FE, bsf, bsfx, curve, ph, fh, hi] = ...
        evalPop(Pop, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

    % CMA-ES model
    cma = cmaesInit(NP, D, lb, ub);

    % Main loop: one LSHADE-SPACMA generation then one EADE-SPA generation
    while FE < maxFE

        % Generation A: LSHADE-SPACMA
        [cr, sf] = drawCrSf(memory_cr, memory_sf, memory_size, NP, FE, maxFE);

        mem_rand_index = ceil(memory_size * rand(NP, 1));
        mem_rand_ratio = rand(NP, 1);
        isClass1 = (memory_1st(mem_rand_index) >= mem_rand_ratio);
        if Hybridization_flag == 0
            isClass1(:) = true;    % everything falls back to the DE branch
        end

        [~, sorted_index] = sort(Fit, 'ascend');

        if size(archive.Pop, 1) ~= 0
            popAll = [Pop; archive.Pop];
        else
            popAll = Pop;
        end
        [r1, r2] = gnR1R2(NP, size(popAll, 1), 1:NP);

        pNP       = max(round(p_best_rate * NP), 2);
        randindex = max(1, ceil(rand(NP, 1) .* pNP));
        pbest     = Pop(sorted_index(randindex), :);

        X = zeros(NP, D);
        if any(isClass1)
            X(isClass1, :) = Pop(isClass1, :) + sf(isClass1, ones(1, D)) .* ...
                (pbest(isClass1, :) - Pop(isClass1, :) + Pop(r1(isClass1), :) - popAll(r2(isClass1), :));
        end
        nClass2 = sum(~isClass1);
        if nClass2 > 0
            temp = repmat(cma.xmean, 1, nClass2) + ...
                   cma.sigma * cma.B * (repmat(cma.diagD, 1, nClass2) .* randn(D, nClass2));
            X(~isClass1, :) = temp';
        end

        if ~isreal(X)
            % Reference guard: abandon the generation and stop hybridising, without consuming budget
            Hybridization_flag = 0;
            continue;
        end

        X = boundConstraint(X, Pop, lu);
        X = binomialCrossover(X, Pop, cr, NP, D);

        [Child_Fit, FE, bsf, bsfx, curve, ph, fh, hi] = ...
            evalPop(X, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

        imp = (Child_Fit <= Fit);
        dif = abs(Fit - Child_Fit);

        goodCR  = cr(imp);
        goodF   = sf(imp);
        dif_val = dif(imp);
        dif_c1  = dif(imp & isClass1);
        dif_c2  = dif(imp & ~isClass1);

        archive = updateArchive(archive, Pop(imp, :), Fit(imp));

        if ~isempty(goodF)
            dif_val = dif_val / sum(dif_val);

            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
            end

            if Hybridization_flag == 1
                memory_1st(memory_pos) = memory_1st(memory_pos) * L_Rate + ...
                    (1 - L_Rate) * (sum(dif_c1) / (sum(dif_c1) + sum(dif_c2)));
                memory_1st(memory_pos) = min(memory_1st(memory_pos), 0.8);
                memory_1st(memory_pos) = max(memory_1st(memory_pos), 0.2);
            end

            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end

        Pop(imp, :) = X(imp, :);
        Fit(imp)    = Child_Fit(imp);

        if Hybridization_flag == 1
            [cma, flag] = cmaesUpdate(FE, Fit, cma, Pop);
            if flag == 0
                % Reproduced verbatim: the reference re-enables hybridisation where 0 was evidently intended
                Hybridization_flag = 1;
            end
        end

        if FE >= maxFE, break; end

        % Generation B: EADE-SPA
        [cr, sf] = drawCrSf(memory_cr, memory_sf, memory_size, NP, FE, maxFE);

        % The reference mutates with an independent F ~ U(0,1); sf only feeds the memory update
        Fmut = rand(NP, 1);
        r    = genR_EADE(Fit);

        X = Pop(r(:, 3), :) + Fmut(:, ones(1, D)) .* (Pop(r(:, 1), :) - Pop(r(:, 2), :));
        X = boundConstraint(X, Pop, lu);
        X = binomialCrossover(X, Pop, cr, NP, D);

        [Child_Fit, FE, bsf, bsfx, curve, ph, fh, hi] = ...
            evalPop(X, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

        imp = (Child_Fit <= Fit);
        dif = abs(Fit - Child_Fit);

        goodCR  = cr(imp);
        goodF   = sf(imp);
        dif_val = dif(imp);

        archive = updateArchive(archive, Pop(imp, :), Fit(imp));

        if ~isempty(goodF)
            dif_val = dif_val / sum(dif_val);
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
            end
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end

        Pop(imp, :) = X(imp, :);
        Fit(imp)    = Child_Fit(imp);

        % Population size and p reduction
        plan_NP     = round((((min_NP - max_NP) / maxFE) * FE) + max_NP);
        p_best_rate = (((p_best_rate_min - p_best_rate_max) / maxFE) * FE) + p_best_rate_max;

        if NP > plan_NP
            reduction_ind_num = NP - plan_NP;
            if NP - reduction_ind_num < min_NP
                reduction_ind_num = NP - min_NP;
            end
            NP = NP - reduction_ind_num;
            for k = 1:reduction_ind_num
                [~, indBest] = sort(Fit, 'ascend');
                worst_ind = indBest(end);
                Pop(worst_ind, :) = [];
                Fit(worst_ind)    = [];
            end
            archive.NP = NP;
            if size(archive.Pop, 1) > archive.NP
                rndpos = randperm(size(archive.Pop, 1));
                rndpos = rndpos(1:floor(archive.NP));
                archive.Pop       = archive.Pop(rndpos, :);
                archive.funvalues = archive.funvalues(rndpos, :);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness       = bsf;
    best_solution      = bsfx;
    population_history = ph;
    fitness_history    = fh;
end

% Helper Functions

function [fit, FE, bsf, bsfx, curve, ph, fh, hi] = evalPop(X, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi)
% Evaluate an n-by-D block and update the best-so-far, the curve and history.
    X = min(max(X, problem.lb), problem.ub);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:);
    n = size(X, 1);
    for i = 1:n
        if fit(i) < bsf
            bsf  = fit(i);
            bsfx = X(i, :);
        end
        ec = FE - n + i;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, X, fit, ph, fh, hi, maxFE);
        end
    end
end

function [cr, sf] = drawCrSf(memory_cr, memory_sf, memory_size, NP, FE, maxFE)
% Success-history Cr; F is U(0.45,0.55) for the first half of the budget, then Cauchy
    mem_rand_index = ceil(memory_size * rand(NP, 1));
    mu_sf = memory_sf(mem_rand_index);
    mu_cr = memory_cr(mem_rand_index);

    cr = mu_cr + 0.1 * randn(NP, 1);
    cr(mu_cr == -1) = 0;
    cr = min(max(cr, 0), 1);

    if FE <= maxFE / 2
        sf  = 0.45 + 0.1 * rand(NP, 1);
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = 0.45 + 0.1 * rand(length(pos), 1);
            pos = find(sf <= 0);
        end
    else
        sf  = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end
    end
    sf = min(sf, 1);
end

function X = binomialCrossover(X, Pop, cr, NP, D)
% Binomial crossover with one forced dimension.
    mask = rand(NP, D) > cr(:, ones(1, D));
    Rnd  = ceil(D * rand(NP, 1));
    mask(sub2ind([NP D], (1:NP)', Rnd)) = false;
    X(mask) = Pop(mask);
end

function r = genR_EADE(Fit)
% Triangular donors: best 10 %, worst 10 %, middle; re-drawn until mutually distinct
    NP = length(Fit);
    [~, idx] = sort(Fit, 'ascend');

    T     = ceil(NP / 10);
    Best  = idx(1:T);
    Mid   = idx(T+1:end-T);
    Worst = idx(end-T+1:end);
    if isempty(Mid)
        Mid = idx;    % degenerate populations keep the middle non-empty
    end

    r = zeros(NP, 3);
    r(:, 1) = Best(max(ceil(length(Best)  * rand(NP, 1)), 1));
    r(:, 2) = Worst(max(ceil(length(Worst) * rand(NP, 1)), 1));
    r(:, 3) = Mid(max(ceil(length(Mid)   * rand(NP, 1)), 1));

    for it = 1:1000
        pos = (r(:, 1) == r(:, 2));
        if ~any(pos), break; end
        r(pos, 2) = Worst(max(ceil(length(Worst) * rand(sum(pos), 1)), 1));
    end
    for it = 1:1000
        pos = (r(:, 2) == r(:, 3));
        if ~any(pos), break; end
        r(pos, 3) = Mid(max(ceil(length(Mid) * rand(sum(pos), 1)), 1));
    end
end

function cma = cmaesInit(NP, D, lb, ub)
% sigma and xmean are scaled to the box; both are unchanged on the CEC one
    cma.sigma = 0.5 * mean(ub - lb) / 200;
    cma.xmean = ((lb + ub) / 2)' + rand(D, 1) .* ((ub - lb)' / 200);

    mu      = NP / 2;
    weights = log(mu + 1/2) - log(1:floor(mu))';
    weights = weights / sum(weights);
    mueff   = sum(weights) ^ 2 / sum(weights .^ 2);

    cma.cc    = (4 + mueff / D) / (D + 4 + 2 * mueff / D);
    cma.cs    = (mueff + 2) / (D + mueff + 5);
    cma.c1    = 2 / ((D + 1.3) ^ 2 + mueff);
    cma.cmu   = min(1 - cma.c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ^ 2 + mueff));
    cma.damps = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cma.cs;

    cma.pc        = zeros(D, 1);
    cma.ps        = zeros(D, 1);
    cma.B         = eye(D, D);
    cma.diagD     = ones(D, 1);
    cma.C         = cma.B * diag(cma.diagD .^ 2) * cma.B';
    cma.invsqrtC  = cma.B * diag(cma.diagD .^ -1) * cma.B';
    cma.eigeneval = 0;
    cma.chiN      = D ^ 0.5 * (1 - 1 / (4 * D) + 1 / (21 * D ^ 2));
end

function [cma, flag] = cmaesUpdate(nfes, fitness, cma, Pop)
% Rank-mu CMA-ES update from the DE population, in try/catch as the reference detects degeneracy
    flag = 1;
    try
        [NP, D] = size(Pop);

        mu      = floor(NP / 2);
        weights = log(NP / 2 + 1/2) - log(1:mu)';
        weights = weights / sum(weights);
        mueff   = sum(weights) ^ 2 / sum(weights .^ 2);

        [~, popindex] = sort(fitness);
        xold  = cma.xmean;
        xmean = Pop(popindex(1:mu), :)' * weights;

        ps = (1 - cma.cs) * cma.ps ...
             + sqrt(cma.cs * (2 - cma.cs) * mueff) * cma.invsqrtC * (xmean - xold) / cma.sigma;
        hsig = sum(ps .^ 2) / (1 - (1 - cma.cs) ^ (2 * nfes / NP)) / D < 2 + 4 / (D + 1);
        pc = (1 - cma.cc) * cma.pc ...
             + hsig * sqrt(cma.cc * (2 - cma.cc) * mueff) * (xmean - xold) / cma.sigma;

        artmp = (1 / cma.sigma) * (Pop(popindex(1:mu), :)' - repmat(xold, 1, mu));
        C = (1 - cma.c1 - cma.cmu) * cma.C ...
            + cma.c1 * (pc * pc' + (1 - hsig) * cma.cc * (2 - cma.cc) * cma.C) ...
            + cma.cmu * artmp * diag(weights) * artmp';

        sigma = cma.sigma * exp((cma.cs / cma.damps) * (norm(ps) / cma.chiN - 1));

        B     = cma.B;
        diagD = cma.diagD;
        invsqrtC  = cma.invsqrtC;
        eigeneval = cma.eigeneval;

        if nfes - eigeneval > NP / (cma.c1 + cma.cmu) / D / 10
            eigeneval = nfes;
            C = triu(C) + triu(C, 1)';
            if any(isnan(C(:))) || any(~isfinite(C(:))) || ~isreal(C)
                flag = 0;
                return;
            end
            [B, Dm] = eig(C);
            e    = real(diag(Dm));
            emax = max(e);
            if ~isfinite(emax) || emax <= 0
                C = eye(D); B = eye(D); e = ones(D, 1);   % fully degenerate -> reset
            else
                e = max(e, emax * 1e-14);   % floor the spectrum so the inverse root stays real
            end
            B        = real(B);
            diagD    = sqrt(e);
            invsqrtC = B * diag(diagD .^ -1) * B';
        end

        cma.xmean     = xmean;
        cma.ps        = ps;
        cma.pc        = pc;
        cma.C         = C;
        cma.sigma     = sigma;
        cma.B         = B;
        cma.diagD     = diagD;
        cma.invsqrtC  = invsqrtC;
        cma.eigeneval = eigeneval;
    catch
        flag = 0;
    end
end

function vi = boundConstraint(vi, pop, lu)
% Violating component moved to the parent/bound midpoint
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
        if sum(pos) == 0, break; end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        if i == 1000
            error('elshade_spacma:r1', 'Cannot generate r1 in 1000 iterations');
        end
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:1000
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        if i == 1000
            error('elshade_spacma:r2', 'Cannot generate r2 in 1000 iterations');
        end
    end
end

function archive = updateArchive(archive, pop, funvalue)
% Append refused parents, drop duplicates, then randomly trim to the cap.
    if archive.NP == 0, return; end

    popAll    = [archive.Pop; pop];
    funvalues = [archive.funvalues; funvalue(:)];

    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll    = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end

    if size(popAll, 1) <= archive.NP
        archive.Pop       = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:floor(archive.NP));
        archive.Pop       = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

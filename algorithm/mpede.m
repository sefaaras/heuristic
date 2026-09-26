% ----------------------------------------------------------------------- %
% Multi-Population Ensemble Differential Evolution (MPEDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP      = 250               % Whole population, re-split every generation
%   lambda  = 0.2               % Share of NP in each strategy's indicator subpopulation
%   ng      = 20                % Generations between reward reassignments
%   c       = 0.1               % Adaptation rate of each strategy's CRm and Fm
%   p       = 0.04              % Greediness of the pbest term (at least 2 members)
%   Afactor = 1                 % Archive holds up to Afactor*NP defeated parents
%
% Algorithm Concept:
%   - Three strategies run side by side: current-to-pbest/1/bin with an archive,
%     current-to-rand/1 without crossover, and rand/1/bin
%   - Each generation the population is shuffled into three indicator subpopulations
%     of lambda*NP; the remaining (1 - 3*lambda)*NP join the rewarded strategy
%   - Every ng generations the reward goes to the strategy with the largest
%     fitness improvement per evaluation spent since the last assignment
%   - Each strategy adapts its own CR ~ N(CRm, 0.1) and F ~ Cauchy(Fm, 0.1),
%     moving CRm to the mean and Fm to the Lehmer mean of successful values
%   - The worst member of each subpopulation is temporarily replaced by the
%     global best, and restored unless that slot's trial improved on it
%   - Violating components are set midway between the parent and the bound
%
% Reference:
% Guohua Wu, Rammohan Mallipeddi, Ponnuthurai N. Suganthan, Rui Wang,
% Huangke Chen,
% Differential evolution with multi-population based ensemble of mutation
% strategies,
% Information Sciences 329 (2016) 329-345.
% https://doi.org/10.1016/j.ins.2015.09.009
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (MPEDE.m in P-N-Suganthan/CODES,
% 2016-INS-MPEDE; MPEDE_LR.m there is a population-reduction variant, not this
% algorithm). Each strategy's subpopulation is evaluated as one batch, and the
% last batch is cut to the evaluations the budget still allows, so FE >= maxFe is
% the only terminator. The reference switches bound handling off for two CEC2005
% functions whose optimum lies outside the initial range; here it is always on.
% As in the reference, the archive stores defeated parents with the offspring's
% fitness; that field is never read back.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mpede(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    NP      = 250;
    lambda  = 0.2;
    ng      = 20;
    c       = 1 / 10;
    pj      = 0.04;
    Afactor = 1;
    nInd    = round(lambda * NP);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    mixPop = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(ub - lb, NP, 1);
    [mixVal, FE] = calculate_fitness(mixPop', problem, FE);
    mixVal = mixVal(:);

    bsf          = inf;
    bsf_solution = mixPop(1, :);
    for i = 1:NP
        if mixVal(i) < bsf
            bsf          = mixVal(i);
            bsf_solution = mixPop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, mixPop, mixVal, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    CRm    = [0.5, 0.5, 0.5];
    Fm     = [0.5, 0.5, 0.5];
    goodCR = {[], [], []};
    goodF  = {[], [], []};

    archive.NP        = Afactor * NP;
    archive.pop       = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    indexBestLN = 1;
    gbestChange = [1, 1, 1];     % fitness improvement per strategy since the last reassignment
    consumedFES = [1, 1, 1];     % trials per strategy over the same span
    gen = 0;

    while FE < maxFE
        gen = gen + 1;

        if mod(gen, ng) == 0
            rate = gbestChange ./ consumedFES;
            [~, indexBestLN] = max(rate);
            if sum(rate == rate(1)) == 3
                indexBestLN = randi(3);
            end
            gbestChange = [0.1, 0.1, 0.1];
            consumedFES = [1, 1, 1];
        end

        % The rewarded strategy takes the third block, which holds the reward share
        perm = randperm(NP);
        B1 = perm(1:nInd);
        B2 = perm(nInd+1:2*nInd);
        B3 = perm(2*nInd+1:end);
        switch indexBestLN
            case 1
                groups = {B3, B2, B1};
            case 2
                groups = {B2, B3, B1};
            otherwise
                groups = {B1, B2, B3};
        end
        consumedFES = consumedFES + cellfun(@numel, groups);

        for s = 1:3
            if FE >= maxFE
                break;
            end
            idx = groups{s};
            ps  = numel(idx);
            pop = mixPop(idx, :);
            val = mixVal(idx);

            % The subpopulation's worst slot carries the global best for this generation
            [~, I1] = sort(mixVal, 'ascend');
            [~, I2] = sort(val, 'descend');
            w = I2(1);
            pop(w, :) = mixPop(I1(1), :);
            val(w)    = mixVal(I1(1));
            preval    = val;

            if ~isempty(goodCR{s}) && sum(goodF{s}) > 0
                CRm(s) = (1 - c) * CRm(s) + c * mean(goodCR{s});
                Fm(s)  = (1 - c) * Fm(s) + c * sum(goodF{s} .^ 2) / sum(goodF{s});   % Lehmer mean
            end
            [F, CR] = randFCR(ps, CRm(s), 0.1, Fm(s), 0.1);

            if s == 1
                % current-to-pbest/1/bin with the archive
                popAll = [pop; archive.pop];
                [r1, r2] = gnR1R2(ps, size(popAll, 1), 1:ps);
                [~, indBest] = sort(val, 'ascend');
                pNP = max(round(pj * ps), 2);
                randindex = max(1, ceil(rand(1, ps) * pNP));
                pbest = pop(indBest(randindex), :);
                vi = pop + F(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
                vi = boundConstraint(vi, pop, lu);
                ui = binomial(vi, pop, CR);
            elseif s == 2
                % current-to-rand/1, no crossover
                [pm1, pm2, pm3] = rotated_triplet(pop);
                vi = pop + repmat(rand(ps, 1), 1, dim) .* (pm1 - pop) + F(:, ones(1, dim)) .* (pm2 - pm3);
                ui = boundConstraint(vi, pop, lu);
            else
                % rand/1/bin
                [pm1, pm2, pm3] = rotated_triplet(pop);
                vi = pm1 + F(:, ones(1, dim)) .* (pm2 - pm3);
                vi = boundConstraint(vi, pop, lu);
                ui = binomial(vi, pop, CR);
            end

            nEval = min(ps, maxFE - FE);
            FE0 = FE;
            [fv, FE] = calculate_fitness(ui(1:nEval, :)', problem, FE);
            fv = fv(:);

            for j = 1:nEval
                if fv(j) < bsf
                    bsf          = fv(j);
                    bsf_solution = ui(j, :);
                end
                curve(FE0 + j) = bsf;
            end

            % I == 2 where the trial wins; trials the budget cut off are never made
            I = ones(ps, 1);
            [val(1:nEval), I(1:nEval)] = min([val(1:nEval), fv], [], 2);

            if s == 1
                archive = updateArchive(archive, pop(I == 2, :), val(I == 2));
            end
            newpop = pop;
            newpop(I == 2, :) = ui(I == 2, :);
            goodCR{s} = CR(I == 2);
            goodF{s}  = F(I == 2);
            gbestChange(s) = gbestChange(s) + sum(preval - val);

            if preval(w) == val(w)
                newpop(w, :) = mixPop(idx(w), :);
                val(w)       = mixVal(idx(w));
            end
            mixPop(idx, :) = newpop;
            mixVal(idx)    = val;

            for ec = FE0 + 1:FE
                [population_history, fitness_history, history_index] = record_history(...
                    ec, mixPop, mixVal, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [F, CR] = randFCR(NP, CRm, CRsigma, Fm, Fsigma)
% CR ~ N(CRm, CRsigma) truncated to [0,1]; F ~ Cauchy(Fm, Fsigma) capped at 1, redrawn if <= 0
    CR = CRm + CRsigma * randn(NP, 1);
    CR = min(1, max(0, CR));

    F = Fm + Fsigma * tan(pi * (rand(NP, 1) - 0.5));
    F = min(1, F);
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = Fm + Fsigma * tan(pi * (rand(length(pos), 1) - 0.5));
        F = min(1, F);
        pos = find(F <= 0);
    end
end

function ui = binomial(vi, pop, CR)
% One coordinate per row always comes from the donor
    [ps, dim] = size(pop);
    mask  = rand(ps, dim) > CR(:, ones(1, dim));
    rows  = (1:ps)';
    cols  = floor(rand(ps, 1) * dim) + 1;
    jrand = sub2ind([ps dim], rows, cols);
    mask(jrand) = false;
    ui = vi;
    ui(mask) = pop(mask);
end

function [pm1, pm2, pm3] = rotated_triplet(pop)
% One permutation rotated by 0, ind(1) and 3 places: three distinct rows when ps >= 4
    ps  = size(pop, 1);
    rot = 0:ps-1;
    ind = randperm(2);
    a1  = randperm(ps);
    rt  = rem(rot + ind(1), ps);
    a2  = a1(rt + 1);
    rt  = rem(rot + ind(2), ps);
    a3  = a2(rt + 1);
    pm1 = pop(a1, :);
    pm2 = pop(a2, :);
    pm3 = pop(a3, :);
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
% r1 ~= r0 from the subpopulation, r2 ~= r1, r0 from subpopulation plus archive
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
% Append, drop duplicates, then randomly thin down to archive.NP
    if archive.NP == 0, return; end

    popAll    = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX]   = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll    = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end

    if size(popAll, 1) <= archive.NP
        archive.pop       = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:archive.NP);
        archive.pop       = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

% ----------------------------------------------------------------------- %
% NL-SHADE-RSP with Midpoint (NL-SHADE-RSP-MID)
% CEC 2022 competition -- 3rd place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NInds      = 5*D initially, 400 after every restart
%   NIndsMin   = 4                   % Floor of the reduction schedule
%   MemorySize = 20*D                % Historical memory for Cr and F
%   M_F = M_CR = 0.2                 % Initial contents, reset by each restart
%   ArchiveSizeParam = 2.1           % Archive size factor
%   min_pop    = 20                  % Floor of the post-restart schedule
%   max_trials = 100                 % Re-sampling attempts per infeasible trial
%   stag_gens  = 8, bound_gens = 9   % Restart triggers
%
% Algorithm Concept:
%   - NL-SHADE-RSP is the base: non-linear population reduction, Cr sorted and
%     re-assigned by fitness rank, archive/population donor mixing
%   - Midpoint: each generation the trial population is split by 2-means and
%     every centroid is evaluated as an extra candidate, replacing the trial
%     nearest to it when better
%   - A silhouette below 1/(4*sqrt(D)), or fewer than 20 individuals, falls
%     back to a single candidate at the overall mean
%   - Trials leaving the box are re-generated whole rather than repaired, up to
%     100 times, redrawing F, Cr and the crossover type after ten failures
%   - Restart when the trial mean moves less than 1e-9 for 8 generations, or an
%     individual sits on a bound for 9; a restart re-samples 400 individuals
%     and switches to a hyperbolic size schedule running down to 20
%
% Reference:
% Rafal Biedrzycki, Jaroslaw Arabas, Eryk Warchulski,
% A Version of NL-SHADE-RSP Algorithm with Midpoint for CEC 2022 Single
% Objective Bound Constrained Problems,
% 2022 IEEE Congress on Evolutionary Computation (CEC), 2022, pp. 1-8.
% https://doi.org/10.1109/CEC55065.2022.9870220
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' competition release (nl_shade_rsp_mid.cpp in
% 2022-SO-BO/Codes_of_Top_ranked_Algorithm). Four deviations: the release ends a
% cycle as soon as the known optimum is reached and the harness has no optimum,
% so only the budget does; mlpack's k-means and silhouette are re-implemented
% (Lloyd's from a random two-point seed) as only k = 2 is ever requested; the
% re-sampling redraw is per individual, where the release writes the redrawn Cr
% into the shared sorted array at the wrong index and flips the generation-wide
% crossover flag, both leaking into later individuals; and zero-width dimensions
% are left out of the on-bound test, which would otherwise restart forever on a
% fixed variable. The stagnation reference starts at +Inf, uninitialised there.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nlshade_rsp_mid(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters (main() and MainCycle() of the reference release)
    NIndsMin           = 4;
    MemorySize         = 20 * dim;
    ArchiveSizeParam   = 2.1;
    RestartInds        = 400;
    MinRestartPop      = 20;
    ShapeConst         = 0.1;
    MaxTrials          = 100;
    TrialsBeforeRedraw = 10;
    BoundGens          = 9;
    StagGens           = 8;
    StagDist           = 1e-9;
    MinSplitPop        = 20;
    MinSilhouette      = 1 / (4 * sqrt(dim));

    % lb == ub dimensions are left out: every point there reads as on-bound
    wide = span > 0;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf          = inf;
    bsf_solution = lb + 0.5 * span;

    NIndsMax  = 5 * dim;   % every later cycle starts from RestartInds instead
    isRestart = false;

    % Each pass is one MainCycle of the reference; it ends on a restart trigger
    while FE < maxFE
        evalsAtStart = FE;

        NInds              = NIndsMax;
        ArchiveCapacity    = NIndsMax * ceil(ArchiveSizeParam);
        ArchiveSize        = floor(NIndsMax * ArchiveSizeParam);
        CurrentArchiveSize = 0;
        Archive            = zeros(ArchiveCapacity, dim);

        MemoryCr   = 0.2 * ones(MemorySize, 1);
        MemoryF    = 0.2 * ones(MemorySize, 1);
        MemoryIter = 1;
        ArchProbs  = 0.5;

        % Per slot, not per individual: the shrink step moves individuals, not counters
        boundCount = zeros(NIndsMax, 1);
        stagCount  = 0;
        meanOld    = inf(1, dim);

        Popul = repmat(lb, NIndsMax, 1) + rand(NIndsMax, dim) .* repmat(span, NIndsMax, 1);
        [FitMass, FE] = calculate_fitness(Popul', problem, FE);
        FitMass = FitMass(:);

        for i = 1:NIndsMax
            if FitMass(i) < bsf
                bsf          = FitMass(i);
                bsf_solution = Popul(i, :);
            end
            ec = evalsAtStart + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Popul, FitMass, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        restartNow = false;
        while FE < maxFE && ~restartNow
            % Fitness ranking: Indexes maps rank -> individual, BackIndexes the reverse
            [~, Indexes] = sort(FitMass(1:NInds), 'ascend');
            BackIndexes  = zeros(NInds, 1);
            BackIndexes(Indexes) = (1:NInds)';

            rankW = exp(-(0:NInds-1)' / NInds);
            rankC = cumsum(rankW / sum(rankW));

            psizeval = max(2, floor(NInds * (0.2 / maxFE * FE + 0.2)));
            psizeval = min(psizeval, NInds);

            % Drawn once per generation; the re-sampling path redraws it per individual
            expo = repmat(rand() < 0.5, NInds, 1);

            memIdx  = randi(MemorySize, NInds, 1);
            CrDrawn = min(1, max(0, MemoryCr(memIdx) + 0.1 * randn(NInds, 1)));
            FGen    = drawF(MemoryF(memIdx));

            % Rank-based selective pressure: sorted Cr re-assigned by fitness rank
            CrDrawn = sort(CrDrawn, 'ascend');
            CrUsed  = CrDrawn(BackIndexes);

            % The binomial branch tests against the budget ramp, not the adapted Cr
            if FE > 0.5 * maxFE
                CrToUse = (FE / maxFE - 0.5) * 2;
            else
                CrToUse = 0;
            end

            PoolAll = [Popul(1:NInds, :); Archive(1:CurrentArchiveSize, :)];
            Trial   = Popul(1:NInds, :);
            useArch = false(NInds, 1);

            % Re-sampling: attempt 1 is the initial build, 2..101 the retries
            rows    = (1:NInds)';
            attempt = 1;
            while true
                if attempt > TrialsBeforeRedraw + 1
                    m = randi(MemorySize, numel(rows), 1);
                    CrUsed(rows) = min(1, max(0, MemoryCr(m) + 0.1 * randn(numel(rows), 1)));
                    FGen(rows)   = drawF(MemoryF(m));
                    expo(rows)   = rand(numel(rows), 1) < 0.5;
                end

                [Trial(rows, :), useArch(rows)] = makeTrial(rows, PoolAll, NInds, ...
                    CurrentArchiveSize, Indexes, rankC, psizeval, FGen, CrUsed, ...
                    expo, ArchProbs, CrToUse, dim);

                bad  = any(Trial(rows, :) < lb | Trial(rows, :) > ub, 2);
                rows = rows(bad);
                if isempty(rows) || attempt > MaxTrials
                    break;
                end
                attempt = attempt + 1;
            end

            % Whatever is still infeasible falls back to a per-component re-sample
            if ~isempty(rows)
                sub = Trial(rows, :);
                oob = sub < lb | sub > ub;
                R   = repmat(lb, numel(rows), 1) + rand(numel(rows), dim) .* repmat(span, numel(rows), 1);
                sub(oob)       = R(oob);
                Trial(rows, :) = sub;
            end

            onBound = any(Trial(:, wide) <= lb(wide) | Trial(:, wide) >= ub(wide), 2);
            boundCount(1:NInds) = (boundCount(1:NInds) + 1) .* onBound;
            if any(boundCount(1:NInds) > BoundGens)
                break;   % restart: an individual has sat on a bound too long
            end

            [FitTemp, FE] = calculate_fitness(Trial', problem, FE);
            FitTemp = FitTemp(:);

            for i = 1:NInds
                if FitTemp(i) < bsf
                    bsf          = FitTemp(i);
                    bsf_solution = Trial(i, :);
                end
                ec = FE - NInds + i;
                if ec >= 1 && ec <= maxFE
                    curve(ec) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, Popul(1:NInds, :), FitMass(1:NInds), population_history, ...
                        fitness_history, history_index, maxFE);
                end
            end

            % Midpoint candidates: 2-means centroids, or the overall mean when no split
            Cand = [];
            if NInds >= MinSplitPop
                [cent, sil] = twoMeans(Trial(1:NInds, :));
                if sil > MinSilhouette
                    Cand = cent;
                end
            end
            if isempty(Cand)
                Cand = mean(Trial(1:NInds, :), 1);
            end

            nCand = size(Cand, 1);
            oob   = Cand < lb | Cand > ub;
            if any(oob(:))
                R = repmat(lb, nCand, 1) + rand(nCand, dim) .* repmat(span, nCand, 1);
                Cand(oob) = R(oob);
            end

            [CandFit, FE] = calculate_fitness(Cand', problem, FE);
            CandFit = CandFit(:);

            for k = 1:nCand
                if CandFit(k) < bsf
                    bsf          = CandFit(k);
                    bsf_solution = Cand(k, :);
                end
                ec = FE - nCand + k;
                if ec >= 1 && ec <= maxFE
                    curve(ec) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, Popul(1:NInds, :), FitMass(1:NInds), population_history, ...
                        fitness_history, history_index, maxFE);
                end
                % The candidate displaces the trial nearest to it, when it is better
                [~, nearest] = min(sum((Trial(1:NInds, :) - Cand(k, :)) .^ 2, 2));
                if CandFit(k) < FitTemp(nearest)
                    FitTemp(nearest)  = CandFit(k);
                    Trial(nearest, :) = Cand(k, :);
                end
            end

            % Stagnation is measured on the trial mean, before selection runs
            meanNow = mean(Trial(1:NInds, :), 1);
            if sqrt(sum((meanNow - meanOld) .^ 2)) < StagDist
                stagCount = stagCount + 1;
                if stagCount > StagGens
                    restartNow = true;
                end
            else
                stagCount = 0;
            end
            meanOld = meanNow;
            if restartNow
                break;
            end

            % Success uses "<" while replacement uses "<=", as in the reference
            succ          = (FitTemp < FitMass(1:NInds));
            SuccessFilled = sum(succ);
            tempSuccessCr = CrUsed(succ);
            tempSuccessF  = FGen(succ);
            FitDelta      = abs(FitMass(succ) - FitTemp(succ));

            repl = (FitTemp <= FitMass(1:NInds));

            parentFit = FitMass(1:NInds);
            rel       = zeros(NInds, 1);
            usable    = repl & isfinite(parentFit) & isfinite(FitTemp) & (parentFit ~= 0);
            rel(usable) = (parentFit(usable) - FitTemp(usable)) ./ parentFit(usable);

            NArchUsages   = sum(repl & useArch);
            ArchSuccess   = sum(rel(repl & useArch));
            NoArchSuccess = sum(rel(repl & ~useArch));

            if any(repl)
                [Archive, CurrentArchiveSize] = copyToArchive(Archive, CurrentArchiveSize, ...
                                                              ArchiveSize, Popul(repl, :));
                Popul(repl, :) = Trial(repl, :);
                FitMass(repl)  = FitTemp(repl);
            end

            if NArchUsages ~= 0
                ArchSuccess   = ArchSuccess / NArchUsages;
                NoArchSuccess = NoArchSuccess / max(NInds - NArchUsages, 1);
                ArchProbs     = ArchSuccess / (ArchSuccess + NoArchSuccess);
                ArchProbs     = max(0.1, min(0.9, ArchProbs));
                if ArchSuccess == 0 || ~isfinite(ArchProbs)
                    ArchProbs = 0.5;
                end
            else
                ArchProbs = 0.5;
            end

            % Hyperbolic size after a restart, non-linear before; the archive always non-linear
            r  = FE / maxFE;
            nl = round((NIndsMin - NIndsMax) * r ^ (1 - r) + NIndsMax);
            if isRestart
                newNInds = hyperbolicPop(NInds, FE, evalsAtStart, maxFE, MinRestartPop, ShapeConst);
            else
                newNInds = nl;
            end
            newNInds = min(max(newNInds, NIndsMin), NIndsMax);

            newArchSize = floor(nl * ArchiveSizeParam);
            if newArchSize < NIndsMin
                newArchSize = NIndsMin;
            end
            ArchiveSize = min(newArchSize, ArchiveCapacity);
            if CurrentArchiveSize >= ArchiveSize
                CurrentArchiveSize = ArchiveSize;
            end

            if newNInds < NInds
                for L = 1:(NInds - newNInds)
                    [~, WorstNum] = max(FitMass(1:NInds));
                    Popul(WorstNum:NInds-1, :) = Popul(WorstNum+1:NInds, :);
                    FitMass(WorstNum:NInds-1)  = FitMass(WorstNum+1:NInds);
                end
                NInds = newNInds;
            end

            if SuccessFilled ~= 0
                MemoryCr(MemoryIter) = meanWLgeneral(tempSuccessCr, FitDelta);
                MemoryF(MemoryIter)  = meanWLgeneral(tempSuccessF,  FitDelta);
                MemoryIter = mod(MemoryIter, MemorySize) + 1;
            else
                % The reference does NOT advance the memory pointer here
                MemoryCr(MemoryIter) = 0.5;
                MemoryF(MemoryIter)  = 0.5;
            end
        end

        NIndsMax  = RestartInds;
        isRestart = true;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function [T, useArch] = makeTrial(rows, PoolAll, NInds, CurArch, Indexes, rankC, ...
                                  psizeval, FGen, CrUsed, expo, ArchProbs, CrToUse, dim)
% One mutation and crossover pass over the rows that still need a feasible trial.
    n      = numel(rows);
    parent = PoolAll(rows, :);

    R0 = Indexes(randi(psizeval, n, 1));
    for it = 1:25
        clash = (R0 == rows);
        if ~any(clash), break; end
        R0(clash) = Indexes(randi(psizeval, sum(clash), 1));
    end

    R1 = randi(NInds, n, 1);
    for it = 1:25
        clash = (R1 == R0);
        if ~any(clash), break; end
        R1(clash) = randi(NInds, sum(clash), 1);
    end

    useArch = (rand(n, 1) <= ArchProbs) & (CurArch > 0);

    R2   = zeros(n, 1);
    nPop = find(~useArch);
    if ~isempty(nPop)
        R2(nPop) = Indexes(rouletteN(rankC, numel(nPop)));
        for it = 1:25
            clash = nPop((R2(nPop) == rows(nPop)) | (R2(nPop) == R0(nPop)) | (R2(nPop) == R1(nPop)));
            if isempty(clash), break; end
            R2(clash) = Indexes(rouletteN(rankC, numel(clash)));
        end
    end
    arc = find(useArch);
    if ~isempty(arc)
        R2(arc) = NInds + randi(CurArch, numel(arc), 1);
    end

    Fcol  = FGen(rows) * ones(1, dim);
    Donor = parent + Fcol .* (PoolAll(R0, :) - parent) + Fcol .* (PoolAll(R1, :) - PoolAll(R2, :));

    T      = parent;
    forced = randi(dim, n, 1);
    bin    = ~expo(rows);

    idx = find(bin);
    if ~isempty(idx)
        take = rand(numel(idx), dim) < CrToUse;
        take(sub2ind([numel(idx) dim], (1:numel(idx))', forced(idx))) = true;
        sub = T(idx, :);
        dn  = Donor(idx, :);
        sub(take) = dn(take);
        T(idx, :) = sub;
    end

    idx = find(~bin);
    if ~isempty(idx)
        % Exponential crossover: the geometric segment length in closed form
        cr       = CrUsed(rows(idx));
        StartLoc = randi(dim, numel(idx), 1);
        u        = rand(numel(idx), 1);
        extra    = zeros(numel(idx), 1);
        growing  = (cr > 0) & (cr < 1);
        extra(growing) = floor(log(u(growing)) ./ log(cr(growing)));
        extra(cr >= 1) = dim;
        EndLoc = min(StartLoc + extra, dim);

        cols = repmat(1:dim, numel(idx), 1);
        take = (cols >= StartLoc) & (cols <= EndLoc);
        sub  = T(idx, :);
        dn   = Donor(idx, :);
        sub(take) = dn(take);
        T(idx, :) = sub;
    end
end

function [cent, sil] = twoMeans(X)
% Lloyd's 2-means from a random two-point seed, plus the overall silhouette score
    n      = size(X, 1);
    sel    = randperm(n, 2);
    cent   = X(sel, :);
    assign = zeros(n, 1);

    for it = 1:1000
        d = [sum((X - cent(1, :)) .^ 2, 2), sum((X - cent(2, :)) .^ 2, 2)];
        [dmin, newAssign] = min(d, [], 2);
        for k = 1:2
            if ~any(newAssign == k)
                % Empty cluster takes the point furthest from its own centroid
                [~, far] = max(dmin);
                newAssign(far) = k;
            end
        end
        if isequal(newAssign, assign), break; end
        assign = newAssign;
        for k = 1:2
            cent(k, :) = mean(X(assign == k, :), 1);
        end
    end

    sq = sum(X .^ 2, 2);
    Dm = sqrt(max(sq + sq' - 2 * (X * X'), 0));
    s  = zeros(n, 1);
    for k = 1:2
        own  = (assign == k);
        nOwn = sum(own);
        if nOwn > 1
            a = sum(Dm(own, own), 2) / (nOwn - 1);
            b = mean(Dm(own, ~own), 2);
            s(own) = (b - a) ./ max(a, b);
        end
    end
    sil = mean(s);
end

function newNInds = hyperbolicPop(NInds, FE, evalsAtStart, maxFE, minPop, shapeConst)
% Post-restart schedule: a hyperbola from NInds down to minPop over the budget left
    left = maxFE - evalsAtStart;
    if left <= 0
        newNInds = minPop;
        return;
    end
    divider = left / shapeConst;
    R       = shapeConst;   % identical to left / divider
    delta   = (minPop * R - R * NInds) ^ 2 - 4 * R * (minPop - NInds);
    if delta <= 0
        newNInds = minPop;
        return;
    end
    b1 = (-(minPop * R - R * NInds) - sqrt(delta)) / (2 * (minPop - NInds));
    a1 = NInds - 1 / b1;
    newNInds = round(a1 + 1 / ((FE - evalsAtStart) / divider + b1));
end

function F = drawF(mu)
% Cauchy(mu, 0.1) redrawn until positive, then capped at 1
    F   = mu + 0.1 * tan(pi * (rand(numel(mu), 1) - 0.5));
    bad = find(F <= 0);
    while ~isempty(bad)
        F(bad) = mu(bad) + 0.1 * tan(pi * (rand(numel(bad), 1) - 0.5));
        bad    = bad(F(bad) <= 0);
    end
    F = min(F, 1);
end

function idx = rouletteN(cumProb, n)
% discretize keeps this O(n log N); an out-of-range draw falls back to the final index
    idx = discretize(rand(n, 1), [0; cumProb(:)]);
    idx(isnan(idx)) = numel(cumProb);
    idx = min(max(idx, 1), numel(cumProb));
end

function [Archive, cur] = copyToArchive(Archive, cur, ArchiveSize, parents)
% Append refused parents while there is room, then overwrite random slots.
    for i = 1:size(parents, 1)
        if cur < ArchiveSize
            cur = cur + 1;
            Archive(cur, :) = parents(i, :);
        elseif ArchiveSize > 0
            Archive(randi(ArchiveSize), :) = parents(i, :);
        end
    end
end

function m = meanWLgeneral(values, deltas)
% Weighted Lehmer mean (g_p = 2, g_m = 1); returns 0.5 on underflow, as the reference does
    sw = sum(deltas);
    if sw <= 0 || ~isfinite(sw)
        w = ones(numel(deltas), 1) / max(numel(deltas), 1);
    else
        w = deltas / sw;
    end
    s = sum(w .* values);
    if abs(s) > 1e-6
        m = sum(w .* values .^ 2) / s;
    else
        m = 0.5;
    end
end

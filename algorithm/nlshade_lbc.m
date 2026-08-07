% ----------------------------------------------------------------------- %
% Non-Linear Population Size Reduction SHADE with Linear Bias Change (NL-SHADE-LBC)
% CEC 2022 competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP_init     = 23 * D         % Initial population, reduced non-linearly to 4
%   MemorySize  = 20 * D         % Historical memory H
%   M_F = 0.5, M_CR = 0.9        % Initial memory contents
%   ArchiveSizeParam = 1         % Archive size as a multiple of the population
%   ArchProb    = 0.5            % Fixed probability of drawing r2 from the archive
%   p = 0.2 -> 0.3 (linear)      % pbest fraction, GROWS over the budget
%   MWLp1 = 3.5, MWLp2 = 1.0, MWLm = 1.5, LBC_fin = 1.5
%
% Algorithm Concept:
%   - LINEAR BIAS CHANGE, the contribution the name refers to: the Lehmer-mean
%     memory update sum(w*v^gp)/sum(w*v^(gp-gm)) has gp swept linearly over the
%     run, gp(F) 3.5 -> 1.5 and gp(CR) 1.0 -> 1.5 with gm = 1.5, so it is the
%     adaptation's BIAS that is scheduled and not only its content
%   - Otherwise the NL-SHADE-RSP lineage: non-linear population reduction
%     NP = round((NPmin-NPmax)*r^(1-r) + NPmax), rank-based third donor with
%     weights exp(-rank/NP), CR values sorted and re-assigned by fitness rank
%   - Differs from NL-SHADE-RSP in a fixed archive probability of 0.5 rather
%     than an adapted one, and in binomial-only crossover
%   - The pbest fraction GROWS 0.2 -> 0.3, so the search becomes less greedy as
%     the population shrinks -- the opposite of jSO
%
% Reference:
% Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin,
% NL-SHADE-LBC algorithm with linear parameter adaptation bias change for
% CEC 2022 Numerical Optimization,
% 2022 IEEE Congress on Evolutionary Computation (CEC), 2022, pp. 1-8.
% https://doi.org/10.1109/CEC55065.2022.9870295
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own C++ competition submission (nl_shade_lbc.cpp, in
% the Codes_of_Top_ranked_Algorithm folder of Suganthan's 2022-SO-BO
% repository); all parameter values come from its Initialize() and MainCycle().
% Two reference properties kept as written: (1) a generation with no successful
% trial RESETS the memory slot to (0.5, 0.5) without advancing the memory index,
% so the same slot is overwritten next time; (2) FindLimits repeats its inner
% dimension loop, applying the same idempotent repair twice.
% One adaptation: out-of-box trials are regenerated for the whole population at
% once rather than one at a time, which is distributionally identical.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nlshade_lbc(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters; 23*D is capped at the budget so the initial population cannot overshoot
    NIndsMax   = max(4, min(23 * D, maxFE));
    NIndsMin   = 4;
    MemorySize = 20 * D;
    ArchiveSizeParam = 1;
    ArchProbs  = 0.5;

    MWLp1   = 3.5;      % initial exponent for F
    MWLp2   = 1.0;      % initial exponent for CR
    MWLm    = 1.5;      % fixed offset
    LBC_fin = 1.5;      % common final exponent

    NInds = NIndsMax;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Popul = repmat(lb, NInds, 1) + rand(NInds, D) .* repmat(ub - lb, NInds, 1);

    [FitMass, FE] = calculate_fitness(Popul', problem, FE);
    FitMass = FitMass(:);

    bsf  = inf;
    bsfx = Popul(1, :);
    for i = 1:NInds
        if FitMass(i) < bsf
            bsf  = FitMass(i);
            bsfx = Popul(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, Popul, FitMass, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    MemoryF  = 0.5 * ones(MemorySize, 1);
    MemoryCr = 0.9 * ones(MemorySize, 1);
    MemoryIter = 1;

    ArchiveSize  = round(NIndsMax * ArchiveSizeParam);
    Archive      = zeros(ArchiveSize, D);
    ArchiveFit   = zeros(ArchiveSize, 1);
    CurArchSize  = 0;

    % Main loop
    while FE < maxFE
        % Fitness ranking; BackIndexes maps individual -> its rank
        [~, Indexes] = sort(FitMass, 'ascend');
        BackIndexes = zeros(NInds, 1);
        BackIndexes(Indexes) = (1:NInds)';

        % Exponential rank weights for the third donor
        rankW  = exp(-(0:NInds-1)' / NInds);
        rankCP = cumsum(rankW / sum(rankW));

        psizeval = max(2, floor(NInds * (0.1 / maxFE * FE + 0.2)));

        % Draw F and CR from the memory
        memIdx = randi(MemorySize, NInds, 1);
        Cr = min(1, max(0, MemoryCr(memIdx) + 0.1 * randn(NInds, 1)));
        F  = drawCauchyPositive(MemoryF(memIdx));

        % RSP: sorted CR values re-assigned by fitness rank
        CrSorted   = sort(Cr, 'ascend');
        CrUsed     = CrSorted(BackIndexes);
        FGenerated = F;

        % Trial generation, with regeneration of out-of-box rows
        popAll = [Popul; Archive(1:CurArchSize, :)];
        ui     = zeros(NInds, D);
        pend   = true(NInds, 1);                  % rows still to be generated
        LBm    = repmat(lb, NInds, 1);
        UBm    = repmat(ub, NInds, 1);

        for attempt = 1:100
            idx = find(pend);
            if isempty(idx)
                break;
            end
            k = numel(idx);

            if attempt > 1
                FGenerated(idx) = drawCauchyPositive(MemoryF(memIdx(idx)));
            end

            % r1: a random member of the top psizeval, not the target
            r1 = Indexes(randi(psizeval, k, 1));
            for t = 1:25
                clash = (r1 == idx);
                if ~any(clash), break; end
                r1(clash) = Indexes(randi(psizeval, sum(clash), 1));
            end

            % r2: uniform over the population, distinct from r1 and the target
            r2 = randi(NInds, k, 1);
            for t = 1:25
                clash = (r2 == idx) | (r2 == r1);
                if ~any(clash), break; end
                r2(clash) = randi(NInds, sum(clash), 1);
            end

            % r3: from the archive w.p. ArchProbs, otherwise a rank-weighted population member
            useArch = (rand(k, 1) <= ArchProbs) & (CurArchSize > 0);
            r3 = zeros(k, 1);
            if any(~useArch)
                n0 = sum(~useArch);
                r3(~useArch) = Indexes(discretizeRank(rankCP, n0));
                for t = 1:25
                    clash = ~useArch & ((r3 == idx) | (r3 == r1) | (r3 == r2));
                    if ~any(clash), break; end
                    r3(clash) = Indexes(discretizeRank(rankCP, sum(clash)));
                end
            end
            if any(useArch)
                r3(useArch) = NInds + randi(CurArchSize, sum(useArch), 1);
            end

            Fk = FGenerated(idx);
            donor = Popul(idx, :) ...
                  + Fk(:, ones(1, D)) .* (Popul(r1, :) - Popul(idx, :)) ...
                  + Fk(:, ones(1, D)) .* (Popul(r2, :) - popAll(r3, :));

            % Binomial crossover, one guaranteed component
            crk  = CrUsed(idx);
            mask = rand(k, D) < crk(:, ones(1, D));
            mask(sub2ind([k D], (1:k)', randi(D, k, 1))) = true;
            trial = Popul(idx, :);
            trial(mask) = donor(mask);

            ui(idx, :) = trial;

            inBox = all(trial >= lb & trial <= ub, 2);
            pend(idx(inBox)) = false;
        end

        % Whatever is still out of bounds gets the midpoint repair
        if any(pend)
            ui(pend, :) = findLimits(ui(pend, :), Popul(pend, :), LBm(pend, :), UBm(pend, :));
        end

        % Evaluate
        nEval = min(NInds, maxFE - FE);
        [FitTemp, FE] = calculate_fitness(ui(1:nEval, :)', problem, FE);
        FitTemp = FitTemp(:);

        for i = 1:nEval
            if FitTemp(i) < bsf
                bsf  = FitTemp(i);
                bsfx = ui(i, :);
            end
            ec = FE - nEval + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Popul, FitMass, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Success bookkeeping (strict improvement)
        succ    = false(NInds, 1);
        succ(1:nEval) = FitTemp < FitMass(1:nEval);
        SuccessF  = FGenerated(succ);
        SuccessCr = CrUsed(succ);
        FitDelta  = abs(FitMass(succ) - FitTemp(succ(1:nEval)));

        % Replacement (ties replace too, and the parent enters the archive)
        repl = false(NInds, 1);
        repl(1:nEval) = FitTemp <= FitMass(1:nEval);
        ridx = find(repl);
        for t = 1:numel(ridx)
            i = ridx(t);
            [Archive, ArchiveFit, CurArchSize] = copyToArchive( ...
                Archive, ArchiveFit, CurArchSize, ArchiveSize, Popul(i, :), FitMass(i));
        end
        Popul(repl, :)  = ui(repl, :);
        FitMass(repl)   = FitTemp(repl(1:nEval));

        % Non-linear population size reduction
        r = min(1, FE / maxFE);
        newNInds = round((NIndsMin - NIndsMax) * r ^ (1 - r) + NIndsMax);
        newNInds = min(max(newNInds, NIndsMin), NIndsMax);

        newArchSize = max(NIndsMin, round(newNInds * ArchiveSizeParam));
        ArchiveSize = newArchSize;
        if CurArchSize > ArchiveSize
            CurArchSize = ArchiveSize;
        end

        if newNInds < NInds
            for L = 1:(NInds - newNInds)
                [~, worst] = max(FitMass);
                Popul(worst, :) = [];
                FitMass(worst)  = [];
            end
            NInds = newNInds;
        end

        % Memory update with the linearly changing bias
        if ~isempty(SuccessF)
            frac  = (maxFE - min(FE, maxFE)) / maxFE;
            FMWL  = LBC_fin + (MWLp1 - LBC_fin) * frac;
            CrMWL = LBC_fin + (MWLp2 - LBC_fin) * frac;

            MemoryF(MemoryIter)  = (MemoryF(MemoryIter)  + meanWLgeneral(SuccessF,  FitDelta, FMWL,  MWLm)) * 0.5;
            MemoryCr(MemoryIter) = (MemoryCr(MemoryIter) + meanWLgeneral(SuccessCr, FitDelta, CrMWL, MWLm)) * 0.5;

            MemoryIter = MemoryIter + 1;
            if MemoryIter > MemorySize
                MemoryIter = 1;
            end
        else
            % Reference behaviour: reset the slot, do NOT advance the index
            MemoryF(MemoryIter)  = 0.5;
            MemoryCr(MemoryIter) = 0.5;
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function F = drawCauchyPositive(mu)
% Cauchy(mu, 0.1), resampled while non-positive, then truncated above at 1.
    n = numel(mu);
    F = mu + 0.1 * tan(pi * (rand(n, 1) - 0.5));
    bad = F <= 0;
    guard = 0;
    while any(bad)
        F(bad) = mu(bad) + 0.1 * tan(pi * (rand(sum(bad), 1) - 0.5));
        bad = F <= 0;
        guard = guard + 1;
        if guard > 1000
            F(bad) = 0.5;
            break;
        end
    end
    F = min(F, 1);
end

function idx = discretizeRank(cp, n)
% Sample n ranks from the cumulative rank-weight distribution.
    idx = discretize(rand(n, 1), [0; cp(:)]);
    idx(isnan(idx)) = numel(cp);
end

function V = findLimits(V, Parent, LBm, UBm)
% Midpoint between the parent and the violated bound.
    lo = V < LBm;
    V(lo) = (LBm(lo) + Parent(lo)) / 2;
    hi = V > UBm;
    V(hi) = (UBm(hi) + Parent(hi)) / 2;
end

function m = meanWLgeneral(V, W, gp, gm)
% Generalised weighted power mean: sum(w V^gp) / sum(w V^(gp-gm)).
    sw = sum(W);
    if sw <= 0
        w = ones(numel(W), 1) / numel(W);
    else
        w = W / sw;
    end
    num = sum(w .* V .^ gp);
    den = sum(w .* V .^ (gp - gm));
    if abs(den) > 1e-6
        m = num / den;
    else
        m = 0.5;
    end
end

function [Archive, ArchiveFit, CurArchSize] = copyToArchive( ...
        Archive, ArchiveFit, CurArchSize, ArchiveSize, parent, parentFit)
% Append while there is room, then overwrite a random archived solution that is no better
    if CurArchSize < ArchiveSize
        CurArchSize = CurArchSize + 1;
        Archive(CurArchSize, :) = parent;
        ArchiveFit(CurArchSize) = parentFit;
    elseif ArchiveSize > 0
        r = randi(ArchiveSize);
        for c = 1:ArchiveSize
            if ArchiveFit(r) >= parentFit
                break;
            end
            r = randi(ArchiveSize);
        end
        Archive(r, :) = parent;
        ArchiveFit(r) = parentFit;
    end
end

% ----------------------------------------------------------------------- %
% Non-Linear SHADE with Rank-based Selective Pressure (NL-SHADE-RSP)
% CEC 2021 competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NIndsMax     = 30 * D      % Initial population (non-linear reduction to 4)
%   NIndsMin     = 4
%   MemorySize   = 20 * D      % Historical memory for Cr and F
%   M_F = M_CR   = 0.2         % Initial memory contents
%   ArchiveSizeParam = 2.1     % Archive size factor
%
% Algorithm Concept:
%   - Non-linear population size reduction:
%       NP = round((NPmin-NPmax) * r^(1-r) + NPmax),  r = FE/maxFE
%   - Rank-based selective pressure (RSP): the Cr values drawn for the whole
%     generation are SORTED and re-assigned by fitness rank, so better
%     individuals receive the smaller crossover rates
%   - The third donor comes from the population (sampled by rank with weights
%     exp(-i/NP)) or the archive, the archive probability being re-estimated
%     every generation from relative improvement and clamped to [0.1, 0.9]
%   - Crossover alternates per generation between binomial and exponential; the
%     binomial branch uses a rate ramping 0 -> 1 over the second half of the
%     budget rather than the adapted Cr
%   - Out-of-bounds components are re-sampled uniformly inside the box
%
% Reference:
% Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin,
% NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for
% CEC 2021 Numerical Optimization,
% 2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. 809-816.
% https://doi.org/10.1109/CEC45853.2021.9504959
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' competition release (nl_shade_rsp.cpp, submission
% E-0125 in the CEC2021 top-methods archive). Three reference behaviours that
% read as wrong are kept: (1) by design, the binomial branch tests rand against
% the budget ramp rather than the adapted Cr, so early binomial trials differ in
% one forced dimension only; (2) by design, success uses "<" while replacement
% uses "<=", so equal-fitness trials are accepted without counting as success;
% (3) a genuine slip -- the F recorded on success is a stale loop variable, so
% the whole generation contributes one F to the memory. It is what won the
% competition. The archive statistic skips parents with zero or non-finite
% fitness, which CEC2020RW's scalarised objective can produce.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nlshade_rsp(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    span  = ub - lb;

    % NL-SHADE-RSP control parameters (main() of the reference release)
    NIndsMax         = 30 * dim;
    NIndsMin         = 4;
    NInds            = NIndsMax;
    MemorySize       = 20 * dim;
    ArchiveSizeParam = 2.1;

    ArchiveSize        = floor(NIndsMax * ArchiveSizeParam);
    ArchiveCapacity    = NIndsMax * ceil(ArchiveSizeParam);
    CurrentArchiveSize = 0;

    MemoryCr   = 0.2 * ones(MemorySize, 1);
    MemoryF    = 0.2 * ones(MemorySize, 1);
    MemoryIter = 1;
    ArchProbs  = 0.5;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial population
    Popul   = repmat(lb, NIndsMax, 1) + rand(NIndsMax, dim) .* repmat(span, NIndsMax, 1);
    Archive = zeros(ArchiveCapacity, dim);

    [FitMass, FE] = calculate_fitness(Popul', problem, FE);
    FitMass = FitMass(:);

    bsf          = inf;
    bsf_solution = Popul(1, :);
    for i = 1:NIndsMax
        if FitMass(i) < bsf
            bsf          = FitMass(i);
            bsf_solution = Popul(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, Popul, FitMass, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        % Fitness ranking: Indexes maps rank -> individual, BackIndexes the reverse
        [~, Indexes] = sort(FitMass(1:NInds), 'ascend');
        BackIndexes  = zeros(NInds, 1);
        BackIndexes(Indexes) = (1:NInds)';

        rankW = exp(-(0:NInds-1)' / NInds);
        rankC = cumsum(rankW / sum(rankW));

        psizeval = max(2, floor(NInds * (0.2 / maxFE * FE + 0.2)));
        psizeval = min(psizeval, NInds);

        CrossExponential = (rand() < 0.5);

        % Draw Cr and F for the whole generation
        memIdx = randi(MemorySize, NInds, 1);

        CrGenerated = min(1, max(0, MemoryCr(memIdx) + 0.1 * randn(NInds, 1)));

        Fraw = MemoryF(memIdx) + 0.1 * tan(pi * (rand(NInds, 1) - 0.5));
        bad  = find(Fraw <= 0);
        while ~isempty(bad)
            Fraw(bad) = MemoryF(memIdx(bad)) + 0.1 * tan(pi * (rand(length(bad), 1) - 0.5));
            bad = find(Fraw <= 0);
        end
        FGenerated = min(Fraw, 1);
        % `F` is a leftover loop variable holding the last raw Cauchy draw, as the reference records it
        F_stale = Fraw(NInds);

        % Rank-based selective pressure: sorted Cr re-assigned by fitness rank
        CrGenerated = sort(CrGenerated, 'ascend');
        CrUsed      = CrGenerated(BackIndexes);

        % Donor indices (0-based in the reference; 1-based here)
        target = (1:NInds)';

        R0 = Indexes(randi(psizeval, NInds, 1));
        for it = 1:25
            clash = (R0 == target);
            if ~any(clash), break; end
            R0(clash) = Indexes(randi(psizeval, sum(clash), 1));
        end

        R1 = randi(NInds, NInds, 1);
        for it = 1:25
            clash = (R1 == R0);
            if ~any(clash), break; end
            R1(clash) = randi(NInds, sum(clash), 1);
        end

        useArch = (rand(NInds, 1) <= ArchProbs) & (CurrentArchiveSize > 0);

        R2 = zeros(NInds, 1);
        nPop = ~useArch;
        if any(nPop)
            idx = find(nPop);
            R2(idx) = Indexes(rouletteN(rankC, numel(idx)));
            for it = 1:25
                clash = idx((R2(idx) == target(idx)) | (R2(idx) == R0(idx)) | (R2(idx) == R1(idx)));
                if isempty(clash), break; end
                R2(clash) = Indexes(rouletteN(rankC, numel(clash)));
            end
        end
        if any(useArch)
            idx = find(useArch);
            R2(idx) = NInds + randi(CurrentArchiveSize, numel(idx), 1);
        end

        % Mutation
        PoolAll = [Popul(1:NInds, :); Archive(1:CurrentArchiveSize, :)];
        Fcol    = FGenerated(:, ones(1, dim));

        Donor = Popul(1:NInds, :) ...
              + Fcol .* (PoolAll(R0, :) - Popul(1:NInds, :)) ...
              + Fcol .* (PoolAll(R1, :) - PoolAll(R2, :));

        % Crossover
        Trial = Popul(1:NInds, :);
        forced = randi(dim, NInds, 1);

        if ~CrossExponential
            % Binomial -- the reference tests against the budget ramp, not Cr
            if FE > 0.5 * maxFE
                CrToUse = (FE / maxFE - 0.5) * 2;
            else
                CrToUse = 0;
            end
            take = rand(NInds, dim) < CrToUse;
            take(sub2ind([NInds dim], target, forced)) = true;
        else
            % Exponential crossover: the geometric segment length drawn in closed form, same distribution
            StartLoc = randi(dim, NInds, 1);
            u     = rand(NInds, 1);
            extra = zeros(NInds, 1);
            growing = (CrUsed > 0) & (CrUsed < 1);
            extra(growing)     = floor(log(u(growing)) ./ log(CrUsed(growing)));
            extra(CrUsed >= 1) = dim;
            EndLoc = min(StartLoc + extra, dim);

            cols = repmat(1:dim, NInds, 1);
            take = (cols >= StartLoc) & (cols <= EndLoc);
        end

        Trial(take) = Donor(take);

        % Out-of-bounds components are re-sampled uniformly inside the box
        oob = Trial < lb | Trial > ub;
        if any(oob(:))
            R = repmat(lb, NInds, 1) + rand(NInds, dim) .* repmat(span, NInds, 1);
            Trial(oob) = R(oob);
        end

        [FitMassTemp, FE] = calculate_fitness(Trial', problem, FE);
        FitMassTemp = FitMassTemp(:);

        for i = 1:NInds
            if FitMassTemp(i) < bsf
                bsf          = FitMassTemp(i);
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

        % Success memory (strict "<", stale F as in the reference)
        succ = (FitMassTemp < FitMass(1:NInds));
        SuccessFilled = sum(succ);
        tempSuccessCr = CrUsed(succ);
        tempSuccessF  = F_stale * ones(SuccessFilled, 1);
        FitDelta      = abs(FitMass(succ) - FitMassTemp(succ));

        % Replacement (non-strict "<=") and archive statistics
        repl = (FitMassTemp <= FitMass(1:NInds));

        parentFit = FitMass(1:NInds);
        rel = zeros(NInds, 1);
        usable = repl & isfinite(parentFit) & isfinite(FitMassTemp) & (parentFit ~= 0);
        rel(usable) = (parentFit(usable) - FitMassTemp(usable)) ./ parentFit(usable);

        NArchUsages   = sum(repl & useArch);
        ArchSuccess   = sum(rel(repl & useArch));
        NoArchSuccess = sum(rel(repl & ~useArch));

        if any(repl)
            [Archive, CurrentArchiveSize] = copyToArchive(Archive, CurrentArchiveSize, ...
                                                          ArchiveSize, Popul(repl, :));
            Popul(repl, :)  = Trial(repl, :);
            FitMass(repl)   = FitMassTemp(repl);
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

        % Non-linear population and archive size reduction
        r = FE / maxFE;
        nl = round((NIndsMin - NIndsMax) * r ^ (1 - r) + NIndsMax);
        newNInds = min(max(nl, NIndsMin), NIndsMax);

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
        end
        NInds = newNInds;

        % Memory update
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

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

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

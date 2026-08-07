% ----------------------------------------------------------------------- %
% L-SHADE with Rank-based Selective Pressure (LSHADE-RSP)
% CEC 2018 competition runner-up
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NIndsMax     = floor(75 * D^(2/3))   % Initial population (linear -> 4)
%   NIndsMin     = 4
%   MemorySize   = 5                     % Historical memory for Cr and F
%   M_CR = 0.8, M_F = 0.3                % Initial memory contents
%   ArchiveSizeParam = 1.0
%   psizeParam   = 0.17                  % pbest fraction, ramps 0.085 -> 0.17
%
% Algorithm Concept:
%   - L-SHADE with rank-based selective pressure: both difference donors are
%     drawn by fitness RANK with linear weights 3*(NP - rank), so better
%     individuals are more likely to contribute a difference vector
%   - A virtual (H+1)-th memory slot is drawn as often as the real ones and
%     yields the fixed pair F ~ Cauchy(0.9, 0.1), Cr ~ N(0.9, 0.1)
%   - Weighted mutation as in jSO: the pbest term uses F2 = 0.7F, 0.8F or 1.2F
%     in the first 20 %, the first 40 % and the remainder of the budget
%   - Budget clamps: F <= 0.7 while FE < 0.6*maxFE, CR >= 0.7 while
%     FE < 0.25*maxFE and CR >= 0.6 while FE < 0.5*maxFE
%   - The archive supplies the second difference term with probability
%     |archive|/(|archive|+NP), its capacity shrinking linearly to 4
%   - Memory is a weighted Lehmer mean averaged with the previous content
%
% Reference:
% Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin,
% LSHADE Algorithm with Rank-Based Selective Pressure Strategy for Solving
% CEC 2017 Benchmark Problems,
% 2018 IEEE Congress on Evolutionary Computation (CEC), 2018, pp. 1-8.
% https://doi.org/10.1109/CEC.2018.8477977
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' competition release (LSHADE_RSP.cpp, "Codes for best
% 3" in Suganthan's CEC2018 repository). Three reference properties are kept:
% ArchiveProb (0.25) is assigned at initialisation and never read, the archive
% decision using |archive|/(|archive|+NP) instead, so it is dropped rather than
% given an effect it never had; success recording and replacement both use "<=",
% so an equal-fitness trial replaces AND counts as a success; and RemoveWorst
% rescans the pre-reduction index range, so dropping more than one individual in
% a generation can see a stale trailing entry.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lshade_rsp(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % LSHADE-RSP control parameters (main() of the reference release)
    NIndsMax         = max(floor(75 * dim ^ (2/3)), 4);
    NIndsMin         = 4;
    NInds            = NIndsMax;
    MemorySize       = 5;
    ArchiveSizeParam = 1.0;
    psizeParam       = 0.17;

    ArchiveSize        = floor(NIndsMax * ArchiveSizeParam);
    ArchiveCapacity    = NIndsMax * ceil(ArchiveSizeParam);
    CurrentArchiveSize = 0;

    MemoryCr   = 0.8 * ones(MemorySize, 1);
    MemoryF    = 0.3 * ones(MemorySize, 1);
    MemoryIter = 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial population
    Popul   = repmat(lb, NIndsMax, 1) + rand(NIndsMax, dim) .* repmat(ub - lb, NIndsMax, 1);
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
        r = FE / maxFE;

        [~, Indexes] = sort(FitMass(1:NInds), 'ascend');   % rank -> individual

        % Linear rank weights 3*(NP - rank), rank counted from 0
        rankW = 3 * (NInds - (0:NInds-1)');
        rankC = cumsum(rankW / sum(rankW));

        psize    = (psizeParam / 2) / maxFE * FE + psizeParam / 2;
        psizeval = floor(NInds * psize);
        if psizeval <= 1
            psizeval = 2;
        end
        psizeval = min(psizeval, NInds);

        target = (1:NInds)';

        % Memory slot: index MemorySize+1 is the virtual 0.9 pair
        memIdx = randi(MemorySize + 1, NInds, 1);
        virt   = (memIdx == MemorySize + 1);

        muF = zeros(NInds, 1);
        muF(~virt) = MemoryF(memIdx(~virt));
        muF(virt)  = 0.9;

        muCr = zeros(NInds, 1);
        muCr(~virt) = MemoryCr(memIdx(~virt));
        muCr(virt)  = 0.9;

        % Scaling factors
        F = muF + 0.1 * tan(pi * (rand(NInds, 1) - 0.5));
        bad = find(F < 0);
        while ~isempty(bad)
            F(bad) = muF(bad) + 0.1 * tan(pi * (rand(length(bad), 1) - 0.5));
            bad = find(F < 0);
        end
        F = min(F, 1);
        if r < 0.6
            F(F > 0.7) = 0.7;
        end

        if r < 0.2
            F2 = 0.7 * F;
        elseif r < 0.4
            F2 = 0.8 * F;
        else
            F2 = 1.2 * F;
        end

        % Crossover rates
        Cr = muCr + 0.1 * randn(NInds, 1);
        Cr(muCr < 0) = 0;
        Cr = min(max(Cr, 0), 1);
        if r < 0.25
            Cr = max(Cr, 0.7);
        end
        if r < 0.5
            Cr = max(Cr, 0.6);
        end

        % Donor indices
        prand = Indexes(randi(psizeval, NInds, 1));
        if r < 0.5
            for it = 1:1000
                clash = (prand == target);
                if ~any(clash), break; end
                prand(clash) = Indexes(randi(psizeval, sum(clash), 1));
                if it == 1000
                    error('lshade_rsp:prand', 'Cannot draw a pbest donor in 1000 iterations');
                end
            end
        end

        Rand1 = Indexes(rouletteN(rankC, NInds));
        for it = 1:1000
            clash = (Rand1 == prand);
            if ~any(clash), break; end
            Rand1(clash) = Indexes(rouletteN(rankC, sum(clash)));
            if it == 1000
                error('lshade_rsp:rand1', 'Cannot draw Rand1 in 1000 iterations');
            end
        end

        Rand2 = Indexes(rouletteN(rankC, NInds));
        for it = 1:1000
            clash = (Rand2 == prand) | (Rand2 == Rand1);
            if ~any(clash), break; end
            Rand2(clash) = Indexes(rouletteN(rankC, sum(clash)));
            if it == 1000
                error('lshade_rsp:rand2', 'Cannot draw Rand2 in 1000 iterations');
            end
        end

        % Mutation
        useArch = rand(NInds, 1) < CurrentArchiveSize / (CurrentArchiveSize + NInds);

        Second = Popul(Rand2, :);
        if any(useArch)
            idx = find(useArch);
            Second(idx, :) = Archive(randi(CurrentArchiveSize, numel(idx), 1), :);
        end

        Donor = Popul(1:NInds, :) ...
              + F2(:, ones(1, dim)) .* (Popul(prand, :) - Popul(1:NInds, :)) ...
              + F(:,  ones(1, dim)) .* (Popul(Rand1, :) - Second);

        Donor = boundConstraint(Donor, Popul(1:NInds, :), lu);

        % Binomial crossover
        take = rand(NInds, dim) < Cr(:, ones(1, dim));
        take(sub2ind([NInds dim], target, randi(dim, NInds, 1))) = true;

        Trial = Popul(1:NInds, :);
        Trial(take) = Donor(take);

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

        % Selection: "<=" for both success and replacement
        succ = (FitTemp <= FitMass(1:NInds));

        tempSuccessCr = Cr(succ);
        tempSuccessF  = F(succ);
        FitDelta      = abs(FitMass(succ) - FitTemp(succ));
        SuccessFilled = sum(succ);

        if SuccessFilled > 0
            [Archive, CurrentArchiveSize] = copyToArchive(Archive, CurrentArchiveSize, ...
                                                          ArchiveSize, Popul(succ, :));
            Popul(succ, :) = Trial(succ, :);
            FitMass(succ)  = FitTemp(succ);
        end

        % Linear population and archive size reduction
        newNInds = floor((NIndsMin - NIndsMax) / maxFE * FE + NIndsMax);
        newNInds = min(max(newNInds, NIndsMin), NIndsMax);

        newArchSize = floor((maxFE - FE) / maxFE * (ArchiveSizeParam * (NIndsMax - NIndsMin)));
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
            Old_F  = MemoryF(MemoryIter);
            Old_Cr = MemoryCr(MemoryIter);

            if MemoryCr(MemoryIter) == -1 || max(tempSuccessCr) == 0
                MemoryCr(MemoryIter) = -1;
            else
                MemoryCr(MemoryIter) = (meanWL(tempSuccessCr, FitDelta) + Old_Cr) / 2;
            end
            MemoryF(MemoryIter) = (meanWL(tempSuccessF, FitDelta) + Old_F) / 2;

            MemoryIter = mod(MemoryIter, MemorySize) + 1;
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function idx = rouletteN(cumProb, n)
% discretize keeps this O(n log N); an out-of-range draw falls back to the end
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

function m = meanWL(values, deltas)
% Weighted Lehmer mean; returns 0.5 on underflow, exactly as the reference does
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

function vi = boundConstraint(vi, pop, lu)
% Violating component moved to the parent/bound midpoint, unchanged from the reference
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

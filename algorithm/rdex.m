% ----------------------------------------------------------------------- %
% Reconstructed Differential Evolution with Exploitation hybrid (RDEx)
% CEC 2025 bound-constrained single-objective track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 20*D -> 4              % Front population, reduced linearly
%   H = 5                       % Memory slots for Cr and F, both initialised to 1
%   meanF = 0.4 + 0.25*tanh(5*SR)  % Scale factor mean, driven by the success rate
%   p = 0.7*exp(-7*SR)          % pbest rate of the base branch
%   p2 = 0.17*(1 - 0.5*t)       % pbest rate of the ordered branch
%   EB_rate = 0.7 (adaptive)    % Share of trials built by the ordered mutation
%   jitter = 0.4                % Chance a non-crossover coordinate is Cauchy-jittered
%
% Algorithm Concept:
%   - L-SRTDE's frame: a front population feeds the trials while successful
%     trials accumulate in a larger pool, and both the scale factor mean and the
%     pbest rate are driven by the measured success rate
%   - Past 70 % of the budget a second branch switches on, whose share adapts to
%     which branch earns the improvement: it re-sorts pbest, r1 and r2 by fitness
%     and steps towards the best of the three along the medium-to-worst difference
%   - That branch draws F and Cr from their own memory, with a sixth slot that
%     returns the fixed 0.9, and caps F at 0.7 early while flooring Cr
%   - Coordinates a trial does not take from the donor are Cauchy-jittered with
%     probability 0.4 instead of copied
%   - Out-of-box coordinates are resampled uniformly, not clamped
%
% Reference:
% Sichen Tao, Ruihan Zhao, Kaiyu Wang, Shangce Gao,
% An Efficient Reconstructed Differential Evolution Variant by Some of the
% Current State-of-the-art Strategies for Solving Single Objective Bound
% Constrained Problems,
% arXiv preprint arXiv:2404.16280 (2024).
% https://doi.org/10.48550/arXiv.2404.16280
% Components: L-SRTDE (Stanovov and Semenkin, CEC 2024) provides the frame;
% the ordered-mutation hybrid and its adaptive share are RDEx's own.
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' RDEx_SOP release (RDEx.cpp in the CEC2025 package),
% which is L-SRTDE with the ordered branch added, so the shared parts follow this
% repository's lsrtde.m. The gate on that branch is kept exactly as written --
% the uniform draw is replaced by the constant 2 while less than 70 % of the
% budget is spent, so the branch can only open early when its own share has
% already grown above 2*(1-t).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rdex(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NIndsFrontMax = 20 * dim;
    PopulSize     = 2 * NIndsFrontMax;
    MemorySize    = 5;
    sigmaF        = 0.02;
    minNInds      = 4;
    eb_rate_init  = 0.7;
    jitter_rate   = 0.4;

    NIndsFront   = NIndsFrontMax;
    NIndsCurrent = NIndsFrontMax;
    SuccessRate  = 0.5;
    MemoryCr     = ones(MemorySize, 1);
    MemoryF      = ones(MemorySize, 1);
    MemoryIter   = 1;
    eb_rate      = eb_rate_init;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Popul  = repmat(lb, PopulSize, 1) + rand(PopulSize, dim) .* repmat(span, PopulSize, 1);
    FitArr = inf(PopulSize, 1);

    bsf          = inf;
    bsf_solution = Popul(1, :);

    for i = 1:NIndsFront
        if FE >= maxFE
            break;
        end
        [fv, FE] = calculate_fitness(Popul(i, :)', problem, FE);
        FitArr(i) = fv(1);
        if FitArr(i) < bsf
            bsf          = FitArr(i);
            bsf_solution = Popul(i, :);
        end
        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            FE, Popul(1:i, :), FitArr(1:i), population_history, fitness_history, ...
            history_index, maxFE);
    end

    [FitArrFront, Indices] = sort(FitArr(1:NIndsFront), 'ascend');
    PopulFront = Popul(Indices, :);
    PFIndex = 1;

    while FE < maxFE
        meanF = 0.4 + tanh(SuccessRate * 5) * 0.25;
        t = FE / maxFE;

        [~, Indices]  = sort(FitArr(1:NIndsFront), 'ascend');       % into Popul
        [~, Indices2] = sort(FitArrFront(1:NIndsFront), 'ascend');  % into PopulFront

        rankW = exp(-(0:NIndsFront-1)' / NIndsFront * 3);
        rankC = cumsum(rankW / sum(rankW));
        pressW = 3 * (NIndsFront - (0:NIndsFront-1)');              % the ordered branch's weights

        psizeval = min(NIndsFront, max(2, floor(NIndsFront * 0.7 * exp(-SuccessRate * 7))));
        psizeval2 = min(NIndsFront, max(2, floor(NIndsFront * 0.17 * (1 - 0.5 * t))));

        SuccessFilled = 0;
        tempSuccessCr = zeros(NIndsFront, 1);
        tempSuccessF  = zeros(NIndsFront, 1);
        FitDelta      = zeros(NIndsFront, 1);
        FitBefore     = FitArrFront(1:NIndsFront);
        FitTrial      = FitArrFront(1:NIndsFront);
        eb_flag       = false(NIndsFront, 1);

        for IndIter = 1:NIndsFront
            if FE >= maxFE
                break;
            end

            TheChosenOne = randi(NIndsFront);
            idx1 = randi(MemorySize);
            idx2 = randi(MemorySize + 1);     % the extra slot returns the fixed 0.9

            Rand_EB = rand;
            if t < 0.7
                Rand_EB = 2;
            end
            use_eb = (Rand_EB * (1 - t)) < eb_rate;
            eb_flag(TheChosenOne) = use_eb;

            base = PopulFront(TheChosenOne, :);

            if use_eb
                prand = Indices(roulette_w(pressW(1:psizeval2)));
                while prand == TheChosenOne
                    prand = Indices(roulette_w(pressW(1:psizeval2)));
                end
                Rand1 = Indices2(roulette_w(pressW));
                while Rand1 == prand
                    Rand1 = Indices2(roulette_w(pressW));
                end
                Rand2 = Indices(roulette_w(pressW));
                while Rand2 == prand || Rand2 == Rand1
                    Rand2 = Indices(roulette_w(pressW));
                end

                [ord_best, ord_med, ord_worst] = eb_order( ...
                    Popul(prand, :), FitArr(prand), ...
                    PopulFront(Rand1, :), FitArrFront(Rand1), ...
                    Popul(Rand2, :), FitArr(Rand2));

                F = -1;
                while F <= 0
                    if idx2 <= MemorySize
                        F = MemoryF(idx2) + 0.1 * tan(pi * (rand - 0.5));
                    else
                        F = 0.9 + 0.1 * tan(pi * (rand - 0.5));
                    end
                end
                F = min(F, 1);
                if t < 0.6 && F > 0.7
                    F = 0.7;
                end

                if idx2 <= MemorySize
                    if MemoryCr(idx2) < 0
                        Cr = 0;
                    else
                        Cr = MemoryCr(idx2) + 0.1 * randn;
                    end
                else
                    Cr = 0.9 + 0.1 * randn;
                end
                Cr = min(max(Cr, 0), 1);
                if t < 0.25
                    Cr = max(Cr, 0.7);
                end
                if t < 0.5
                    Cr = max(Cr, 0.6);
                end

                donor = base + F * (ord_best - base) + F * (ord_med - ord_worst);
            else
                prand = Indices(randi(psizeval));
                while prand == TheChosenOne
                    prand = Indices(randi(psizeval));
                end
                Rand1 = Indices2(roulette(rankC));
                while Rand1 == prand
                    Rand1 = Indices2(roulette(rankC));
                end
                Rand2 = Indices(randi(NIndsFront));
                while Rand2 == prand || Rand2 == Rand1
                    Rand2 = Indices(randi(NIndsFront));
                end

                F = meanF + sigmaF * randn();
                while F < 0.0 || F > 1.0
                    F = meanF + sigmaF * randn();
                end

                Cr = MemoryCr(idx1) + 0.05 * randn();
                Cr = min(max(Cr, 0.0), 1.0);

                donor = base + F * (Popul(prand, :) - base) ...
                             + F * (PopulFront(Rand1, :) - Popul(Rand2, :));
            end

            take = rand(1, dim) < Cr;
            take(randi(dim)) = true;

            if rand < jitter_rate
                Trial = base + 0.1 * tan(pi * (rand(1, dim) - 0.5));
            else
                Trial = base;
            end
            Trial(take) = donor(take);

            oob = take & (Trial < lb | Trial > ub);
            if any(oob)
                Trial(oob) = lb(oob) + rand(1, sum(oob)) .* span(oob);
            end

            ActualCr = sum(take) / dim;

            [fv, FE] = calculate_fitness(Trial', problem, FE);
            TempFit  = fv(1);
            FitTrial(TheChosenOne) = TempFit;

            if TempFit <= FitArrFront(TheChosenOne)
                slot = NIndsCurrent + SuccessFilled + 1;
                Popul(slot, :) = Trial;
                FitArr(slot)   = TempFit;

                PopulFront(PFIndex, :) = Trial;
                FitArrFront(PFIndex)   = TempFit;

                if TempFit < bsf
                    bsf          = TempFit;
                    bsf_solution = Trial;
                end

                SuccessFilled = SuccessFilled + 1;
                tempSuccessCr(SuccessFilled) = ActualCr;
                tempSuccessF(SuccessFilled)  = F;
                % Measured after the front slot was overwritten, as in the reference
                FitDelta(SuccessFilled) = abs(FitArrFront(TheChosenOne) - TempFit);

                PFIndex = mod(PFIndex, NIndsFront) + 1;
            end

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, PopulFront(1:NIndsFront, :), FitArrFront(1:NIndsFront), ...
                population_history, fitness_history, history_index, maxFE);
        end

        % Which branch earned the improvement sets next generation's share
        improved = FitTrial <= FitBefore;
        sum_eb   = sum(FitBefore(improved & eb_flag) - FitTrial(improved & eb_flag));
        sum_base = sum(FitBefore(improved & ~eb_flag) - FitTrial(improved & ~eb_flag));
        if sum_eb ~= 0 && sum_base ~= 0
            eb_rate = min(max(sum_eb / (sum_eb + sum_base), 0), 1);
        else
            eb_rate = eb_rate_init;
        end

        SuccessRate = SuccessFilled / NIndsFront;

        newNIndsFront = max(minNInds, floor((minNInds - NIndsFrontMax) / maxFE * FE + NIndsFrontMax));
        if newNIndsFront < NIndsFront
            for L = 1:(NIndsFront - newNIndsFront)
                [~, WorstNum] = max(FitArrFront(1:NIndsFront));
                PopulFront(WorstNum:NIndsFront-1, :) = PopulFront(WorstNum+1:NIndsFront, :);
                FitArrFront(WorstNum:NIndsFront-1)   = FitArrFront(WorstNum+1:NIndsFront);
            end
        end
        NIndsFront = newNIndsFront;
        if PFIndex > NIndsFront
            PFIndex = 1;
        end

        if SuccessFilled > 0
            MemoryCr(MemoryIter) = 0.5 * (meanWL(tempSuccessCr(1:SuccessFilled), ...
                                                 FitDelta(1:SuccessFilled)) + MemoryCr(MemoryIter));
            MemoryF(MemoryIter) = meanWL(tempSuccessF(1:SuccessFilled), FitDelta(1:SuccessFilled));
            MemoryIter = mod(MemoryIter, MemorySize) + 1;
        end

        NIndsCurrent = NIndsFront + SuccessFilled;
        if NIndsCurrent > NIndsFront
            [srt, ord]   = sort(FitArr(1:NIndsCurrent), 'ascend');
            NIndsCurrent = NIndsFront;
            Popul(1:NIndsCurrent, :) = Popul(ord(1:NIndsCurrent), :);
            FitArr(1:NIndsCurrent)   = srt(1:NIndsCurrent);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function idx = roulette(cumProb)
% find() can return empty when the last cumulative entry rounds below 1
    idx = find(rand() <= cumProb, 1);
    if isempty(idx)
        idx = numel(cumProb);
    end
end

function idx = roulette_w(w)
    c = cumsum(w);
    idx = find(rand() * c(end) <= c, 1);
    if isempty(idx)
        idx = numel(w);
    end
end

% The three donors re-sorted into best, medium and worst
function [b, m, w] = eb_order(x1, f1, x2, f2, x3, f3)
    X = [x1; x2; x3];
    [~, o] = sort([f1; f2; f3]);
    b = X(o(1), :);
    m = X(o(2), :);
    w = X(o(3), :);
end

function m = meanWL(values, deltas)
% Weighted Lehmer mean; returns 1.0 on underflow, exactly as the reference does
    sw = sum(deltas);
    if sw <= 0
        w = ones(numel(deltas), 1) / numel(deltas);
    else
        w = deltas / sw;
    end
    s = sum(w .* values);
    if s == 0
        m = 1.0;
    else
        m = sum(w .* values .* values) / s;
    end
end

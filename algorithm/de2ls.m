% ----------------------------------------------------------------------- %
% Differential Evolution with Late-Stage Local Search (DE-2LS)
% CEC 2026 bound-constrained single-objective track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 18*D -> 4 (linear)      % Front population
%   H = 5                        % Memory slots for Cr and F, both initialised to 1
%   meanF = 0.4 + 0.25*tanh(5*SR), sigmaF = 0.025  % Standard-branch F
%   p = 0.65*exp(-6.5*SR)        % pbest rate of the standard branch
%   p2 = 0.17*(1 - 0.5*t)        % pbest rate of the ordered branch
%   EB_rate = 0.7 (adaptive)     % Share of the ordered branch once it may open
%   EB_SS = 0.75, EB_OW = 0.65   % Late smoothing: start fraction, weight of the old share
%   jitter = 0.4                 % Chance a trial Cauchy-jitters its non-crossover coordinates
%
% Algorithm Concept:
%   - RDEx's engine: a front population feeds the trials while successful trials
%     collect in a pool; the success rate sets the F mean and the pbest window
%   - An ordered branch re-sorts pbest, r1 and r2 by fitness and steps towards
%     the best along the medium-to-worst difference; it opens after 70 % of FEs
%   - Its share follows which branch earned the improvement; past 75 % of the
%     budget the new share is smoothed as 0.65*old + 0.35*new
%   - Ordered-branch F and Cr come from their own memory draw plus a fixed 0.9
%     slot; F is capped at 0.7 early and Cr floored at 0.7, then 0.6
%   - The paper's coordinate-pattern local search is absent from the released code
%
% Reference:
% Dikshit Chauhan,
% DE-2LS: Differential Evolution with Late-Stage local-search for
% Unconstrained Single-Objective Numerical Optimization,
% arXiv preprint arXiv:2606.27762 (2026).
% https://doi.org/10.48550/arXiv.2606.27762
% Components: RDEx-SOP (Sichen Tao et al., arXiv:2603.27089, 2026) is the engine.
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from DE_2LS.cpp (identical in the organisers' CEC 2026 BC-SOP package and
% the authors' repository), which is the paper's pre-local-search stage (variant
% tag lsv1_s75sf025_xi065k65): late EB smoothing and sigmaF = 0.025, but pbest
% coefficients 0.65 and 6.5 where Table II gives 0.75 and 7.5, and no coordinate-
% pattern local search at all. The port follows the file. Kept from its RDEx base:
% 1 - NFEval/MaxFEval is integer division, so the ordered branch is shut before
% 70 % of the budget and opens with probability EB_rate after; per-slot trial
% fitness and branch flags persist across generations (start Inf and 0 here); the
% front write index is not reset when the front shrinks. Changed: jittered
% coordinates that leave the box are resampled like crossover ones.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = de2ls(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NIndsFrontMax = 18 * dim;
    PopulSize     = 2 * NIndsFrontMax;
    MemorySize    = 5;
    sigmaF        = 0.025;
    psize_xi      = 0.65;
    psize_k       = 6.5;
    minNInds      = 4;
    eb_rate_init  = 0.7;
    late_start    = 0.75;
    late_old_w    = 0.65;
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
    FitMass    = FitArrFront;                  % front fitness when the generation starts
    FitTemp    = inf(NIndsFrontMax, 1);        % last trial fitness per front slot, never reset
    eb_flag    = false(NIndsFrontMax, 1);      % branch of that last trial, never reset
    PFIndex    = 1;

    while FE < maxFE
        meanF = 0.4 + tanh(SuccessRate * 5) * 0.25;

        [~, Indices]  = sort(FitArr(1:NIndsFront), 'ascend');       % into Popul
        [~, Indices2] = sort(FitArrFront(1:NIndsFront), 'ascend');  % into PopulFront

        rankW = exp(-(0:NIndsFront-1)' / NIndsFront * 3);
        rankC = cumsum(rankW / sum(rankW));
        pressW = 3 * (NIndsFront - (0:NIndsFront-1)');              % the ordered branch's weights

        psizeval = min(NIndsFront, max(2, floor(NIndsFront * psize_xi * exp(-SuccessRate * psize_k))));
        psizeval2 = floor(NIndsFront * 0.17 * (1 - 0.5 * FE / maxFE));
        if psizeval2 <= 1
            psizeval2 = 2;
        end

        SuccessFilled = 0;
        tempSuccessCr = zeros(NIndsFront, 1);
        tempSuccessF  = zeros(NIndsFront, 1);
        FitDelta      = zeros(NIndsFront, 1);

        for IndIter = 1:NIndsFront
            if FE >= maxFE
                break;
            end
            t = FE / maxFE;                    % the reference reads the live FE count per trial

            TheChosenOne = randi(NIndsFront);
            idx1 = randi(MemorySize);
            idx2 = randi(MemorySize + 1);      % the extra slot returns the fixed 0.9

            Rand_EB = rand;
            if t < 0.7
                Rand_EB = 2;
            end
            % The reference multiplies by (1 - NFEval/MaxFEval) in integer arithmetic, i.e. by 1
            use_eb = Rand_EB < eb_rate;
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
                while F < 0
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

            % The reference resamples only crossover coordinates and evaluates jittered ones outside the box
            oob = Trial < lb | Trial > ub | ~isfinite(Trial);
            if any(oob)
                Trial(oob) = lb(oob) + rand(1, sum(oob)) .* span(oob);
            end

            ActualCr = sum(take) / dim;

            [fv, FE] = calculate_fitness(Trial', problem, FE);
            TempFit  = fv(1);
            FitTemp(TheChosenOne) = TempFit;

            if TempFit < bsf
                bsf          = TempFit;
                bsf_solution = Trial;
            end

            if TempFit <= FitArrFront(TheChosenOne)
                slot = NIndsCurrent + SuccessFilled + 1;
                Popul(slot, :) = Trial;
                FitArr(slot)   = TempFit;

                PopulFront(PFIndex, :) = Trial;
                FitArrFront(PFIndex)   = TempFit;

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
        improved = FitTemp(1:NIndsFront) <= FitMass(1:NIndsFront);
        gain     = FitMass(1:NIndsFront) - FitTemp(1:NIndsFront);
        sum_eb   = sum(gain(improved & eb_flag(1:NIndsFront)));
        sum_base = sum(gain(improved & ~eb_flag(1:NIndsFront)));
        if sum_eb ~= 0 && sum_base ~= 0
            raw = sum_eb / (sum_eb + sum_base);
            if raw > 1
                raw = 1;
            elseif raw < 0
                raw = 0;
            end
            if FE / maxFE >= late_start
                eb_rate = late_old_w * eb_rate + (1 - late_old_w) * raw;
            else
                eb_rate = raw;
            end
        else
            eb_rate = eb_rate_init;
        end
        FitMass(1:NIndsFront) = FitArrFront(1:NIndsFront);

        SuccessRate = SuccessFilled / NIndsFront;

        newNIndsFront = max(minNInds, floor((minNInds - NIndsFrontMax) / maxFE * FE + NIndsFrontMax));
        if newNIndsFront < NIndsFront
            % Each pass rescans the old front length, stale tail included, as the reference does
            for L = 1:(NIndsFront - newNIndsFront)
                [~, WorstNum] = max(FitArrFront(1:NIndsFront));
                PopulFront(WorstNum:NIndsFront-1, :) = PopulFront(WorstNum+1:NIndsFront, :);
                FitArrFront(WorstNum:NIndsFront-1)   = FitArrFront(WorstNum+1:NIndsFront);
                FitMass(WorstNum:NIndsFront-1)       = FitMass(WorstNum+1:NIndsFront);
            end
        end
        NIndsFront = newNIndsFront;

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
% Weighted Lehmer mean; 1.0 when the weights are undefined or the mean vanishes, as in MeanWL
    w = deltas / sum(deltas);
    s = sum(w .* values);
    if abs(s) > 1e-8
        m = sum(w .* values .* values) / s;
    else
        m = 1.0;
    end
end

% ----------------------------------------------------------------------- %
% Success Rate-based Adaptive Differential Evolution (L-SRTDE)
% CEC 2024 competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NIndsFrontMax = 20 * D        % Initial front size (linear reduction to 4)
%   PopulSize     = 2 * 20 * D    % Capacity of the main population buffer
%   MemorySize    = 5             % Historical memory for Cr (initialised to 1.0)
%   SuccessRate   = 0.5           % Initial success rate
%   sigmaF        = 0.02          % Std of the scaling-factor distribution
%
% Algorithm Concept:
%   - L-NTADE-style two-population scheme: a "front" holding the newest
%     successful trials and a main population holding the best ones
%   - The scaling factor is NOT success-history adapted: its mean follows the
%     previous generation's SUCCESS RATE, and the elite fraction shrinks as it
%     grows -- meanF = 0.4 + tanh(SuccessRate*5)*0.25, F ~ N(meanF, 0.02), and
%     psizeval = max(2, floor(NIndsFront*0.7*exp(-SuccessRate*7)))
%   - Mutation r-new-to-pbest/1 across the two populations:
%       u = xF_k + F*(x_prand - xF_k) + F*(xF_r1 - x_r2)
%     with xF_r1 drawn by rank from the front (weights exp(-i/NIndsFront*3))
%   - Cr memory stores the ACTUAL crossover ratio (fraction of dimensions taken
%     from the donor), not the sampled Cr, by a half-weighted Lehmer mean
%   - The front shrinks linearly, the main population is best-selected each generation
%
% Reference:
% Vladimir Stanovov, Eugene Semenkin,
% Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024
% Competition, 2024 IEEE Congress on Evolutionary Computation (CEC), 2024.
% https://doi.org/10.1109/CEC60901.2024.10611907
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' reference C++ release (L-SRTDE.cpp), including two
% quirks of the published code that are deliberately NOT "fixed": the success
% delta is measured AFTER the front slot has been overwritten, so a trial
% landing on its own parent slot contributes a zero weight; and RemoveWorst
% rescans the pre-reduction index range, so dropping more than one individual in
% a generation can see a stale trailing entry.
% The algorithm is genuinely sequential -- the front is mutated inside the
% generation loop -- so it is evaluated one trial at a time.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lsrtde(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    span  = ub - lb;

    % L-SRTDE control parameters (main.cc of the reference release)
    NIndsFrontMax = 20 * dim;
    PopulSize     = 2 * NIndsFrontMax;
    MemorySize    = 5;
    sigmaF        = 0.02;
    minNInds      = 4;

    NIndsFront   = NIndsFrontMax;
    NIndsCurrent = NIndsFrontMax;
    SuccessRate  = 0.5;
    MemoryCr     = ones(MemorySize, 1);
    MemoryIter   = 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Populations
    Popul   = repmat(lb, PopulSize, 1) + rand(PopulSize, dim) .* repmat(span, PopulSize, 1);
    FitArr  = inf(PopulSize, 1);

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

    % Main loop
    while FE < maxFE
        meanF = 0.4 + tanh(SuccessRate * 5) * 0.25;

        [~, Indices]  = sort(FitArr(1:NIndsFront), 'ascend');       % into Popul
        [~, Indices2] = sort(FitArrFront(1:NIndsFront), 'ascend');  % into PopulFront

        % Rank weights for the front donor, exp(-i/NIndsFront*3)
        rankW = exp(-(0:NIndsFront-1)' / NIndsFront * 3);
        rankC = cumsum(rankW / sum(rankW));

        psizeval = max(2, floor(NIndsFront * 0.7 * exp(-SuccessRate * 7)));
        psizeval = min(psizeval, NIndsFront);

        SuccessFilled = 0;
        tempSuccessCr = zeros(NIndsFront, 1);
        FitDelta      = zeros(NIndsFront, 1);

        for IndIter = 1:NIndsFront
            if FE >= maxFE
                break;
            end

            TheChosenOne       = randi(NIndsFront);
            MemoryCurrentIndex = randi(MemorySize);

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

            Cr = MemoryCr(MemoryCurrentIndex) + 0.05 * randn();
            Cr = min(max(Cr, 0.0), 1.0);

            base  = PopulFront(TheChosenOne, :);
            donor = base + F * (Popul(prand, :) - base) ...
                         + F * (PopulFront(Rand1, :) - Popul(Rand2, :));

            take = rand(1, dim) < Cr;
            take(randi(dim)) = true;

            Trial = base;
            Trial(take) = donor(take);

            % Out-of-bounds components are re-sampled uniformly in the box
            oob = take & (Trial < lb | Trial > ub);
            if any(oob)
                Trial(oob) = lb(oob) + rand(1, sum(oob)) .* span(oob);
            end

            ActualCr = sum(take) / dim;

            [fv, FE] = calculate_fitness(Trial', problem, FE);
            TempFit  = fv(1);

            if TempFit <= FitArrFront(TheChosenOne)
                slot = NIndsCurrent + SuccessFilled + 1;
                Popul(slot, :)  = Trial;
                FitArr(slot)    = TempFit;

                PopulFront(PFIndex, :) = Trial;
                FitArrFront(PFIndex)   = TempFit;

                if TempFit < bsf
                    bsf          = TempFit;
                    bsf_solution = Trial;
                end

                SuccessFilled = SuccessFilled + 1;
                tempSuccessCr(SuccessFilled) = ActualCr;
                % Measured after the front slot was overwritten, as in the reference: a self-overwrite gives 0
                FitDelta(SuccessFilled) = abs(FitArrFront(TheChosenOne) - TempFit);

                PFIndex = mod(PFIndex, NIndsFront) + 1;
            end

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, PopulFront(1:NIndsFront, :), FitArrFront(1:NIndsFront), ...
                population_history, fitness_history, history_index, ...
                maxFE);
        end

        SuccessRate = SuccessFilled / NIndsFront;

        % Linear reduction of the front population
        newNIndsFront = floor((minNInds - NIndsFrontMax) / maxFE * FE + NIndsFrontMax);
        newNIndsFront = max(newNIndsFront, minNInds);

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

        % Cr memory: half-weighted Lehmer mean of the actual crossover
        if SuccessFilled > 0
            MemoryCr(MemoryIter) = 0.5 * (meanWL(tempSuccessCr(1:SuccessFilled), ...
                                                 FitDelta(1:SuccessFilled)) + ...
                                          MemoryCr(MemoryIter));
            MemoryIter = mod(MemoryIter, MemorySize) + 1;
        end

        % Keep the best NIndsFront individuals in the main population
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

% Helper Functions

function idx = roulette(cumProb)
% find() can return empty when the last cumulative entry rounds below 1; falls back to the end
    idx = find(rand() <= cumProb, 1);
    if isempty(idx)
        idx = numel(cumProb);
    end
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
    if abs(s) > 1e-8
        m = sum(w .* values .^ 2) / s;
    else
        m = 1.0;
    end
end

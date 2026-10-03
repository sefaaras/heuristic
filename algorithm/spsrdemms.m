% ----------------------------------------------------------------------- %
% Structured Population Size Reduction DE with Multiple Mutation Strategies (SPSRDEMMS)
% CEC 2013 competition -- 11th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 100 -> 50 -> 25 -> 13    % Halved once g > maxFe/(pmax*NP) generations
%   pmax = 4, NPmin = 10          % Reduction stages; no halving below NPmin
%   NPbest = 6                    % Last NPbest slots run best/1 around the vector xbb
%   tau1 = tau2 = 0.1             % jDE: chance to redraw F in [0.1, 1] and CR in [0, 1]
%   rDiffBH = 0.5                 % Migration threshold on the normalised position ratio
%
% Algorithm Concept:
%   - jDE self-adaptation: every individual carries its own F and CR, redrawn with
%     probability 0.1 and inherited only when the trial wins
%   - Structured population: slots 1..NP-NPbest run rand/1/bin, the last NPbest run
%     best/1/bin around xbb, a best vector kept apart from the population
%   - xbb takes the rand part's best whenever that is better; after a generation in
%     which xbb beats it and moved enough, xbb overwrites the rand part's best member
%   - Halving keeps the fitter of x_i and x_{NP+i} for each surviving slot
%   - Binomial crossover with no forced mutant gene; trials clipped to the box;
%     generational greedy selection (strictly better)
%
% Reference:
% Ales Zamuda, Janez Brest, Efren Mezura-Montes,
% Structured Population Size Reduction Differential Evolution with Multiple
% Mutation Strategies on CEC 2013 real parameter optimization,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, 1925-1931.
% https://doi.org/10.1109/CEC.2013.6557794
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' C++ release SPSRDEMMS-code.zip in the organiser archive
% (github.com/P-N-Suganthan/CEC2013): ZADE strategy 9 with option 35 (NPbest = 6,
% NP = 100), the setting its README runs. Kept as released: a generation writes a
% second buffer that is then swapped in, so halving 25 -> 13 compares slot 13 with a
% stale row 26 left from an earlier 50-member generation; the migration test sums
% xbb_j/xchg_j (box-normalised, over xchg_j > 0) and divides by D. The rand part of a
% generation is evaluated as one batch, which changes nothing (no trial there depends on
% another's result). NaN fitness loses every comparison, as in the release.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = spsrdemms(problem)

    D = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    NPmax = 100;
    pmax = 4;
    NPmin = 10;
    NPbest = 6;
    rDiffBH = 0.5;
    NP = NPmax;
    iF = D + 1;
    iCR = D + 2;
    iFE = D + 3;

    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    FE = 0;
    bsf = inf;

    % Rows are [x, F, CR, f]; both buffers start at the release's 555 fill
    P = 555 * ones(NPmax, D + 3);
    Q = P;
    for i = 1:NP
        P(i, iF) = 0.1 + 0.9 * rand;
        P(i, iCR) = rand;
        P(i, 1:D) = lb + rand(1, D) .* (ub - lb);
    end
    bsf_solution = P(1, 1:D);
    n_eval = min(NP, maxFE);
    [fv, FE] = calculate_fitness(P(1:n_eval, 1:D)', problem, FE);
    P(1:n_eval, iFE) = fv(:);
    track(P(1:n_eval, 1:D), P(1:n_eval, iFE), P(1:NP, :));

    best = P(1, :);                    % best of the rand/1 part (the release's "best")
    for i = 2:n_eval
        if better(P(i, iFE), best(iFE))
            best = P(i, :);
        end
    end
    ibb = max(NP - NPbest, 0) + 1;
    for i = ibb + 1:NP
        if better(P(i, iFE), P(ibb, iFE))
            ibb = i;
        end
    end
    xbb = P(ibb, :);                   % bestBestPop
    xchg = xbb;                        % xbb as of its last exchange with the rand part

    g = 0;
    while FE < maxFE
        genp = floor(maxFE / (pmax * NP));
        if ~(g <= genp || floor((NP + 1) / 2) < NPmin)
            NP = floor((NP + 1) / 2);
            for i = 1:NP
                if better(P(NP + i, iFE), P(i, iFE))
                    P(i, :) = P(NP + i, :);
                end
            end
        end
        nS = max(NP - NPbest, 0);

        % rand/1 part: trials depend on P only, so the release's sequential loop is one batch
        n_r = min(nS, maxFE - FE);
        T = zeros(n_r, D + 3);
        for i = 1:n_r
            T(i, :) = jde_trial(P, i, NP, [], lb, ub, iF, iCR);
        end
        if n_r > 0
            [fv, FE] = calculate_fitness(T(:, 1:D)', problem, FE);
            T(:, iFE) = fv(:);
            for i = 1:n_r
                if better(T(i, iFE), P(i, iFE))
                    Q(i, :) = T(i, :);
                    if better(T(i, iFE), best(iFE))
                        best = T(i, :);
                    end
                else
                    Q(i, :) = P(i, :);
                end
            end
            snap = P(1:NP, :);
            snap(1:n_r, :) = Q(1:n_r, :);
            track(T(:, 1:D), T(:, iFE), snap);
        end

        % best/1 part: xbb moves after every trial, so these run one at a time
        for i = nS + 1:NP
            if FE >= maxFE
                break;
            end
            if i == nS + 1 && better(best(iFE), xbb(iFE))
                xbb = best;
                xchg = xbb;
            end
            t = jde_trial(P, i, NP, xbb(1:D), lb, ub, iF, iCR);
            [fv, FE] = calculate_fitness(t(1:D)', problem, FE);
            t(iFE) = fv(1);
            if better(t(iFE), P(i, iFE))
                Q(i, :) = t;
                if better(t(iFE), xbb(iFE))
                    xbb = t;
                end
            else
                Q(i, :) = P(i, :);
            end
            snap = P(1:NP, :);
            snap(1:i, :) = Q(1:i, :);
            track(t(1:D), t(iFE), snap);
        end

        if better(xbb(iFE), best(iFE)) && differs(xbb(1:D), xchg(1:D), lb, ub, rDiffBH)
            ib = 1;
            for i = 2:nS
                if better(Q(i, iFE), Q(ib, iFE))
                    ib = i;
                end
            end
            Q(ib, :) = xbb;
            xchg = xbb;
        end

        tmp = P;
        P = Q;
        Q = tmp;
        g = g + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsf_solution;

    % Best-so-far per evaluation, then one history sample per evaluation of S (finished slots + old rest)
    function track(X, f, S)
        n = size(X, 1);
        for q = 1:n
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = X(q, :);
            end
            ec = FE - n + q;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, S(:, 1:D), S(:, iFE), population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end
end

function tf = better(f1, f2)
% isBetterFirst with C = 0: a NaN fitness loses, a finite one beats NaN
    tf = ~isnan(f1) && (isnan(f2) || f1 < f2);
end

function t = jde_trial(P, i, NP, base, lb, ub, iF, iCR)
% jDE parameters, rand/1 (base empty) or best/1 donor, binomial crossover, box clipping
    D = numel(lb);
    t = zeros(1, D + 3);
    if rand < 0.1
        t(iF) = 0.1 + 0.9 * rand;
    else
        t(iF) = P(i, iF);
    end
    if rand < 0.1
        t(iCR) = rand;
    else
        t(iCR) = P(i, iCR);
    end
    r1 = floor(rand * NP) + 1;
    while r1 == i
        r1 = floor(rand * NP) + 1;
    end
    r2 = floor(rand * NP) + 1;
    while r2 == i || r2 == r1
        r2 = floor(rand * NP) + 1;
    end
    r3 = floor(rand * NP) + 1;
    while r3 == i || r3 == r1 || r3 == r2
        r3 = floor(rand * NP) + 1;
    end
    if isempty(base)
        v = P(r1, 1:D) + t(iF) * (P(r2, 1:D) - P(r3, 1:D));
    else
        v = base + t(iF) * (P(r2, 1:D) - P(r3, 1:D));
    end
    keep = rand(1, D) > t(iCR);
    v(keep) = P(i, keep);
    t(1:D) = min(max(v, lb), ub);
end

function tf = differs(xbb, xchg, lb, ub, r)
% Mean over D of the box-normalised ratio xbb_j / xchg_j, summed where xchg_j > 0
    span = ub - lb;
    a = (xbb - lb) ./ span;
    b = (xchg - lb) ./ span;
    m = b > 0;
    diffBH = sum(a(m) ./ b(m)) / numel(lb);
    tf = diffBH < 1 - r || diffBH > 1 + r;
end

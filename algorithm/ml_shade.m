% ----------------------------------------------------------------------- %
% Modified L-SHADE (mL-SHADE)
% CEC 2019 100-Digit Challenge -- 11th place (score 78.2)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 18*D -> 4 (linear)        % Population size, LPSR over the budget
%   H = 6, MF = MCR = 0.5          % Memory slots and their initial contents
%   rarc = 1.0, p = 0.11           % Archive size round(rarc*NP); pbest share
%   mr = 0.05                      % Chance a trial is also polynomial-mutated
%   pm = 1/D, eta = 10             % Polynomial mutation rate and distribution index
%   Nstuck = 400                   % Generations without a memory update before a perturbation
%
% Algorithm Concept:
%   - L-SHADE: current-to-pbest/1/bin with archive, F ~ Cauchy(MF, 0.1),
%     CR ~ N(MCR, 0.1) clamped to [0, 1], midpoint repair towards the parent
%   - No terminal value: a zero CR memory slot keeps drawing from N(0, 0.1) and can
%     be overwritten, so CR is never frozen at 0
%   - With probability mr the trial is polynomial-mutated and the better of the two
%     trials competes with the target
%   - Nstuck consecutive generations without a memory update flip the next slot to
%     its opposite end: MF = 1 - MF, MCR = 1 - MCR
%   - Memory by weighted Lehmer mean of F and CR; linear population size reduction
%
% Reference:
% Jia-Fong Yeh, Ting-Yu Chen, Tsung-Che Chiang,
% Modified L-SHADE for Single Objective Real-Parameter Optimization,
% 2019 IEEE Congress on Evolutionary Computation (CEC), 2019, pp. 381-386.
% https://doi.org/10.1109/CEC.2019.8789991
% ----------------------------------------------------------------------- %
% Implementation Note:
% The competition entry is ported from the paper (Table I and the slides' parameter
% table); the shared operators follow the authors' C++ (src/alg_mL-SHADE.cpp in
% github.com/danney9512/Bound-Constrained-Opt_mpmL-SHADE_mL-SHADE_L-SHADE). That
% file is a later revision (no polynomial mutation, random memory perturbation with
% probability no-success/FE, Cauchy scale 0.1 -> 0.2, rarc 2.6, CR = |CR|) and is
% not the entry scored 78.2. From the paper: the stuck counter resets on a memory
% update ("consecutive generations"); Nstuck = 6 for F3/F9 is per-function tuning
% and not transferred. A mutation that changes no gene costs no evaluation. A memory
% update that comes out non-finite keeps the old slot value.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ml_shade(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NPinit     = round(18 * D);
    Nmin       = 4;
    H          = 6;
    rarc       = 1.0;
    pbest_rate = 0.11;
    mr         = 0.05;
    pm         = 1 / D;
    eta        = 10;
    Nstuck     = 400;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    NP = min(NPinit, maxFE);
    X  = lb + rand(NP, D) .* (ub - lb);
    [fv, FE] = calculate_fitness(X', problem, FE);
    fx = fv(:);

    bsf  = inf;
    bsfx = X(1, :);
    for e = 1:NP
        if fx(e) < bsf
            bsf  = fx(e);
            bsfx = X(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fx, population_history, fitness_history, history_index, maxFE);
    end
    [fx, o] = sort(fx);
    X = X(o, :);

    Amax  = round(rarc * NP);
    A     = zeros(0, D);
    MF    = 0.5 * ones(1, H);
    MCR   = 0.5 * ones(1, H);
    k     = 1;
    stuck = 0;

    while FE < maxFE
        ii = (1:NP)';

        % Table I lines 8-10, repaired as in L-SHADE
        r  = randi(H, NP, 1);
        CR = min(max(MCR(r)' + 0.1 * randn(NP, 1), 0), 1);
        F  = MF(r)' + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
        bad = F <= 0;
        while any(bad)
            F(bad) = MF(r(bad))' + 0.1 * tan(pi * (rand(nnz(bad), 1) - 0.5));
            bad = F <= 0;
        end
        F = min(F, 1);

        % current-to-pbest/1 as CurtopBest_DonorVec: r1, r2 are redrawn together
        pNP = max(2, round(NP * pbest_rate));
        pb  = randi(pNP, NP, 1);
        NA  = size(A, 1);
        r1  = randi(NP, NP, 1);
        r2  = randi(NP + NA, NP, 1);
        bad = r1 == ii | r2 == ii | r1 == r2;
        while any(bad)
            r1(bad) = randi(NP, nnz(bad), 1);
            r2(bad) = randi(NP + NA, nnz(bad), 1);
            bad = r1 == ii | r2 == ii | r1 == r2;
        end
        XA = [X; A];
        V  = X + F .* (X(pb, :) - X + X(r1, :) - XA(r2, :));

        take = rand(NP, D) <= CR;
        take(sub2ind([NP, D], ii, randi(D, NP, 1))) = true;
        U = X;
        U(take) = V(take);
        LB = repmat(lb, NP, 1);
        UB = repmat(ub, NP, 1);
        hi = U > UB;
        U(hi) = (X(hi) + UB(hi)) / 2;
        lo = U < LB;
        U(lo) = (X(lo) + LB(lo)) / 2;

        n = min(NP, maxFE - FE);
        [fu, FE] = calculate_fitness(U(1:n, :)', problem, FE);
        fu = fu(:);
        for e = 1:n
            if fu(e) < bsf
                bsf  = fu(e);
                bsfx = U(e, :);
            end
            fe_e = FE - n + e;
            curve(fe_e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                fe_e, X, fx, population_history, fitness_history, history_index, maxFE);
        end

        % Table I lines 12-17: polynomial mutation of the trial, the better trial is kept
        for i = find(rand(n, 1) <= mr)'
            if FE >= maxFE
                break;
            end
            [y, changed] = poly_mutation(U(i, :), lb, ub, pm, eta);
            if ~changed
                continue;
            end
            [fy, FE] = calculate_fitness(y', problem, FE);
            if fy(1) < bsf
                bsf  = fy(1);
                bsfx = y;
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fx, population_history, fitness_history, history_index, maxFE);
            if fy(1) <= fu(i)
                U(i, :) = y;
                fu(i)   = fy(1);
            end
        end

        % Lines 19-30: selection over the evaluated trials
        sel    = (1:n)';
        better = fu <= fx(sel);
        strict = fu < fx(sel);
        SF  = F(sel(strict));
        SCR = CR(sel(strict));
        df  = abs(fu(strict) - fx(sel(strict)));
        A   = [A; X(sel(strict), :)]; %#ok<AGROW>
        X(sel(better), :) = U(sel(better), :);
        fx(sel(better))   = fu(better);

        if size(A, 1) > Amax
            A = A(randperm(size(A, 1), Amax), :);
        end

        % Lines 34-43: memory update, or the opposite-end perturbation after Nstuck
        if ~isempty(SF)
            w = df / sum(df);
            MF(k)  = lehmer(w, SF, MF(k));
            MCR(k) = lehmer(w, SCR, MCR(k));
            k = mod(k, H) + 1;
            stuck = 0;
        else
            stuck = stuck + 1;
            if stuck >= Nstuck
                MCR(k) = 1 - MCR(k);
                MF(k)  = 1 - MF(k);
                stuck = 0;
                k = mod(k, H) + 1;
            end
        end

        % Line 44-48: LPSR removes the worst, the archive shrinks at random
        NPnew = round(((Nmin - NPinit) / maxFE) * FE + NPinit);
        [fx, o] = sort(fx);
        X = X(o, :);
        if NPnew < NP
            NP = NPnew;
            X  = X(1:NP, :);
            fx = fx(1:NP);
        end
        Amax = round(rarc * NP);
        if size(A, 1) > Amax
            A = A(randperm(size(A, 1), Amax), :);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function m = lehmer(w, S, old)
% Weighted Lehmer mean as update_memory: 0 when the denominator is 0, old slot if non-finite
    den = sum(w .* S);
    if den == 0
        m = 0;
    else
        m = sum(w .* S .^ 2) / den;
    end
    if ~isfinite(m)
        m = old;
    end
end

function [y, changed] = poly_mutation(x, lb, ub, pm, eta)
% Deb's polynomial mutation as alg_mutation.cpp: each gene with probability pm, result clamped
    y = x;
    j = find(rand(1, numel(x)) <= pm & ub > lb);
    changed = ~isempty(j);
    if ~changed
        return;
    end
    span = ub(j) - lb(j);
    d1   = (x(j) - lb(j)) ./ span;
    d2   = (ub(j) - x(j)) ./ span;
    rnd  = rand(1, numel(j));
    mp   = 1 / (eta + 1);
    dq   = zeros(1, numel(j));
    lo   = rnd <= 0.5;
    val  = 2 * rnd(lo) + (1 - 2 * rnd(lo)) .* (1 - d1(lo)) .^ (eta + 1);
    dq(lo) = val .^ mp - 1;
    val  = 2 * (1 - rnd(~lo)) + 2 * (rnd(~lo) - 0.5) .* (1 - d2(~lo)) .^ (eta + 1);
    dq(~lo) = 1 - val .^ mp;
    y(j) = min(max(x(j) + dq .* span, lb(j)), ub(j));
end

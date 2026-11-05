% ----------------------------------------------------------------------- %
% Memetic Algorithm with CMA-ES Local Search Chains (MA-CMA-Chains)
% MA-LS-Chains with the CMA-ES local search, as in the Rmalschains package
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 50                 % Steady-state RCGA population (package default)
%   istep    = 500                % FEs per local-search application (chain step)
%   effort   = 0.5                % LS share of the FEs: 500 RCGA FEs between LS steps
%   alpha    = 0.5, n_nam = 3     % BLX-alpha, NAM candidates for the father
%   p_mut    = 0.125              % BGA mutation of one gene, per offspring
%   delta    = 1e-9               % Least improvement that keeps a chain (threshold/10)
%   CMA-ES   = cmaes.c 3.02.03 defaults, lambda = 4 + floor(3*ln(D))
%
% Algorithm Concept:
%   - Steady-state RCGA: a random mother, as father the farthest of 3 random
%     candidates (NAM); BLX-0.5 child, BGA mutation; it replaces the worst if better
%   - Every 500 RCGA FEs the best individual not yet marked failed gets 500 FEs of
%     CMA-ES, started at it with per-coordinate std 0.5*nearest-neighbour distance
%   - LS chains: the CMA-ES state is stored with the individual and resumed when
%     it is picked again; the LS result replaces the individual only if it improves
%   - A step that gains less than delta marks the individual failed and drops its
%     chain; a replaced individual loses its chain; with all marked, pick at random
%
% Reference:
% Daniel Molina, Manuel Lozano, Carlos Garcia-Martinez, Francisco Herrera,
% Memetic Algorithms for Continuous Optimisation Based on Local Search Chains,
% Evolutionary Computation 18 (1) (2010) 27-63.
% https://doi.org/10.1162/evco.2010.18.1.18102
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the C++ core of Rmalschains 0.2-12 (cran/Rmalschains: malschains.cc,
% ssga.cc, cmaeshan.cc, Hansen's cmaes.c 3.02.03) with malschains.control() defaults,
% not the paper's MA-LSCh-CMA settings (pop 60, delta 1e-8). Kept as released, unlike
% the paper: with every individual marked a random one is improved, not the best; a
% step gaining < delta neither replaces the individual nor keeps its chain; sigma is
% per coordinate, 0.5*the least coordinate distance + 0.001 (individual 1 skipped,
% 0.001 alone for the 1st or 2nd), not half the nearest-neighbour distance; the first
% RCGA phase has 450 FEs; raw coordinates. Changed: the best is kept per FE (the
% release returns the population best); a CMA-ES generation cut by maxFe is not used;
% a non-positive eigenvalue stops the chain (the release samples NaN once).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = macmachains(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    pop_size = 50;
    istep    = 500;
    effort   = 0.5;
    alpha    = 0.5;
    n_nam    = 3;
    p_mut    = 0.125;
    delta    = 1e-8 / 10;                    % Running::getThreshold()/10 in hasImprovedEnough
    frec     = floor(effort * istep / (1 - effort));   % calculateFrec(0, 0, istep, effort)

    cm = cma_defaults(D);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    n0 = min(pop_size, maxFE);
    P  = lb + (ub - lb) .* rand(n0, D);
    [F, FE] = calculate_fitness(P', problem, FE);
    F = F(:)';
    for k = 1:n0
        if isempty(bsfx) || F(k) < bsf
            bsf  = F(k);
            bsfx = P(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history( ...
            k, P, F, population_history, fitness_history, history_index, maxFE);
    end
    F(isnan(F)) = inf;                       % NaN ranks last

    nonimp = zeros(1, n0);                   % the individual's "non_improved" counter
    chain  = cell(1, n0);                    % LSParametersMemory: one CMA-ES state per slot
    sub    = n0;                             % the RCGA's sub-running already counts the initial FEs

    while FE < maxFE && n0 == pop_size
        % RCGA phase (SSGA::realApply) until its sub-running holds frec FEs
        while sub < frec && FE < maxFE
            [mom, dad] = nam_select(P, n_nam);
            child = blx(P(mom, :), P(dad, :), alpha, lb, ub);
            if rand <= p_mut
                g = floor(D * rand) + 1;
                child(g) = bga(child(g), lb(g), ub(g));
            end
            [fc, FE] = calculate_fitness(child', problem, FE);
            fc = fc(1);
            sub = sub + 1;
            if fc < bsf
                bsf  = fc;
                bsfx = child;
            end
            if isnan(fc)
                fc = inf;
            end
            [~, wi] = max(F);
            if fc < F(wi)
                P(wi, :) = child;
                F(wi) = fc;
                nonimp(wi) = 0;
                chain{wi} = [];              % notifyChange: a replaced individual loses its chain
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                FE, P, F, population_history, fitness_history, history_index, maxFE);
        end
        sub = 0;
        if FE >= maxFE
            break;
        end

        % SelectBestToImprove: best individual whose counter is 0, else a random one
        cand = find(nonimp == 0);
        if ~isempty(cand)
            [~, kb] = min(F(cand));
            pos = cand(kb);
        else
            pos = floor(pop_size * rand) + 1;
        end
        st = chain{pos};
        if isempty(st)
            st = cma_new(P(pos, :)', init_std(P, pos), cm);
        end
        sol    = P(pos, :);
        fit    = F(pos);
        oldfit = fit;

        % CMAESHansen::apply for istep FEs
        maxeval   = istep - cm.lam;
        counteval = 0;
        while ~cma_stop(st, cm) && counteval < maxeval && FE < maxFE
            [st, X] = cma_sample(st, cm);
            if st.dead
                break;
            end
            X  = min(max(X, lb'), ub');      % domain->clip on the sampled points themselves
            ne = min(cm.lam, maxFE - FE);
            [f, FE] = calculate_fitness(X(:, 1:ne), problem, FE);
            f = f(:)';
            counteval = counteval + ne;
            for k = 1:ne
                if f(k) < bsf
                    bsf  = f(k);
                    bsfx = X(:, k)';
                end
                curve(FE - ne + k) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    FE - ne + k, P, F, population_history, fitness_history, history_index, maxFE);
            end
            if ne < cm.lam
                break;
            end
            f(isnan(f)) = inf;
            st = cma_update(st, X, f, cm);
        end
        if counteval > 0 && st.has_best
            fb = st.fbest;
            xb = st.xbest;
            % The best estimator of the optimum is the mean, so it is evaluated once
            if FE < maxFE
                [fm, FE] = calculate_fitness(st.xmean, problem, FE);
                fm = fm(1);
                if fm < bsf
                    bsf  = fm;
                    bsfx = st.xmean';
                end
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    FE, P, F, population_history, fitness_history, history_index, maxFE);
                if fm < fb
                    fb = fm;
                    xb = st.xmean;
                end
            end
            if fb < fit
                sol = xb';
                fit = fb;
            end
        end

        % hasImprovedEnough: strictly better by at least delta
        if fit < oldfit && abs(fit - oldfit) >= delta
            P(pos, :) = sol;
            F(pos) = fit;
            chain{pos} = st;
        else
            nonimp(pos) = nonimp(pos) + 1;
            chain{pos} = [];
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% SelectNAM(3): random mother, then the farthest (distreal) of 3 further distinct individuals
function [mom, dad] = nam_select(P, n_nam)
    idx = randperm(size(P, 1), n_nam + 1);
    mom = idx(1);
    cand = idx(2:end);
    d = sqrt(sum((P(cand, :) - P(mom, :)) .^ 2, 2)) / size(P, 2);
    [~, b] = max(d);
    dad = cand(b);
end

% CrossBLX(alpha), the interval clipped to the box
function c = blx(x, y, alpha, lb, ub)
    lo = min(x, y);
    hi = max(x, y);
    I  = (hi - lo) * alpha;
    A  = max(lo - I, lb);
    B  = min(hi + I, ub);
    c  = A + rand(size(x)) .* (B - A);
end

% MutationBGA: +-0.1*range*sum(a_i*2^-i), a_i = 1 with probability 1/16, clipped to the box
function v = bga(v, lo, hi)
    s = sum((rand(1, 16) < 1 / 16) .* 2 .^ -(0:15));
    if s == 0
        return;
    end
    r = 0.1 * (hi - lo);
    if rand < 0.5
        v = min(v + r * s, hi);
    else
        v = max(v - r * s, lo);
    end
end

% getInitOptions: 0.5*min_vector_distance + 0.001, with the release's index skip
function sd = init_std(P, pos)
    x = P(pos, :);
    first = 1;
    if ~isequal(P(1, :), x)
        first = 2;                           % 'if (ind->sol() != x)' skips individual 1, as released
    end
    md = abs(x - P(first, :));
    for i = first + 1:size(P, 1)
        d = abs(x - P(i, :));
        upd = d < md & d > 0;
        md(upd) = d(upd);
    end
    sd = (md * 0.5 + 0.001)';
end

% readpara_SupplementDefaults of cmaes.c 3.02.03 for dimension N
function cm = cma_defaults(N)
    cm.N   = N;
    cm.lam = 4 + floor(3 * log(N));
    cm.mu  = floor(cm.lam / 2);
    w = log(cm.mu + 1) - log(1:cm.mu)';
    cm.mueff = sum(w) ^ 2 / sum(w .^ 2);
    cm.w   = w / sum(w);
    cm.cs  = (cm.mueff + 2) / (N + cm.mueff + 3);
    cm.cc  = 4 / (N + 4);
    cm.mucov = cm.mueff;
    t1 = 2 / ((N + 1.4142) ^ 2);
    t2 = min(1, (2 * cm.mueff - 1) / ((N + 2) ^ 2 + cm.mueff));
    cm.ccov = (1 / cm.mucov) * t1 + (1 - 1 / cm.mucov) * t2;
    max_fe   = 900 * (N + 3) ^ 2;
    max_iter = ceil(max_fe / cm.lam);
    cm.damps = (1 + 2 * max(0, sqrt((cm.mueff - 1) / (N + 1)) - 1)) * ...
               max(0.3, 1 - N / (1e-6 + min(max_iter, max_fe / cm.lam))) + cm.cs;
    cm.modulo  = 1 / cm.ccov / N / 10;
    cm.chiN    = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));
    cm.histlen = 10 + ceil(30 * N / cm.lam);
    cm.dmax    = 2 ^ 53 / 1000;              % dMaxSignifKond
end

% cmaes_init with xstart and per-coordinate initial stds
function st = cma_new(x, sd, cm)
    N = cm.N;
    tr = sum(sd .^ 2);
    st.sigma = sqrt(tr / N);
    st.Dv    = sd * sqrt(N / tr);
    st.C     = diag(st.Dv .^ 2);
    st.B     = eye(N);
    st.minEW = min(st.Dv) ^ 2;
    st.maxEW = max(st.Dv) ^ 2;
    st.xmean = x;
    st.xold  = x;
    st.pc    = zeros(N, 1);
    st.ps    = zeros(N, 1);
    st.gen   = 0;
    st.state = 0;
    st.geneig   = 0;
    st.uptodate = true;
    st.hist  = zeros(cm.histlen, 1);
    st.fvals = zeros(1, cm.lam);
    st.xbest = zeros(N, 1);
    st.fbest = 0;
    st.has_best = false;                     % no full generation evaluated yet
    st.istd  = sd;
    st.dead  = false;
end

% cmaes_TestForTermination with the release's settings (TolX = 0 and MaxIter disabled)
function stop = cma_stop(st, cm)
    stop = st.dead;
    if stop
        return;
    end
    if st.gen > 0
        h = st.hist(1:min(st.gen, cm.histlen));
        if max(max(h), max(st.fvals)) - min(min(h), min(st.fvals)) <= 1e-12
            stop = true;
            return;
        end
    end
    if st.gen > cm.histlen && max(st.hist) - min(st.hist) <= 1e-13
        stop = true;
        return;
    end
    sd = st.sigma * sqrt(diag(st.C));
    if any(sd > 1e3 * st.istd) || st.maxEW >= st.minEW * cm.dmax
        stop = true;
        return;
    end
    if any(all(st.xmean == st.xmean + 0.1 * st.sigma * (st.B .* st.Dv'), 1)) || ...
            any(st.xmean == st.xmean + 0.2 * sd)
        stop = true;
    end
end

% cmaes_SamplePopulation: lazy eigendecomposition every modulo generations
function [st, X] = cma_sample(st, cm)
    X = [];
    if ~st.uptodate && st.gen >= st.geneig + cm.modulo
        if ~all(isfinite(st.C(:)))
            st.dead = true;
            return;
        end
        [Bn, Em] = eig((st.C + st.C') / 2);
        d = diag(Em);
        if ~all(d > 0)
            st.dead = true;
            return;
        end
        st.B = Bn;
        st.minEW = min(d);
        st.maxEW = max(d);
        st.Dv = sqrt(d);
        st.uptodate = true;
        st.geneig = st.gen;
    end
    X = st.xmean + st.sigma * (st.B * (st.Dv .* randn(cm.N, cm.lam)));
    if st.state == 3 || st.gen == 0
        st.gen = st.gen + 1;
    end
    st.state = 1;
end

% cmaes_UpdateDistribution with Adapt_C2; X holds the clipped samples, as the release keeps them
function st = cma_update(st, X, f, cm)
    lam = cm.lam;
    st.fvals = f;
    [~, r] = sort(f(end:-1:1));
    idx = lam + 1 - r;                       % Sorted_index: on ties the later sample first
    if f(idx(1)) == f(idx(floor(lam / 2) + 1))
        st.sigma = st.sigma * exp(0.2 + cm.cs / cm.damps);
    end
    st.hist = [f(idx(1)); st.hist(1:end-1)];
    if st.fbest > f(idx(1)) || st.gen == 1
        st.xbest = X(:, idx(1));
        st.fbest = f(idx(1));
    end
    st.has_best = true;
    sel = idx(1:cm.mu);
    st.xold  = st.xmean;
    st.xmean = X(:, sel) * cm.w;
    BDz = sqrt(cm.mueff) * (st.xmean - st.xold) / st.sigma;
    z = (st.B' * BDz) ./ st.Dv;
    st.ps = (1 - cm.cs) * st.ps + sqrt(cm.cs * (2 - cm.cs)) * (st.B * z);
    psxps = st.ps' * st.ps;
    hsig = sqrt(psxps) / sqrt(1 - (1 - cm.cs) ^ (2 * st.gen)) / cm.chiN < 1.4 + 2 / (cm.N + 1);
    st.pc = (1 - cm.cc) * st.pc + hsig * sqrt(cm.cc * (2 - cm.cc)) * BDz;
    if cm.ccov ~= 0
        Y = (X(:, sel) - st.xold) / st.sigma;
        st.C = (1 - cm.ccov) * st.C + cm.ccov * (1 / cm.mucov) * ...
               (st.pc * st.pc' + (1 - hsig) * cm.cc * (2 - cm.cc) * st.C) + ...
               cm.ccov * (1 - 1 / cm.mucov) * ((Y .* cm.w') * Y');
        st.uptodate = false;
    end
    st.sigma = st.sigma * exp((sqrt(psxps) / cm.chiN - 1) * cm.cs / cm.damps);
    st.state = 3;
end

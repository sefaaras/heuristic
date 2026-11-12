% ----------------------------------------------------------------------- %
% Cat Swarm Optimization (CSO)
% Stored as catso: the acronym CSO collides with cso (Competitive Swarm Optimizer)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 160                     % Number of cats
%   MR = 0.02                   % Mixture ratio: share of cats put in tracing mode
%   SMP = 5, SPC = true         % Seeking memory pool; the present position is a candidate
%   SRD = 0.2                   % Seeking range: a changed coordinate is scaled by 1 +- 0.2
%   CDC = 0.8                   % Share of the coordinates changed in each copy
%   c1 = 2.0                    % Tracing acceleration constant
%   Vmax = (ub - lb)/2          % Tracing velocity cap per dimension
%
% Algorithm Concept:
%   - Every generation all cats are evaluated and the best one so far is kept;
%     floor(MR*N) randomly picked cats trace, the others seek
%   - Seeking: SMP-1 copies, each with floor(CDC*D) random coordinates multiplied by
%     1 + SRD or 1 - SRD (a coin per coordinate); the cat moves to one of SMP candidates
%   - A candidate is drawn with probability |FS_i - FS_max|/(FS_max - FS_min), so
%     the worst never wins; all-equal fitnesses give a uniform draw
%   - Tracing: v = v + U(0,1)*c1*(x_best - x) per dimension, capped at Vmax, x = x + v
%
% Reference:
% Shu-Chuan Chu, Pei-wei Tsai, Jeng-Shyang Pan,
% Cat swarm optimization,
% PRICAI 2006: Trends in Artificial Intelligence, Lecture Notes in Computer Science
% 4099 (2006) 854-858.
% https://doi.org/10.1007/978-3-540-36668-3_94
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the authors' extended version (Chu, Tsai, IJICIC 3(1) 2007 163-173, Secs. 3.2-3.4,
% Tables 2-4), as the LNCS paper is closed; floor counts as in Tsai et al., ACI 1(1) 2021.
% "Plus or minus SRD percents the present values" is read literally, x*(1 +- SRD) with a
% coin per coordinate: the ACI paper's U(-1,1)*SRD, or one factor per copy, miss the Figs.
% Step 3 re-evaluates every cat each generation, the seeking cats' chosen candidates too;
% kept and counted, as ACI 2021 Alg. 3 counts it. The tracing cap is not given: Vmax =
% Xmax as in the paper's PSO settings (Table 4). The paper searches unbounded; positions
% are clamped here, velocities untouched. A coordinate at 0 never moves in seeking (kept).
% IJICIC 2007 Figs. 2, 6, 4 (D = 30, 2000 generations, read off the plots) against 10
% seeded runs here; then U(-1,1) per coordinate (10 runs) and one factor per copy (5 runs):
%      Rosenbrock   ~30 / 28.2         U(-1,1) 38.5   per copy 27.2
%      Rastrigin    ~1.4 / 1.93        U(-1,1) 98.6   per copy 0 (collapse onto the origin)
%      Griewank     ~6e-10 / 7.6e-6    U(-1,1) 0.040  per copy 0
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = catso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N    = 160;
    MR   = 0.02;
    SMP  = 5;
    SRD  = 0.2;
    CDC  = 0.8;
    c1   = 2.0;
    vmax = span / 2;

    ntrace = floor(MR * N);
    nchg   = max(1, floor(CDC * D));
    ncopy  = SMP - 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;
    v = (2 * rand(N, D) - 1) .* vmax;
    f = inf(N, 1);
    trace = false(N, 1);
    trace(randperm(N, ntrace)) = true;

    xbest = x(1, :);
    fbest = inf;
    bsf          = inf;
    bsf_solution = x(1, :);
    % Mirror for the recorder: each cat at the position its fitness was measured
    xrec = x;
    frec = inf(N, 1);
    nrec = 0;

    while FE < maxFE
        % Step 3: evaluate every cat and keep the best one so far in memory
        n = min(N, maxFE - FE);
        [fv, FE] = calculate_fitness(x(1:n, :)', problem, FE);
        f(1:n) = fv(:);
        xrec(1:n, :) = x(1:n, :);
        frec(1:n)    = f(1:n);
        nrec = max(nrec, n);
        for k = 1:n
            if f(k) < bsf
                bsf          = f(k);
                bsf_solution = x(k, :);
            end
            ec = FE - n + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, xrec(1:nrec, :), frec(1:nrec), population_history, fitness_history, ...
                history_index, maxFE);
        end
        [fm, km] = min(f(1:n));
        if fm < fbest
            fbest = fm;
            xbest = x(km, :);
        end
        if FE >= maxFE
            break;
        end

        % Step 4, tracing cats (Eqs. 2-3)
        for k = find(trace)'
            vk = v(k, :) + rand(1, D) .* c1 .* (xbest - x(k, :));
            vk = min(max(vk, -vmax), vmax);
            v(k, :) = vk;
            x(k, :) = min(max(x(k, :) + vk, lb), ub);
        end

        % Step 4, seeking cats: their copies depend on nothing shared, so one batch
        seek = find(~trace);
        ns   = numel(seek);
        cand = repelem(x(seek, :), ncopy, 1);
        for j = 1:size(cand, 1)
            sel = randperm(D, nchg);
            cand(j, sel) = (1 + (1 - 2 * (rand(1, nchg) < 0.5)) * SRD) .* cand(j, sel);
        end
        cand = min(max(cand, lb), ub);

        nc = min(ns * ncopy, maxFE - FE);
        [fv, FE] = calculate_fitness(cand(1:nc, :)', problem, FE);
        fc = inf(ns * ncopy, 1);
        fc(1:nc) = fv(:);

        for s = 1:ns
            k   = seek(s);
            idx = (s - 1) * ncopy + (1:ncopy);
            idx = idx(idx <= nc);
            if isempty(idx)
                continue;
            end
            % SPC: the present position, already measured in Step 3, is candidate 1
            fs = [f(k); fc(idx)];
            q  = seek_pick(fs);
            if q > 1
                x(k, :) = cand(idx(q - 1), :);
                f(k)    = fs(q);
            end
            xrec(k, :) = x(k, :);
            frec(k)    = f(k);
        end
        for j = 1:nc
            if fc(j) < bsf
                bsf          = fc(j);
                bsf_solution = cand(j, :);
            end
            ec = FE - nc + j;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, xrec, frec, population_history, fitness_history, history_index, maxFE);
        end

        % Step 5: re-pick the tracing cats
        trace(:) = false;
        trace(randperm(N, ntrace)) = true;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Eq. 1 roulette; a non-finite FS gets probability 0 and the all-equal rule uses the finite ones
function q = seek_pick(fs)
    ok = isfinite(fs);
    w  = zeros(size(fs));
    if any(ok)
        fmax = max(fs(ok));
        fmin = min(fs(ok));
        if fmax > fmin
            w(ok) = (fmax - fs(ok)) / (fmax - fmin);
        else
            w(ok) = 1;
        end
    else
        w(:) = 1;
    end
    c = cumsum(w);
    q = find(c > rand * c(end), 1);
    % rand*c(end) can round up to c(end): fall back to the last candidate with weight
    if isempty(q)
        q = find(w > 0, 1, 'last');
    end
end

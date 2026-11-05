% ----------------------------------------------------------------------- %
% Charged System Search (CSS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20, CMS = N/4               % Charged particles (CPs), charged-memory size
%   a = 1                           % Radius between the r/a^3 and 1/r^2 force laws
%   CMCR = 0.95, PAR = 0.10         % Memory-considering and pitch-adjusting rates
%   bw = 0.01*(ub - lb)             % Pitch-adjustment bandwidth, per dimension
%   ka = 0.5 -> 1, kv = 0.5 -> 0    % Eq. (24), linear in iter/itermax
%   itermax = ceil((maxFe - N)/N)   % Iterations the budget allows
%   eps = 1e-10                     % Singularity guard of r_ij, box-normalised units
%
% Algorithm Concept:
%   - Charge q_i = (f_i - f_worst)/(f_best - f_worst), f_best the best so far
%   - Separation r_ij = |X_i - X_j|/(|(X_i + X_j)/2 - X_best| + eps)
%   - CP i pulls CP j by q_i*r_ij/a^3*(X_i - X_j) inside radius a and by
%     q_i/r_ij^2*(X_i - X_j) outside; a better i always pulls, a worse one by chance
%   - The CMS best solutions found so far pull in place of the CMS worst CPs
%   - X_new = X + rand*ka*F + rand*kv*V, V = X_new - X: the pull grows, inertia fades
%   - A coordinate outside the box comes from the memory w.p. CMCR (pitch-adjusted
%     w.p. PAR), else uniformly from the box; the memory keeps the CMS best
%
% Reference:
% Ali Kaveh, Siamak Talatahari,
% A novel heuristic optimization method: charged system search,
% Acta Mechanica 213 (2010) 267-289.
% https://doi.org/10.1007/s00707-009-0270-4
% ----------------------------------------------------------------------- %
% Implementation Note:
% Eq. (19) as printed, (f_i - f_best)/(f_j - f_i) > rand, is never positive when CP i
% is the worse one, so the rule would collapse to Eq. (18); the ratio is read with i and
% j exchanged. kv follows Eq. (24), not the (1 + t) that Eq. (25) prints. Unspecified,
% chosen: eps = 1e-10, bw = 0.01*(ub - lb), a uniform CM member per coordinate, the
% pitch step around that value then a clamp; V of Eq. (26) is taken before the Rule 7
% correction, the pseudo-code's order. r_ij uses box-normalised coordinates (the same on
% a uniform box); being a ratio, it keeps a = 1 box-free. Non-finite fitness ranks as
% the worst (+Inf, NaN) or best (-Inf) finite one, equal fitness gives unit charges.
% The last iteration evaluates only the CPs the budget still pays for. At the Table 2
% setting (50 runs, 1e-4): Hartman6 42/50, Rastrigin 28/50 vs 50/50; the rest >= 48/50.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = css(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;
    unit  = span;
    unit(unit <= 0) = 1;   % a zero-width dimension adds nothing to r_ij

    N    = 20;
    CMS  = N / 4;
    a    = 1;
    CMCR = 0.95;
    PAR  = 0.10;
    bw   = 0.01 * span;
    tiny = 1e-10;
    itermax = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = lb + rand(N, D) .* span;   % Eq. (16)
    V = zeros(N, D);               % Eq. (17)

    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(X(1:m, :)', problem, FE);
    fit = inf(N, 1);
    fit(1:m) = fv(:);

    best_fitness  = inf;
    best_solution = X(1, :);
    for k = 1:m
        if fit(k) < best_fitness
            best_fitness  = fit(k);
            best_solution = X(k, :);
        end
        curve(k) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            k, X(1:m, :), fit(1:m), population_history, fitness_history, history_index, maxFE);
    end

    % Charged memory: the CMS best so far, NaN sorted last
    nCM = min(CMS, m);
    [~, order] = sort(fit(1:m));
    CM  = X(order(1:nCM), :);
    CMf = fit(order(1:nCM));

    Xrec = X;
    frec = fit;
    iter = 0;
    while FE < maxFE
        iter = iter + 1;
        t  = min(iter / itermax, 1);
        ka = 0.5 * (1 + t);   % Eq. (24)
        kv = 0.5 * (1 - t);

        g  = ranked([fit; CMf]);
        gc = g(1:N);
        fbest  = min(g);
        fworst = max(gc);
        if fworst > fbest
            q = min(max((g - fworst) / (fbest - fworst), 0), 1);   % Eq. (14)
        else
            q = ones(size(g));
        end

        % Rule 6: the memory attracts in place of the nCM worst CPs
        [~, rk] = sort(gc);
        keep = rk(1:N - nCM);
        XA = [X(keep, :); CM];
        fA = [gc(keep); g(N + 1:end)];
        qA = [q(keep); q(N + 1:end)];
        nA = numel(fA);
        [~, ib] = min(gc);

        % Eq. (15) on box-normalised coordinates; nA x N, attractor i by attracted j
        UA  = permute((XA - lb) ./ unit, [1 3 2]);
        UC  = permute((X - lb) ./ unit, [3 1 2]);
        UB  = permute((X(ib, :) - lb) ./ unit, [3 1 2]);
        num = sqrt(sum((UA - UC).^2, 3));
        den = sqrt(sum(((UA + UC) / 2 - UB).^2, 3)) + tiny;
        r   = num ./ den;

        % Eq. (20): i1 = 1 inside the radius, i2 = 1 outside
        inside = r < a;
        w = qA .* (inside .* r / a^3 + ~inside ./ max(r, a).^2);

        % Eq. (19) with the ratio's i and j exchanged (see the note); ties fall to IEEE 0/0, x/0
        fC = gc';
        p  = (fA < fC) | ((fC - fbest) ./ (fA - fC) > rand(nA, N));

        W = w .* p;
        W(sub2ind([nA, N], 1:N - nCM, keep')) = 0;   % no CP attracts itself

        % Eq. (25): F_j/m_j, where q_j of Eq. (20) cancels against m_j = q_j
        dX = permute(XA, [1 3 2]) - permute(X, [3 1 2]);
        F  = reshape(sum(W .* dX, 1), N, D);

        Xn = rand(N, 1) .* ka .* F + rand(N, 1) .* kv .* V + X;   % Eq. (22), dt = 1
        V  = Xn - X;                                               % Eq. (26)
        V(~isfinite(V)) = 0;   % guard, never fires on a finite swarm

        % Rule 7, Eq. (27): harmony-search correction of each coordinate outside the box
        bad = Xn < lb | Xn > ub | ~isfinite(Xn);
        if any(bad(:))
            [~, ci] = find(bad);
            nb  = numel(ci);
            lo  = reshape(lb(ci), [], 1);
            hi  = reshape(ub(ci), [], 1);
            val = CM(sub2ind(size(CM), randi(nCM, nb, 1), ci));
            from_cm = rand(nb, 1) < CMCR;
            adjust  = from_cm & rand(nb, 1) < PAR;
            step    = reshape(bw(ci), [], 1) .* (2 * rand(nb, 1) - 1);
            val(adjust) = val(adjust) + step(adjust);
            fresh = lo + rand(nb, 1) .* (hi - lo);
            val(~from_cm) = fresh(~from_cm);
            Xn(bad) = min(max(val, lo), hi);
        end

        m = min(N, maxFE - FE);
        [fv, FE] = calculate_fitness(Xn(1:m, :)', problem, FE);
        fn = fv(:);
        Xrec(1:m, :) = Xn(1:m, :);
        frec(1:m)    = fn;
        for k = 1:m
            if fn(k) < best_fitness
                best_fitness  = fn(k);
                best_solution = Xn(k, :);
            end
            ec = FE - m + k;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, history_index, maxFE);
        end
        if FE >= maxFE
            break;
        end

        X   = Xn;
        fit = fn;

        % Step 5: merge and keep the nCM best; the stable sort keeps old entries on ties
        [~, order] = sort([CMf; fit]);
        pool = [CM; X];
        pf   = [CMf; fit];
        CM   = pool(order(1:nCM), :);
        CMf  = pf(order(1:nCM));
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

% Finite stand-ins for ranking: +Inf and NaN as the worst finite value, -Inf as the best
function g = ranked(f)
    g  = f;
    ok = isfinite(f);
    if ~any(ok)
        g(:) = 0;
        return;
    end
    g(isnan(f) | f == inf) = max(f(ok));
    g(f == -inf) = min(f(ok));
end

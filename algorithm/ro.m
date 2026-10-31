% ----------------------------------------------------------------------- %
% Ray Optimization (RO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nAgents = 24                           % Rays (light-ray agents)
%   eta     = nd/nt = 0.45                 % Refraction index ratio of the two media
%   stoch   = 0.35                         % Chance a sub-space vector is redrawn at random
%   dDiv    = 7.5                          % d: divides the sub-space diagonal a
%   ite     = ceil((maxFe - nAgents)/nAgents)  % Iterations of the cockshut schedule
%   shrink  = 0.9                          % Share of the way to the box crossing a violator moves
%   onStep  = 0.001                        % Times rand: step length of a ray sitting on O
%
% Algorithm Concept:
%   - Variables split into consecutive 2-D sub-spaces, plus one 3-D sub-space when
%     D is odd; each sub-space carries its own movement vector
%   - Cockshut point O = ((ite + k)*GB + (ite - k)*LB_i)/(2*ite): from the midpoint of
%     the global and the personal best at k = 0 to the global best at k = ite
%   - Snell refraction, n = (X - O)/|X - O|, d the last movement: new vector
%     (-n*sqrt(1 - eta^2*sin^2) + eta*(d - (d.n)n))*|X - O|, bent toward O
%   - With probability stoch the sub-space vector is instead a random direction of
%     length rand*a/d, a the diagonal of the sub-space's box
%   - A ray that would leave the box moves 0.9 of the way to the crossing point
%
% Reference:
% A. Kaveh, M. Khayatazad,
% A new meta-heuristic method: Ray Optimization,
% Computers & Structures 112-113 (2012) 283-294.
% https://doi.org/10.1016/j.compstruc.2012.09.003
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' restatement in A. Kaveh, M. Khayatazad, Optimal design of
% cantilever retaining walls using ray optimization method, IJST Trans. Civ. Eng. 38(C1+)
% (2014) 261-274, Sec. 3, Eq. (8), App. B; the 2012 paper and Kaveh's MATLAB chapter are
% paywalled, not read. nd/nt, stoch, d and 24 agents are that paper's (a 7-variable design).
% Readings: sin^2 in (B-1) is of the incidence angle, 1 - (d.n)^2; the 3-D sub-space is the
% last three variables; (B-2)'s outer branches are taken as d - (d.n)n, their exact value (as
% printed, rounding becomes an O(1) sideways step when d is parallel to n). Harness: ite from
% the budget. Guard: a zero movement vector (a ray rounded onto the box face, NaN as written)
% is read as d = -n, or a random direction on O. Mean evaluations to f* + 1e-4, 30 runs:
%     Tsoulos set                 AP   Bf2  Camel  CM   EXP2  EXP8  EXP16        Griewank  GoldPrice
%     2012 paper, in IRO Table 3  331  582  332    802  136   1287  17236 (46%)  1091      451
%     this port, maxFe = 2e4      355  668  360    853  142   1504  15165 (70%)  1148      449
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ro(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    nAgents = 24;
    eta     = 0.45;
    stoch   = 0.35;
    dDiv    = 7.5;
    shrink  = 0.9;
    onStep  = 0.001;
    ite     = max(1, ceil((maxFE - nAgents) / nAgents));

    % Sub-spaces: D/2 pairs, or (D - 3)/2 pairs and the last three variables as one triple
    nPair = floor(D / 2) - mod(D, 2);
    P1 = 1:2:2*nPair;
    P2 = P1 + 1;
    T3 = [];
    if mod(D, 2) == 1
        T3 = D-2:D;
    end
    span  = ub - lb;
    aPair = sqrt(span(P1).^2 + span(P2).^2);   % Eq. (8): a of each 2-D sub-space
    aTrip = sqrt(sum(span(T3).^2));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Scattering, Eqs. (6)-(7); movement vectors normalised per sub-space
    X = lb + rand(nAgents, D) .* span;
    V = -1 + 2 * rand(nAgents, D);
    V = unit_subspaces(V, P1, P2, T3);

    m = min(nAgents, maxFE);
    [fit, FE] = calculate_fitness(X(1:m, :)', problem, FE);
    fit = fit(:)';
    Xrec = X(1:m, :);
    frec = fit;
    best_fitness  = inf;
    best_solution = X(1, :);
    stamp_batch(X(1:m, :), fit, m);
    LB  = X;
    LBf = fit;

    k = 0;
    while FE < maxFE
        % Motion refinement: a violator's vector shrinks to 0.9 of the way to the box
        Xn = X + V;
        tB = inf(nAgents, D);
        over  = Xn > ub;
        under = Xn < lb;
        UB  = repmat(ub, nAgents, 1);
        LBx = repmat(lb, nAgents, 1);
        tB(over)  = (UB(over) - X(over)) ./ V(over);
        tB(under) = (LBx(under) - X(under)) ./ V(under);
        sB  = min(tB, [], 2);
        out = isfinite(sB);
        V(out, :) = shrink * sB(out) .* V(out, :);
        X = min(max(X + V, lb), ub);   % the clamp only absorbs rounding at the box face

        m = min(nAgents, maxFE - FE);
        [fit, FE] = calculate_fitness(X(1:m, :)', problem, FE);
        fit = fit(:)';
        stamp_batch(X(1:m, :), fit, m);
        if FE >= maxFE, break; end

        better = fit < LBf;
        LB(better, :) = X(better, :);
        LBf(better)   = fit(better);

        k = k + 1;
        [~, ig] = min(LBf);
        GB = LB(ig, :);

        % Cockshut points, Eq. (5), and new movement vectors, Eq. (8)
        O  = ((ite + k) * GB + (ite - k) * LB) / (2 * ite);
        Xd = X - O;
        uS = rand(nAgents, nPair + ~isempty(T3));   % stoch draw per sub-space
        vR = -1 + 2 * rand(nAgents, D);             % random direction components
        mR = rand(nAgents, nPair + ~isempty(T3));   % random length factor per sub-space
        Vnew = V;

        if nPair > 0
            r  = sqrt(Xd(:, P1).^2 + Xd(:, P2).^2);
            n1 = Xd(:, P1) ./ r;
            n2 = Xd(:, P2) ./ r;
            vn = sqrt(V(:, P1).^2 + V(:, P2).^2);
            d1 = V(:, P1) ./ vn;
            d2 = V(:, P2) ./ vn;
            noD = ~(vn > 0) | ~isfinite(vn);
            d1(noD) = -n1(noD);
            d2(noD) = -n2(noD);
            dn = d1 .* n1 + d2 .* n2;
            c  = sqrt(1 - eta^2 * max(0, 1 - dn.^2));   % (B-1)
            t1 = (-n1 .* c + eta * (d1 - dn .* n1)) .* r;
            t2 = (-n2 .* c + eta * (d2 - dn .* n2)) .* r;

            vr1 = vR(:, P1);
            vr2 = vR(:, P2);
            mP  = mR(:, 1:nPair);
            rn  = sqrt(vr1.^2 + vr2.^2);
            s1  = vr1 ./ rn .* aPair / dDiv .* mP;
            s2  = vr2 ./ rn .* aPair / dDiv .* mP;

            % On O: the old direction scaled to rand*0.001; a zero vector takes the random one
            o1 = d1 .* mP * onStep;
            o2 = d2 .* mP * onStep;
            o1(noD) = vr1(noD) ./ rn(noD) .* mP(noD) * onStep;
            o2(noD) = vr2(noD) ./ rn(noD) .* mP(noD) * onStep;

            onO   = r == 0;
            isRnd = ~onO & uS(:, 1:nPair) < stoch;
            isRef = ~onO & ~isRnd;
            W1 = zeros(size(r));
            W2 = zeros(size(r));
            W1(isRef) = t1(isRef);   W2(isRef) = t2(isRef);
            W1(isRnd) = s1(isRnd);   W2(isRnd) = s2(isRnd);
            W1(onO)   = o1(onO);     W2(onO)   = o2(onO);
            Vnew(:, P1) = W1;
            Vnew(:, P2) = W2;
        end

        if ~isempty(T3)
            g  = nPair + 1;
            xd = Xd(:, T3);
            r  = sqrt(sum(xd.^2, 2));
            nv = xd ./ r;
            vn = sqrt(sum(V(:, T3).^2, 2));
            dv = V(:, T3) ./ vn;
            noD = ~(vn > 0) | ~isfinite(vn);
            dv(noD, :) = -nv(noD, :);
            nd = sum(nv .* dv, 2);
            dt = dv - nd .* nv;
            % (B-2), |n.d| > 0.05: j* is d's part normal to n, so t2*.j* = eta*dt exactly
            c  = sqrt(1 - eta^2 * max(0, 1 - nd.^2));
            tv = -c .* nv + eta * dt;
            % |n.d| <= 0.05: j* = d itself, d* = (n.d, 1) normalised
            mid = abs(nd) <= 0.05;
            sm  = sqrt(1 + nd(mid).^2);
            cm  = sqrt(1 - eta^2 * (1 - (nd(mid) ./ sm).^2));
            tv(mid, :) = -cm .* nv(mid, :) + eta * dv(mid, :) ./ sm;
            tv = tv .* r;

            vr = vR(:, T3);
            sv = vr ./ sqrt(sum(vr.^2, 2)) * aTrip / dDiv .* mR(:, g);
            ov = dv .* mR(:, g) * onStep;
            ov(noD, :) = vr(noD, :) ./ sqrt(sum(vr(noD, :).^2, 2)) .* mR(noD, g) * onStep;

            onO   = r == 0;
            isRnd = ~onO & uS(:, g) < stoch;
            isRef = ~onO & ~isRnd;
            W = zeros(nAgents, 3);
            W(isRef, :) = tv(isRef, :);
            W(isRnd, :) = sv(isRnd, :);
            W(onO, :)   = ov(onO, :);
            Vnew(:, T3) = W;
        end
        V = Vnew;
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Best-so-far and one history sample per evaluation; agent g's slot holds its new position
    function stamp_batch(Xb, fb, mb)
        for gg = 1:mb
            if fb(gg) < best_fitness
                best_fitness  = fb(gg);
                best_solution = Xb(gg, :);
            end
            Xrec(gg, :) = Xb(gg, :);
            frec(gg)    = fb(gg);
            ec = FE - mb + gg;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end
end

% Normalise each 2-D and 3-D sub-space of every row to unit length
function V = unit_subspaces(V, P1, P2, T3)
    vn = sqrt(V(:, P1).^2 + V(:, P2).^2);
    V(:, P1) = V(:, P1) ./ vn;
    V(:, P2) = V(:, P2) ./ vn;
    if ~isempty(T3)
        V(:, T3) = V(:, T3) ./ sqrt(sum(V(:, T3).^2, 2));
    end
end

% ----------------------------------------------------------------------- %
% Dynamic Multi-Swarm Particle Swarm Optimizer with Local Search (DMS-L-PSO)
% CEC 2005 competition -- 3rd by Friedman rank (Garcia et al. 2009, D = 10)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   group_num = 20, group_ps = 3   % Sub-swarms and particles per sub-swarm (60 in all)
%   w = 0.729, c1 = c2 = 1.49445   % Inertia weight and acceleration constants
%   Vmax = 0.2*(ub - lb)           % Velocity clamp
%   R = 5                          % Regrouping period, generations
%   L = 100, L_FES = 100           % Local-search period (generations) and its evaluations
%   L_num = ceil(0.25*group_num)   % Sub-swarm leaders refined per local-search round
%   phases = 0.95 / 0.05 of maxFe  % Multi-swarm search, then a final local search
%
% Algorithm Concept:
%   - The swarm is split into small sub-swarms of 3; each particle follows its own
%     pbest and the best pbest of its sub-swarm (lbest)
%   - Every R generations the sub-swarms are re-drawn at random, so information
%     travels between them while each stays small and diverse
%   - A moved particle keeps each coordinate of its pbest with probability 0.5
%   - A particle outside the box is not evaluated and its pbest is not updated
%   - Every L generations the lbest of the best 25 % of sub-swarms is refined by a
%     quasi-Newton search; at 95 % of the budget the best lbest gets 5 % of it
%   - The remaining budget runs a plain global-best PSO around the refined best
%
% Reference:
% J. J. Liang, P. N. Suganthan,
% Dynamic Multi-Swarm Particle Swarm Optimizer with Local Search,
% 2005 IEEE Congress on Evolutionary Computation, vol. 1, 2005, 522-528.
% https://doi.org/10.1109/CEC.2005.1554727
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release DMS_PSO_func.m (P-N-Suganthan/CODES,
% 2005-IEEE-SIS-DMS_PSO_func.zip). It ships no driver: group_num = 20, group_ps = 3
% are the paper's; R = 5 and L_FES = 100 are the code's (the paper says 10 and 200).
% Its local search is fminunc quasi-Newton (Optimization Toolbox, unbounded); a BFGS
% projected onto the box stands in, with fminusub's start (re-evaluated x0, step
% min(1/|g|inf,1)), forward differences, 1e-6 tolerances and 400 iterations, capped
% at the stated evaluations (fminunc overruns them). Inside the box it can stop on a
% bound that fminunc would cross. Every phase is held to maxFe. Hang guard: after 1000
% generations with no particle in the box, the outside ones are clamped onto it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dmslpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    group_num = 20;
    group_ps  = 3;
    ps        = group_num * group_ps;
    w         = 0.729;
    c1        = 1.49445;
    c2        = 1.49445;
    R         = 5;
    L         = 100;
    L_FES     = 100;
    L_num     = ceil(0.25 * group_num);
    mv        = 0.2 * span;
    stall_max = 1000;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = (lb + ub) / 2;

    pos = lb + span .* rand(ps, D);
    inb = true(ps, 1);
    n0 = min(ps, maxFE);
    e = inf(ps, 1);
    e(1:n0) = evaluate(pos(1:n0, :));
    inb(n0 + 1:end) = false;
    for q = 1:n0
        record(q);
    end

    vel = -mv + 2 .* mv .* rand(ps, D);
    pbest = pos;
    pbestval = e;
    group_id = reshape(1:ps, group_ps, group_num)';
    pos_group = zeros(1, ps);
    gbest = zeros(group_num, D);
    gbestval = zeros(1, group_num);
    regroup_leaders();

    it = 0;
    stall = 0;
    while FE < 0.95 * maxFE && FE < maxFE
        it = it + 1;
        n_before = FE;
        for k = 1:ps
            if FE >= maxFE
                break;
            end
            g = pos_group(k);
            aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + c2 .* rand(1, D) .* (gbest(g, :) - pos(k, :));
            vel(k, :) = min(max(w .* vel(k, :) + aa, -mv), mv);
            pos1 = pos(k, :) + vel(k, :);
            keep_d = rand(1, D) < 0.5;
            pos(k, :) = keep_d .* pbest(k, :) + (1 - keep_d) .* pos1;
            move_particle(k);
            if inb(k) && pbestval(k) < gbestval(g)
                gbest(g, :) = pbest(k, :);
                gbestval(g) = pbestval(k);
            end
        end
        stall = check_stall(FE == n_before, stall);

        if mod(it, L) == 0
            [~, tmpid] = sort(gbestval);
            for k = 1:L_num
                if FE >= maxFE
                    break;
                end
                gk = tmpid(k);
                [x, fval] = quasi_newton(gbest(gk, :), L_FES);
                if fval < gbestval(gk)
                    [~, gid] = min(pbestval(group_id(gk, :)));
                    pbest(group_id(gk, gid), :) = x;
                    pbestval(group_id(gk, gid)) = fval;
                    gbest(gk, :) = x;
                    gbestval(gk) = fval;
                end
            end
        end

        if mod(it, R) == 0
            group_id = reshape(randperm(ps), group_ps, group_num)';
            regroup_leaders();
        end
    end

    [~, tmpid] = sort(gbestval);
    gb = gbest(tmpid(1), :);
    gbv = gbestval(tmpid(1));
    if FE < maxFE
        [x, fval] = quasi_newton(gb, floor(0.05 * maxFE));
        if fval < gbv
            gb = x;
            gbv = fval;
        end
    end

    while FE < maxFE
        n_before = FE;
        for k = 1:ps
            if FE >= maxFE
                break;
            end
            aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + c2 .* rand(1, D) .* (gb - pos(k, :));
            vel(k, :) = min(max(w .* vel(k, :) + aa, -mv), mv);
            pos(k, :) = pos(k, :) + vel(k, :);
            move_particle(k);
            if inb(k) && pbestval(k) < gbv
                gb = pbest(k, :);
                gbv = pbestval(k);
            end
        end
        stall = check_stall(FE == n_before, stall);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of X (in the box, within the budget), tracking the best per evaluation
    function f = evaluate(X)
        [fr, FE_new] = calculate_fitness(X', problem, FE);
        f = fr(:);
        for q2 = 1:numel(f)
            if f(q2) < bsf
                bsf = f(q2);
                bsf_solution = X(q2, :);
            end
            curve(FE + q2) = bsf;
        end
        FE = FE_new;
    end

    % Population = the particles inside the box, each at the value of its current position
    function record(ec)
        if any(inb)
            [population_history, fitness_history, history_index] = record_history( ...
                ec, pos(inb, :), e(inb), population_history, fitness_history, history_index, maxFE);
        else
            [population_history, fitness_history, history_index] = record_history( ...
                ec, pbest, pbestval, population_history, fitness_history, history_index, maxFE);
        end
    end

    % Evaluates particle k only if it lies in the box; pbest takes ties, as in the release
    function move_particle(k)
        inb(k) = all(pos(k, :) <= ub) && all(pos(k, :) >= lb);
        if inb(k)
            e(k) = evaluate(pos(k, :));
            if ~(pbestval(k) < e(k))
                pbest(k, :) = pos(k, :);
                pbestval(k) = e(k);
            end
            record(FE);
        end
    end

    % Sub-swarm membership and leaders from the current group_id
    function regroup_leaders()
        for gi = 1:group_num
            pos_group(group_id(gi, :)) = gi;
            [gbestval(gi), gid2] = min(pbestval(group_id(gi, :)));
            gbest(gi, :) = pbest(group_id(gi, gid2), :);
        end
    end

    % Hang guard: a swarm that stays wholly outside the box is clamped onto it
    function stall = check_stall(idle, stall)
        if ~idle
            stall = 0;
            return;
        end
        stall = stall + 1;
        if stall >= stall_max
            out = find(~inb)';
            for kk = out
                if FE >= maxFE
                    break;
                end
                pos(kk, :) = min(max(pos(kk, :), lb), ub);
                move_particle(kk);
            end
            stall = 0;
        end
    end

    % fminunc stand-in (its quasi-Newton loop, fminusub): projected BFGS, forward differences
    function [x, fx] = quasi_newton(x0, nmax)
        used = 0;
        x = min(max(x0, lb), ub);
        fx = probe(x);
        if ~isfinite(fx)
            return;
        end
        [gr, ok] = fd_gradient(x, fx);
        if ~ok
            return;
        end
        g0n = norm(gr, inf);
        H = eye(D);
        fresh = true;
        first = true;
        for iter = 1:400
            act = (x <= lb & gr > 0) | (x >= ub & gr < 0);
            if all(act) || norm(gr(~act), inf) < 1e-6 * (1 + g0n)
                break;
            end
            % Quasi-Newton direction, cut where it would leave an active bound
            dvec = -(H * gr')';
            dvec((x <= lb & dvec < 0) | (x >= ub & dvec > 0)) = 0;
            if ~(gr * dvec' < 0)
                H = eye(D);
                fresh = true;
                first = true;
                dvec = -gr;
                dvec(act) = 0;
            end
            % fminusub's trial step: min(1/|g|inf, 1) on a fresh H, else 1
            if first
                a = min(1 / norm(gr, inf), 1);
            else
                a = 1;
            end
            accepted = false;
            while a > 1e-16
                if used >= nmax || FE >= maxFE
                    return;
                end
                xn = min(max(x + a * dvec, lb), ub);
                fn = probe(xn);
                slope = gr * (xn - x)';
                if fn <= fx + 1e-4 * slope
                    accepted = true;
                    break;
                end
                % Minimiser of the quadratic through f(0), f'(0), f(a), kept in [0.1a, 0.5a]
                den = 2 * (fn - fx - slope);
                if isfinite(fn) && den > 0
                    a = min(max(-slope * a / den, 0.1 * a), 0.5 * a);
                else
                    a = 0.1 * a;
                end
            end
            % A failed search restarts once from steepest descent before giving up
            if ~accepted
                if fresh
                    break;
                end
                H = eye(D);
                fresh = true;
                first = true;
                continue;
            end
            [gn, ok] = fd_gradient(xn, fn);
            if ~ok
                x = xn;
                fx = fn;
                return;
            end
            % The step doubles while the curvature (Wolfe) condition still fails and nothing is clipped
            while gn * dvec' < 0.9 * (gr * dvec') && all(xn == x + a * dvec) && ...
                    used + D < nmax && FE + D < maxFE
                xt = min(max(x + 2 * a * dvec, lb), ub);
                ft = probe(xt);
                if ~(ft < fn)
                    break;
                end
                [gt, ok2] = fd_gradient(xt, ft);
                if ~ok2
                    break;
                end
                a = 2 * a;
                xn = xt;
                fn = ft;
                gn = gt;
            end
            s = xn - x;
            x = xn;
            fx = fn;
            yv = gn - gr;
            sy = s * yv';
            if sy > 1e-10 * norm(s) * norm(yv)
                if first
                    H = (sy / (yv * yv')) * eye(D);
                end
                rho = 1 / sy;
                V = eye(D) - rho * (s' * yv);
                H = V * H * V' + rho * (s' * s);
                first = false;
            end
            gr = gn;
            % fminusub's relative step test; it ends the search only on a fresh H
            if norm(s ./ (1 + abs(x)), inf) < 1e-6
                if fresh
                    break;
                end
                H = eye(D);
                fresh = true;
                first = true;
                continue;
            end
            fresh = false;
        end

        % One evaluation inside the local search, recorded against the swarm
        function fv = probe(xv)
            fv = evaluate(xv);
            used = used + 1;
            record(FE);
        end

        % fminunc's forward step sqrt(eps)*max(|x|, 1), taken backwards where it would leave the box
        function [gv, okv] = fd_gradient(xc, fc)
            gv = zeros(1, D);
            h = sqrt(eps) * max(abs(xc), 1);
            flip = xc + h > ub;
            h(flip) = -h(flip);
            pts = repmat(xc, D, 1);
            pts(1:D + 1:end) = xc + h;
            n_ev = min([D, nmax - used, maxFE - FE]);
            okv = false;
            if n_ev < 1
                return;
            end
            fe0 = FE;
            fv = evaluate(pts(1:n_ev, :));
            used = used + n_ev;
            for q3 = 1:n_ev
                record(fe0 + q3);
            end
            okv = n_ev == D && all(isfinite(fv));
            if okv
                gv = (fv' - fc) ./ h;
            end
        end
    end
end

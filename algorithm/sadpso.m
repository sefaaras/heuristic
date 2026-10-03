% ----------------------------------------------------------------------- %
% Self-Adaptive Dynamic Particle Swarm Optimizer (SaDPSO)
% CEC 2015 learning-based track -- 8th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   group_num = 10/20/30/50 (D <= 10/30/50/more), 3 each  % Sub-swarms and their size
%   c1 = 1.49445, Vmax = 0.2*(ub - lb)  % Cognitive constant and velocity clamp
%   w ~ U[0.4,0.9], c2 ~ U[0.5,2.5] -> N(median, 0.1)  % Per sub-swarm, learned from a memory
%   LP = 8, giter_num = 10        % Parameter memory length; generations per iteration
%   R = 5, L = 100 iterations     % Regrouping and local-search periods
%   L_FES = 100, L_num = ceil(0.25*group_num)  % Local-search evaluations and leaders
%   phases = 0.95 / 0.05 of maxFe % Multi-swarm search, then a final local search
%
% Algorithm Concept:
%   - DMS-PSO frame: sub-swarms of 3 follow their own lbest and are re-drawn at
%     random; a moved particle keeps each pbest coordinate with probability 0.5
%   - Each iteration (10 generations) every sub-swarm gets its own w and c2; the pair
%     of the sub-swarm with most lbest improvements enters an 8-slot memory
%   - Once the memory is full and the swarm improved more than 10 times, w and c2 are
%     drawn from normals around the memory medians
%   - Out-of-box coordinates are redrawn within 25 % of the range from the bound
%   - Periodic and final quasi-Newton refinement of the best lbests, as DMS-L-PSO;
%     a global-best PSO then spends the rest, feeding successful pairs to the memory
%
% Reference:
% J. J. Liang, L. Guo, R. Liu, B. Y. Qu,
% A self-adaptive dynamic particle swarm optimizer,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, 3206-3213.
% https://doi.org/10.1109/CEC.2015.7257290
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (P-N-Suganthan/CEC2015-Learning-Based,
% 15460-SaDPSO2 code.zip, folder SaDPSO2 = the submitted "fixed c1, w c2 adapt").
% group_num follows its driver's D = 10/30/50/100 table, other D taking the next row.
% Its local search is fminunc quasi-Newton (Optimization Toolbox, unbounded); a BFGS
% projected onto the box stands in, with fminusub's start (re-evaluated x0, step
% min(1/|g|inf,1)), forward differences, 1e-6 tolerances and 400 iterations, capped
% at the stated evaluations (fminunc overruns them). The release's stop at
% error < 1e-8 needs the known optimum and is dropped; every phase is held to maxFe;
% normrnd is replaced by median + 0.1*randn.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sadpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    group_table = [10, 10; 30, 20; 50, 30; Inf, 50];
    group_num = group_table(find(D <= group_table(:, 1), 1), 2);
    group_ps  = 3;
    ps        = group_num * group_ps;
    c1        = 1.49445;
    LP        = 8;
    giter_num = 10;
    R         = 5;
    L         = 100;
    L_FES     = 100;
    L_num     = ceil(0.25 * group_num);
    mv        = 0.2 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = (lb + ub) / 2;

    pos = lb + span .* rand(ps, D);
    n0 = min(ps, maxFE);
    e = inf(ps, 1);
    e(1:n0) = evaluate(pos(1:n0, :));
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

    success_num = zeros(1, group_num);
    parameter_set = zeros(0, 2);
    it = 0;
    while FE < 0.95 * maxFE && FE < maxFE
        it = it + 1;
        if size(parameter_set, 1) < LP || sum(success_num) <= giter_num
            iwt = 0.5 * rand(1, group_num) + 0.4;
            cc2 = 2 * rand(1, group_num) + 0.5;
        else
            cc2 = median(parameter_set(:, 1)) + 0.1 * randn(1, group_num);
            iwt = median(parameter_set(:, 2)) + 0.1 * randn(1, group_num);
        end
        success_num = zeros(1, group_num);
        for giter = 1:giter_num
            for k = 1:ps
                if FE >= maxFE
                    break;
                end
                g = pos_group(k);
                aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + cc2(g) .* rand(1, D) .* (gbest(g, :) - pos(k, :));
                vel(k, :) = min(max(iwt(g) .* vel(k, :) + aa, -mv), mv);
                pos1 = pos(k, :) + vel(k, :);
                keep_d = rand(1, D) < 0.5;
                pos(k, :) = keep_d .* pbest(k, :) + (1 - keep_d) .* pos1;
                move_particle(k);
                if pbestval(k) < gbestval(g)
                    success_num(g) = success_num(g) + 1;
                    gbest(g, :) = pbest(k, :);
                    gbestval(g) = pbestval(k);
                end
            end
        end
        [~, maxindex] = max(success_num);
        if size(parameter_set, 1) == LP
            parameter_set(1, :) = [];
        end
        parameter_set = [parameter_set; cc2(maxindex), iwt(maxindex)]; %#ok<AGROW>

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

    % Only reachable on a budget too small for one multi-swarm iteration
    if isempty(parameter_set)
        parameter_set = [2 * rand + 0.5, 0.5 * rand + 0.4];
    end
    while FE < maxFE
        c2  = median(parameter_set(:, 1)) + 0.1 * randn;
        iwt = median(parameter_set(:, 2)) + 0.1 * randn;
        for k = 1:ps
            if FE >= maxFE
                break;
            end
            aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + c2 .* rand(1, D) .* (gb - pos(k, :));
            vel(k, :) = min(max(iwt .* vel(k, :) + aa, -mv), mv);
            pos(k, :) = pos(k, :) + vel(k, :);
            improved = move_particle(k);
            % A pbest replacement (ties included) feeds the pair to the memory, FIFO
            if improved
                parameter_set(1, :) = [];
                parameter_set = [parameter_set; c2, iwt]; %#ok<AGROW>
            end
            if pbestval(k) < gbv
                gb = pbest(k, :);
                gbv = pbestval(k);
            end
        end
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

    % Population = the swarm, each particle at the value of its current position
    function record(ec)
        [population_history, fitness_history, history_index] = record_history( ...
            ec, pos, e, population_history, fitness_history, history_index, maxFE);
    end

    % Redraws out-of-box coordinates near the violated bound, evaluates, updates pbest (ties included)
    function improved = move_particle(k)
        p = pos(k, :);
        p = ((p >= lb) & (p <= ub)) .* p + (p < lb) .* (lb + 0.25 .* span .* rand(1, D)) + ...
            (p > ub) .* (ub - 0.25 .* span .* rand(1, D));
        pos(k, :) = p;
        e(k) = evaluate(p);
        improved = ~(pbestval(k) < e(k));
        if improved
            pbest(k, :) = p;
            pbestval(k) = e(k);
        end
        record(FE);
    end

    % Sub-swarm membership and leaders from the current group_id
    function regroup_leaders()
        for gi = 1:group_num
            pos_group(group_id(gi, :)) = gi;
            [gbestval(gi), gid2] = min(pbestval(group_id(gi, :)));
            gbest(gi, :) = pbest(group_id(gi, gid2), :);
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

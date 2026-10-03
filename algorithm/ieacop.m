% ----------------------------------------------------------------------- %
% Improved Evolutionary Algorithm for Complex-Process Optimization (iEACOP)
% CEC 2024 competition -- 6th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 2*ceil((1 + sqrt(1 + 40*D))/4), rounded up to even  % Population (scatter-search rule)
%   m = 10*D                    % Uniform initial sample; n/2 best + n/2 random kept
%   n_change = 20               % Failed generations before a member is redrawn
%   epsilon = 1e-3              % Relative distance under which a member is a duplicate
%   balance = 0.5               % Quality vs novelty weight of the second local search
%   n1 = 1, n2 = 10             % Evaluation and generation spacing of the local searches
%   Powell: xtol = ftol = 1e-4, 1000*D evaluations per call  % Local search, start method
%
% Algorithm Concept:
%   - Scatter-search combination: every ordered pair (i, j) yields one child drawn
%     uniformly in a hyper-rectangle around x_i, biased by the rank gap of i and j
%   - A parent beaten by its best child is replaced after a go-beyond walk that keeps
%     extrapolating past the child, doubling the reach every two successes
%   - Members within relative distance epsilon of an earlier one are redrawn
%   - Local search (Powell; on failure a quasi-Newton retry from the same start)
%     from the best, or from the offspring best in quality-plus-novelty rank
%   - A local optimum that beats the best replaces the worst member
%
% Reference:
% Andrea Tangherloni, Vasco Coelho, Francesca M. Buffa, Paolo Cazzaniga,
% A Modified EACOP Implementation for Real-Parameter Single Objective Optimization
% Problems,
% 2024 IEEE Congress on Evolutionary Computation (CEC), 2024, pp. 1-8.
% https://doi.org/10.1109/CEC60901.2024.10611920
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' Python release (iEACOP.py, identical in BC-SOPs-Codes.zip
% of the CEC 2024 repository and github.com/andreatangherloni/iEACOP), run as its
% run_cec17.py does: uniform initialisation, Powell first and changeable. scipy is
% replaced: Powell is a line-by-line port of scipy's bounded Powell and its fminbound
% line search (same f, nfev and x as scipy from shared starts); L-BFGS-B is replaced by
% a projected L-BFGS (10 pairs) with forward differences (step 1e-8), an interpolating
% Armijo search and L-BFGS-B's default limits (15000 evaluations, pgtol 1e-5, ftol
% 2.2e-9, 20 trials, one memory reset on failure). The release checks its budget only
% between generations and adds local-search evaluations afterwards; here every
% evaluation is guarded. Line-search points are clamped to the box against rounding.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ieacop(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    n_change = 20;
    epsilon  = 1e-3;
    balance  = 0.5;
    n1       = 1;
    n2       = 10;
    xtol     = 1e-4;              % scipy Powell defaults
    ftol     = 1e-4;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = lb + 0.5 * span;
    done = false;
    ctxX = zeros(0, D);           % population shown to record_history
    ctxF = zeros(0, 1);

    % Population size: the scatter-search heuristic, made even
    n = ceil((1 + sqrt(1 + 4 * 10 * D)) / 2);
    if mod(n, 2) == 1
        n = n + 1;
    end
    m = 10 * D;

    local_fit_evals = 0;          % evaluations outside the local searches since the last one
    P0 = lb + rand(m, D) .* span;
    F0 = evaluate(P0, true);
    if done
        finish();
        return;
    end
    [F0, o] = sort(F0);
    P0 = P0(o, :);
    keep = [1:n / 2, randi([n / 2 + 1, m], 1, n - n / 2)];   % the release draws the rest from ranks n/2+1..m
    X = P0(keep, :);
    F = F0(keep);
    [F, o] = sort(F);
    X = X(o, :);
    stuck = zeros(n, 1);
    best_x = X(1, :);
    best_f = F(1);

    iterations         = 0;
    apply_local_search = false;
    last_best_local    = 0;
    last_restart_local = 0;
    local_solutions    = zeros(0, D);
    opt_method         = 1;       % 1 = Powell, 2 = quasi-Newton (L-BFGS-B in the release)
    OX = [];
    OF = [];

    while ~done
        check_diversity();
        if done, break; end
        combination_method();
        if done, break; end
        update_population();
        if done, break; end

        if apply_local_search
            if last_best_local == 0
                apply_local1();
                if done, break; end
                if size(local_solutions, 1) > 1
                    apply_local2();
                end
            else
                if isempty(local_solutions)
                    if local_fit_evals >= n1
                        apply_local1();
                    end
                elseif local_fit_evals >= n2 && mod(iterations, n2) == 0
                    apply_local2();
                end
            end
            local_fit_evals = 0;
        end
        if done, break; end

        if last_restart_local >= n_change
            apply_local_search = true;
            last_restart_local = 0;
        end
        iterations = iterations + 1;
    end

    finish();

    function finish()
        curve(min(max(FE, 1), maxFE):end) = bsf;
        best_fitness  = bsf;
        best_solution = bsfx;
    end

    % Evaluates rows of Xr within the budget (rows past it score inf and set done)
    function f = evaluate(Xr, counted)
        k = size(Xr, 1);
        f = inf(k, 1);
        k_ok = min(k, maxFE - FE);
        if k_ok > 0
            [fv, FE_new] = calculate_fitness(Xr(1:k_ok, :)', problem, FE);
            f(1:k_ok) = fv(:);
            for q = 1:k_ok
                if f(q) < bsf
                    bsf  = f(q);
                    bsfx = Xr(q, :);
                end
                ec = FE + q;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, [ctxX; Xr(1:q, :)], [ctxF; f(1:q)], population_history, ...
                    fitness_history, history_index, maxFE);
            end
            FE = FE_new;
            if counted
                local_fit_evals = local_fit_evals + k_ok;
            end
        end
        if FE >= maxFE
            done = true;
        end
    end

    function x = draw_uniform()
        x = lb + rand(1, D) .* span;
    end

    % A later member within relative distance epsilon of an earlier one is redrawn
    function check_diversity()
        ctxX = X;
        ctxF = F;
        for i = 1:n - 1
            xi = X(i, :);
            for j = i + 1:n
                v = abs((xi - X(j, :)) ./ X(j, :));
                if ~any(isnan(v)) && max(v) < epsilon      % numpy's max propagates NaN
                    xn = draw_uniform();
                    fn = evaluate(xn, true);
                    if done, return; end
                    X(j, :) = xn;
                    F(j) = fn;
                    stuck(j) = 0;
                    ctxX = X;
                    ctxF = F;
                end
            end
        end
    end

    % One child per ordered pair, in the hyper-rectangle of the release's EACOP combination
    function combination_method()
        OX = zeros(n, n - 1, D);
        Ochild = zeros(n * (n - 1), D);
        r = 0;
        for i = 1:n
            c = 0;
            for j = 1:n
                if i == j
                    continue;
                end
                if i < j
                    alpha = 1;
                else
                    alpha = -1;
                end
                beta  = (abs(j - i) - 1) / (n - 2);
                delta = (X(j, :) - X(i, :)) / 2;
                c1 = min(max(X(i, :) - delta * (1 + alpha * beta), lb), ub);
                c2 = min(max(X(i, :) + delta * (1 - alpha * beta), lb), ub);
                val = min(max(c1 + (c2 - c1) .* rand(1, D), lb), ub);
                c = c + 1;
                r = r + 1;
                OX(i, c, :) = reshape(val, 1, 1, D);
                Ochild(r, :) = val;
            end
        end
        ctxX = X;
        ctxF = F;
        fall = evaluate(Ochild, true);
        OF = reshape(fall, n - 1, n)';
        for i = 1:n
            [OF(i, :), o] = sort(OF(i, :));
            OX(i, :, :) = OX(i, o, :);
        end
    end

    % Extrapolates past an improving child, doubling the reach after every two successes
    function [xr, fr] = go_beyond(idx)
        xpr = X(idx, :);
        fpr = F(idx);
        xch = reshape(OX(idx, 1, :), 1, D);
        fch = OF(idx, 1);
        improvement = 1;
        lambd = 1.0;
        while fch < fpr
            c1 = min(max(xch - (xpr - xch) / lambd, lb), ub);
            c2 = min(max(xch, lb), ub);
            val = min(max(c1 + (c2 - c1) .* rand(1, D), lb), ub);
            fv = evaluate(val, true);
            if done
                break;
            end
            xpr = xch;
            fpr = fch;
            xch = val;
            fch = fv;
            improvement = improvement + 1;
            if improvement == 2
                lambd = lambd / 2;
                improvement = 0;
            end
        end
        xr = xpr;
        fr = fpr;
    end

    function update_population()
        ctxX = X;
        ctxF = F;
        for i = 1:n
            if OF(i, 1) < F(i)
                [xr, fr] = go_beyond(i);
                if done, return; end
                X(i, :) = xr;
                F(i) = fr;
                stuck(i) = 0;
            else
                stuck(i) = stuck(i) + 1;
                if stuck(i) > n_change
                    xn = draw_uniform();
                    fn = evaluate(xn, true);
                    if done, return; end
                    X(i, :) = xn;
                    F(i) = fn;
                    stuck(i) = 0;
                end
            end
            ctxX = X;
            ctxF = F;
        end
        [F, o] = sort(F);
        X = X(o, :);
        stuck = stuck(o);
        if F(1) < best_f
            best_x = X(1, :);
            best_f = F(1);
            last_best_local = 0;
            last_restart_local = 0;
            if iterations >= n_change * 2
                apply_local_search = true;
            end
        else
            last_restart_local = last_restart_local + 1;
        end
    end

    function [zx, zf] = local_search(x0)
        if opt_method == 1
            [zx, zf] = powell(x0);
        else
            [zx, zf] = bfgs_box(x0);
        end
    end

    % A local optimum that beats the best replaces the worst member
    function ok = check_local_search(zx, zf)
        if zf < best_f
            last_best_local = 0;
            last_restart_local = 0;
            apply_local_search = true;
            X(end, :) = zx;
            F(end) = zf;
            stuck(end) = 0;
            [F, o] = sort(F);
            X = X(o, :);
            stuck = stuck(o);
            best_x = X(1, :);
            best_f = F(1);
            ok = true;
        else
            last_best_local = last_best_local + 1;
            last_restart_local = last_restart_local + 1;
            apply_local_search = false;
            ok = false;
        end
    end

    function add_local(zx)
        if ~any(all(local_solutions == zx, 2))
            local_solutions(end + 1, :) = zx;
        end
    end

    % On failure the other method is tried from the same start and kept only if it does better
    function evaluate_local_search(zx, zf, z1x)
        if check_local_search(zx, zf)
            add_local(zx);
            return;
        end
        old_method = opt_method;
        opt_method = 3 - opt_method;
        [nx, nf] = local_search(z1x);
        if done, return; end
        check_local_search(nx, nf);
        if nf < zf
            add_local(nx);
        else
            opt_method = old_method;
            add_local(zx);
        end
    end

    function apply_local1()
        z1x = best_x;
        [zx, zf] = local_search(z1x);
        if done, return; end
        evaluate_local_search(zx, zf, z1x);
    end

    % Start from the offspring with the best mean of quality rank and novelty rank
    function apply_local2()
        Y  = reshape(permute(OX, [2 1 3]), n * (n - 1), D);
        yq = reshape(OF', [], 1);
        L  = size(Y, 1);
        yd = zeros(L, 1);
        for k = 1:L
            yd(k) = min(sqrt(sum((local_solutions - Y(k, :)) .^ 2, 2)));
        end
        [~, oq] = sort(yq, 'ascend');
        [~, od] = sort(yd, 'descend');
        rq = zeros(L, 1);
        rd = zeros(L, 1);
        rq(oq) = 0:L - 1;
        rd(od) = 0:L - 1;
        score = (1 - balance) * rq + balance * rd;
        [~, idx] = min(score);
        z1x = Y(idx, :);
        [zx, zf] = local_search(z1x);
        if done, return; end
        evaluate_local_search(zx, zf, z1x);
    end

    % scipy's bounded Powell (_minimize_powell): coordinate line searches plus one extrapolation
    function [x, fval] = powell(x0)
        maxfun = 1000 * D;
        maxiter = 1000 * D;
        ncalls = 0;
        abort = false;
        direc = eye(D);
        x = x0;
        ctxX = X;
        ctxF = F;
        fval = pf(x);
        if abort
            return;
        end
        x1 = x;
        iter = 0;
        while true
            fx = fval;
            bigind = 1;
            delta = 0.0;
            for i = 1:D
                fx2 = fval;
                [fv, xn] = linesearch(x, direc(i, :), fval);
                if abort, return; end
                fval = fv;
                x = xn;
                if (fx2 - fval) > delta
                    delta = fx2 - fval;
                    bigind = i;
                end
            end
            iter = iter + 1;
            bnd = ftol * (abs(fx) + abs(fval)) + 1e-20;
            if 2.0 * (fx - fval) <= bnd || ncalls >= maxfun || iter >= maxiter || (isnan(fx) && isnan(fval))
                break;
            end
            direc1 = x - x1;
            x1 = x;
            if ~any(direc1)
                break;                % scipy would fail in _line_for_search here
            end
            [~, lmax] = line_bounds(x, direc1);
            x2 = x + min(lmax, 1) * direc1;
            fx2 = pf(x2);
            if abort, return; end
            if fx > fx2
                t = 2.0 * (fx + fx2 - 2.0 * fval);
                temp = fx - fval - delta;
                t = t * temp * temp;
                temp = fx - fx2;
                t = t - delta * temp * temp;
                if t < 0.0
                    [fv, xn, dn] = linesearch(x, direc1, fval);
                    if abort, return; end
                    fval = fv;
                    x = xn;
                    if any(dn)
                        direc(bigind, :) = direc(end, :);
                        direc(end, :) = dn;
                    end
                end
            end
        end

        % scipy's maxfun wrapper raises before the call that would exceed it
        function f = pf(z)
            if ncalls >= maxfun || done
                abort = true;
                f = NaN;
                return;
            end
            ncalls = ncalls + 1;
            f = evaluate(min(max(z, lb), ub), false);
            if done
                abort = true;
            end
        end

        function [fv, xn, dn] = linesearch(p, xi, f0)
            if ~any(xi)
                fv = f0;
                xn = p;
                dn = xi;
                return;
            end
            [lo, hi] = line_bounds(p, xi);
            [a, fv] = fminbound(@(al) pf(p + al * xi), lo, hi, xtol);
            xn = p + a * xi;
            dn = a * xi;
        end

        % scipy's _minimize_scalar_bounded (Brent on a closed interval), maxiter 500
        function [xf, fx] = fminbound(fun, x1b, x2b, xatol)
            maxfn = 500;
            sqrt_eps = sqrt(2.2e-16);
            golden_mean = 0.5 * (3.0 - sqrt(5.0));
            a = x1b;
            b = x2b;
            fulc = a + golden_mean * (b - a);
            nfc = fulc;
            xf = fulc;
            rat = 0.0;
            e = 0.0;
            xx = xf;
            fx = fun(xx);
            if abort, return; end
            num = 1;
            ffulc = fx;
            fnfc = fx;
            xm = 0.5 * (a + b);
            tol1 = sqrt_eps * abs(xf) + xatol / 3.0;
            tol2 = 2.0 * tol1;
            while abs(xf - xm) > (tol2 - 0.5 * (b - a))
                golden = 1;
                if abs(e) > tol1
                    golden = 0;
                    r = (xf - nfc) * (fx - ffulc);
                    q = (xf - fulc) * (fx - fnfc);
                    p = (xf - fulc) * q - (xf - nfc) * r;
                    q = 2.0 * (q - r);
                    if q > 0.0
                        p = -p;
                    end
                    q = abs(q);
                    r = e;
                    e = rat;
                    if (abs(p) < abs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))
                        rat = (p + 0.0) / q;
                        xx = xf + rat;
                        if ((xx - a) < tol2) || ((b - xx) < tol2)
                            si = sign(xm - xf) + ((xm - xf) == 0);
                            rat = tol1 * si;
                        end
                    else
                        golden = 1;
                    end
                end
                if golden
                    if xf >= xm
                        e = a - xf;
                    else
                        e = b - xf;
                    end
                    rat = golden_mean * e;
                end
                si = sign(rat) + (rat == 0);
                xx = xf + si * max(abs(rat), tol1);
                fu = fun(xx);
                if abort, return; end
                num = num + 1;
                if fu <= fx
                    if xx >= xf
                        a = xf;
                    else
                        b = xf;
                    end
                    fulc = nfc;
                    ffulc = fnfc;
                    nfc = xf;
                    fnfc = fx;
                    xf = xx;
                    fx = fu;
                else
                    if xx < xf
                        a = xx;
                    else
                        b = xx;
                    end
                    if (fu <= fnfc) || (nfc == xf)
                        fulc = nfc;
                        ffulc = fnfc;
                        nfc = xx;
                        fnfc = fu;
                    elseif (fu <= ffulc) || (fulc == xf) || (fulc == nfc)
                        fulc = xx;
                        ffulc = fu;
                    end
                end
                xm = 0.5 * (a + b);
                tol1 = sqrt_eps * abs(xf) + xatol / 3.0;
                tol2 = 2.0 * tol1;
                if num >= maxfn
                    break;
                end
            end
        end
    end

    % scipy's _line_for_search: the step interval that keeps x0 + l*d inside the box
    function [lmin, lmax] = line_bounds(x0, d)
        nz = d ~= 0;
        low  = (lb(nz) - x0(nz)) ./ d(nz);
        high = (ub(nz) - x0(nz)) ./ d(nz);
        pos  = d(nz) > 0;
        lo = high;
        lo(pos) = low(pos);
        hi = low;
        hi(pos) = high(pos);
        lmin = max(lo);
        lmax = min(hi);
        if lmax < lmin
            lmin = 0;
            lmax = 0;
        end
    end

    % L-BFGS-B stand-in: projected L-BFGS (10 pairs), forward differences, interpolating Armijo
    function [x, f] = bfgs_box(x0)
        maxfun = 15000;
        pgtol  = 1e-5;
        factr  = 2.220446049250313e-09;
        h      = 1e-8;
        mem    = 10;
        nf     = 0;
        x = min(max(x0, lb), ub);
        ctxX = X;
        ctxF = F;
        f = qf(x);
        if done, return; end
        [g, ok] = fd_grad(x, f);
        if ~ok, return; end
        Sm = zeros(0, D);
        Ym = zeros(0, D);
        for it = 1:15000
            pgrad = min(max(x - g, lb), ub) - x;
            if max(abs(pgrad)) <= pgtol
                break;
            end
            free = ~((x <= lb & g > 0) | (x >= ub & g < 0));
            d = zeros(1, D);
            d(free) = -two_loop(g(free), free);
            if ~(g * d' < 0)
                Sm = zeros(0, D);
                Ym = zeros(0, D);
                d = zeros(1, D);
                d(free) = -g(free);
            end
            a = 1;                            % L-BFGS-B starts every line search at a unit step on a fully boxed problem
            accepted = false;
            for ls = 1:20
                xn = min(max(x + a * d, lb), ub);
                fn = qf(xn);
                if done, return; end
                slope = g * (xn - x)';
                if fn <= f + 1e-3 * slope
                    accepted = true;
                    break;
                end
                % Safeguarded quadratic interpolation, so a step can shrink tenfold per trial
                aq = -slope * a / (2 * (fn - f - slope));
                if ~isfinite(aq)
                    aq = 0.1 * a;
                end
                a = min(0.5 * a, max(0.1 * a, aq));
            end
            if ~accepted
                if ~isempty(Sm)
                    Sm = zeros(0, D);         % L-BFGS-B refreshes its memory once before giving up
                    Ym = zeros(0, D);
                    continue;
                end
                break;
            end
            s = xn - x;
            fold = f;
            x = xn;
            f = fn;
            if (fold - f) / max([abs(fold), abs(f), 1]) <= factr || nf >= maxfun
                break;
            end
            [gn, ok] = fd_grad(x, f);
            if ~ok, return; end
            yv = gn - g;
            if s * yv' > eps * (yv * yv')
                Sm = [Sm(max(1, end - mem + 2):end, :); s];
                Ym = [Ym(max(1, end - mem + 2):end, :); yv];
            end
            g = gn;
        end

        % Two-loop recursion on the free coordinates, start matrix scaled by the newest pair
        function r = two_loop(q, fr)
            k = size(Sm, 1);
            if k == 0
                r = q;
                return;
            end
            rho = 1 ./ sum(Sm .* Ym, 2);
            al = zeros(k, 1);
            for i = k:-1:1
                al(i) = rho(i) * (Sm(i, fr) * q');
                q = q - al(i) * Ym(i, fr);
            end
            r = (1 / (rho(k) * (Ym(k, :) * Ym(k, :)'))) * q;
            for i = 1:k
                be = rho(i) * (Ym(i, fr) * r');
                r = r + (al(i) - be) * Sm(i, fr);
            end
        end

        function fv = qf(z)
            nf = nf + 1;
            fv = evaluate(z, false);
        end

        % One-sided step of 1e-8, taken backwards where it would leave the box
        function [gr, ok2] = fd_grad(xc, fc)
            gr = zeros(1, D);
            hs = h * ones(1, D);
            hs(xc + hs > ub) = -h;
            hs = (xc + hs) - xc;              % the step actually taken, as scipy divides by
            Pts = repmat(xc, D, 1) + diag(hs);
            fv = evaluate(Pts, false);
            nf = nf + D;
            ok2 = ~done && all(isfinite(fv));
            if ok2
                gr = (fv' - fc) ./ hs;
            end
        end
    end
end

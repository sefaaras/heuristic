% ----------------------------------------------------------------------- %
% Mean-Variance Mapping Optimization, population-based hybrid with new mapping (MVMO-PHM)
% CEC 2016 learning-based track (CEC 2015 suite) -- 1st place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_par = 10                  % Particles, each with its own solution archive
%   n_tosave = 5                % Archive size per particle
%   good share = 0.8 -> 0.5     % Particles ranked good, linear in t = FE/maxFe, +/-15 % noise
%   m = min(5, D)               % Coordinates mutated per child
%   fs = 1 -> 20                % Shape scaling, quadratic in t, times 1 + 4*(rand - 0.2)
%   alpha = 1                   % Scale of the bad particles' multi-parent step
%   local search = 5 runs from t = 0.25  % Gradient polish of top-ranked particles
%
% Algorithm Concept:
%   - Search runs in the unit cube; each particle keeps an archive of its best
%     solutions whose per-coordinate mean and variance shape its mutation
%   - Mutation: m coordinates (one in sequence, the rest random) are redrawn from a
%     uniform (early) or a narrowing normal (late) draw and bent towards the mean
%   - The bending is a hyperbolic mapping centred at 0.5 (Mapping #3), switched to
%     an exponential one (#2) once the local-search allowance is spent
%   - Particles are ranked by archive best: good ones continue from their own best,
%     bad ones from x_one + beta*(x_best - x_worst) over good-group members
%   - After a quarter of the budget the top-ranked particles are polished by a
%     gradient-based local search, five runs in all
%
% Reference:
% Jose L. Rueda, Istvan Erlich,
% Solving the CEC2016 Real-Parameter Single Objective Optimization Problems
% through MVMO-PHM, Technical Report, Delft University of Technology and
% University Duisburg-Essen, 2016.
% https://github.com/P-N-Suganthan/CEC2016
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release for the learning-based track (mvmo.m in
% CEC-2015-Learning-Based.rar of the repository above). The release tunes every
% function and dimension separately (Parameter_MVMO-SH_<f>_D<d>.txt); those tables
% are not ported, the parameters are its round-valued baseline block (the F1/F2
% entry, with the report's example fs 1 -> 20). Its local search calls fmincon
% interior-point (Optimization Toolbox); a projected BFGS with central differences
% in the normalised box stands in, under fmincon's limits: the whole remaining
% budget, 1000 iterations, 1e-6 first-order tolerance. The stop at error < 1e-8 is
% removed and the feasibility flag (error == 0) dropped, a no-op without constraints.
% The bad-particle rebuild, repeated until one beta fits all coordinates, is capped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mvmo_phm(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    n_par          = 10;
    n_to_save      = 5;
    fs_start       = 1;
    fs_end         = 20;
    local_prob     = 100;
    min_eval_LS    = round(0.25 * maxFE);
    ratio_gute_max = 0.8;
    ratio_gute_min = 0.5;
    n_rand_ini     = 5;
    n_rand_last    = 5;
    local_i_max    = 5;
    alpha          = 1;
    max_passes     = 100;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = lb + 0.5 * span;

    % Drawn particle by particle, coordinate by coordinate, as the release's nested loops
    x_norm    = rand(D, n_par)';
    Shape_dyn = ones(n_par, D);
    shape     = zeros(n_par, D);
    meann     = 0.5 * ones(n_par, D);
    meann_app = meann;
    izm       = zeros(1, n_par);
    IX        = 1:n_par;
    arch_u    = nan(n_to_save, D, n_par);
    arch_r    = nan(n_to_save, D, n_par);
    arch_f    = inf(n_to_save, n_par);
    no_in     = zeros(1, n_par);
    no_inin   = zeros(1, n_par);
    goodbad   = true(1, n_par);
    local_search = false(1, n_par);

    local_i          = 0;
    local_i_selected = 0;
    ls_all_selected  = false;
    mod_method       = 3;
    l_vari           = n_to_save;
    n_rand_last      = min(max(n_rand_last, 1), D);
    n_rand_ini       = min(n_rand_ini, D);
    vary_n_rand      = n_rand_ini ~= n_rand_last;
    n_randomly       = n_rand_last;
    delta_nrand      = n_rand_ini - n_rand_last;
    fs_factor0       = fs_start;
    local_search0    = local_prob / 100;
    firsttime        = true;
    changed_best     = false;
    stop             = false;

    while FE < maxFE && ~stop
        ff   = FE / maxFE;
        ff2  = ff * ff;
        vvqq = 10 ^ (-(5.3 + ff * 5.3));   % tolerance below which archive values count as equal

        border_gute = n_par * (ratio_gute_max - ff * (ratio_gute_max - ratio_gute_min));
        border_gute = round(normalfunc(border_gute, border_gute * 0.15));
        if border_gute < 3 && n_par > 3
            border_gute = 3;
        end
        if border_gute > n_par
            border_gute = n_par;
        end
        if vary_n_rand
            n_randomly_X = round(n_rand_ini - ff * delta_nrand);
            n_randomly = round(n_rand_last + rand * (n_randomly_X - n_rand_last));
        end
        if fs_start ~= fs_end
            fs_factor0 = fs_start + ff2 * (fs_end - fs_start);
        end

        if local_search0 > 0 && FE > min_eval_LS && ~ls_all_selected
            for iuz = 1:border_gute
                if rand <= local_search0 && local_i_selected < local_i_max
                    if rand <= local_search0
                        local_search(IX(iuz)) = true;
                        local_i_selected = local_i_selected + 1;
                        if local_i_selected >= local_i_max
                            ls_all_selected = true;
                            break;
                        end
                    end
                end
            end
        end

        % IX is re-sorted inside the sweep, so a particle can be visited twice or skipped, as in the release
        ipx = 0;
        while ipx < n_par
            ipx = ipx + 1;
            ipp = IX(ipx);

            if no_inin(ipp) >= 1 && ~local_search(ipp)
                [considered, izm(ipp)] = select_variables(D, n_randomly, izm(ipp));
                if no_inin(ipp) < l_vari
                    considered(:) = true;
                end
                for ivar = find(considered)
                    if rand > ff2
                        xv = rand;
                    else
                        xv = -1;
                        while xv > 1 || xv < 0
                            xv = normalfunc(0.5, (1 - ff2) ^ 2 + 1e-3);
                        end
                    end
                    if shape(ipp, ivar) > 0
                        sss1 = shape(ipp, ivar);
                        if sss1 > Shape_dyn(ipp, ivar)
                            Shape_dyn(ipp, ivar) = Shape_dyn(ipp, ivar) * 1.1;
                        else
                            Shape_dyn(ipp, ivar) = Shape_dyn(ipp, ivar) / 1.1;
                        end
                        grosser = max(Shape_dyn(ipp, ivar), sss1);
                        kleiner = min(Shape_dyn(ipp, ivar), sss1);
                        if rand > 0.5
                            s1 = grosser;
                            s2 = kleiner;
                        else
                            s1 = kleiner;
                            s2 = grosser;
                        end
                        fs_factor = fs_factor0 * (1 + 4 * (rand - 0.2));
                        xv = h_function(meann_app(ipp, ivar), s1 * fs_factor, s2 * fs_factor, xv, mod_method);
                    end
                    x_norm(ipp, ivar) = xv;
                end
            end
            bad = ~isfinite(x_norm(ipp, :));
            x_norm(ipp, bad) = rand(1, nnz(bad));

            if FE >= maxFE
                stop = true;
                break;
            end
            if local_search(ipp)
                [x_norm(ipp, :), f_new] = local_search_run(x_norm(ipp, :), ipp);
                local_i = local_i + 1;
                if local_i >= local_i_max
                    local_search0 = 0;
                    local_search(:) = false;
                    mod_method = 2;
                end
            else
                f_new = evaluate(lb + span .* x_norm(ipp, :));
            end
            x_raw = lb + span .* x_norm(ipp, :);

            % The mean used by the next mapping of this particle comes from a random good particle
            meann_app(ipp, :) = meann(ipp, :);
            if border_gute > 2
                irandom = ipp;
                while ipp == irandom
                    irandom = IX(round(rand * (border_gute - 1)) + 1);
                end
                meann_app(ipp, :) = meann(irandom, :);
            end

            fill_archive(ipp, x_norm(ipp, :), x_raw, f_new, vvqq);
            record(0, [], []);

            if no_inin(ipp) > l_vari
                if changed_best || firsttime
                    [~, IX] = sort(arch_f(1, :));
                    firsttime = false;
                end
                goodbad(:) = false;
                goodbad(IX(1:border_gute)) = true;
                if ~goodbad(ipp)
                    bests = reshape(arch_u(1, :, :), D, n_par);
                    x_norm(ipp, :) = multi_parent(bests, IX, border_gute, n_par, ff2, alpha, max_passes);
                else
                    x_norm(ipp, :) = min(0.999 * arch_u(1, :, ipp) + 0.001 * arch_u(1, :, IX(1)), 1);
                end
            else
                x_norm(ipp, :) = arch_u(1, :, ipp);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of Xr (raw coordinates, within the budget), tracking the best per evaluation
    function f = evaluate(Xr)
        [fv, FE_new] = calculate_fitness(Xr', problem, FE);
        f = fv(:);
        for q = 1:numel(f)
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = Xr(q, :);
            end
            curve(FE + q) = bsf;
        end
        FE = FE_new;
    end

    % Population = each visited particle's archive leader; during a local search its row is the current iterate
    function record(row, xr, fr)
        vis = no_in > 0;
        Pm = reshape(arch_r(1, :, :), D, n_par)';
        Fm = arch_f(1, :)';
        if row > 0
            Pm(row, :) = xr;
            Fm(row) = fr;
            vis(row) = true;
        end
        [population_history, fitness_history, history_index] = record_history( ...
            FE, Pm(vis, :), Fm(vis), population_history, fitness_history, history_index, maxFE);
    end

    % Sorted n-best archive of particle ipp, and the mean/shape it implies once full
    function fill_archive(ipp, u, xr, f, vvqq)
        no_in(ipp) = no_in(ipp) + 1;
        changed = false;
        changed_best = false;
        i_position = 0;
        if no_in(ipp) == 1
            arch_f(:, ipp) = 1e200;
            arch_u(1, :, ipp) = u;
            arch_r(1, :, ipp) = xr;
            arch_f(1, ipp) = f;
            no_inin(ipp) = no_inin(ipp) + 1;
            changed_best = true;
        else
            for ij = 1:n_to_save
                if f < arch_f(ij, ipp)
                    i_position = ij;
                    changed = true;
                    if ij < n_to_save
                        no_inin(ipp) = no_inin(ipp) + 1;
                    end
                    break;
                end
            end
        end
        if changed
            nn = n_to_save;
            if i_position == 1
                changed_best = true;
            end
            if no_inin(ipp) < n_to_save
                nn = no_inin(ipp);
            end
            isdx = nn:-1:(i_position + 1);
            arch_u(isdx, :, ipp) = arch_u(isdx - 1, :, ipp);
            arch_r(isdx, :, ipp) = arch_r(isdx - 1, :, ipp);
            arch_f(isdx, ipp)    = arch_f(isdx - 1, ipp);
            arch_u(i_position, :, ipp) = u;
            arch_r(i_position, :, ipp) = xr;
            arch_f(i_position, ipp)    = f;
            if no_inin(ipp) >= n_to_save
                for jv = 1:D
                    [meann(ipp, jv), shape(ipp, jv)] = mv_noneq(arch_u(1:nn, jv, ipp), ...
                        meann(ipp, jv), shape(ipp, jv), vvqq);
                end
            end
        end
    end

    % fmincon stand-in: projected BFGS in the unit box, central differences, Armijo backtracking
    function [u, fu] = local_search_run(u0, row)
        u = min(max(u0, 0), 1);
        fu = evaluate(lb + span .* u);
        record(row, lb + span .* u, fu);
        if ~isfinite(fu)
            return;
        end
        [g, ok] = fd_gradient(u, fu, row);
        if ~ok
            return;
        end
        tol = 1e-6 * max(1, norm(g, inf));
        H = eye(D);
        scaled = false;
        for it = 1:1000
            % First-order optimality on a box: the gradient over coordinates not pinned by their bound
            free = ~((u <= 0 & g > 0) | (u >= 1 & g < 0));
            if ~any(free) || norm(g(free), inf) <= tol
                break;
            end
            d = zeros(1, D);
            d(free) = -(H(free, free) * g(free)')';
            if ~(g * d' < 0)
                H = eye(D);
                scaled = false;
                d = zeros(1, D);
                d(free) = -g(free);
            end
            if scaled
                a = 1;
            else
                a = min(1, 1 / norm(d, inf));
            end
            accepted = false;
            while a > 1e-16
                if FE >= maxFE
                    return;
                end
                un = min(max(u + a * d, 0), 1);
                fn = evaluate(lb + span .* un);
                if fn <= fu + 1e-4 * (g * (un - u)')
                    accepted = true;
                    break;
                end
                a = a / 2;
            end
            if ~accepted
                record(row, lb + span .* u, fu);
                break;
            end
            s = un - u;
            u = un;
            fu = fn;
            record(row, lb + span .* u, fu);
            if norm(s, inf) < 1e-10
                break;
            end
            [gn, ok] = fd_gradient(u, fu, row);
            if ~ok
                return;
            end
            yv = gn - g;
            sy = s * yv';
            if sy > 1e-10 * norm(s) * norm(yv)
                if ~scaled
                    H = (sy / (yv * yv')) * eye(D);
                    scaled = true;
                end
                rho = 1 / sy;
                V = eye(D) - rho * (s' * yv);
                H = V * H * V' + rho * (s' * s);
            end
            g = gn;
        end
    end

    % Central differences with fmincon's default step eps^(1/3), one-sided where a bound cuts it
    function [g, ok] = fd_gradient(u, fu, row)
        g = zeros(1, D);
        h = eps ^ (1 / 3);
        lo = max(u - h, 0);
        hi = min(u + h, 1);
        pts = repmat(u, 2 * D, 1);
        pts(sub2ind([2 * D, D], 1:D, 1:D)) = hi;
        pts(sub2ind([2 * D, D], D + (1:D), 1:D)) = lo;
        n_eval = min(2 * D, maxFE - FE);
        if n_eval < 1
            ok = false;
            return;
        end
        f = evaluate(lb + span .* pts(1:n_eval, :));
        record(row, lb + span .* u, fu);
        ok = n_eval == 2 * D && all(isfinite(f));
        if ok
            g = (f(1:D) - f(D + 1:end))' ./ (hi - lo);
        end
    end
end

% Variable selection mode 4: one coordinate in cyclic order, the rest distinct random ones
function [considered, izm_i] = select_variables(D, n_randomly, izm_i)
    considered = false(1, D);
    izm_i = izm_i - 1;
    if izm_i < 1
        izm_i = D;
    end
    considered(izm_i) = true;
    for ii = 1:n_randomly - 1
        inn = round(rand * (D - 1)) + 1;
        while considered(inn)
            inn = round(rand * (D - 1)) + 1;
        end
        considered(inn) = true;
    end
end

% Mean over the distinct archive values (0.9 new, 0.1 old) and shape -log(variance) about it
function [vmean, vshape] = mv_noneq(values, vmean_old, vshape, vvqq)
    distinct = values(1);
    for ii = 2:numel(values)
        if ~any(abs(distinct - values(ii)) < vvqq)
            distinct(end + 1) = values(ii); %#ok<AGROW>
        end
    end
    if numel(distinct) > 1
        vmean = 0.1 * vmean_old + 0.9 * (sum(distinct) / numel(distinct));
        if vmean > 1
            vmean = 1;
        end
        vv = sum((distinct - vmean) .^ 2) / numel(distinct);
        if vv > 1e-50
            vshape = -log(vv);
        end
    else
        vmean = vmean_old;
    end
end

% Mapping #2 (exponential) or #3 (hyperbolic): both pass through the mean at x_curr = 0.5
function xnew = h_function(x_mean, s1, s2, x_curr, mod_method)
    if x_curr < 0.5
        if (1 - x_mean) > 1e-15
            s11 = s1 / (1 - x_mean);
        else
            s11 = s1 / 1e-15;
        end
        if mod_method == 2
            hm = x_mean * (1 - exp(-0.5 * s11));
            hf = x_mean * (1 - exp(-x_curr * s11));
        else
            hm = -x_mean / (0.5 * s11 + 1) + x_mean;
            hf = -x_mean / (x_curr * s11 + 1) + x_mean;
        end
        xnew = hf + (x_mean - hm) * 2 * x_curr;
    else
        if x_mean > 1e-15
            s11 = s2 / x_mean;
        else
            s11 = s2 / 1e-15;
        end
        if mod_method == 2
            hm = (1 - x_mean) * exp(-0.5 * s11);
            hb = (1 - x_mean) * exp(-(1 - x_curr) * s11) + x_mean;
        else
            hm = (1 - x_mean) / (0.5 * s11 + 1);
            hb = (1 - x_mean) / ((1 - x_curr) * s11 + 1) + x_mean;
        end
        xnew = hb - hm * 2 * (1 - x_curr);
    end
end

% Bad particle: x_one + beta*(x_best - x_worst) per coordinate, beta and parents redrawn on a box exit
function x = multi_parent(bests, IX, border_gute, n_par, ff2, alpha, max_passes)
    D = size(bests, 1);
    x = zeros(1, D);
    [bestp, onep1, worstp] = must(IX, border_gute, n_par, ff2);
    bbb = 1.1 + (rand - 0.5) * 2;
    beta1 = alpha * 3 * bbb * ((1 + 2.5 * ff2) * rand - (1 - ff2) * 0.8);
    beta10 = 0;
    passes = 0;
    while beta1 ~= beta10 && passes < max_passes
        beta10 = beta1;
        passes = passes + 1;
        for jx = 1:D
            ccc = bests(jx, bestp) - bests(jx, worstp);
            if abs(ccc) > 1e-8
                xj = bests(jx, onep1) + beta1 * ccc;
            else
                xj = bests(jx, onep1);
            end
            bj = bests(jx, bestp);
            if bj <= 0.85 && bj >= 0.15 && rand > 0.015
                redraw = true;
            elseif bj > 0.85 || (bj < 0.15 && rand < 0.985)
                redraw = true;
            else
                redraw = false;
            end
            if redraw
                while xj > 1 || xj < 0
                    [bestp, onep1, worstp] = must(IX, border_gute, n_par, ff2);
                    ccc = bests(jx, bestp) - bests(jx, worstp);
                    if abs(ccc) > 1e-8
                        bbb = 1.1 + (rand - 0.5) * 2;
                        beta1 = alpha * 3 * bbb * ((1 + 2.5 * ff2) * rand - (1 - ff2) * 0.3);
                        xj = bests(jx, onep1) + beta1 * ccc;
                    else
                        xj = bests(jx, onep1);
                    end
                end
            else
                xj = min(max(xj, 0), 1);
            end
            x(jx) = xj;
        end
    end
end

% Parent triple by rank: best among the top few, worst just past the good border, one in between
function [bestp, onep1, worstp] = must(IX, border_gute, n_par, ff2)
    iup = round(5 * (1 - ff2)) + 1;
    if iup > border_gute
        iup = round((border_gute - 1) * (1 - ff2)) + 1;
    end
    bestp = round(rand * (iup - 1) + 1);
    worstp = -1;
    iup = min(15, n_par);
    while worstp <= bestp || worstp > n_par
        worstp = round(rand * iup);
        worstp = round(border_gute + (worstp - 3));
    end
    onep1 = round(rand * ((worstp - 1) - (bestp + 1)) + bestp + 1);
    onep1 = IX(onep1);
    bestp = IX(bestp);
    worstp = IX(worstp);
end

% Approximate normal: mean + spread * (sum of 12 uniforms - 6) / 6, bounded by +/- spread
function v = normalfunc(mittel, streuung)
    v = mittel + streuung * (sum(rand(1, 12)) - 6) / 6;
end

% ----------------------------------------------------------------------- %
% Mean-Variance Mapping Optimization, CEC 2014 version (MVMO)
% CEC 2014 competition -- top-5 entry (organiser code release); mvmo is the 2013 one
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_par = 70/100/100/150      % Particles at D = 10/30/50/100, each with an archive
%   n_tosave = 25               % Archive size per particle
%   m = 5/15/15/30 -> 1         % Coordinates mutated, a uniform draw below a falling cap
%   fs = 1 -> 20                % Shape scaling, linear in t = FE/maxFe, with random spread
%   good share = 0.7 -> 0.1     % Particles ranked good, linear in t
%   local search = 10 %         % Chance per evaluation in t = 0.5..0.9 (0.01..0.1 at D=30)
%   independent phase = 2*n_par % Evaluations before particles are ranked
%
% Algorithm Concept:
%   - Search runs in the unit cube; each particle's archive of its best solutions
%     gives a per-coordinate mean and shape -log(variance) for its mapping
%   - Mutation: m coordinates (one in sequence, the rest random) get a uniform draw
%     bent towards the archive mean by the h-function; near-bound values snap to it
%   - The two shape factors of a coordinate are its archive shape and a dynamic one
%     that chases it by a random factor in [1, 1.4]
%   - Ranked by archive best: good particles continue from their own best, bad ones
%     from x_one + beta*(x_best - x_worst) over the good group
%   - A gradient-based local search may start from a particle's next parent
%
% Reference:
% Istvan Erlich, Jose L. Rueda, Sebastian Wildenhues, Fekadu Shewarega,
% Evaluating the Mean-Variance Mapping Optimization on the IEEE-CEC 2014 test suite,
% 2014 IEEE Congress on Evolutionary Computation (CEC), 2014, pp. 1625-1632.
% https://doi.org/10.1109/CEC.2014.6900516
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (MVMO-IEEE-CEC2014-PartA-Real-num-opt.zip
% in Top-Methods-Part-A of github.com/P-N-Suganthan/CEC2014): one script per D = 10,
% 30, 50, 100, differing only in the four tabled values; the nearest D is used (ties
% to the smaller). Its local search calls fmincon interior-point with the whole
% remaining budget; mvmo_phm's projected BFGS (central differences, normalised box,
% 1000 iterations, 1e-6 tolerance) stands in. Kept as released: a local-search result
% counts as feasible only if its value equals the last evaluation's, and an archive
% slot is compared with particle 1's feasibility flag, so such results rarely enter
% an archive. Guards: the bad-particle rebuild is capped at 1000 redraws, non-finite
% coordinates are redrawn, and points are clamped to the box against rounding.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mvmo14(problem)

    D       = problem.dimension;
    lb      = problem.lb(:)';
    ub      = problem.ub(:)';
    maxFE   = problem.maxFe;
    scaling = ub - lb;

    % Per-dimension scripts of the release (mvmo_CEC2014_Part_A_D<d>.m)
    [~, kd] = min(abs([10 30 50 100] - D));
    n_par_tab   = [70 100 100 150];
    ls_from_tab = [0.5 0.01 0.5 0.5];
    ls_to_tab   = [0.9 0.1 0.9 0.9];
    n_ini_tab   = [5 15 15 30];

    n_par           = n_par_tab(kd);
    n_to_save       = 25;
    fs_factor_start = 1;
    fs_factor_end   = 20;
    delta_Shape_dyn = 0.2;
    local_prob      = 10.0;
    min_eval_LS     = round(ls_from_tab(kd) * maxFE);
    max_eval_LS     = round(ls_to_tab(kd) * maxFE);
    ratio_gute_max  = 0.7;
    ratio_gute_min  = 0.1;
    n_randomly_ini  = n_ini_tab(kd);
    n_randomly_last = 1;
    indpendent_runs = n_par * 2;
    max_redraws     = 1000;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;
    bsf = inf;
    bsf_solution = lb + 0.5 * scaling;
    last_f = inf;

    % Particle by particle, coordinate by coordinate, as the release's nested loops
    xx = lb + rand(D, n_par)' .* scaling;
    x_norm = (xx - lb) ./ scaling;
    Shape_dyn = 40 * rand(D, n_par)';

    n_randomly_last = min(max(n_randomly_last, 1), D);
    n_randomly_ini  = min(n_randomly_ini, D);
    yes_n_randomly  = n_randomly_ini ~= n_randomly_last;
    n_randomly      = n_randomly_last;
    delta_nrandomly = n_randomly_ini - n_randomly_last;
    delta_Shape_dyn0 = delta_Shape_dyn;
    delta_Shape_dyn1 = 1 + delta_Shape_dyn0;
    local_search0   = local_prob / 100;

    bests   = nan(n_to_save, D, n_par);
    arch_r  = nan(n_to_save, D, n_par);
    fit     = inf(n_to_save, n_par);
    feas    = zeros(n_to_save, n_par);
    lead_x  = zeros(n_par, D);
    lead_f  = inf(n_par, 1);
    shape   = zeros(n_par, D);
    variance = ones(n_par, D);
    meann   = x_norm;
    meann_app = x_norm;
    izm     = zeros(1, n_par);
    considered = true(n_par, D);
    local_search = false(n_par, 1);
    no_in   = zeros(1, n_par);
    no_inin = zeros(1, n_par);
    goodbad = zeros(n_par, 1);
    IX      = (1:n_par)';
    amax    = 0;
    firsttime    = true;
    changed_best = false;
    stop = false;

    while ~stop
        for ipp = 1:n_par
            ff = FE / maxFE;
            ff2 = ff * ff;
            one_minus_ff2 = 1 - ff2 * 0.99;
            streu = one_minus_ff2 * 0.5;
            border_gute0 = ratio_gute_max - ff * (ratio_gute_max - ratio_gute_min);
            border_gute = round(n_par * border_gute0);
            if border_gute < 3 || n_par - border_gute < 1
                border_gute = n_par;
            end
            if yes_n_randomly
                n_randomly_X = round(n_randomly_ini - ff * delta_nrandomly);
                n_randomly = round(n_randomly_last + rand * (n_randomly_X - n_randomly_last));
            end
            fs_factor0 = fs_factor_start + ff * (fs_factor_end - fs_factor_start);

            bad = ~isfinite(x_norm(ipp, :));
            x_norm(ipp, bad) = rand(1, nnz(bad));
            x_raw = min(max(lb + scaling .* x_norm(ipp, :), lb), ub);
            if local_search(ipp)
                [x_raw, f_new] = local_search_run(x_raw, ipp);
                local_search(ipp) = false;
                % As released: feasible only when the returned value is the last one evaluated
                feasible = f_new == last_f;
            else
                f_new = evaluate(x_raw);
                feasible = true;
            end
            if FE >= maxFE
                stop = true;
                break;
            end

            x_norm(ipp, :) = (x_raw - lb) ./ scaling;
            fill_archive(ipp, f_new, feasible, x_raw);
            meann_app(ipp, :) = meann(ipp, :);
            record(0, [], []);

            if FE > indpendent_runs && border_gute < n_par
                if changed_best || firsttime
                    A = fit(1, :)';
                    if firsttime
                        amax = max(A);
                    end
                    firsttime = false;
                    for ia = 1:n_par
                        if ~feas(1, ia)
                            A(ia) = A(ia) + amax;
                        end
                    end
                    [~, IX] = sort(A);
                end
                goodbad(IX(border_gute+1:n_par)) = 0;
                goodbad(IX(1:border_gute)) = 1;
                iec = randi(border_gute - 2, 1, 1);
                bestp  = IX(1);
                onep   = IX(iec + 1);
                worstp = IX(border_gute);
                if ~goodbad(ipp)
                    % Bad particle: rebuilt from three members of the good group
                    shift = streu;
                    beta1 = 2.5 * (rand - shift);
                    for jl = 1:D
                        x_norm(ipp, jl) = bests(1, jl, onep) + beta1 * (bests(1, jl, bestp) - bests(1, jl, worstp));
                        tries = 0;
                        while (x_norm(ipp, jl) > 1 || x_norm(ipp, jl) < 0) && tries < max_redraws
                            beta2 = 2.5 * (rand - shift);
                            x_norm(ipp, jl) = bests(1, jl, onep) + beta2 * (bests(1, jl, bestp) - bests(1, jl, worstp));
                            tries = tries + 1;
                        end
                        x_norm(ipp, jl) = min(max(x_norm(ipp, jl), 0), 1);
                    end
                    meann_app(ipp, :) = x_norm(ipp, :);
                else
                    x_norm(ipp, :) = bests(1, :, ipp);
                end
            else
                x_norm(ipp, :) = bests(1, :, ipp);
            end
            considered(ipp, :) = false;

            % A local search, if drawn, starts from this parent unmutated at the particle's next turn
            local_search(ipp) = rand < local_search0 && FE > min_eval_LS && FE < max_eval_LS;
            if ~local_search(ipp)
                variable_select(ipp);
            end

            for ivar = 1:D
                if considered(ipp, ivar)
                    x_norm(ipp, ivar) = rand;
                    if shape(ipp, ivar) > 0
                        sss1 = shape(ipp, ivar);
                        sss2 = sss1;
                        delta_ddd_x = delta_Shape_dyn0 * (rand - 0.5) * 2 + delta_Shape_dyn1;
                        if shape(ipp, ivar) > Shape_dyn(ipp, ivar)
                            Shape_dyn(ipp, ivar) = Shape_dyn(ipp, ivar) * delta_ddd_x;
                        else
                            Shape_dyn(ipp, ivar) = Shape_dyn(ipp, ivar) / delta_ddd_x;
                        end
                        if rand < 0.5
                            sss1 = Shape_dyn(ipp, ivar);
                        else
                            sss2 = Shape_dyn(ipp, ivar);
                        end
                        if rand > 0.5
                            fs_factor = fs_factor0 * (1 + rand);
                        else
                            fs_factor = 1 + fs_factor0 * (1 - rand) * 0.25;
                        end
                        sss1 = sss1 * fs_factor;
                        sss2 = sss2 * fs_factor;
                        x_norm(ipp, ivar) = h_function(meann_app(ipp, ivar), sss1, sss2, x_norm(ipp, ivar));
                        if x_norm(ipp, ivar) > 0.98 && rand < 0.2
                            x_norm(ipp, ivar) = 1.0;
                        elseif x_norm(ipp, ivar) < 0.02 && rand < 0.2
                            x_norm(ipp, ivar) = 0.0;
                        end
                    end
                end
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
        last_f = f(end);
    end

    % Population = each visited particle's archive leader; during a local search its row is the current iterate
    function record(row, xr, fr)
        if row > 0 || no_in(n_par) == 0
            vis = no_in > 0;
            Pm = lead_x;
            Fm = lead_f;
            if row > 0
                Pm(row, :) = xr;
                Fm(row) = fr;
                vis(row) = true;
            end
            [population_history, fitness_history, history_index] = record_history( ...
                FE, Pm(vis, :), Fm(vis), population_history, fitness_history, history_index, maxFE);
        else
            [population_history, fitness_history, history_index] = record_history( ...
                FE, lead_x, lead_f, population_history, fitness_history, history_index, maxFE);
        end
    end

    % Sorted n-best archive of particle ipp (the release's Fill_solution_archive)
    function fill_archive(ipp, f, feasible, xr)
        no_in(ipp) = no_in(ipp) + 1;
        changed = false;
        changed_best = false;
        i_position = 0;
        if no_in(ipp) == 1
            fit(:, ipp)  = 1e200;
            feas(:, ipp) = 0;
            bests(1, :, ipp)  = x_norm(ipp, :);
            arch_r(1, :, ipp) = xr;
            fit(1, ipp)  = f;
            feas(1, ipp) = feasible;
            no_inin(ipp) = no_inin(ipp) + 1;
            changed_best = true;
        else
            for ij = 1:n_to_save
                % As released: the flag compared is particle 1's (a linear index into the table)
                if (f < fit(ij, ipp) && feasible == feas(ij, 1)) || (feas(ij, ipp) < feasible)
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
            isdx = nn:-1:i_position+1;
            bests(isdx, :, ipp)  = bests(isdx-1, :, ipp);
            arch_r(isdx, :, ipp) = arch_r(isdx-1, :, ipp);
            fit(isdx, ipp)  = fit(isdx-1, ipp);
            feas(isdx, ipp) = feas(isdx-1, ipp);
            bests(i_position, :, ipp)  = x_norm(ipp, :);
            arch_r(i_position, :, ipp) = xr;
            fit(i_position, ipp)  = f;
            feas(i_position, ipp) = feasible;
            if no_inin(ipp) >= 2
                [meann(ipp, :), variance(ipp, :)] = mv_noneq(bests(1:nn, :, ipp));
                id_nonzero = variance(ipp, :) > 1.1e-100;
                shape(ipp, id_nonzero) = -log(variance(ipp, id_nonzero));
            end
        end
        lead_x(ipp, :) = arch_r(1, :, ipp);
        lead_f(ipp) = fit(1, ipp);
    end

    % Mode 4: one coordinate in cyclic order, the rest distinct random ones
    function variable_select(ipp)
        % The release draws once for a mode switch that can never fire
        rand;
        izm(ipp) = izm(ipp) - 1;
        if izm(ipp) < 1
            izm(ipp) = D;
        end
        considered(ipp, izm(ipp)) = true;
        if n_randomly > 1
            for ii = 1:n_randomly-1
                inn = round(rand * (D - 1)) + 1;
                while considered(ipp, inn)
                    inn = round(rand * (D - 1)) + 1;
                end
                considered(ipp, inn) = true;
            end
        end
    end

    % fmincon stand-in: projected BFGS in the unit box, central differences, Armijo backtracking
    function [xr, fu] = local_search_run(xr0, row)
        u = min(max((xr0 - lb) ./ scaling, 0), 1);
        fu = evaluate(lb + scaling .* u);
        xr = lb + scaling .* u;
        record(row, xr, fu);
        if ~isfinite(fu) || FE >= maxFE
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
            dd = zeros(1, D);
            dd(free) = -(H(free, free) * g(free)')';
            if ~(g * dd' < 0)
                H = eye(D);
                scaled = false;
                dd = zeros(1, D);
                dd(free) = -g(free);
            end
            if scaled
                a = 1;
            else
                a = min(1, 1 / norm(dd, inf));
            end
            accepted = false;
            while a > 1e-16
                if FE >= maxFE
                    return;
                end
                un = min(max(u + a * dd, 0), 1);
                fn = evaluate(lb + scaling .* un);
                if fn <= fu + 1e-4 * (g * (un - u)')
                    accepted = true;
                    break;
                end
                a = a / 2;
            end
            if ~accepted
                record(row, xr, fu);
                break;
            end
            s = un - u;
            u = un;
            fu = fn;
            xr = lb + scaling .* u;
            record(row, xr, fu);
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
        fd = evaluate(lb + scaling .* pts(1:n_eval, :));
        record(row, lb + scaling .* u, fu);
        ok = n_eval == 2 * D && all(isfinite(fd));
        if ok
            g = (fd(1:D) - fd(D + 1:end))' ./ (hi - lo);
        end
    end
end

% Mean and variance per column over the DISTINCT archive values, summed in archive order
function [vmean, vvar] = mv_noneq(V)
    [nn, D] = size(V);
    vmean = zeros(1, D);
    vvar = zeros(1, D);
    if nn > 1
        sv = sort(V, 1);
        has_dup = any(abs(diff(sv, 1, 1)) < 1e-70, 1);
    else
        has_dup = false(1, D);
    end
    for c = 1:D
        if has_dup(c) || nn == 1
            vals = V(1, c);
            for r = 2:nn
                if ~any(abs(vals - V(r, c)) < 1e-70)
                    vals(end + 1) = V(r, c); %#ok<AGROW>
                end
            end
        else
            vals = V(:, c)';
        end
        iz = numel(vals);
        cs = cumsum(vals);
        if iz > 1
            vmean(c) = cs(end) / iz;
            cv = cumsum((vals - vmean(c)) .* (vals - vmean(c)));
            vvar(c) = cv(end) / iz;
        else
            vmean(c) = vals(1);
            vvar(c) = 1.0e-100;
        end
    end
end

% The mapping: a uniform draw is bent towards the archive mean by the two shape factors
function x = h_function(x_bar, s1, s2, x_p)
    H  = x_bar .* (1 - exp(-x_p .* s1)) + (1 - x_bar) .* exp(-(1 - x_p) .* s2);
    H0 = (1 - x_bar) .* exp(-s2);
    H1 = x_bar .* exp(-s1);
    x  = H + H1 .* x_p + H0 .* (x_p - 1);
end

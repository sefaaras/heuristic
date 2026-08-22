function [pop_hist, fit_hist, hist_idx] = record_history(current_fe, population, fitness, pop_hist, fit_hist, hist_idx, max_fe)
% Samples the population at T evenly spaced FE points and stores its scalar and
% per-dimension metrics (see metric_labels) instead of the raw population.
% max_fe is the run's FE budget (problem.maxFe) and is all the caller needs to
% know about sampling. Every metric comes from THIS sample's (X, f) alone, so a
% row reads the same however the caller reached it.
%
% Commands, pushed once per parfor worker before the run (see main.m):
%   record_history('set_samples', T)        time samples; also drops the cached
%                                           problem, so call it FIRST
%   record_history('set_problem', problem)  bounds that normalise the distance
%                                           metrics, and on CEC2020RW the handle
%                                           used to read constraint violation
%   record_history('get_samples')           returns T

    persistent T_samples PROB LBv SPANv ZUBv

    if isempty(T_samples)
        T_samples = 1000;
    end

    if nargin <= 2 && (ischar(current_fe) || isstring(current_fe))
        switch lower(char(current_fe))
            case 'set_samples'
                T_samples = max(1, round(double(population)));
                PROB = []; LBv = []; SPANv = []; ZUBv = [];  % new run => no stale problem
            case 'set_problem'
                [PROB, LBv, SPANv, ZUBv] = cache_problem(population);
            case 'get_samples'
                pop_hist = T_samples;
            otherwise
                error('record_history:badCommand', ...
                      'Unknown command: %s', char(current_fe));
        end
        return;
    end

    T = T_samples;
    step = max(1, floor(max(1, round(double(max_fe))) / T));

    % Row T has to hold the END of the run, but only the caller knows which of its
    % samples is the last one. So once row T is filled, it keeps taking any later
    % sample that at least HALVES the FE still missing to max_fe: the refinement
    % converges onto the caller's final call, and because each write halves what
    % is left it costs at most log2(step) extra samples. It stops as soon as row T
    % reaches max_fe, so an algorithm that overshoots the budget cannot spin here.
    refine = false;
    if hist_idx > T
        if T < 2 || isempty(pop_hist) || size(pop_hist, 1) < T
            return;                     % T = 1 keeps its t=0 meaning
        end
        fe_T = pop_hist(T, 1);
        refine = isfinite(fe_T) && fe_T < max_fe && ...
                 double(current_fe) >= fe_T + max(1, (double(max_fe) - fe_T) / 2);
        if ~refine
            return;
        end
    end

    % Row 1 fires on the very first call so the STARTING state is on record and
    % every diversity curve has its t=0 reference (XPL%/XPT% normalise by it).
    % The slot is computed FROM the FE, not by advancing one per accepted call, so
    % the row index is the same time axis for every algorithm. Slots no call
    % reaches stay NaN; save_run holds the last measured row forward.
    %
    % Row h is the first sample at or after h*step, so the row index IS the
    % fraction of the budget spent (h/T) and row T is the state at max_fe. Both
    % caller shapes hit it exactly: a per-FE caller records at fe = h*step, and a
    % generational one whose population divides step records there too.
    if refine
        slot = T;
    elseif hist_idx == 1
        slot = 1;
    else
        slot = min(T, max(1, floor(max(0, double(current_fe)) / step)));
        if slot < hist_idx
            % Inside the last cell with row T still empty: the caller's generation
            % does not divide max_fe, so no sample reaches T*step and this is the
            % closest one to the end of the run.
            if hist_idx == T && double(current_fe) > (T - 1) * step
                slot = T;
            else
                return;
            end
        end
    end

    X = double(population);
    f = double(fitness(:));
    [N, dim] = size(X);

    if hist_idx == 1
        LB = metric_labels();
        % Both stay DOUBLE: eps(single(2100)) = 2.4e-4 against CEC error targets of
        % 1e-8, and single(ub) rounds ABOVE ub, so a population on the bound would
        % read as out of the box. save_run downcasts what it safely can on disk.
        pop_hist = nan(T, numel(LB.pop));
        fit_hist = nan(T, dim, numel(LB.dim));
    end

    % Order statistics keep +-Inf (a -Inf optimum is legitimate on CEC2020RW F25)
    % and skip NaN; the moments run on the finite entries only, because one Inf
    % turns mean/std/skew/kurt to NaN. nonfinite_frac reports what was dropped.
    finite_f = isfinite(f);
    ff = f(finite_f);
    fnn = f(~isnan(f));
    [f_best, ib] = min(f);
    if isempty(ib)
        ib = 1;
    end
    xbest = X(ib, :);

    % Aggregate distances are measured in box units, the per-dimension layers stay
    % raw. See metric_labels for why.
    [Z, box_norm] = box_scale(X, LBv, SPANv);

    centroid = mean(X, 1);
    if N > 1
        dstd = std(X, 0, 1);
        zstd = std(Z, 0, 1);
        zvar = var(Z, 0, 1);
    else
        dstd = zeros(1, dim);
        zstd = zeros(1, dim);
        zvar = zeros(1, dim);
    end
    dmin = min(X, [], 1);
    dmax = max(X, [], 1);
    mad_best = mean(abs(X - xbest), 1);
    ent_dim  = dim_entropy(X, dmin, dmax);

    zcen   = mean(Z, 1);
    dist_c = sqrt(sum((Z - zcen) .^ 2, 2));
    dap    = mean(dist_c);
    zbest  = Z(ib, :);
    dist_b = sqrt(sum((Z - zbest) .^ 2, 2));
    zbbox  = max(Z, [], 1) - min(Z, [], 1);

    [div_mpd, diam, nn] = pairwise_stats(Z);
    [fdc_p, fdc_r] = fdc_pair(f, dist_b, ib);
    [f_skew, f_kurt] = skew_kurt(ff);
    [eff_dim, pc1_frac] = eff_dim_pca(Z);
    [bound_frac, oob_frac] = box_occupancy(Z, ZUBv, box_norm);

    % One inert call to the constraint functions covers columns 27-29 and 33-35
    % together, and the raw objective it returns is also what ranks the elite on
    % CEC2020RW -- the fitness the search minimises is a run-private scalarisation
    % there, so ranking on it would make the elite mean different things in two
    % runs of the same problem.
    S = pop_constraints(X, f, PROB);
    [ekey, egrp] = elite_key(f, S);
    [elite_ratio, elite_shift] = elite_geometry(Z, zcen, dap, ekey, egrp);

    row = nan(1, size(pop_hist, 2));
    row(1)  = current_fe;                   % fe
    row(2)  = N;                            % pop_size
    row(3)  = f_best;                       % f_best
    row(4)  = max(f);                       % f_worst
    row(5)  = mean_safe(ff);                % f_mean
    row(6)  = median_safe(fnn);             % f_median
    row(7)  = std_safe(ff);                 % f_std
    row(8)  = iqr_manual(fnn);              % f_iqr
    row(9)  = dap;                          % div_dap
    row(10) = mean(zstd);                   % div_dim
    row(11) = sum(zvar);                    % div_var_sum
    row(12) = max(dist_c);                  % radius
    row(13) = sqrt(sum(zbbox .^ 2));        % bbox_diag
    row(14) = norm(zcen - zbest);           % centroid_to_best
    row(15) = mean(dist_b);                 % mean_to_best
    row(16) = fdc_p;                        % fdc
    row(17) = f_skew;                       % f_skew
    row(18) = f_kurt;                       % f_kurt
    row(19) = median(dist_c);               % median_to_centroid
    row(20) = div_mpd;                      % div_mpd
    row(21) = diam;                         % diameter
    row(22) = mean(nn);                     % div_nnd
    row(23) = mean(mean(abs(Z - median(Z, 1)))); % div_mad
    row(24) = mean(ent_dim);                % div_ent
    row(25) = uniq_fraction(f);             % uniq_frac
    row(26) = eff_dim;                      % eff_dim
    row(27) = mean(S.feas);                 % feas_frac
    row(28) = mean_safe(finite_of(S.viol)); % viol_mean
    row(29) = max_safe(finite_of(S.vmax));  % viol_max
    row(30) = fdc_r;                        % fdc_rank
    row(31) = 1 - nnz(finite_f) / max(1, N);% nonfinite_frac
    row(32) = double(box_norm);             % box_norm
    row(33) = min_safe(S.obj);              % obj_best
    row(34) = min_safe(S.obj(S.feas > 0));  % obj_best_feas
    row(35) = median_safe(finite_of(S.viol)); % viol_median
    row(36) = row_unique_fraction(X);       % x_uniq_frac
    row(37) = mean(any(~isfinite(X), 2));   % x_nonfinite_frac
    row(38) = bound_frac;                   % bound_frac
    row(39) = oob_frac;                     % oob_frac
    row(40) = median(nn);                   % nnd_median
    row(41) = nnd_variation(nn, dap);       % nnd_cv
    row(42) = pc1_frac;                     % pc1_frac
    row(43) = elite_ratio;                  % elite_div_ratio
    row(44) = elite_shift;                  % elite_centroid_shift

    pop_hist(slot, :)    = row;
    fit_hist(slot, :, 1) = centroid; % centroid
    fit_hist(slot, :, 2) = dstd;     % std
    fit_hist(slot, :, 3) = dmin;     % min
    fit_hist(slot, :, 4) = dmax;     % max
    fit_hist(slot, :, 5) = xbest;    % x_best
    fit_hist(slot, :, 6) = mad_best; % mad_best
    fit_hist(slot, :, 7) = ent_dim;  % ent
    hist_idx = slot + 1;
end

function [prob, lb, span, zub] = cache_problem(p)
% Keep the problem the current run is solving: its bounds normalise the distance
% metrics and on CEC2020RW its handle reads the population's own violation.
% Bounds are dropped unless lb and ub agree in length.
%
% zub is where the UPPER bound lands in box units: 1 on every ordinary axis, but
% 0 on a degenerate one (lb == ub), where span falls back to 1 and the box
% collapses to the point z = 0. bound_frac and oob_frac compare against zub
% rather than a hard 1, so such an axis reads as fully on the bound.
    prob = []; lb = []; span = []; zub = [];
    if isempty(p) || ~isstruct(p)
        return;
    end
    prob = p;
    if ~isfield(p, 'lb') || ~isfield(p, 'ub') || isempty(p.lb) || isempty(p.ub)
        return;
    end
    l = double(p.lb(:).');
    u = double(p.ub(:).');
    if numel(l) ~= numel(u)
        return;
    end
    lb = l;
    span = u - l;
    span(~isfinite(span) | span <= 0) = 1;   % degenerate axis => leave it raw
    zub = (u - l) ./ span;
end

function [Z, ok] = box_scale(X, lb, span)
% Map the population into box units z = (x - lb) ./ (ub - lb). Without this,
% CEC2020RW's per-dimension bounds (1e-2 .. 1e7) let the widest dimension dictate
% every distance: collapsing four of five dimensions moves div_dap by 0.0%. Falls
% back to raw coordinates when no matching problem was pushed.
    ok = ~isempty(span) && numel(span) == size(X, 2);
    if ~ok
        Z = X;
        return;
    end
    Z = (X - lb) ./ span;
end

function S = pop_constraints(X, f, prob)
% Constraint state of the CURRENT population, read straight off the constraint
% functions rather than from the evaluator's cumulative counters, which describe
% the points evaluated since the previous sample. count_this = false keeps the
% call out of the FE budget and off the scalarisation reference.
%
% Returns per-individual vectors, not aggregates: obj_best_feas needs the
% objective and the feasibility flag TOGETHER, and the elite ranking needs both
% again. The raw objective comes free, as one of calculate_fitness's outputs.
%
% Off CEC2020RW there is nothing to read -- f IS the objective, everything is
% feasible and every violation is 0 -- and the call is skipped entirely.
    N = size(X, 1);
    S = struct('obj', double(f(:)), 'feas', ones(N, 1), 'viol', zeros(N, 1), ...
               'vmax', zeros(N, 1), 'is_rw', false);
    if isempty(prob) || ~isstruct(prob) || ~isfield(prob, 'fhd') ...
            || ~isfield(prob, 'dimension') || prob.dimension ~= size(X, 2) ...
            || ~contains(func2str(prob.fhd), 'cec20rw')
        return;
    end
    S.is_rw = true;
    try
        [~, ~, is_feas, viol, obj, vmax] = calculate_fitness(X.', prob, 0, false);
        S.obj  = double(obj(:));
        S.feas = double(logical(is_feas(:)));
        S.viol = double(viol(:));
        S.vmax = double(vmax(:));
    catch
        % NaN rather than 0: an evaluator that would not answer is not the same
        % as a population with nothing to violate.
        S.obj = nan(N, 1); S.feas = nan(N, 1);
        S.viol = nan(N, 1); S.vmax = nan(N, 1);
    end
end

function [key, grp] = elite_key(f, S)
% Ranking behind the elite columns: plain fitness off CEC2020RW, feasibility
% first on it. grp is 0 for feasible and 1 for infeasible so that the pair sorts
% lexicographically -- feasible individuals then rank by RAW objective and
% infeasible ones by violation, the order the competition guidelines use. A
% non-finite key is pushed to +Inf so an individual the evaluator could not score
% never displaces one it could, and an unknown feasibility counts as infeasible.
    if S.is_rw
        grp = double(~(S.feas > 0));
        key = S.obj;
        key(grp > 0) = S.viol(grp > 0);
    else
        grp = zeros(numel(f), 1);
        key = double(f(:));
    end
    key = key(:);
    key(~isfinite(key)) = Inf;
end

function [ratio, shift] = elite_geometry(Z, zcen, dap, key, grp)
% Spread of the best ceil(0.2*N) individuals about THEIR OWN centroid, relative
% to the whole population's, plus the offset between the two centroids. This is
% what div_dap cannot separate: a wide population whose elite has collapsed is an
% exploitation core forming, a wide elite is genuine exploration.
%
% The cut is taken by VALUE, not by position, so everyone tied with the cut
% individual joins the elite. A converging population is nearly all ties, and a
% count-based cut would hand the set to whatever order sort happened to produce.
    ELITE_FRAC = 0.2;
    ratio = NaN; shift = NaN;
    N = size(Z, 1);
    if N < 2
        return;
    end
    ne = min(N, max(2, ceil(ELITE_FRAC * N)));
    [~, ord] = sortrows([grp(:), key(:)]);
    cut = ord(ne);
    is_e = grp < grp(cut) | (grp == grp(cut) & key <= key(cut));
    if nnz(is_e) < 2
        return;
    end
    E  = Z(is_e, :);
    ze = mean(E, 1);
    edap = mean(sqrt(sum((E - ze) .^ 2, 2)));
    if isfinite(dap) && dap > 0
        ratio = edap / dap;
    end
    shift = norm(ze - zcen);
end

function [on_bound, outside] = box_occupancy(Z, zub, ok)
% Share of COORDINATES sitting on a bound and share of INDIVIDUALS with at least
% one coordinate outside the box, in box units where the lower bound is 0 and the
% upper one is zub (1 on every ordinary axis). See metric_labels for what the two
% columns separate and how BOUND_TOL is set.
%
% on_bound tests the DISTANCE to a bound, not "at or past" it: a coordinate that
% has escaped the box is not sitting on the bound and belongs to outside alone.
% As a one-sided comparison the two columns would rise together, and bound_frac
% would read highest for exactly the algorithms that never clamp.
    BOUND_TOL = 1e-9;
    if ~ok
        on_bound = NaN; outside = NaN;   % no bounds pushed => no box to measure
        return;
    end
    on_bound = mean(abs(Z) <= BOUND_TOL | abs(zub - Z) <= BOUND_TOL, 'all');
    outside  = mean(any(Z < -BOUND_TOL | Z > zub + BOUND_TOL, 2));
end

function u = row_unique_fraction(X)
% Fraction of distinct ROWS: the copies an elitist accept-or-keep-parent step
% leaves in the population, which uniq_frac cannot separate from symmetric optima
% or plateaus. Compares exactly, so unlike div_nnd it has no noise floor. unique
% treats NaN as distinct from itself, so a row with a non-finite coordinate always
% counts as its own -- x_nonfinite_frac reports how many of those there are.
    N = size(X, 1);
    if N < 1
        u = 0;
        return;
    end
    u = size(unique(X, 'rows'), 1) / N;
end

function cv = nnd_variation(nn, dap)
% Spread of the nearest-neighbour distances over their mean: what separates an
% evenly spaced cloud from one that is half exact duplicates and half far-flung
% outliers, both of which carry the same div_nnd. Undefined on a collapsed
% population, where every distance sits on the Gram identity's floor; the test is
% the one metric_labels prescribes, div_nnd/div_dap below 1e-7.
    COLLAPSE = 1e-7;
    cv = NaN;
    if numel(nn) < 2
        return;
    end
    m = mean(nn);
    if ~isfinite(m) || m <= 0 || ~isfinite(dap) || dap <= 0 || m / dap < COLLAPSE
        return;
    end
    cv = std(nn) / m;
end

function v = finite_of(x)
    v = double(x(:));
    v = v(isfinite(v));
end

function [mpd, diam, nn] = pairwise_stats(X)
% Mean (div_mpd) and max (diameter) pairwise distance, plus the per-individual
% nearest-neighbour distance VECTOR that div_nnd, nnd_median and nnd_cv are read
% off. O(N^2*dim), but it runs only at the T sample points.
%
% The Gram identity is evaluated on the CENTRED cloud: on raw coordinates
% sum(x.^2) swamps ||xi-xj||^2 and the subtraction loses every significant digit,
% driving div_nnd to exactly 0 across the whole exploitation phase. The distance
% matrix is consumed one row block at a time and never held whole -- 3.7 GB at
% the largest population the pool reaches (olshade, N = 13689), times 24 workers.
% Neither changes a result: every pair is still visited exactly once.
    N = size(X, 1);
    if N < 2
        mpd = 0; diam = 0; nn = zeros(N, 1);
        return;
    end
    Xc = X - mean(X, 1);
    sq = sum(Xc .^ 2, 2);

    blk = max(1, min(N, floor(4e6 / N)));   % one block stays near 32 MB

    total = 0;                      % sum over ORDERED pairs, diagonal contributes 0
    diam  = 0;
    nn    = zeros(N, 1);
    for a = 1:blk:N
        b = min(a + blk - 1, N);
        D2 = sq(a:b) + sq.' - 2 * (Xc(a:b, :) * Xc.');
        D2(D2 < 0) = 0;             % FP rounding can make a squared distance negative
        Dm = sqrt(D2);
        % The Gram identity leaves ~eps*|x|^2 on the diagonal where a self-distance
        % is exactly zero, which moves div_mpd by 1e-9 at small N.
        dix = sub2ind(size(Dm), 1:(b - a + 1), a:b);
        Dm(dix) = 0;
        total = total + sum(Dm, 'all');
        diam  = max(diam, max(Dm, [], 'all'));
        Dm(dix) = Inf;              % exclude self before the nearest-neighbour search
        nn(a:b) = min(Dm, [], 2);
    end
    mpd = total / (N * (N - 1));    % ordered-pair sum over ordered-pair count
end

function hs = dim_entropy(X, dmin, dmax)
% Per-dimension Shannon entropy over k equal-width bins spanning [dmin, dmax],
% normalised to [0, 1]. The bins follow the population's own range, so it is a
% shape measure and never a spread measure.
    k = 10;
    [N, dim] = size(X);
    hs = zeros(1, dim);
    if N < 2
        return;
    end
    for j = 1:dim
        w = dmax(j) - dmin(j);
        if ~isfinite(w) || w <= 0
            continue;
        end
        b = min(floor((X(:, j) - dmin(j)) / w * k), k - 1) + 1;
        b = b(isfinite(b));
        if isempty(b)
            continue;
        end
        c = accumarray(b, 1, [k, 1]);
        p = c(c > 0) / numel(b);
        hs(j) = -sum(p .* log(p)) / log(k);
    end
end

function u = uniq_fraction(f)
% Fraction of distinct fitness values, all NaNs counted as one; low values flag
% phenotypic collapse.
    n = numel(f);
    if n < 1
        u = 0;
        return;
    end
    u = (numel(unique(f(~isnan(f)))) + double(any(isnan(f)))) / n;
end

function [e, pc1] = eff_dim_pca(X)
% Effective spread dimension: PCA participation ratio (sum(lam))^2/sum(lam.^2) of
% the covariance eigenvalues, 1 = spread along a single direction. Needs box
% units; in raw coordinates a uniformly filled 5-D CEC2020RW box scores 1.000.
%
% pc1 is the share of the variance on the leading component, from the same
% eigenvalues. Unlike e it does not move with the population size, so it reads
% anisotropy directly under LPSR. NaN where there is no variance to apportion.
    pc1 = NaN;
    if size(X, 1) < 2
        e = 0;
        return;
    end
    C = cov(X);
    if ~all(isfinite(C(:)))
        e = NaN;
        return;
    end
    lam = eig((C + C') / 2);
    lam(lam < 0) = 0;
    s = sum(lam);
    if s <= 0
        e = 0;
    else
        e = s ^ 2 / sum(lam .^ 2);
        pc1 = max(lam) / s;
    end
end

function m = mean_safe(f)
% NaN instead of erroring on an empty input.
    if isempty(f)
        m = NaN;
    else
        m = mean(f);
    end
end

function m = median_safe(f)
    if isempty(f)
        m = NaN;
    else
        m = median(f);
    end
end

function m = max_safe(f)
    if isempty(f)
        m = NaN;
    else
        m = max(f);
    end
end

function m = min_safe(f)
% NaN on an empty input -- which is what obj_best_feas gets when the population
% has no feasible individual at all, a state that lasts most of a hard
% CEC2020RW run and is not the same as an objective of 0.
    if isempty(f)
        m = NaN;
    else
        m = min(f);
    end
end

function s = std_safe(f)
% 0 for a single element, NaN when nothing finite is left to measure.
    if numel(f) > 1
        s = std(f);
    elseif isscalar(f)
        s = 0;
    else
        s = NaN;
    end
end

function v = iqr_manual(f)
% Q3-Q1 without the Statistics Toolbox. Uses the linear-interpolation (type 7)
% convention of R and numpy, which differs from MATLAB's quantile().
    n = numel(f);
    if n < 2
        v = NaN;
        return;
    end
    fs = sort(f(:));
    v = q_interp(fs, 0.75) - q_interp(fs, 0.25);
end

function q = q_interp(fs, p)
% p-th quantile of a sorted vector by linear interpolation between order statistics.
    n = numel(fs);
    pos = p * (n - 1) + 1;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        q = fs(lo);
    else
        q = fs(lo) + (pos - lo) * (fs(hi) - fs(lo));
    end
end

function [rp, rs] = fdc_pair(f, d, ib)
% Fitness-distance correlation against the distance to the best individual, as
% Pearson (rp) and Spearman (rs).
%
% The reference individual is EXCLUDED: it is argmin f at distance 0, so keeping
% it correlates it with itself -- +0.72 at N=4, +0.27 at N=30, +0.03 at N=400,
% which under LPSR alone would make fdc climb through the run. rs is reported
% alongside because CEC objectives span decades and Pearson is then decided by
% one or two outliers. Both are NaN when either series is constant.
    keep = true(numel(f), 1);
    if ib >= 1 && ib <= numel(f)
        keep(ib) = false;
    end
    keep = keep & isfinite(f(:)) & isfinite(d(:));
    fv = f(keep);
    dv = d(keep);
    if numel(fv) < 3
        rp = NaN; rs = NaN;
        return;
    end
    rp = pearson(fv, dv);
    rs = pearson(tied_rank(fv), tied_rank(dv));
end

function r = pearson(a, b)
% NaN when either series has zero variance.
%
% Each centred series is scaled by its largest magnitude first: the correlation
% is scale free, the denominator is not. A converged population put both series
% near 1e-80 and their product at 1e-319 -- subnormal, four digits left -- and
% fdc came back as 1.0000218, outside the [-1, 1] it is defined on.
    a = a - mean(a);
    b = b - mean(b);
    sa = max(abs(a));
    sb = max(abs(b));
    if sa > 0, a = a / sa; end
    if sb > 0, b = b / sb; end
    den = sqrt(sum(a .^ 2) * sum(b .^ 2));
    if ~isfinite(den) || den <= 0
        r = NaN;
    else
        r = sum(a .* b) / den;
    end
end

function r = tied_rank(x)
% Ranks with ties averaged (tiedrank without the Statistics Toolbox).
    x = x(:);
    n = numel(x);
    [xs, ord] = sort(x);
    r = zeros(n, 1);
    i = 1;
    while i <= n
        j = i;
        while j < n && xs(j + 1) == xs(i)
            j = j + 1;
        end
        r(ord(i:j)) = (i + j) / 2;
        i = j + 1;
    end
end

function [sk, ku] = skew_kurt(f)
% Skewness and excess kurtosis (normal = 0) from the population std, so no
% Statistics Toolbox call is needed. NaN on a constant or empty sample: 0 would
% read as "symmetric, normal-tailed" for a collapsed population.
    n = numel(f);
    if n < 2
        sk = NaN; ku = NaN; return;
    end
    mu = mean(f);
    s  = std(f, 1);
    if ~isfinite(s) || s <= 0
        sk = NaN; ku = NaN; return;
    end
    z = (f - mu) / s;
    sk = mean(z .^ 3);
    ku = mean(z .^ 4) - 3;
end

% ----------------------------------------------------------------------- %
% Incremental Adapted Maximum-Likelihood Gaussian Model Iterated Density-Estimation Evolutionary Algorithm (iAMaLGaM)
% Parameter-free variant with a full covariance matrix
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_base    = floor(10*sqrt(D))          % Base population size
%   m x n     = 2^(s/2) x (1+s/2)*n_base at even restart s, 1 x 2^(1+floor(s/2))*n_base at odd
%   tau       = 0.35                       % Truncation selection fraction
%   eta_Sigma = 1 - exp(-1.1*|S|^1.2/D^1.6)   % Covariance learning rate
%   eta_Shift = 1 - exp(-1.2*|S|^0.31/D^0.5)  % Learning rate of the mean shift
%   DEC = 0.9, INC = 1/DEC, theta_SDR = 1  % AVS multiplier steps and SDR trigger
%   alpha_AMS = 0.5*tau*n/(n-1), delta_AMS = 2  % Share of samples shifted, shift factor
%   NIS_max   = 25 + D                     % Stagnant generations before c_mult < 1
%
% Algorithm Concept:
%   - Each population keeps its best and samples n - 1 new points from a Gaussian
%     fitted to its best tau*n; draws outside the box are redrawn
%   - The covariance is learned incrementally, (1-eta)*Sigma + eta*Sigma_ML of the
%     selection; the mean is the selection's maximum-likelihood mean
%   - AVS: c_mult scales Sigma, times INC on an improvement whose SDR exceeds 1,
%     times DEC after NIS_max stagnant generations, when the mean jumps to the best
%   - SDR = max |L^-1 (x_imp - mu)|, how many standard deviations the average
%     improvement lies from the mean
%   - AMS moves a share of the samples by delta_AMS*c_mult*shift, the smoothed
%     generation-to-generation mean shift
%   - A population stops at c_mult < 1e-10; restarts alternate one larger
%     population with 2^k parallel ones, clustered from a uniform start
%
% Reference:
% Peter A. N. Bosman, Joern Grahl, Dirk Thierens,
% Benchmarking Parameter-Free AMaLGaM on Functions With and Without Noise,
% Evolutionary Computation 21(3) (2013) 445-469.
% https://doi.org/10.1162/EVCO_a_00094
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from iAMaLGaM-Full-Free.c (21 May 2012) in AMaLGaM.zip from Bosman's
% page homepages.cwi.nl/~bosman/source_code.php, the variant the paper calls
% iAMaLGaM-Full; MATLAB's generator replaces the release's, so runs match it in
% distribution only. Its command-line inputs: no value-to-reach, fitness-variance
% tolerance 0, no cap on parallel populations (the paper has none); the box is
% both its initialisation range and its hard bounds. Kept as released: ties sort
% later index first; a failed Cholesky falls back to the standard deviations; the
% first covariance is diagonal; AMS moves the first n_AMS samples. Changed: the
% last batch stops at maxFe; the best is kept per FE; NaN fitness ranks last;
% history records the union of the run's parallel populations.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = iamalgam(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    tau        = 0.35;
    mult_dec   = 0.9;
    mult_inc   = 1.0 / mult_dec;
    sdr_thresh = 1.0;
    delta_ams  = 2.0;
    nis_max    = 25 + dim;
    var_tol    = 0;                          % the release's command-line fitness-variance tolerance
    n_base     = floor(10.0 * dim ^ 0.5);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    best_fitness  = inf;
    best_solution = [];
    P = zeros(0, dim);   % the run's m populations stacked, population i in rows (i-1)*n+1 : i*n
    F = zeros(1, 0);

    n_starts = 0;
    while FE < maxFE
        % run(): alternately 2^(s/2) parallel populations and one population twice as large
        if mod(n_starts, 2) == 0
            n = (1 + n_starts / 2) * n_base;
            m = 2 ^ (n_starts / 2);
        else
            n = 2 ^ (1 + floor(n_starts / 2)) * n_base;
            m = 1;
        end
        n_starts = n_starts + 1;

        sel       = floor(tau * n);
        alpha_ams = 0.5 * tau * (n / (n - 1));
        n_ams     = floor(alpha_ams * (n - 1));
        eta_p     = 1.0 - exp(-1.2 * sel ^ 0.31 / dim ^ 0.50);
        eta_s     = 1.0 - exp(-1.1 * sel ^ 1.20 / dim ^ 1.60);

        n_init = min(m * n, maxFE - FE);
        P = lb + (ub - lb) .* rand(n_init, dim);
        [F, FE] = calculate_fitness(P', problem, FE);
        F = F(:)';
        for k = 1:n_init
            stamp(FE - n_init + k, P(k, :), F(k));
        end
        if n_init < m * n
            break;
        end
        if m > 1
            [P, F] = cluster_populations(P, F, m, n);
        end

        term    = false(1, m);
        mult    = ones(1, m);
        nis     = zeros(1, m);
        mu      = zeros(m, dim);
        mu_prev = zeros(m, dim);
        ams     = zeros(m, dim);
        aggC    = zeros(dim, dim, m);
        Lf      = zeros(dim, dim, m);
        Ldiag   = false(1, m);       % Cholesky failed and Lf holds the standard deviations only
        Ssel    = zeros(sel, dim, m);
        fsel    = zeros(sel, m);
        drawn   = zeros(1, m);
        oob     = zeros(1, m);
        rk      = zeros(1, m * n);
        for i = 1:m
            r = (i - 1) * n + (1:n);
            rk(r) = dense_ranks(F(r));
        end
        gen = 0;

        while FE < maxFE
            % cumsum adds left to right as the release does; sum does not, and the order decides this stop
            for i = find(~term)
                fi = F((i - 1) * n + (1:n));
                cs = cumsum(fi);
                avg = cs(end) / n;
                cs = cumsum((fi - avg) .^ 2);
                if cs(end) / n <= var_tol
                    term(i) = true;
                end
            end
            term(mult < 1e-10) = true;
            if all(term)
                break;
            end
            act = find(~term);

            for i = act
                r = (i - 1) * n + (1:n);
                idx = select_truncation(P(r, :), F(r), rk(r), sel);
                S = P(r(idx), :);
                Ssel(:, :, i) = S;
                fsel(:, i) = F(r(idx))';
                if gen > 0
                    mu_prev(i, :) = mu(i, :);
                end
                cs = cumsum(S, 1);
                mu(i, :) = cs(end, :) / sel;     % sequential sum: a collapsed selection keeps the release's ulp offset
                if mult(i) < 1
                    mu(i, :) = S(1, :);          % AVS below 1: focus on the best selected
                end
                Dv = S - mu(i, :);
                Cg = (Dv' * Dv) / sel;
                if gen == 0
                    aggC(:, :, i) = diag(diag(Cg));
                else
                    if gen == 1
                        ams(i, :) = mu(i, :) - mu_prev(i, :);
                    else
                        ams(i, :) = (1.0 - eta_p) * ams(i, :) + eta_p * (mu(i, :) - mu_prev(i, :));
                    end
                    aggC(:, :, i) = (1.0 - eta_s) * aggC(:, :, i) + eta_s * Cg;   % Eq. (6)
                end
            end

            for i = act
                r1 = (i - 1) * n + 1;
                P(r1, :) = Ssel(1, :, i);
                F(r1) = fsel(1, i);
            end

            for i = act
                if FE >= maxFE
                    break;
                end
                Cm = aggC(:, :, i) * mult(i);
                [L, p] = chol(Cm, 'lower');
                Ldiag(i) = p > 0;
                if Ldiag(i)
                    L = diag(sqrt(diag(Cm)));    % the release's choleskyDecomposition fallback
                end
                Lf(:, :, i) = L;
                [Xn, drawn(i), oob(i)] = sample_in_box(mu(i, :), L, n - 1, lb, ub);
                if gen > 0 && n_ams > 0
                    Xn(1:n_ams, :) = shift_ams(Xn(1:n_ams, :), delta_ams * mult(i) * ams(i, :), lb, ub);
                end

                ne = min(n - 1, maxFE - FE);
                [fn, FE] = calculate_fitness(Xn(1:ne, :)', problem, FE);
                fn = fn(:)';
                r0 = (i - 1) * n + 1;
                for k = 1:ne
                    P(r0 + k, :) = Xn(k, :);
                    F(r0 + k) = fn(k);
                    stamp(FE - ne + k, Xn(k, :), fn(k));
                end
            end
            if FE >= maxFE
                break;
            end

            for i = act
                r = (i - 1) * n + (1:n);
                rk(r) = dense_ranks(F(r));
            end

            for i = act
                r = (i - 1) * n + (1:n);
                if oob(i) / drawn(i) > 0.9
                    mult(i) = mult(i) * 0.5;
                end
                fs = fsel(:, i)';
                fi = F(r);
                ibs = first_min(fs);
                ibp = first_min(fi);
                better = fi < fs(ibs);
                sdr = 0;
                if any(better)
                    xavg = sum(P(r(better), :), 1) / nnz(better);
                    sdr = sd_ratio(Lf(:, :, i), Ldiag(i), xavg - mu(i, :));
                end
                if abs(fs(ibs) - fi(ibp)) == 0
                    if mult(i) <= 1
                        nis(i) = nis(i) + 1;
                    end
                    if mult(i) > 1 || nis(i) >= nis_max
                        mult(i) = mult(i) * mult_dec;
                    end
                    if nis(i) < nis_max && mult(i) < 1
                        mult(i) = 1;
                    end
                else
                    nis(i) = 0;
                    if mult(i) < 1
                        mult(i) = 1;
                    end
                    if sdr > sdr_thresh
                        mult(i) = mult(i) * mult_inc;
                    end
                end
            end
            gen = gen + 1;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Best-so-far at evaluation fe, then one history sample of the stacked populations (P, F)
    function stamp(fe, x, f)
        if isempty(best_solution)
            best_solution = x;
        end
        if f < best_fitness
            best_fitness  = f;
            best_solution = x;
        end
        curve(fe) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            fe, P, F, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

% The release's ranks: its merge sort puts the later index first on ties, rank + 1 per new value
function rk = dense_ranks(f)
    nf = numel(f);
    [~, r] = sort(f(end:-1:1));
    ord = nf + 1 - r;
    fs = f(ord);
    rk = zeros(1, nf);
    rk(ord) = cumsum([0, fs(2:end) ~= fs(1:end-1)]);
end

% makeSelectionsForOnePopulation: the best sel by rank, or a diverse subset when sel or more tie at rank 0
function idx = select_truncation(X, f, rk, sel)
    nf = numel(rk);
    [~, r] = sort(rk(end:-1:1));
    ord = nf + 1 - r;
    if rk(ord(sel)) == 0
        idx = select_diverse(X, f, rk, sel);
    else
        idx = ord(1:sel);
    end
end

% Farthest-point selection among the rank-0 solutions, removal by swapping in the last, as released
function idx = select_diverse(X, f, rk, sel)
    pre = find(rk == 0);
    [~, far] = max(f(pre));
    idx = zeros(1, sel);
    idx(1) = pre(far);
    pre(far) = pre(end);
    pre(end) = [];
    nn = sqrt(sum((X(pre, :) - X(idx(1), :)) .^ 2, 2));
    for s = 2:sel
        [~, far] = max(nn);
        idx(s) = pre(far);
        pre(far) = pre(end);
        nn(far) = nn(end);
        pre(end) = [];
        nn(end) = [];
        nn = min(nn, sqrt(sum((X(pre, :) - X(idx(s), :)) .^ 2, 2)));
    end
end

% initializePopulationsAndFitnessValues for m > 1: farthest-point leaders, then capacity-n nearest leader
function [P, F] = cluster_populations(P, F, m, n)
    [total, dim] = size(P);
    o = floor(dim * rand) + 1;
    c = P(:, o);
    lead = zeros(m, dim);
    lead(1, :) = P(find(c == min(c), 1, 'last'), :);
    dneg = -sqrt(sum((P - lead(1, :)) .^ 2, 2));
    for q = 2:m
        lead(q, :) = P(find(dneg == min(dneg), 1, 'last'), :);
        dneg = max(dneg, -sqrt(sum((P - lead(q, :)) .^ 2, 2)));
    end
    Dl = zeros(total, m);
    for q = 1:m
        Dl(:, q) = sqrt(sum((P - lead(q, :)) .^ 2, 2));
    end
    cnt = zeros(1, m);
    dest = zeros(total, 1);
    for k = 1:total
        d = Dl(k, :);
        d(cnt >= n) = inf;
        [~, q] = min(d);
        cnt(q) = cnt(q) + 1;
        dest(k) = (q - 1) * n + cnt(q);
    end
    P(dest, :) = P;
    F(dest) = F;
end

% generateNewSolution for k points: a Gaussian draw is redone while outside the box, uniform after 100
function [X, drawn, oob] = sample_in_box(mu, L, k, lb, ub)
    dim = numel(mu);
    X = zeros(k, dim);
    tries = zeros(k, 1);
    todo = (1:k)';
    while ~isempty(todo)
        g = todo(tries(todo) < 100);
        u = todo(tries(todo) >= 100);
        X(g, :) = mu + randn(numel(g), dim) * L';
        X(u, :) = lb + (ub - lb) .* rand(numel(u), dim);
        tries(todo) = tries(todo) + 1;
        bad = any(X(todo, :) < lb | X(todo, :) > ub | isnan(X(todo, :)), 2);
        todo = todo(bad);
    end
    drawn = sum(tries);
    oob = drawn - k;
end

% AMS with the release's halving of the shift, 1 down to 2^-34, until the point is inside the box
function X = shift_ams(X, step, lb, ub)
    todo = (1:size(X, 1))';
    shrink = 2;
    while ~isempty(todo) && shrink > 1e-10
        shrink = shrink * 0.5;
        Xa = X(todo, :) + shrink * step;
        ok = ~any(Xa < lb | Xa > ub | isnan(Xa), 2);
        X(todo(ok), :) = Xa(ok, :);
        todo = todo(~ok);
    end
end

% getStDevRatio; on the diagonal fallback LINPACK's DTRDI stops at the last zero, leaving earlier entries raw
function sdr = sd_ratio(L, is_diag, d)
    if is_diag
        s = diag(L)';
        k0 = find(s == 0, 1, 'last');
        if isempty(k0)
            k0 = 0;
        end
        inv_s = s;
        inv_s(k0 + 1:end) = 1 ./ s(k0 + 1:end);
        z = abs(inv_s .* d);
    else
        [z, ~] = linsolve(L, d', struct('LT', true));   % two outputs: no near-singular warning
        z = abs(z');
    end
    sdr = max([0, z(~isnan(z))]);
end

% The release's argmin: first index under strict <, so a NaN in front is never replaced
function b = first_min(v)
    if isnan(v(1))
        b = 1;
    else
        [~, b] = min(v);
    end
end

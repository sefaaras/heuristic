% ----------------------------------------------------------------------- %
% CMA-ES Super-fit Scheme for the Re-sampled Inheritance Search (CMAES-RIS)
% CEC 2013 competition -- 12th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4 + floor(3*ln(D)), mu = floor(lambda/2)  % CMA-ES defaults, log weights
%   sigma0 = 0.2                  % Initial CMA-ES step, absolute (not scaled to the box)
%   cma_share = 0.3               % Budget share of the super-fit CMA-ES phase
%   alpha_e = 0.5                 % Inheritance factor, CR = 0.5^(1/(alpha_e*D))
%   radius = 0.2                  % Initial short-distance step, fraction of each range
%   xi = 1e-7                     % Step size below which the local search stops
%
% Algorithm Concept:
%   - Super-fit phase: a CMA-ES started at one uniform point spends 30 % of the
%     budget; samples are wrapped toroidally into the box only for evaluation
%   - Re-sampled inheritance: the elite is recombined with a fresh uniform point by
%     exponential crossover, so a random segment of genes is re-sampled
%   - Each offspring is refined by 3SOME's short-distance search: per coordinate a
%     step of -SR, then +SR/2, judged against the value at the start of the sweep
%   - SR halves after a sweep without improvement, until max(SR) <= xi; then a new
%     offspring is built; the elite takes any better offspring or refinement
%
% Reference:
% Fabio Caraffini, Giovanni Iacca, Ferrante Neri, Lorenzo Picinali, Ernesto Mininno,
% A CMA-ES super-fit scheme for the re-sampled inheritance search,
% 2013 IEEE Congress on Evolutionary Computation, 2013, 1123-1130.
% https://doi.org/10.1109/CEC.2013.6557692
% Components:
% CMA-ES -- N. Hansen, A. Ostermeier, Evol. Comput. 9(2) (2001) 159-195, doi:10.1162/106365601750190398
% RIS -- F. Caraffini, F. Neri, B. N. Passow, G. Iacca, Soft Comput. 17(12) (2013) 2235-2256, doi:10.1007/s00500-013-1106-7
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' Java in the SOS platform (github.com/facaraff/SOS,
% src/algorithms/CMAES_RIS.java with the CMAEvolutionStrategy.java it embeds and
% MemesLibrary.ThreeSome_ShortDistance); the 2013 entry's own code is not public.
% Release quirks kept: the elite is re-evaluated when RIS starts; the crossover start
% is round((D-1)*rand), so the two end positions are half as likely; a coordinate move
% is judged against the sweep-start value and an exact tie resets the whole sweep;
% bounds use SOS's default toroidal correction (the code comment says saturate). The
% CMA-ES phase finishes its last generation, overrunning 30 % by < lambda evaluations.
% The eigen update keeps the release's lazy count rule but not its wall-clock trigger;
% NaN fitness is ranked as +Inf and a non-finite sample coordinate is redrawn uniformly.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cmaes_ris(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    sigma0    = 0.2;
    cma_share = 0.3;
    alpha_e   = 0.5;
    radius    = 0.2;
    xi        = 1e-7;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = lb + rand(1, D) .* span;

    % Hansen's Java defaults with stopMaxIter = stopMaxFunEvals = Long.MAX_VALUE
    lambda = floor(4 + 3 * log(D));
    mu     = floor(lambda / 2);
    w      = log(mu + 1) - log((1:mu)');
    w      = w / sum(w);
    mueff  = sum(w) ^ 2 / sum(w .^ 2);
    cs     = (mueff + 2) / (D + mueff + 3);
    damps  = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cs;
    cc     = 4 / (D + 4);
    mucov  = mueff;
    ccov   = 2 / (D + 1.41) ^ 2 / mucov + (1 - 1 / mucov) * min(1, (2 * mueff - 1) / (mueff + (D + 2) ^ 2));
    chiN   = sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ^ 2));
    eig_gap = 1 / ccov / D / 5;

    best  = lb + span .* rand(1, D);
    fBest = NaN;
    xmean = best';
    sigma = sigma0;
    pc = zeros(D, 1);
    ps = zeros(D, 1);
    C  = eye(D);
    B  = eye(D);
    dD = ones(D, 1);
    n_upd = 0;
    countiter = 0;
    f_sorted = [];

    % Java integer division 3*maxEvaluations/10
    budget = floor(round(cma_share * 10) * maxFE / 10);
    j = 0;
    while j < budget && FE < maxFE
        % samplePopulation: up to five eigen updates, each failure adds |min eig|*k to diag(C)
        k_rep = 1;
        for it = 1:5
            if n_upd > 0 && n_upd >= eig_gap
                [V, E] = eig((C + C') / 2);
                [e, ord] = sort(real(diag(E)));
                B = real(V(:, ord));
                if e(1) < 0
                    dD = e;
                    C = C + abs(e(1)) * k_rep * eye(D);
                    k_rep = k_rep * 1.5;
                    continue;
                end
                dD = sqrt(e);
                n_upd = 0;
            end
            if min(dD) > 0
                break;
            end
        end
        countiter = countiter + 1;

        % testAndCorrectNumerics: flat-fitness step increase, then rescale C and sigma
        if countiter > 1 && f_sorted(1) == f_sorted(min(lambda - 1, floor(lambda / 2) + 1))
            sigma = sigma * exp(0.2 + cs / damps);
        end
        fac = 1;
        if max(dD) < 1e-6
            fac = 1 / max(dD);
        elseif min(dD) > 1e4
            fac = 1 / min(dD);
        end
        if fac ~= 1
            sigma = sigma / fac;
            pc = pc * fac;
            dD = dD * fac;
            C = C * fac ^ 2;
        end

        ARX = xmean + sigma * (B * (dD .* randn(D, lambda)));
        n_ev = min(lambda, maxFE - FE);
        P = toroidal(ARX(:, 1:n_ev)', lb, ub);
        fv = evaluate(P);
        fitness = inf(lambda, 1);
        fitness(1:n_ev) = fv;
        for q = 1:n_ev
            if j == 0 || fv(q) < fBest
                fBest = fv(q);
                best = P(q, :);
            end
            j = j + 1;
            [population_history, fitness_history, history_index] = record_history( ...
                FE - n_ev + q, P, fv, population_history, fitness_history, history_index, maxFE);
        end

        % updateDistribution on the unrepaired samples arx
        [f_sorted, idx] = sort(fitness);
        xold = xmean;
        xmean = ARX(:, idx(1:mu)) * w;
        BDz = sqrt(mueff) * (xmean - xold) / sigma;
        zz = (B' * BDz) ./ dD;
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs)) * (B * zz);
        psn = norm(ps);
        hsig = psn / sqrt(1 - (1 - cs) ^ (2 * countiter)) / chiN < 1.4 + 2 / (D + 1);
        pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc)) * BDz;
        Y = (ARX(:, idx(1:mu)) - xold) / sigma;
        C = (1 - ccov) * C + ccov * (1 / mucov) * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) + ...
            ccov * (1 - 1 / mucov) * ((Y .* w') * Y');
        C = (C + C') / 2;
        n_upd = n_upd + 1;
        sigma = sigma * exp((psn / chiN - 1) * cs / damps);
    end

    CR = 0.5 ^ (1 / (D * alpha_e));
    x = best;
    first = true;
    while FE < maxFE
        if ~first
            x = crossover_exp(best, lb + span .* rand(1, D), CR);
        end
        first = false;
        fx = evaluate(x);
        record(x, fx);
        if fx < fBest
            fBest = fx;
            best = x;
        end
        [x, fx] = short_distance(x, fx);
        if fx < fBest
            fBest = fx;
            best = x;
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of X (in the box, within the budget); NaN is ranked as +Inf
    function f = evaluate(X)
        [fr, FE_new] = calculate_fitness(X', problem, FE);
        f = fr(:);
        f(isnan(f)) = inf;
        for q2 = 1:numel(f)
            if f(q2) < bsf
                bsf = f(q2);
                bsf_solution = X(q2, :);
            end
            curve(FE + q2) = bsf;
        end
        FE = FE_new;
    end

    % RIS phase population: the offspring being refined, at its current value
    function record(xr, fr)
        [population_history, fitness_history, history_index] = record_history( ...
            FE, xr, fr, population_history, fitness_history, history_index, maxFE);
    end

    % 3SOME short-distance search (precision-stopped variant), refines sol in place
    function [sol, fit] = short_distance(sol, fit)
        SR = span * radius;
        improve = true;
        while max(SR) > xi && FE < maxFE
            Xk = sol;
            Xk_orig = sol;
            f_orig = fit;
            if ~improve
                SR = SR / 2;
            end
            improve = false;
            kk = 1;
            while kk <= D && FE < maxFE
                Xk(kk) = Xk(kk) - SR(kk);
                Xk = toroidal(Xk, lb, ub);
                fXk = evaluate(Xk);
                if fXk < fit
                    fit = fXk;
                    sol = Xk;
                end
                record(sol, fit);
                if FE < maxFE
                    if fXk == f_orig
                        Xk = Xk_orig;
                    elseif fXk > f_orig
                        Xk(kk) = Xk_orig(kk) + 0.5 * SR(kk);
                        Xk = toroidal(Xk, lb, ub);
                        fXk = evaluate(Xk);
                        if fXk < fit
                            fit = fXk;
                            sol = Xk;
                        end
                        record(sol, fit);
                        if fXk >= f_orig
                            Xk(kk) = Xk_orig(kk);
                        else
                            improve = true;
                        end
                    else
                        improve = true;
                    end
                end
                kk = kk + 1;
            end
        end
    end
end

% SOS crossOverExp: start at round((n-1)*rand), copy y while rand <= CR, wrapping once
function xo = crossover_exp(x, y, CR)
    n = numel(x);
    xo = x;
    s = round((n - 1) * rand) + 1;
    xo(s) = y(s);
    idx = mod(s, n) + 1;
    while rand <= CR && idx ~= s
        xo(idx) = y(idx);
        idx = mod(idx, n) + 1;
    end
end

% SOS toro correction: out-of-range coordinates wrap around the box
function X = toroidal(X, lb, ub)
    U = (X - lb) ./ (ub - lb);
    bad = ~isfinite(U);
    U(bad) = rand(nnz(bad), 1);
    hi = U > 1;
    U(hi) = U(hi) - fix(U(hi));
    lo = U < 0;
    U(lo) = 1 - abs(U(lo) - fix(U(lo)));
    X = U .* (ub - lb) + lb;
end

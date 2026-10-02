% ----------------------------------------------------------------------- %
% Iterated CMA-ES with MTS local search (iCMAES-ILS)
% CEC 2013 competition -- 2nd by mean aggregated rank (1st by Friedman test)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   learn_budget = 0.15         % Share of the budget each component gets to prove itself
%   lambda0 = 4 + floor(9.687*ln N), mu = floor(lambda/1.614)  % Tuned CMA-ES sizes
%   lambda = floor(3.245*lambda), at most 200  % IPOP growth at every restart
%   sigma0 = 0.6825 * box width % Initial step size of every CMA-ES restart
%   TolFun, TolHistFun, TolX = 10^-9.023, 10^-10.82, 10^-16.26  % CMA-ES restart triggers
%   step0 = 0.6703 * box width  % Initial step of the local search, reset every cycle
%   bias = 0.0191               % Pull of the incumbent in a local search restart
%
% Algorithm Concept:
%   - IPOP-CMA-ES and the MTS-LS1 iterated local search compete from the same
%     random start, each for 15 % of the budget
%   - The local search takes the remaining 70 % only if it beat the CMA-ES
%     best; otherwise CMA-ES resumes from its own best point
%   - MTS-LS1 sweeps the coordinates one at a time, trying a step back and then
%     half a step forward, halving the step after a sweep that improves nothing
%   - A cycle of D sweeps that leaves the best unchanged restarts the search from
%     a random point pulled almost all the way to the incumbent
%
% Reference:
% Tianjun Liao, Thomas Stuetzle,
% Benchmark results for a simple hybrid algorithm on the CEC 2013 benchmark set
% for real-parameter optimization,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 1938-1944.
% https://doi.org/10.1109/CEC.2013.6557796
% Components: IPOP-CMA-ES (Auger and Hansen) and MTS-LS1 (Tseng and Chen).
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released C++ (icmaesils.cc) with all tuned settings of
% its README command line, since the code carries no defaults. The CMA-ES side is
% this repository's ipop_cmaes core (Hansen's MATLAB update, boundary handling and
% stopping tests) run with the release's lambda, mu, IPOP factor and tolerances.
% Kept from the release: the local search's convergence test compares a value with
% itself, so the second step collapse in a cycle ends that cycle; its best starts
% at the CMA-ES value, so it is deployed only if it beats CMA-ES strictly; it
% re-evaluates its current point at every coordinate. Changed: out-of-box local
% search points are clamped instead of scored with the release's growing penalty.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = icmaes_ils(problem)

    N     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;

    learn_budget = 0.15;
    cma.lambda0  = 4 + floor(9.687 * log(N));
    cma.mu_div   = 1.614;
    cma.ipop     = 3.245;
    cma.lam_max  = 200;
    cma.sigma    = 0.6825;
    cma.tol      = 10 .^ [-9.023, -10.82, -16.26];   % TolFun, TolHistFun, TolX
    step_rate    = 0.6703;
    bias         = 0.01910;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x0 = lb + rand(N, 1) .* (ub - lb);
    [f0, FE] = calculate_fitness(x0, problem, FE);
    bsf  = f0(1);
    bsfx = x0';
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            FE, bsfx, bsf, population_history, fitness_history, history_index, maxFE);
    end

    % Competition phase: both components start from x0
    [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, cma_best, cma_x] = ...
        run_cmaes(problem, FE, min(maxFE, FE + round(learn_budget * maxFE)), maxFE, curve, ...
                  population_history, fitness_history, history_index, bsf, bsfx, x0, f0(1), cma);

    improve = true;   % the release keeps this flag global, so it carries into the next call
    [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, ls_best, ls_x, improve] = ...
        run_mtsls1(problem, FE, min(maxFE, FE + round(learn_budget * maxFE)), maxFE, curve, ...
                   population_history, fitness_history, history_index, bsf, bsfx, x0', ...
                   cma_best, x0', improve, step_rate, bias);

    % Deployment phase: the winner spends what is left from its own best point
    if cma_best > ls_best
        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
            run_mtsls1(problem, FE, maxFE, maxFE, curve, population_history, fitness_history, ...
                       history_index, bsf, bsfx, ls_x, ls_best, ls_x, improve, step_rate, bias);
    else
        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
            run_cmaes(problem, FE, maxFE, maxFE, curve, population_history, fitness_history, ...
                      history_index, bsf, bsfx, cma_x(:), cma_best, cma);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% IPOP-CMA-ES: the first run starts from xstart, later ones from random points with a larger population
function [FE, curve, ph, fh, hidx, bsf, bsfx, phase_best, phase_x] = run_cmaes( ...
        problem, FE, stopFE, maxFE, curve, ph, fh, hidx, bsf, bsfx, xstart, fstart, cma)

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);
    sigma0 = cma.sigma * (ub - lb);
    phase_best = fstart;
    phase_x = xstart(:)';
    lambda = cma.lambda0;
    first = true;

    while FE < stopFE && stopFE - FE >= 4
        if first
            x0 = min(max(xstart(:), lb), ub);
            first = false;
        else
            x0 = lb + rand(N, 1) .* (ub - lb);
        end
        lam = max(4, min(lambda, stopFE - FE));
        [FE, curve, ph, fh, hidx, bsf, bsfx, run_best, run_x] = cmaesRun(problem, FE, stopFE, ...
            curve, ph, fh, hidx, bsf, bsfx, x0, sigma0, lam, cma);
        if run_best < phase_best
            phase_best = run_best;
            phase_x = run_x;
        end
        lambda = min(cma.lam_max, floor(cma.ipop * lambda));
    end
    curve(min(max(FE, 1), maxFE)) = bsf;
end

% MTS-LS1 inside an iterated local search, as in the release's ILSmtsls1
function [FE, curve, ph, fh, hidx, bsf, bsfx, ls_best, ls_x, improve] = run_mtsls1( ...
        problem, FE, stopFE, maxFE, curve, ph, fh, hidx, bsf, bsfx, xstart, ls_best, ls_x, ...
        improve, step_rate, bias)

    N  = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    span = ub - lb;
    s0 = step_rate * span;

    xk = min(max(xstart(:)', lb), ub);
    if FE < stopFE
        [f, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
        [ls_best, ls_x] = keep_best(f, xk, ls_best, ls_x);
    end
    s = s0;

    while FE < stopFE
        accept_before = ls_best;
        collapses = 0;
        for sweep = 1:N                                  % the release's maxiter = 1*dim
            if ~improve
                s = s / 2;
                if max(s) < 1e-20
                    collapses = collapses + 1;
                    s = (0.3 + 0.3 * rand) * span;
                end
            end
            improve = false;

            for i = 1:N
                if FE >= stopFE
                    break;
                end
                [before1, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
                [ls_best, ls_x] = keep_best(before1, xk, ls_best, ls_x);
                if FE >= stopFE
                    break;
                end

                xi = xk(i);
                xk(i) = max(min(xi - s(i), ub(i)), lb(i));
                [after1, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
                [ls_best, ls_x] = keep_best(after1, xk, ls_best, ls_x);

                if abs(after1 - before1) <= 1e-20
                    xk(i) = xi;
                elseif after1 > before1
                    xk(i) = max(min(xi + 0.5 * s(i), ub(i)), lb(i));   % half a step the other way
                    if FE >= stopFE
                        break;
                    end
                    [after2, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
                    [ls_best, ls_x] = keep_best(after2, xk, ls_best, ls_x);
                    if after2 >= before1
                        xk(i) = xi;
                    else
                        improve = true;
                    end
                else
                    improve = true;
                end
            end
            % The release's convergence test compares a value with itself: a second collapse ends the cycle
            if FE >= stopFE || collapses >= 2
                break;
            end
        end
        if FE >= stopFE
            break;
        end

        if accept_before - ls_best < 1e-20
            srand = lb + rand(1, N) .* span;
            xk = srand + ((1 - bias) * rand(1, N) + bias) .* (ls_x - srand);
        end
        s = s0;
    end
end

function [best, bestx] = keep_best(f, x, best, bestx)
    if best - f > 1e-30
        best = f;
        bestx = x;
    end
end

function [f, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(x, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx)
    [fv, FE] = calculate_fitness(x(:), problem, FE);
    f = fv(1);
    if f < bsf
        bsf = f;
        bsfx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hidx] = record_history(FE, x, f, ph, fh, hidx, maxFE);
    end
end

% Helper Functions

function [FE, curve, ph, fh, hidx, bsf, bsfx, run_best, run_x] = cmaesRun( ...
        problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx, ...
        xstart, insigma, lambda, cma)
% One (mu/mu_w, lambda)-CMA-ES run under Hansen's stopping criteria or the budget; returns its own best

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);

    run_best = inf;
    run_x    = xstart(:)';

    % Strategy parameters (cmaes.m defaults except the release's mu)
    mu      = floor(lambda / cma.mu_div);
    weights = log(max(mu, lambda / 2) + 0.5) - log(1:mu)';
    mueff   = sum(weights) ^ 2 / sum(weights .^ 2);
    weights = weights / sum(weights);

    cc     = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
    cs     = (mueff + 2) / (N + mueff + 3);
    ccov1  = 2 / ((N + 1.3) ^ 2 + mueff);
    ccovmu = min(1 - ccov1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ^ 2 + mueff));
    damps  = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    chiN   = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));

    % Dynamic state
    if isscalar(insigma)
        insigma = insigma * ones(N, 1);
    end
    sigma = max(insigma);
    diagD = insigma / sigma;
    diagC = diagD .^ 2;
    B     = eye(N);
    BD    = B .* diagD';
    C     = diag(diagC);
    pc    = zeros(N, 1);
    ps    = zeros(N, 1);

    xmean = min(max(xstart, lb), ub);
    xold  = xmean;

    maxdx = (ub - lb) / 2;
    if any(sigma * sqrt(diagC) > maxdx)
        sigma = min(maxdx ./ sqrt(diagC));
    end

    % Termination thresholds
    stopTolX       = cma.tol(3);
    stopTolUpX     = 1e3   * max(insigma);
    stopTolFun     = cma.tol(1);
    stopTolHistFun = cma.tol(2);
    stopMaxIter    = 1e3 * (N + 5) ^ 2 / sqrt(lambda);

    histLen        = 10 + ceil(3 * 10 * N / lambda);
    fitHist        = NaN(1, histLen);
    histBest       = [];
    histMedian     = [];
    arrEqualFunvals = zeros(1, 10 + N);

    % Adaptive boundary penalty state
    bndWeights    = zeros(N, 1);
    bndScale      = ones(N, 1);
    bndDfithist   = 1;
    bndValidfit   = 0;
    bndIniphase   = 1;

    % Evaluate the initial mean (cmaes.m EvalInitialX defaults to on)
    if FE < maxFE
        [f0, FE] = calculate_fitness(xmean, problem, FE);
        f0       = f0(1);
        fitHist(1) = f0;
        run_best = f0;
        run_x    = xmean';
        if f0 < bsf
            bsf  = f0;
            bsfx = xmean';
        end
        if FE >= 1 && FE <= maxFE
            curve(FE) = bsf;
            [ph, fh, hidx] = record_history(FE, xmean', f0, ph, fh, hidx, maxFE);
        end
    end

    countiter = 0;

    % Generation loop
    while FE < maxFE
        countiter = countiter + 1;

        nsample = min(lambda, maxFE - FE);
        arz = randn(N, nsample);
        arx = xmean(:, ones(1, nsample)) + sigma * (BD * arz);
        arxvalid = min(max(arx, lb), ub);

        [fraw, FE] = calculate_fitness(arxvalid, problem, FE);
        fraw = fraw(:)';

        % Best-so-far, curve and history
        popRows = arxvalid';
        for k = 1:nsample
            if fraw(k) < bsf
                bsf  = fraw(k);
                bsfx = popRows(k, :);
            end
            if fraw(k) < run_best
                run_best = fraw(k);
                run_x    = popRows(k, :);
            end
            ec = FE - nsample + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [ph, fh, hidx] = record_history(ec, popRows, fraw', ph, fh, hidx, maxFE);
            end
        end

        % A truncated final generation cannot drive the adaptation
        if nsample < lambda
            break;
        end

        % Adaptive boundary penalty
        q   = myprctile(fraw, [25 75]);
        val = (q(2) - q(1)) / N / mean(diagC) / sigma ^ 2;
        if ~isfinite(val)
            val = max(bndDfithist);
        elseif val == 0
            pos = bndDfithist(bndDfithist > 0);
            if isempty(pos)
                val = eps;          % see the header note
            else
                val = min(pos);
            end
        elseif bndValidfit == 0
            bndDfithist = [];
            bndValidfit = 1;
        end

        if numel(bndDfithist) < 20 + (3 * N) / lambda
            bndDfithist = [bndDfithist val];
        else
            bndDfithist = [bndDfithist(2:end) val];
        end

        tx = min(max(xmean, lb), ub);
        ti = (xmean ~= tx);

        if bndIniphase && any(ti)
            bndWeights(:) = 2.0002 * median(bndDfithist);
            dd            = diagC / mean(diagC);
            bndWeights    = bndWeights ./ dd;
            if bndValidfit && countiter > 2
                bndIniphase = 0;
            end
        end

        if any(ti)
            txd = xmean - tx;
            idx = ti & (abs(txd) > 3 * max(1, sqrt(N) / mueff) * sigma * sqrt(diagC));
            idx = idx & (sign(txd) == sign(xmean - xold));
            bndWeights(idx) = 1.2 ^ (min(1, mueff / 10 / N)) * bndWeights(idx);
        end

        fsel = fraw + (bndWeights ./ bndScale)' * (arxvalid - arx) .^ 2;

        % Sort and record the fitness histories
        fraw_s  = sort(fraw);
        [fsel_s, idxsel] = sort(fsel);

        fitHist = [fraw_s(1), fitHist(1:end-1)];
        if numel(histBest) < 120 + ceil(30 * N / lambda)
            histBest   = [fraw_s(1)      histBest];
            histMedian = [median(fraw_s) histMedian];
        else
            histBest   = [fraw_s(1)      histBest(1:end-1)];
            histMedian = [median(fraw_s) histMedian(1:end-1)];
        end

        % Recombination
        xold  = xmean;
        xmean = arx(:, idxsel(1:mu)) * weights;
        zmean = arz(:, idxsel(1:mu)) * weights;

        % Evolution paths
        ps   = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (B * zmean);
        hsig = norm(ps) / sqrt(1 - (1 - cs) ^ (2 * countiter)) / chiN < 1.4 + 2 / (N + 1);
        pc   = (1 - cc) * pc + hsig * (sqrt(cc * (2 - cc) * mueff) / sigma) * (xmean - xold);

        % Covariance matrix: rank-one plus rank-mu
        arpos = (arx(:, idxsel(1:mu)) - xold(:, ones(1, mu))) / sigma;
        C = (1 - ccov1 - ccovmu + (1 - hsig) * ccov1 * cc * (2 - cc)) * C ...
            + ccov1 * (pc * pc') ...
            + ccovmu * (arpos * ((weights * ones(1, N)) .* arpos'));
        diagC = diag(C);

        % Step size
        sigma = sigma * exp(min(1, (sqrt(sum(ps .^ 2)) / chiN - 1) * cs / damps));

        % Eigen decomposition, on Hansen's lazy schedule
        if (ccov1 + ccovmu) > 0 && mod(countiter, 1 / (ccov1 + ccovmu) / N / 10) < 1
            C = triu(C) + triu(C, 1)';
            [Btmp, Dtmp] = eig(C);
            dtmp = diag(Dtmp);
            if any(~isfinite(dtmp)) || any(~isfinite(Btmp(:)))
                break;                                   % conditioncov
            end
            if min(dtmp) <= 0 || max(dtmp) > 1e14 * min(dtmp)
                break;                                   % conditioncov
            end
            B     = Btmp;
            diagC = diag(C);
            diagD = sqrt(dtmp);
            BD    = B .* diagD';
        end

        % Numerical error management (StopOnWarnings is on by default)
        if any(sigma * sqrt(diagC) > maxdx)
            sigma = min(maxdx ./ sqrt(diagC));
        end
        if any(xmean == xmean + 0.2 * sigma * sqrt(diagC))
            break;                                       % noeffectcoord
        end
        iax = 1 + floor(mod(countiter, N));
        if all(xmean == xmean + 0.1 * sigma * BD(:, iax))
            break;                                       % noeffectaxis
        end
        keq = min(lambda, 1 + ceil(0.1 + lambda / 4));
        if fsel_s(1) == fsel_s(keq)
            arrEqualFunvals = [countiter arrEqualFunvals(1:end-1)];
            if arrEqualFunvals(end) > countiter - 3 * numel(arrEqualFunvals)
                break;                                   % equalfunvals
            end
        end
        if countiter > 2 && myrange([fitHist fsel_s(1)]) == 0
            break;                                       % equalfunvalhist
        end

        % Stop criteria
        if all(sigma * max(abs(pc), sqrt(diagC)) < stopTolX)
            break;                                       % tolx
        end
        if any(sigma * sqrt(diagC) > stopTolUpX)
            break;                                       % tolupx
        end
        if sigma * max(diagD) == 0
            break;
        end
        if countiter > 2 && myrange([fsel_s fitHist]) <= stopTolFun
            break;                                       % tolfun
        end
        if countiter >= histLen && myrange(fitHist) <= stopTolHistFun
            break;                                       % tolhistfun
        end
        l = floor(numel(histBest) / 3);
        if countiter > N * (5 + 100 / lambda) && numel(histBest) > 100 && ...
                median(histMedian(1:l)) >= median(histMedian(end-l:end)) && ...
                median(histBest(1:l))   >= median(histBest(end-l:end))
            break;                                       % stagnation
        end
        if countiter >= stopMaxIter
            break;                                       % maxiter
        end
    end
end

function r = myrange(x)
% Hansen's myrange; max/min skip the NaNs that pad an unfilled history.
    r = max(x) - min(x);
end

function res = myprctile(inar, perc)
% Hansen's myprctile: linear interpolation between order statistics at 100*((1:N)-0.5)/N, clamped
    N   = numel(inar);
    sar = sort(inar(:))';
    avail = 100 * ((1:N) - 0.5) / N;
    res = zeros(1, numel(perc));
    for k = 1:numel(perc)
        p = perc(k);
        if p <= avail(1)
            res(k) = sar(1);
        elseif p >= avail(end)
            res(k) = sar(end);
        else
            i = find(avail <= p, 1, 'last');
            res(k) = sar(i) + (sar(i+1) - sar(i)) * (p - avail(i)) / (avail(i+1) - avail(i));
        end
    end
end

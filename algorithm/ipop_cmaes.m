% ----------------------------------------------------------------------- %
% Restart CMA-ES with Increasing Population Size (IPOP-CMA-ES)
% Also known as G-CMA-ES; CEC 2005 real-parameter competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda_default = 4 + floor(3*log(N))   % Hansen's default population size
%   lambda         = 2^i_restart * lambda_default
%   sigma0         = (ub - lb) / 2         % half the box width, per the paper
%   x0 = U(lb, ub)                         % Fresh draw for every restart
%   mu = floor(lambda/2), superlinear recombination weights
%
% Algorithm Concept:
%   - CMA-ES is a local optimiser whose one real weakness is multimodality; a
%     LARGER POPULATION fixes that but wastes budget on unimodal problems
%   - IPOP declines to choose: run CMA-ES at the default population until an
%     internal termination criterion fires, then restart from a fresh random
%     point with the population DOUBLED, so early restarts are cheap local
%     searches and later ones increasingly global
%   - Doubling (not 1.5x or 3x) bounds the loss against an oracle that knew the
%     right population size at about a factor of two
%   - Restarts are driven entirely by CMA-ES's own stopping criteria, with no
%     extra machinery and nothing to tune
%   - Bound handling is the standard adaptive penalty: samples are projected in
%     for evaluation and the squared distance back is charged to SELECTION only
%
% Reference:
% Anne Auger, Nikolaus Hansen,
% A Restart CMA Evolution Strategy With Increasing Population Size,
% 2005 IEEE Congress on Evolutionary Computation (CEC), 2005, pp. 1769-1776.
% https://doi.org/10.1109/CEC.2005.1554902
% ----------------------------------------------------------------------- %
% Implementation Note:
% The CMA-ES core, boundary penalty and termination criteria are ported from
% Hansen's canonical cmaes.m 3.61.beta, the same code family the paper cites.
% The restart policy comes from the paper, which fixes what the code leaves to
% the caller: uniform restart point, sigma0 = (B-A)/2, population doubling, and
% TolX = 1e-12*sigma0 rather than Hansen's 1e-11*max(insigma).
% Eleven termination criteria are ported; everything cmaes.m has off by default
% (active CMA, diagonal-only phase, noise handling, logging) is left out.
% Two adaptations shared with bipop_cmaes.m: lambda is capped at the evaluations
% left, and the penalty's `val == 0` branch falls back on eps when min is empty.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ipop_cmaes(problem)

    N     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = (lb + ub)' / 2;

    lambda_def = max(4, 4 + floor(3 * log(N)));
    sigma_def  = (ub - lb) / 2;

    i_restart = 0;

    % Restart loop: one independent CMA-ES per pass, lambda doubling each time
    while maxFE - FE >= 4
        lambda = floor(lambda_def * 2 ^ i_restart);
        lambda = max(4, min(lambda, maxFE - FE));     % never overshoot the budget

        xstart = lb + rand(N, 1) .* (ub - lb);

        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
            cmaesRun(problem, FE, maxFE, curve, population_history, fitness_history, ...
                     history_index, bsf, bsfx, ...
                     xstart, sigma_def, lambda);

        i_restart = i_restart + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [FE, curve, ph, fh, hidx, bsf, bsfx, used] = cmaesRun( ...
        problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx, ...
        xstart, insigma, lambda)
% One (mu/mu_w, lambda)-CMA-ES run under Hansen's stopping criteria or the budget; returns FEs used

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);

    used = 0;

    % Strategy parameters (cmaes.m defaults)
    mu      = floor(lambda / 2);
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
    stopTolX       = 1e-12 * max(insigma);      % IPOP paper value
    stopTolUpX     = 1e3   * max(insigma);
    stopTolFun     = 1e-12;
    stopTolHistFun = 1e-13;
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
        used     = used + 1;
        f0       = f0(1);
        fitHist(1) = f0;
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
        used = used + nsample;

        % Best-so-far, curve and history
        popRows = arxvalid';
        for k = 1:nsample
            if fraw(k) < bsf
                bsf  = fraw(k);
                bsfx = popRows(k, :);
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

% ----------------------------------------------------------------------- %
% NBIPOP Restart Active CMA-ES (NBIPOP-aCMA-ES)
% CEC 2013 competition -- 1st by mean aggregated rank (2nd by Friedman test)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda_default = 4 + floor(3*log(N))
%   sigma_default  = (ub - lb) / 2
%   large regime:  lambda = 2^i * lambda_default, sigma0 = 1.6^-i * sigma_default
%   small regime:  lambda = lambda_default,       sigma0 = 0.01^U[0,1] * sigma_default
%   budget ratio   = 0.5 or 2.0     % Which regime may spend more, set by results
%   neg.ccov = (1 - ccovmu) * 0.25 * mueff / ((N+2)^1.5 + 2*mueff)
%
% Algorithm Concept:
%   - Like BIPOP, two restart regimes share the budget, but here the share is
%     not equal: whichever regime holds the better result may spend twice what
%     the other has spent, so the run commits to what is working
%   - A large restart doubles the population AND shrinks its initial step size
%     by 1.6 each time, instead of always starting from the default width
%   - A small restart keeps the default population and only randomises the step
%     size, and is capped at half the large regime's spending in iterations
%   - The covariance update is active: the worst mu samples of a generation are
%     subtracted, with a learning rate that is clipped whenever the subtraction
%     would take more than a third of the variance out of any direction
%
% Reference:
% Ilya Loshchilov,
% CMA-ES with restarts for solving CEC 2013 benchmark problems,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 369-376.
% https://doi.org/10.1109/CEC.2013.6557593
% Components: the CMA-ES core, boundary penalty and termination criteria are
% Hansen's cmaes.m, shared with this repository's ipop_cmaes and bipop_cmaes.
% ----------------------------------------------------------------------- %
% Implementation Note:
% The core is this repository's bipop_cmaes verbatim; what differs is the
% restart policy and the active update, both taken from the author's released
% package (xacmes.m and cmaes3.55.m of NBIPOPaCMA.zip, CEC2013 archive) with its
% surrogate machinery left out, since NBIPOP-aCMA-ES is the entry that runs
% without it. The released code holds the budget ratio at 1.0 until a first
% result exists on each side, then sets it to 0.5 or 2.0, and that is what is
% coded here. The active update is flgActiveCMA = 1: the weighted negative
% update with the residual-variance check that clips neg.ccov.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nbipop_cmaes(problem)

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

    budget_large = 0;
    budget_small = 0;
    i_large      = 0;      % lambda_large = 2^i_large * lambda_def
    best_large   = inf;    % best value each regime has reached
    best_small   = inf;
    ratio        = 1.0;    % how much the small regime may spend, relative to the large

    i_run     = 0;
    first_run = true;

    % Restart loop: the small regime runs while its spending is under the ratio
    while maxFE - FE >= 4
        i_run = i_run + 1;
        maxiter = inf;
        if first_run
            % First run uses the default population and sigma, booked against the large regime
            lambda    = lambda_def;
            sigma0    = sigma_def;
            is_large  = true;
            first_run = false;
        elseif i_run > 2 && budget_small < ratio * budget_large
            % Small regime: default population, only the step size is randomised
            lambda   = lambda_def;
            sigma0   = sigma_def * 0.01 ^ rand;
            is_large = false;
            maxiter  = 0.5 * budget_large / lambda;
        else
            i_large  = i_large + 1;
            lambda   = lambda_def * 2 ^ i_large;
            % A large restart also starts from a narrower step size than the last
            sigma0   = sigma_def * 1.6 ^ (-i_large);
            is_large = true;
        end

        lambda = max(4, min(lambda, maxFE - FE));     % never overshoot the budget
        xstart = lb + rand(N, 1) .* (ub - lb);

        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, used, run_best] = ...
            cmaesRun(problem, FE, maxFE, curve, population_history, fitness_history, ...
                     history_index, bsf, bsfx, ...
                     xstart, sigma0, lambda, maxiter);

        if is_large
            budget_large = budget_large + used;
            best_large   = min(best_large, run_best);
        else
            budget_small = budget_small + used;
            best_small   = min(best_small, run_best);
        end

        % The regime holding the better result is allowed to spend more
        if isfinite(best_large) && isfinite(best_small)
            if best_large < best_small
                ratio = 0.5;
            else
                ratio = 2.0;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [FE, curve, ph, fh, hidx, bsf, bsfx, used, run_best] = cmaesRun( ...
        problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx, ...
        xstart, insigma, lambda, maxiter)
% One (mu/mu_w, lambda)-CMA-ES run under Hansen's stopping criteria or the budget; returns FEs used

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);

    used = 0;
    run_best = inf;      % best this run reached, which decides the budget ratio

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
    stopMaxIter    = min(1e3 * (N + 5) ^ 2 / sqrt(lambda), maxiter);

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
        run_best = min(run_best, f0);
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
        run_best = min(run_best, min(fraw));
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

        % Covariance matrix: rank-one, rank-mu and the active negative update
        arpos = (arx(:, idxsel(1:mu)) - xold(:, ones(1, mu))) / sigma;
        neg_ccov = (1 - ccovmu) * 0.25 * mueff / ((N + 2) ^ 1.5 + 2 * mueff);
        neg_alphaold = 0.5;
        neg_minresidual = 0.66;
        arzneg = arz(:, idxsel(lambda:-1:lambda - mu + 1));
        [arnorms, idxnorms] = sort(sqrt(sum(arzneg .^ 2, 1)));
        [~, idxnorms] = sort(idxnorms);                 % inverse permutation
        arnormfacs = arnorms(end:-1:1) ./ arnorms;
        arnorms = arnorms(end:-1:1);
        arzneg = arzneg .* repmat(arnormfacs(idxnorms), N, 1);
        Ccheck = arzneg * diag(weights) * arzneg';
        artmp = BD * arzneg;
        Cneg = artmp * diag(weights) * artmp';
        neg_ccovfinal = neg_ccov;
        % Clipped whenever the subtraction would take too much variance out
        if 1 - neg_ccov * (arnorms(idxnorms) .^ 2) * weights < neg_minresidual
            maxeig = max(eig(Ccheck));
            if isfinite(maxeig) && maxeig > 0
                neg_ccovfinal = min(neg_ccov, (1 - ccovmu) * (1 - neg_minresidual) / maxeig);
            end
        end
        C = (1 - ccov1 - ccovmu + neg_alphaold * neg_ccovfinal + (1 - hsig) * ccov1 * cc * (2 - cc)) * C ...
            + ccov1 * (pc * pc') ...
            + (ccovmu + (1 - neg_alphaold) * neg_ccovfinal) * (arpos * ((weights * ones(1, N)) .* arpos')) ...
            - neg_ccovfinal * Cneg;
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

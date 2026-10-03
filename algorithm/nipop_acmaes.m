% ----------------------------------------------------------------------- %
% NIPOP Restart Active CMA-ES (NIPOP-aCMA-ES)
% CEC 2013 competition -- 5th place (mean aggregated rank)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 2^(k-1) * (4 + floor(3*log(N)))   % k-th run of a ten-run cycle
%   sigma0 = 1.6^-(k-1) * 0.6*(ub - lb)        % Capped at (ub-lb)/2 after the first update
%   x0     = U(lb, ub)                         % Fresh draw for every run
%   cc = 4/(N+4), TolX = 1e-11 * 0.6*max(ub-lb), TolUpX = 1e3 * 0.6*max(ub-lb)
%   MaxIter = 1e3*(N+5)^2/sqrt(lambda), TolHistFun = 1e-13
%   neg.ccov = (1 - ccovmu) * 0.25 * mueff / ((N+2)^1.5 + 2*mueff)
%
% Algorithm Concept:
%   - IPOP restarts: whenever one of CMA-ES's own stopping criteria fires, the
%     run restarts from a uniform point with the population doubled
%   - New restart rule: each larger restart also starts from a step size 1.6
%     times smaller, so the big populations search progressively more locally
%   - Every run ends by evaluating its final mean; after ten runs the cycle
%     starts over at the default population and step size
%   - Active covariance update: the worst mu samples are subtracted, with the
%     rate clipped when that would remove over a third of any direction's variance
%   - Bounds: samples are projected into the box and an adaptive penalty on the
%     projection distance enters selection only
%
% Reference:
% Ilya Loshchilov,
% CMA-ES with restarts for solving CEC 2013 benchmark problems,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 369-376.
% https://doi.org/10.1109/CEC.2013.6557593
% ----------------------------------------------------------------------- %
% Implementation Note:
% The release is NBIPOPaCMA.zip (CEC2013 archive) run with settings.BIPOP = 0,
% newRestartRules = 1, CMAactive = 1, withSurr = 0 (Preprocc.m's NIPOPaCMA).
% The CMA-ES core is the package's cmaes3.55 split, shared with nbipop_cmaes;
% BIPOP = 0 skips xacmes.m's option overrides, so cmaes3.55 defaults hold for
% cc, ccov, TolX, TolUpX, TolHistFun and MaxIter, insigma being 0.6*(ub-lb).
% Adapter.m calls xacmes afresh after its Restarts = 9, so the schedule restarts
% every ten runs; the RNG draws follow the package's order. Harness adaptations:
% the package evaluates CEC2013 unbounded and stops at an error of 1e-9; here
% Hansen's bound penalty keeps points in the box and the whole budget is spent.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nipop_acmaes(problem)

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

    lambda_def = 4 + floor(3 * log(N));
    sigma_def  = 0.6 * (ub - lb);         % xacmes.m: sigma0 = 200*0.6 on [-100, 100]

    % cmaes3.55 defaults; xacmes.m overrides them only when BIPOP = 1
    opt.bipop      = false;
    opt.tolx       = 1e-11 * max(sigma_def);
    opt.tolupx     = 1e3 * max(sigma_def);
    opt.tolhistfun = 1e-13;

    % Each pass is one xacmes.m call from Adapter.m: ten IPOP runs (Restarts = 9)
    while FE < maxFE
        rand(20, 1);                      % xacmes.m calls cmaes_initialize twice, each drawing rand(10,1)
        irun = 0;
        while irun <= 9 && FE < maxFE
            irun   = irun + 1;
            lambda = floor(lambda_def * 2 ^ (irun - 1));
            xstart = lb + rand(N, 1) .* (ub - lb);
            % New restart rule: each larger run also starts 1.6 times narrower
            sigma0 = sigma_def * 1.6 ^ (-(irun - 1));
            opt.histLambda = lambda;
            opt.maxiter    = 1e3 * (N + 5) ^ 2 / sqrt(lambda);

            [FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
                cmaesRun(problem, FE, maxFE, curve, population_history, fitness_history, ...
                         history_index, bsf, bsfx, xstart, sigma0, lambda, opt);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [FE, curve, ph, fh, hidx, bsf, bsfx, used, run_best] = cmaesRun( ...
        problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx, xstart, insigma, lambda, opt)
% One active CMA-ES run of the package's cmaes3.55 split; returns FEs used and the run's best

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);

    used     = 0;
    run_best = inf;      % xacmes.m's algo.Fmin for this run

    % Strategy parameters (cmaes3.55, superlinear weights)
    mu      = floor(lambda / 2);
    weights = log(mu + 0.5) - log(1:mu)';
    mueff   = sum(weights) ^ 2 / sum(weights .^ 2);
    weights = weights / sum(weights);

    cs = (mueff + 2) / (N + mueff + 3);
    if opt.bipop
        % xacmes.m overrides these only when BIPOP = 1
        cc   = ((N + 4 + 2 * mueff / N) / (4 + mueff / N)) ^ -1;
        cfac = min(2, lambda / 3);
    else
        cc   = 4 / (N + 4);
        cfac = 2;
    end
    ccov1    = cfac / ((N + 1.3) ^ 2 + mueff);
    ccovmu   = min(1 - ccov1, cfac * (mueff - 2 + 1 / mueff) / ((N + 2) ^ 2 + mueff));
    damps    = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    chiN     = N ^ 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));
    neg_ccov = (1 - ccovmu) * 0.25 * mueff / ((N + 2) ^ 1.5 + 2 * mueff);
    neg_alphaold    = 0.5;
    neg_minresidual = 0.66;

    % Dynamic state; xacmes.m sets sigma after cmaes_initializeRun, so there is no initial clip
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

    stopTolFun  = 1e-12;
    stopMaxIter = opt.maxiter;

    % cmaes_initializeRun sizes the history with its own lambda, before xacmes.m overrides it
    fitHist         = NaN(1, 10 + ceil(3 * 10 * N / opt.histLambda));
    histBest        = [];
    histMedian      = [];
    arrEqualFunvals = zeros(1, 10 + N);

    % Adaptive boundary penalty state
    bndWeights  = zeros(N, 1);
    bndScale    = ones(N, 1);
    bndDfithist = 1;
    bndValidfit = 0;
    bndIniphase = 1;

    % EvalInitialX
    if FE >= maxFE
        return;
    end
    [f0, FE] = calculate_fitness(xmean, problem, FE);
    f0 = f0(1);
    used = used + 1;
    fitHist(1) = f0;
    run_best = min(run_best, f0);
    if f0 < bsf
        bsf  = f0;
        bsfx = xmean';
    end
    curve(FE) = bsf;
    [ph, fh, hidx] = record_history(FE, xmean', f0, ph, fh, hidx, maxFE);

    countiter = 0;
    stopped   = false;

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
                val = eps;          % the package's min over an empty set errors out here
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
        if numel(histBest) < 120 + ceil(30 * N / lambda) || ...
                (mod(countiter, 5) == 0 && numel(histBest) < 2e4)
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
            neg_ccovfinal = min(neg_ccov, (1 - ccovmu) * (1 - neg_minresidual) / maxeig);
        end
        C = (1 - ccov1 - ccovmu + neg_alphaold * neg_ccovfinal + (1 - hsig) * ccov1 * cc * (2 - cc)) * C ...
            + ccov1 * pc * pc' ...
            + (ccovmu + (1 - neg_alphaold) * neg_ccovfinal) * arpos * (repmat(weights, 1, N) .* arpos') ...
            - neg_ccovfinal * Cneg;
        diagC = diag(C);

        % Step size
        sigma = sigma * exp(min(1, (sqrt(sum(ps .^ 2)) / chiN - 1) * cs / damps));

        % Eigen decomposition, on the lazy schedule that counts the negative rate too
        if mod(countiter, 1 / (ccov1 + ccovmu + neg_ccov) / N / 10) < 1
            C = triu(C) + triu(C, 1)';
            [Btmp, Dtmp] = eig(C);
            dtmp = diag(Dtmp);
            if any(~isfinite(dtmp)) || any(~isfinite(Btmp(:)))
                stopped = true;                          % the package raises an error here
                break;
            end
            if min(dtmp) <= 0 || max(dtmp) > 1e14 * min(dtmp)
                stopped = true;                          % warnconditioncov
                break;
            end
            B     = Btmp;
            diagC = diag(C);
            diagD = sqrt(dtmp);
            BD    = B .* diagD';
        end

        % Rescale a sigma that dwarfs the axes
        if sigma > 1e10 * max(diagD)
            fac   = sigma / max(diagD);
            sigma = sigma / fac;
            pc    = fac * pc;
            diagD = fac * diagD;
            C     = fac ^ 2 * C;
            BD    = B .* diagD';
            diagC = fac ^ 2 * diagC;
        end

        % Numerical error management (StopOnWarnings is on by default)
        if any(sigma * sqrt(diagC) > maxdx)
            sigma = min(maxdx ./ sqrt(diagC));
        end
        if any(xmean == xmean + 0.2 * sigma * sqrt(diagC))
            stopped = true;                              % warnnoeffectcoord
            break;
        end
        if all(xmean == xmean + 0.1 * sigma * BD(:, 1 + floor(mod(countiter, N))))
            stopped = true;                              % warnnoeffectaxis
            break;
        end
        if fsel_s(1) == fsel_s(min(lambda, 1 + ceil(0.1 + lambda / 4)))
            arrEqualFunvals = [countiter arrEqualFunvals(1:end-1)];
            if arrEqualFunvals(end) > countiter - 3 * numel(arrEqualFunvals)
                stopped = true;                          % equalfunvals
                break;
            end
        end
        if countiter > 2 && myrange([fitHist fsel_s(1)]) == 0
            stopped = true;                              % warnequalfunvalhist
            break;
        end

        % Stop criteria
        l = floor(numel(histBest) / 3);
        if all(sigma * max(abs(pc), sqrt(diagC)) < opt.tolx) ...
                || any(sigma * sqrt(diagC) > opt.tolupx) ...
                || sigma * max(diagD) == 0 ...
                || (countiter > 2 && myrange([fsel_s fitHist]) <= stopTolFun) ...
                || (countiter >= numel(fitHist) && myrange(fitHist) <= opt.tolhistfun) ...
                || (countiter > N * (5 + 100 / lambda) && numel(histBest) > 100 && ...
                    median(histMedian(1:l)) >= median(histMedian(end-l:end)) && ...
                    median(histBest(1:l)) >= median(histBest(end-l:end))) ...
                || countiter >= stopMaxIter
            stopped = true;
            break;
        end
    end

    % cmaes_finalize evaluates the final mean of every run that stopped on a criterion
    if stopped && FE < maxFE
        xm = min(max(xmean, lb), ub);
        [fm, FE] = calculate_fitness(xm, problem, FE);
        fm = fm(1);
        used = used + 1;
        run_best = min(run_best, fm);
        if fm < bsf
            bsf  = fm;
            bsfx = xm';
        end
        curve(FE) = bsf;
        [ph, fh, hidx] = record_history(FE, xm', fm, ph, fh, hidx, maxFE);
    end
end

function r = myrange(x)
% cmaes3.55's myrange; max/min skip the NaNs that pad an unfilled history.
    r = max(x) - min(x);
end

function res = myprctile(inar, perc)
% cmaes3.55's myprctile: linear interpolation between order statistics at 100*((1:N)-0.5)/N, clamped
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
            i = find(p > avail, 1, 'last');
            res(k) = sar(i) + (sar(i+1) - sar(i)) * (p - avail(i)) / (avail(i+1) - avail(i));
        end
    end
end

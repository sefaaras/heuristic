% ----------------------------------------------------------------------- %
% Iterated CMA-ES with MTS local search (iCMAES-ILS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   learn_budget = 0.15         % Share of the budget each component gets to prove itself
%   sigma0 = 0.6825 * box width % Initial step size of every CMA-ES restart
%   step0 = 0.6703 * box width  % Initial step of the local search
%   step reset = U[0.3, 0.6]    % Step redrawn when it collapses below 1e-20
%   bias = 0.0191               % Pull of the incumbent in a local search restart
%   lambda = 4 + floor(3*log(N))
%
% Algorithm Concept:
%   - Two components that do not interact during a run: restart CMA-ES and the
%     MTS-LS1 line search, each given the other's best point to start from
%   - Each gets 15 % of the budget to prove itself; the one that comes back with
%     the better value then spends the remaining 70 % alone
%   - MTS-LS1 sweeps the coordinates one at a time, trying a step back and then
%     half a step forward, halving the step after a sweep that improves nothing
%   - When the step collapses the search restarts from a random point pulled
%     almost all the way towards the incumbent, which is the ILS part
%
% Reference:
% Tianjun Liao, Thomas Stuetzle,
% Benchmark results for a simple hybrid algorithm on the CEC 2013 benchmark set
% for real-parameter optimization,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 1938-1944.
% https://doi.org/10.1109/CEC.2013.6557796
% Components: restart CMA-ES (Hansen) and MTS-LS1 (Tseng and Chen).
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released C++ (icmaesils.cc), with the tuned settings
% their README gives on its example command line -- learn_perbudget 0.15,
% mtsls1_initstep_rate 0.6703, mtsls1_iterbias_choice 0.01910, ttunec 0.6825 --
% since the code itself carries no defaults. The CMA-ES side is this
% repository's ipop_cmaes core rather than their bundled cmaes.c, so its
% termination criteria are Hansen's MATLAB ones; both stop on the same budget.
% The local search re-evaluates its current point at every coordinate, as the
% release does, which costs an evaluation per coordinate but is what its budget
% accounting assumes.
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
    sigma_rate   = 0.6825;
    step_rate    = 0.6703;
    bias         = 0.01910;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = (lb + ub)' / 2;

    x0 = lb + rand(N, 1) .* (ub - lb);
    [f0, FE] = calculate_fitness(x0, problem, FE);
    bsf  = f0(1);
    bsfx = x0';
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            FE, bsfx, bsf, population_history, fitness_history, history_index, maxFE);
    end

    % Learning phase: each component gets the same share, starting from the other's best
    [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, cma_best] = ...
        run_cmaes(problem, FE, min(maxFE, FE + round(learn_budget * maxFE)), maxFE, curve, ...
                  population_history, fitness_history, history_index, bsf, bsfx, bsfx', sigma_rate);

    [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, ls_best] = ...
        run_mtsls1(problem, FE, min(maxFE, FE + round(learn_budget * maxFE)), maxFE, curve, ...
                   population_history, fitness_history, history_index, bsf, bsfx, bsfx', ...
                   step_rate, bias);

    % Deployment phase: the winner spends what is left
    if cma_best < ls_best
        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, ~] = ...
            run_cmaes(problem, FE, maxFE, maxFE, curve, population_history, fitness_history, ...
                      history_index, bsf, bsfx, bsfx', sigma_rate);
    else
        [FE, curve, population_history, fitness_history, history_index, bsf, bsfx, ~] = ...
            run_mtsls1(problem, FE, maxFE, maxFE, curve, population_history, fitness_history, ...
                       history_index, bsf, bsfx, bsfx', step_rate, bias);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Restart CMA-ES: the first run starts from the incumbent, later ones from random points
function [FE, curve, ph, fh, hidx, bsf, bsfx, phase_best] = run_cmaes( ...
        problem, FE, stopFE, maxFE, curve, ph, fh, hidx, bsf, bsfx, xstart, sigma_rate)

    N  = problem.dimension;
    lb = problem.lb(:);
    ub = problem.ub(:);
    lambda_def = max(4, 4 + floor(3 * log(N)));
    sigma0 = sigma_rate * (ub - lb);
    phase_best = inf;
    first = true;

    while FE < stopFE && stopFE - FE >= 4
        if first
            x0 = min(max(xstart(:), lb), ub);
            first = false;
        else
            x0 = lb + rand(N, 1) .* (ub - lb);
        end
        lambda = max(4, min(lambda_def, stopFE - FE));
        [FE, curve, ph, fh, hidx, bsf, bsfx, ~] = cmaesRun(problem, FE, stopFE, curve, ph, fh, ...
            hidx, bsf, bsfx, x0, sigma0, lambda);
        phase_best = min(phase_best, bsf);
    end
    curve(min(max(FE, 1), maxFE):min(max(FE, 1), maxFE)) = bsf;
end

% MTS-LS1 with an iterated restart, Eq. (1)-(3) of the reference
function [FE, curve, ph, fh, hidx, bsf, bsfx, phase_best] = run_mtsls1( ...
        problem, FE, stopFE, maxFE, curve, ph, fh, hidx, bsf, bsfx, xstart, step_rate, bias)

    N  = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    span = ub - lb;

    xk = min(max(xstart(:)', lb), ub);
    best_x = xk;
    phase_best = inf;
    s = step_rate * span;
    improved = true;

    while FE < stopFE
        if ~improved
            s = s / 2;
            if max(s) < 1e-20
                % The step has collapsed: restart from a random point pulled to the incumbent
                srand = lb + rand(1, N) .* span;
                xk = srand + ((1 - bias) * rand(1, N) + bias) .* (best_x - srand);
                s = (0.3 + 0.3 * rand) * span;
            end
        end
        improved = false;

        for i = 1:N
            if FE >= stopFE
                break;
            end
            [before1, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
            if before1 < phase_best
                phase_best = before1;
                best_x = xk;
            end
            if FE >= stopFE
                break;
            end

            xk(i) = xk(i) - s(i);
            xk = min(max(xk, lb), ub);
            [after1, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
            if after1 < phase_best
                phase_best = after1;
                best_x = xk;
            end

            if abs(after1 - before1) <= 1e-20
                xk(i) = xk(i) + s(i);
            elseif after1 > before1
                xk(i) = xk(i) + 1.5 * s(i);          % undo, then half a step the other way
                xk = min(max(xk, lb), ub);
                if FE >= stopFE
                    break;
                end
                [after2, FE, curve, ph, fh, hidx, bsf, bsfx] = probe(xk, problem, FE, maxFE, curve, ph, fh, hidx, bsf, bsfx);
                if after2 < phase_best
                    phase_best = after2;
                    best_x = xk;
                end
                if after2 >= before1
                    xk(i) = xk(i) - 0.5 * s(i);
                else
                    improved = true;
                end
            else
                improved = true;
            end
            xk = min(max(xk, lb), ub);
        end
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

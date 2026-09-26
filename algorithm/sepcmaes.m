% ----------------------------------------------------------------------- %
% Separable Covariance Matrix Adaptation Evolution Strategy (sep-CMA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4 + floor(3*log(N))        % Offspring per generation (Hansen's default)
%   mu = floor(lambda/2)                % Parents, log-decreasing recombination weights
%   ccov1_sep = ccov1*(N+1.5)/3         % Rank-one rate with the sep-CMA correction
%   ccovmu_sep = ccovmu*(N+1.5)/3       % Rank-mu rate with the sep-CMA correction
%   cs = (mueff+2)/(N+mueff+3)          % Cumulation of the step-size path ps
%   cc = (4+mueff/N)/(N+4+2*mueff/N)    % Cumulation of the rank-one path pc
%   sigma0 = 0.3*(ub - lb)              % Initial per-coordinate std
%
% Algorithm Concept:
%   - Samples N(m, sigma^2*diag(C)): only the diagonal of C is learned, so
%     sampling and update cost O(N) per offspring
%   - The mean moves to the weighted recombination of the best mu offspring
%   - diag(C) adapts by the rank-one (pc) and rank-mu updates restricted to the
%     diagonal, their learning rates raised by (N+1.5)/3 to compensate
%   - The step size sigma adapts by cumulative step-size adaptation (path ps)
%   - Out-of-box samples are evaluated clamped; an adaptive quadratic penalty
%     on the clamp distance steers the unclamped update back into the box
%
% Reference:
% Raymond Ros, Nikolaus Hansen,
% A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity,
% Parallel Problem Solving from Nature -- PPSN X, Lecture Notes in Computer
% Science 5199 (2008) 296-305.
% https://doi.org/10.1007/978-3-540-87700-4_30
% ----------------------------------------------------------------------- %
% Implementation Note:
% Built from Hansen's cmaes.m 3.61.beta with DiagonalOnly = 1, cross-checked
% against PyPop7 SEPCMAES; the code scales ccov1 and ccovmu by (N+1.5)/3 where
% the paper writes (n+2)/3, and the code is followed. Boundary penalty as in
% bipop_cmaes.m (its val == 0 branch falls back on eps when min is empty). No
% restarts and no stop criteria: the budget is the only terminator, and the
% numerical warnings take cmaes.m's own no-stop branches (StopOnWarnings off,
% StopOnEqualFunctionValues 0), which inflate sigma, and diag(C) on a
% no-effect coordinate. DiffMinChange = eps*(ub-lb) (0 in cmaes.m) floors every
% coordinate std, and diag(C) is floored at max(diag(C))*1e-14, the condition
% limit cmaes.m applies only in its full-covariance branch.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sepcmaes(problem)

    N     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    lambda  = 4 + floor(3 * log(N));
    mu      = floor(lambda / 2);
    weights = log(max(mu, lambda / 2) + 0.5) - log(1:mu)';
    mueff   = sum(weights) ^ 2 / sum(weights .^ 2);
    weights = weights / sum(weights);

    cc     = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
    cs     = (mueff + 2) / (N + mueff + 3);
    ccov1  = 2 / ((N + 1.3) ^ 2 + mueff);
    ccovmu = min(1 - ccov1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ^ 2 + mueff));
    ccov1_sep  = min(1, ccov1 * (N + 1.5) / 3);
    ccovmu_sep = min(1 - ccov1_sep, ccovmu * (N + 1.5) / 3);
    damps  = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    chiN   = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));

    insigma = 0.3 * (ub - lb);   % cmaes.m's default when every coordinate is bounded
    maxdx   = (ub - lb) / 2;     % DiffMaxChange, clipped to half the box as cmaes.m does
    mindx   = eps * (ub - lb);   % DiffMinChange

    sigma = max(insigma);
    diagD = insigma / sigma;
    diagC = diagD .^ 2;
    pc    = zeros(N, 1);
    ps    = zeros(N, 1);

    xmean = lb + rand(N, 1) .* (ub - lb);
    xold  = xmean;

    fitHist = NaN(1, 10 + ceil(3 * 10 * N / lambda));

    % Adaptive boundary penalty state (bnd.flgscale = 0, so the scale is one)
    bndWeights  = zeros(N, 1);
    bndDfithist = 1;
    bndValidfit = 0;
    bndIniphase = 1;

    bsf  = inf;
    bsfx = xmean';

    % EvalInitialX is on in cmaes.m
    [f0, FE] = calculate_fitness(xmean, problem, FE);
    f0 = f0(1);
    fitHist(1) = f0;
    if f0 < bsf
        bsf  = f0;
        bsfx = xmean';
    end
    curve(FE) = bsf;
    [population_history, fitness_history, history_index] = record_history( ...
        FE, xmean', f0, population_history, fitness_history, history_index, maxFE);

    countiter = 0;

    while FE < maxFE
        countiter = countiter + 1;

        nsample = min(lambda, maxFE - FE);
        arz = randn(N, nsample);
        arx = xmean + (sigma * diagD) .* arz;
        bad = ~isfinite(arx);
        if any(bad(:))
            draw = lb + rand(N, nsample) .* (ub - lb);
            arx(bad) = draw(bad);
        end
        arxvalid = min(max(arx, lb), ub);

        [fraw, FE] = calculate_fitness(arxvalid, problem, FE);
        fraw = fraw(:)';

        popRows = arxvalid';
        for k = 1:nsample
            if fraw(k) < bsf
                bsf  = fraw(k);
                bsfx = popRows(k, :);
            end
            ec = FE - nsample + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                ec, popRows, fraw', population_history, fitness_history, history_index, maxFE);
        end

        % A truncated final generation cannot drive the adaptation
        if nsample < lambda
            break;
        end

        q   = myprctile(fraw, [25 75]);
        val = (q(2) - q(1)) / N / mean(diagC) / sigma ^ 2;
        if ~isfinite(val)
            val = max(bndDfithist);
        elseif val == 0
            pos = bndDfithist(bndDfithist > 0);
            if isempty(pos)
                val = eps;
            else
                val = min(pos);
            end
        elseif bndValidfit == 0
            bndDfithist = [];
            bndValidfit = 1;
        end

        if numel(bndDfithist) < 20 + (3 * N) / lambda
            bndDfithist = [bndDfithist val]; %#ok<AGROW>
        else
            bndDfithist = [bndDfithist(2:end) val];
        end

        tx = min(max(xmean, lb), ub);
        ti = (xmean ~= tx);

        if bndIniphase && any(ti)
            bndWeights(:) = 2.0002 * median(bndDfithist);
            dd = diagC / mean(diagC);
            bndWeights = bndWeights ./ dd;
            if bndValidfit && countiter > 2
                bndIniphase = 0;
            end
        end

        if any(ti)
            txd = xmean - tx;
            idx = ti & (abs(txd) > 3 * max(1, sqrt(N) / mueff) * sigma * sqrt(diagC));
            % Only while the mean is still moving away from the box
            idx = idx & (sign(txd) == sign(xmean - xold));
            bndWeights(idx) = 1.2 ^ (min(1, mueff / 10 / N)) * bndWeights(idx);
        end

        fsel = fraw + bndWeights' * (arxvalid - arx) .^ 2;

        fraw_s = sort(fraw);
        [fsel_s, idxsel] = sort(fsel);
        fitHist = [fraw_s(1), fitHist(1:end-1)];

        xold  = xmean;
        xmean = arx(:, idxsel(1:mu)) * weights;
        zmean = arz(:, idxsel(1:mu)) * weights;

        % B is the identity when C is diagonal
        ps   = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * zmean;
        hsig = norm(ps) / sqrt(1 - (1 - cs) ^ (2 * countiter)) / chiN < 1.4 + 2 / (N + 1);
        pc   = (1 - cc) * pc + hsig * (sqrt(cc * (2 - cc) * mueff) / sigma) * (xmean - xold);

        diagC = (1 - ccov1_sep - ccovmu_sep + (1 - hsig) * ccov1_sep * cc * (2 - cc)) * diagC ...
                + ccov1_sep * pc .^ 2 ...
                + ccovmu_sep * (diagC .* (arz(:, idxsel(1:mu)) .^ 2 * weights));
        diagC = max(diagC, max(diagC) * 1e-14);
        diagD = sqrt(diagC);

        sigma = sigma * exp(min(1, (norm(ps) / chiN - 1) * cs / damps));

        % cmaes.m re-aligns the scales of sigma and C when sigma runs away
        if sigma > 1e10 * max(diagD) && sigma > 8e14 * max(insigma)
            fac   = sigma;
            sigma = 1;
            pc    = fac * pc;
            diagD = fac * diagD;
            diagC = fac ^ 2 * diagC;
        end

        if any(sigma * sqrt(diagC) > maxdx)
            sigma = min(maxdx ./ sqrt(diagC));
        end
        if any(sigma * sqrt(diagC) < mindx)
            sigma = max(mindx ./ sqrt(diagC)) * exp(0.05 + cs / damps);
        end
        noeff = xmean == xmean + 0.2 * sigma * sqrt(diagC);
        if any(noeff)
            diagC = diagC + (ccov1_sep + ccovmu_sep) * (diagC .* noeff);
            sigma = sigma * exp(0.05 + cs / damps);
        end
        if all(xmean == xmean + 0.1 * sigma * diagD)
            sigma = sigma * exp(0.2 + cs / damps);
        end
        if fsel_s(1) == fsel_s(1 + ceil(0.1 + lambda / 4))
            sigma = sigma * exp(0.2 + cs / damps);
        end
        if countiter > 2 && myrange([fitHist fsel_s(1)]) == 0
            sigma = sigma * exp(0.2 + cs / damps);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

function r = myrange(x)
% Hansen's myrange; max/min skip the NaNs that pad an unfilled history
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

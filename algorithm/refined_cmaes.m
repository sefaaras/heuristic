% ----------------------------------------------------------------------- %
% Refined CMA-ES (R-CMA-ES)
% CEC 2026 competition -- 6th place; stored as refined_cmaes, rcmaes is Robust CMA-ES
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4*N, mu = lambda/2           % Tuned: replaces the usual 4 + floor(3*ln N)
%   sigma0 = 0.035*(ub - lb)              % Tuned: sigma = 7 on the release's [-100, 100] box
%   cc = 4/(N+4), ccov with mucov = mueff % R package cmaes learning rates
%   presamples = 1                        % Start mean = best of this many uniform points
%   TolX = 1e-12*sigma                    % With a non-PD C, the run's only stops
%   ipop_restarts = 0, flat-fitness escape off, midpoint evaluation off
%
% Algorithm Concept:
%   - Plain (mu/mu_w, lambda)-CMA-ES: log-weighted recombination, cumulative
%     step-size adaptation, rank-one plus rank-mu covariance update
%   - Out-of-box samples are evaluated at their projection, fitness multiplied
%     by 1 + squared distance to the box
%   - A run starts from one uniform presample after spending lambda evaluations
%     on a uniform population, and has no restart of its own
%   - It ends when C stops being numerically positive definite (min eig below
%     sqrt(eps)*max eig) or all sqrt(eig(C)) < TolX; see the note for what follows
%
% Reference:
% Adam Stelmaszczyk, Rafal Biedrzycki, Jaroslaw Arabas,
% Refined CMA-ES for CEC 2026 Bound-Constrained Single-Objective Optimization,
% 2026 IEEE Congress on Evolutionary Computation (CEC 2026), Maastricht, 2026.
% No DOI yet: the CEC 2026 proceedings are not indexed (Crossref, 2026-10-03).
% https://github.com/AdamStelmaszczyk/refined-cma-es
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from CMAES.R at the repository root (= data/ESCAPE_TUNED/CMAES.R), run
% as CEC2017ParallelBenchmark.R runs it: control = list(diag.bestVal = TRUE),
% so ipop_restarts = 0 and only the tolx test of ipop_stop is live.
% The release stops for good at its first termination and leaves the rest of
% the budget unspent; here the release is run again from scratch (presample,
% start population, sigma, C) until the budget is spent. Its own ipop_restarts
% branch is not used: the submission has it off, and its sigma falls back to a
% stale 0.5. Adaptations: box-normalised coordinates (exact on a uniform box),
% TolX in units of the widest side, and a negative fitness is divided by the
% penalty instead of multiplied, so the penalty always worsens it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = refined_cmaes(problem)

    N     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    w     = ub - lb;              % search coordinates y = (x - lb) ./ w
    wref  = max(w);               % problem-unit scale of sigma in the TolX test

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = (lb + ub)' / 2;

    sigma0  = 7 / 200;            % control default sigma = 7 on [-100, 100]
    sc_tolx = 1e-12;
    chiN    = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));

    lambda  = 4 * N;
    mu      = floor(lambda / 2);
    weights = log(mu + 1) - log(1:mu)';
    weights = weights / sum(weights);
    mueff   = sum(weights) ^ 2 / sum(weights .^ 2);
    cc      = 4 / (N + 4);
    cs      = (mueff + 2) / (N + mueff + 3);
    mucov   = mueff;
    ccov    = (1 / mucov) * 2 / (N + 1.4) ^ 2 + (1 - 1 / mucov) * ((2 * mucov - 1) / ((N + 2) ^ 2 + 2 * mucov));
    damps   = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;

    % The release's single run, started afresh whenever it terminates early
    while FE < maxFE
        runStart = FE;

        % best_of_random with presamples = 1
        xmean = rand(N, 1);
        [~, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
            evalCols(min(max(lb + w .* xmean, lb), ub), problem, FE, maxFE, curve, ...
                     population_history, fitness_history, history_index, bsf, bsfx);
        if FE >= maxFE
            break;
        end

        % The release evaluates lambda uniform points as its first "work arrays"
        nsample = min(lambda, maxFE - FE);
        [~, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
            evalCols(min(max(lb + w .* rand(N, nsample), lb), ub), problem, FE, maxFE, curve, ...
                     population_history, fitness_history, history_index, bsf, bsfx);

        sigma = sigma0;
        pc = zeros(N, 1);
        ps = zeros(N, 1);
        B  = eye(N);
        BD = eye(N);
        C  = eye(N);

        while FE < maxFE
            nsample = min(lambda, maxFE - FE);
            arz = randn(N, lambda);
            arx = xmean + sigma * (BD * arz);
            vx  = min(max(arx, 0), 1);
            pen = 1 + sum((w .* (arx - vx)) .^ 2, 1);
            pen(~isfinite(pen)) = realmax / 2;

            [yfit, FE, curve, population_history, fitness_history, history_index, bsf, bsfx] = ...
                evalCols(min(max(lb + w .* vx(:, 1:nsample), lb), ub), problem, FE, maxFE, curve, ...
                         population_history, fitness_history, history_index, bsf, bsfx);
            if nsample < lambda
                break;
            end

            arfitness = penalise(yfit, pen);
            [~, arindex] = sort(arfitness);
            aripop = arindex(1:mu);
            xmean  = arx(:, aripop) * weights;
            zmean  = arz(:, aripop) * weights;

            ps   = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (B * zmean);
            % The release's counteval: this run's evaluations, presample included
            hsig = norm(ps) / sqrt(1 - (1 - cs) ^ (2 * (FE - runStart) / lambda)) / chiN < 1.4 + 2 / (N + 1);
            pc   = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (BD * zmean);

            BDz = BD * arz(:, aripop);
            C = (1 - ccov) * C ...
                + ccov * (1 / mucov) * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
                + ccov * (1 - 1 / mucov) * (BDz .* weights') * BDz';

            sigma = sigma * exp((norm(ps) / chiN - 1) * cs / damps);

            % The release breaks out on a C that is not numerically positive definite
            if any(~isfinite(C(:)))
                break;
            end
            [Bn, e] = eigDesc(C);
            if ~all(e >= sqrt(eps) * abs(e(1)))
                break;
            end
            B  = Bn;
            d  = sqrt(e);
            BD = B .* d';

            % ipop_stop with ipop_restarts = 0 tests tolx only
            tolx = sc_tolx * sigma * wref;
            if all(d < tolx) && all(sigma * wref * pc < tolx)
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [f, FE, curve, ph, fh, hidx, bsf, bsfx] = evalCols(X, problem, FE, maxFE, ...
                                                           curve, ph, fh, hidx, bsf, bsfx)
% Evaluates the columns of X, which are also the population handed to the recorder
    k = size(X, 2);
    [f, FE] = calculate_fitness(X, problem, FE);
    f = f(:)';
    rows = X';
    for j = 1:k
        if f(j) < bsf
            bsf  = f(j);
            bsfx = X(:, j)';
        end
        ec = FE - k + j;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hidx] = record_history(ec, rows, f', ph, fh, hidx, maxFE);
        end
    end
end

function fp = penalise(f, pen)
% The release multiplies by the penalty; a negative fitness is divided so it still worsens
    fp = f .* pen;
    neg = f < 0;
    fp(neg) = f(neg) ./ pen(neg);
end

function [V, e] = eigDesc(C)
% R's eigen(symmetric = TRUE): reads the lower triangle, values in decreasing order
    Cs = tril(C) + tril(C, -1)';
    [V, E] = eig(Cs);
    [e, o] = sort(diag(E), 'descend');
    V = V(:, o);
end

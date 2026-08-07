% ----------------------------------------------------------------------- %
% Cross-Entropy CMA-ES (CE-CMAES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   I_NP = 100        % Population size
%   Ne   = 0.2*I_NP   % Number of elite samples (Cross-Entropy)
%   alpha = 0.9, beta = 0.1, q = 5   % CE smoothing parameters
%
% Algorithm Concept:
%   - Phase 1 (first half of the budget): Cross-Entropy method - sample from
%     N(mu, sigma2), keep the elite fraction, and smooth-update mu/sigma2.
%   - Phase 2 (second half): CMA-ES local exploitation seeded from the CE
%     best (covariance C, evolution paths psc/pc, step size sigma).
%
% Reference:
% D. Dabhi, K. Pandya, J. Soares, F. Lezama, Z. Vale,
% Cross Entropy Covariance Matrix Adaptation Evolution Strategy,
% Energies 15(13) (2022) 4838.
% https://doi.org/10.3390/en15134838
% Components:
%   Cross-Entropy method - R. Y. Rubinstein, D. P. Kroese, The Cross-Entropy
%     Method, Springer, 2004.
%   CMA-ES - N. Hansen, A. Ostermeier, "Completely Derandomized Self-Adaptation
%     in Evolution Strategies," Evolutionary Computation 9(2) (2001) 159-195.
%     https://doi.org/10.1162/106365601750190398
% ----------------------------------------------------------------------- %
% Implementation Note:
% The Energies paper above is the closest published CE-CMAES; this file is a
% generic-benchmark adaptation of the same idea rather than a port of it.
% Covariance eigenvalues are floored at emax*1e-14 before the inverse square
% root, and C is reset to the identity when it degenerates entirely. Without
% the floor, rounding drives the smallest eigenvalue slightly negative once C
% loses positive definiteness, sqrt returns a complex number, and the whole
% population turns complex -- observed on CEC2017 F30 and CEC2022 F12.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cecmaes(problem)

    % Extract problem parameters
    D = problem.dimension;
    Xmin = problem.lb;
    Xmax = problem.ub;
    maxFE = problem.maxFe;

    I_NP = 100;
    I_itermax = ceil(maxFE / I_NP) + 1;

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Ne = round(0.2 * I_NP);

    % Cross-Entropy method (phase 1)
    mu = (Xmin + Xmax) / 2;
    sigma2 = (Xmax - Xmin) / 4;
    alpha = 0.9;
    beta = 0.1;
    q = 5;
    xmin = repmat(Xmin, I_NP, 1);
    xmax = repmat(Xmax, I_NP, 1);

    bsf = inf;
    bsf_sol = mu;
    gbestval = inf;
    gbest = mu;

    gen = 1;
    while gen < round(0.5 * I_itermax) && FE < maxFE
        pos = zeros(I_NP, D);
        for ii = 1:I_NP
            pos(ii, :) = sigma2 .* randn(1, D) + mu;
        end
        % check limits
        changemax = pos > xmax; pos(changemax) = xmax(changemax);
        changemin = pos < xmin; pos(changemin) = xmin(changemin);

        ev0 = FE;
        [solFitness_M, FE] = calculate_fitness(pos', problem, FE);
        solFitness_M = solFitness_M(:);

        [m, mi] = min(solFitness_M);
        if m < bsf, bsf = m; bsf_sol = pos(mi, :); end
        [curve, population_history, fitness_history, history_index] = rec_block(...
            pos, solFitness_M, ev0 + 1, FE, bsf, curve, population_history, fitness_history, ...
            history_index, maxFE);

        mu_old = mu;
        sigma2_old = sigma2;
        [yy, I] = sort(solFitness_M, 'ascend');
        pos = pos(I, :);
        gbestval = yy(1);
        gbest = pos(1, :);
        xx = pos(1:Ne, :);
        mu = mean(xx);
        sigma2 = sqrt(var(xx, 1));

        mu = alpha * mu + (1 - alpha) * mu_old;
        sigmaStd1 = beta - beta * (1 - 1 / (gen + 1))^q;
        sigma2 = sigma2 * sigmaStd1 + sigma2_old * (1 - sigmaStd1);

        gen = gen + 1;
    end

    % CMA-ES local exploitation (phase 2)
    FVr_bestmemit = gbest;
    xmean = FVr_bestmemit';
    sigma = 1.0;
    lambda = I_NP;
    mu = lambda / 2;
    weights = log(mu + 1 / 2) - log(1:mu)';
    mu = floor(mu);
    weights = weights / sum(weights);
    mueff = sum(weights)^2 / sum(weights.^2);

    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D);
    cs = (mueff + 2) / (D + mueff + 5);
    c1 = 2 / ((D + 1.3)^2 + mueff);
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2)^2 + mueff));
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (D + 1)) - 1) + cs;

    pc = zeros(D, 1); psc = zeros(D, 1);
    B = eye(D, D);
    F = ones(D, 1);
    C = B * diag(F.^2) * B';
    invsqrtC = B * diag(F.^-1) * B';
    eigeneval = 0;
    chiN = D^0.5 * (1 - 1 / (4 * D) + 1 / (21 * D^2));
    counteval = 0;

    while gen >= round(0.5 * I_itermax) && gen < I_itermax && FE < maxFE
        arx = zeros(D, lambda);
        for k = 1:lambda
            arx(:, k) = xmean + sigma * B * (F .* randn(D, 1));
            counteval = counteval + 1;
        end
        arx_trans1 = arx';
        changemax = arx_trans1 > xmax; arx_trans1(changemax) = xmax(changemax);
        changemin = arx_trans1 < xmin; arx_trans1(changemin) = xmin(changemin);
        arx_trans = arx_trans1;
        % arx kept at the CLAMPED draw: xmean and the rank-mu term of C are both built from it
        arx = arx_trans';
        pos = arx_trans;

        ev0 = FE;
        [solFitness_M, FE] = calculate_fitness(pos', problem, FE);
        solFitness_M = solFitness_M(:);

        [~, arindex] = sort(solFitness_M);
        [tmpgbestval, gbestid] = min(solFitness_M);
        if tmpgbestval < gbestval
            gbestval = tmpgbestval;
            gbest = arx_trans(gbestid, :);
        end
        if tmpgbestval < bsf, bsf = tmpgbestval; bsf_sol = arx_trans(gbestid, :); end

        [curve, population_history, fitness_history, history_index] = rec_block(...
            pos, solFitness_M, ev0 + 1, FE, bsf, curve, population_history, fitness_history, ...
            history_index, maxFE);

        FVr_bestmemit = gbest;
        xold = xmean;
        xmean = arx(:, arindex(1:mu)) * weights;

        psc = (1 - cs) * psc + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
        hsig = norm(psc) / sqrt(1 - (1 - cs)^(2 * counteval / lambda)) / chiN < 1.4 + 2 / (D + 1);
        pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

        artmp = (1 / sigma) * (arx(:, arindex(1:mu)) - repmat(xold, 1, mu));
        C = (1 - c1 - cmu) * C ...
            + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
            + cmu * artmp * diag(weights) * artmp';

        sigma = sigma * exp((cs / damps) * (norm(psc) / chiN - 1));

        if counteval - eigeneval > lambda / (c1 + cmu) / D / 10
            eigeneval = counteval;
            C = triu(C) + triu(C, 1)';
            [B, F] = eig(C);
            e = real(diag(F));
            emax = max(e);
            if ~isfinite(emax) || emax <= 0
                C = eye(D); B = eye(D); e = ones(D, 1);   % fully degenerate -> reset
            else
                e = max(e, emax * 1e-14);
            end
            B = real(B);
            F = sqrt(e);
            invsqrtC = B * diag(F.^-1) * B';
        end

        gen = gen + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_solution = bsf_sol;
    best_fitness = bsf;
end

% Record best-so-far curve + population metrics over an FE block
function [curve, ph, fh, hidx] = rec_block(pop, costs, fe_from, fe_to, bestval, curve, ph, fh, hidx, maxFE)
    fe_to = min(fe_to, maxFE);
    if fe_from < 1, fe_from = 1; end
    if fe_to < fe_from, return; end
    costs = costs(:)';
    for ec = fe_from:fe_to
        curve(ec) = bestval;
        [ph, fh, hidx] = record_history(ec, pop, costs, ph, fh, hidx, maxFE);
    end
end

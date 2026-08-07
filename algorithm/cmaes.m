% ----------------------------------------------------------------------- %
% Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4 + round(3*log(nVar))        % Offspring population size (Hansen's default)
%   mu     = round(lambda/2)               % Number of parents
%
% Algorithm Concept:
%   - Samples offspring from N(m, sigma^2 * C)
%   - Adapts the mean by weighted recombination of the best mu offspring
%   - Controls step size sigma via the conjugate evolution path (cs/ds)
%   - Adapts covariance C via rank-one (pc) + rank-mu updates
%
% Reference:
% Nikolaus Hansen, Andreas Ostermeier,
% Completely Derandomized Self-Adaptation in Evolution Strategies,
% Evolutionary Computation 9 (2) (2001) 159-195.
% https://doi.org/10.1162/106365601750190398
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from a MATLAB release, not from Hansen's paper; it stored ps/pc/C/
% sigma/M per generation, this port keeps only the current one (identical).
% Six deliberate deviations from it: lambda is Hansen's default, not ten times
% it; the CSA step drops its ^0.3 damping; ps uses M_Step/Rchol, not /chol(C)'
% (R'R = C, so dividing by R' gives inv(R*R'), not inv(C)); the step is
% re-derived from the CLAMPED position, since the mean and rank-mu term build on
% it and pre-clamp steps collapse C to rank 1; enforce_pd floors eigenvalues at
% emax*1e-14 every generation, where the release clamped only negatives to 0 and
% left C singular so chol threw on real CEC2014 runs; and MaxIt = maxFE makes
% the FE guard the sole terminator, the release's estimate spending 50-88 %.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cmaes(problem)

    % Extract problem parameters
    nVar = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    VarSize = [1 nVar];

    % CMA-ES settings
    MaxIt = maxFE;

    lambda = 4 + round(3 * log(nVar));           % population size (offspring)
    mu = round(lambda / 2);                      % number of parents

    w = log(mu + 0.5) - log(1:mu);               % parent weights
    w = w / sum(w);
    mu_eff = 1 / sum(w.^2);                       % effective number of parents

    sigma0 = 0.3 * (VarMax - VarMin);
    cs = (mu_eff + 2) / (nVar + mu_eff + 5);
    ds = 1 + cs + 2 * max(sqrt((mu_eff - 1) / (nVar + 1)) - 1, 0);
    ENN = sqrt(nVar) * (1 - 1 / (4 * nVar) + 1 / (21 * nVar^2));

    cc = (4 + mu_eff / nVar) / (4 + nVar + 2 * mu_eff / nVar);
    c1 = 2 / ((nVar + 1.3)^2 + mu_eff);
    alpha_mu = 2;
    cmu = min(1 - c1, alpha_mu * (mu_eff - 2 + 1 / mu_eff) / ((nVar + 2)^2 + alpha_mu * mu_eff / 2));
    hth = (1.4 + 2 / (nVar + 1)) * ENN;

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialization (current-generation state)
    ps = zeros(VarSize);
    pc = zeros(VarSize);
    C = eye(nVar);
    sigma = sigma0;

    empty_individual.Position = [];
    empty_individual.Step = [];
    empty_individual.Cost = [];

    M_Position = unifrnd(VarMin, VarMax, VarSize);
    M_Step = zeros(VarSize); %#ok<NASGU>
    [M_Cost, FE] = calculate_fitness(M_Position', problem, FE);

    BestSol_Position = M_Position;
    BestSol_Cost = M_Cost;

    if FE <= maxFE
        curve(FE) = BestSol_Cost;
    end

    % CMA-ES main loop
    for g = 1:MaxIt
        if FE >= maxFE, break; end
        FE_before = FE;

        % Generate samples
        pop = repmat(empty_individual, lambda, 1);
        for i = 1:lambda
            pop(i).Step = mvnrnd(zeros(VarSize), C);
            pop(i).Position = M_Position + sigma .* pop(i).Step;
            pop(i).Position = max(pop(i).Position, VarMin);
            pop(i).Position = min(pop(i).Position, VarMax);
            % Step re-derived from the CLAMPED position: pre-clamp steps collapse C to rank 1 at large sigma
            pop(i).Step = (pop(i).Position - M_Position) ./ sigma;
            [pop(i).Cost, FE] = calculate_fitness(pop(i).Position', problem, FE);
            if pop(i).Cost < BestSol_Cost
                BestSol_Cost = pop(i).Cost;
                BestSol_Position = pop(i).Position;
            end
            if FE <= maxFE
                curve(FE) = BestSol_Cost;
            end
            if FE >= maxFE, break; end
        end

        % Sort population
        Costs = [pop.Cost];
        [~, SortOrder] = sort(Costs);
        pop = pop(SortOrder);

        % Record history for this generation's FE block (top-100)
        [population_history, fitness_history, history_index] = record_cmaes(...
            pop, nVar, FE_before + 1, min(FE, maxFE), ...
            population_history, fitness_history, history_index, maxFE);

        if g == MaxIt || FE >= maxFE
            break;
        end

        % Update mean
        M_Step = 0;
        for j = 1:mu
            M_Step = M_Step + w(j) * pop(j).Step;
        end
        M_Position_new = M_Position + sigma .* M_Step;
        M_Position_new = max(M_Position_new, VarMin);
        M_Position_new = min(M_Position_new, VarMax);
        [M_Cost_new, FE] = calculate_fitness(M_Position_new', problem, FE);
        if M_Cost_new < BestSol_Cost
            BestSol_Cost = M_Cost_new;
            BestSol_Position = M_Position_new;
        end
        if FE <= maxFE
            curve(FE) = BestSol_Cost;
        end

        % Update step size (robust factor: never errors on a near-singular C)
        Rchol = safe_chol(C);
        % chol gives R'R = C, so the ROW vector's conjugate step is M_Step/R, not M_Step/R'
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (M_Step / Rchol);
        % Canonical CSA update - no extra damping exponent.
        sigma = sigma * exp(cs / ds * (norm(ps) / ENN - 1));

        % Update covariance matrix
        if norm(ps) / sqrt(1 - (1 - cs)^(2 * (g + 1))) < hth
            hs = 1;
        else
            hs = 0;
        end
        delta = (1 - hs) * cc * (2 - cc);
        pc = (1 - cc) * pc + hs * sqrt(cc * (2 - cc) * mu_eff) * M_Step;
        C = (1 - c1 - cmu) * C + c1 * (pc' * pc + delta * C);
        for j = 1:mu
            C = C + cmu * w(j) * pop(j).Step' * pop(j).Step;
        end

        % Symmetry and strict positive definiteness enforced so the next chol/mvnrnd cannot fail
        C = enforce_pd(C, nVar);

        M_Position = M_Position_new;
        M_Cost = M_Cost_new; %#ok<NASGU>
    end

    curve(min(FE, maxFE):end) = BestSol_Cost;

    best_solution = BestSol_Position;
    best_fitness = BestSol_Cost;
end

% Robust Cholesky factor: diagonal-jitter fallback if C is not PD
function R = safe_chol(C)
    [R, p] = chol(C);
    if p == 0, return; end
    n = size(C, 1);
    d = mean(diag(C));
    if ~isfinite(d) || d <= 0, d = 1; end
    jitter = 1e-12 * d;
    for k = 1:30
        [R, p] = chol(C + jitter * eye(n));
        if p == 0, return; end
        jitter = jitter * 10;
    end
    R = eye(n);   % last resort: whitening reduces to identity
end

% Force C symmetric and strictly positive definite (capped condition)
function C = enforce_pd(C, nVar)
    C = (C + C') / 2;                    % strip roundoff asymmetry
    [V, E] = eig(C);
    e = real(diag(E));
    emax = max(e);
    if ~isfinite(emax) || emax <= 0
        C = eye(nVar);                   % fully degenerate / NaN -> reset
        return;
    end
    e = max(e, emax * 1e-14);            % floor eigenvalues -> strict PD, cond <= 1e14
    C = V * diag(e) * V';
    C = (C + C') / 2;
end

% Record the metrics of the CMA-ES population over an FE block
function [pop_hist, fit_hist, hist_idx] = record_cmaes(pop, dim, fe_from, fe_to, pop_hist, fit_hist, hist_idx, maxFE)
    if fe_to < fe_from, return; end
    n = numel(pop);
    rec_pop = zeros(n, dim);
    rec_fit = zeros(1, n);
    for i = 1:n
        rec_pop(i, :) = pop(i).Position;
        rec_fit(i) = pop(i).Cost;
    end
    for eval_count = fe_from:fe_to
        [pop_hist, fit_hist, hist_idx] = record_history(...
            eval_count, rec_pop, rec_fit, pop_hist, fit_hist, hist_idx, ...
            maxFE);
    end
end

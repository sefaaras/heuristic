% ----------------------------------------------------------------------- %
% Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
% for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = (4 + round(3*log(nVar)))*10   % Offspring population size
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
%
% Note: the original stores ps/pc/C/sigma/M for every generation in cell/struct
% arrays of length MaxIt; this port keeps only the current generation's values
% (identical computations) to avoid a MaxIt x nVar x nVar allocation.
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cmaes(problem)

    % Extract problem parameters
    nVar = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    VarSize = [1 nVar];

    %% CMA-ES settings
    MaxIt = maxFE;

    lambda = (4 + round(3 * log(nVar))) * 10;   % population size (offspring)
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
    history_pop_size = 100;
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, history_pop_size, nVar);
    fitness_history = zeros(history_size, history_pop_size);
    history_index = 1;

    %% Initialization (current-generation state)
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

    %% CMA-ES main loop
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
            pop, history_pop_size, nVar, FE_before + 1, min(FE, maxFE), ...
            population_history, fitness_history, history_index, sampling_interval, history_size);

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
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (M_Step / Rchol');
        sigma = sigma * exp(cs / ds * (norm(ps) / ENN - 1))^0.3;

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

        % Enforce symmetry and strict positive definiteness so the next
        % generation's chol(C) / mvnrnd(.,C) cannot fail on a degenerate C.
        C = enforce_pd(C, nVar);

        M_Position = M_Position_new;
        M_Cost = M_Cost_new; %#ok<NASGU>
    end

    curve(min(FE, maxFE):end) = BestSol_Cost;

    best_solution = BestSol_Position;
    best_fitness = BestSol_Cost;
end

%% --- Robust Cholesky factor: diagonal-jitter fallback if C is not PD ---
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

%% --- Force C symmetric and strictly positive definite (capped condition) ---
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

%% --- Record top-k of the sorted CMA-ES population over an FE block ---
function [pop_hist, fit_hist, hist_idx] = record_cmaes(pop, history_pop_size, dim, fe_from, fe_to, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    if fe_to < fe_from, return; end
    top_k = min(history_pop_size, numel(pop));
    rec_pop = NaN(history_pop_size, dim);
    rec_fit = NaN(1, history_pop_size);
    for i = 1:top_k
        rec_pop(i, :) = pop(i).Position;
        rec_fit(i) = pop(i).Cost;
    end
    for eval_count = fe_from:fe_to
        [pop_hist, fit_hist, hist_idx] = record_history(...
            eval_count, rec_pop, rec_fit, pop_hist, fit_hist, hist_idx, ...
            sampling_interval, history_size);
    end
end

% ----------------------------------------------------------------------- %
% Dynamic Search Fireworks Algorithm with Covariance Mutation (dynFWACM)
% CEC 2015 learning-based track -- 10th place, tied (organiser's draft ranking)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_fw = 5                    % Fireworks
%   m = 150                     % Explosion sparks shared out by fitness, each in [6, 120]
%   A_cf = ub - lb              % Core firework amplitude at the start, per dimension
%   Ca = 1.2, Cr = 0.9          % Core amplitude x Ca after an improvement, x Cr otherwise
%   A_hat = 0.2*(ub - lb)       % Amplitude budget of the non-core fireworks
%   n_gauss = D                 % Covariance-mutation sparks per iteration
%
% Algorithm Concept:
%   - Better fireworks get more sparks; a spark moves each coordinate with chance 1/2
%     by a uniform offset within the firework's amplitude
%   - The core firework (the best) has a dynamic amplitude, amplified when the
%     best improves and reduced when it does not
%   - The other fireworks' amplitudes grow with their fitness gap to the best
%   - Covariance mutation: D sparks drawn from N(mean, C) of the core firework's
%     spark cluster, C from its better half about the cluster mean
%   - Next fireworks: the best of everything, plus four drawn uniformly at random
%     from the explosion sparks and the old fireworks
%
% Reference:
% Chao Yu, Ling Chen Kelley, Ying Tan,
% Dynamic search fireworks algorithm with covariance mutation for solving the CEC
% 2015 learning based competition problems,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 1106-1112.
% https://doi.org/10.1109/CEC.2015.7257013
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (15642matlab_dynFWACM_simplify.zip in
% github.com/P-N-Suganthan/CEC2015-Learning-Based). Its main tunes (Ca, Cr) for each
% CEC 2015 function, from (1.11-1.25, 0.80-0.90); those are not ported, and the
% dynFWA defaults of the same group, Ca = 1.2 and Cr = 0.9, are used instead. The
% release's amplitudes 200 and 40 on its +/-100 box are scaled to each range (exact
% there). mvnrnd is replaced by the same factorisation (cholcov's chol, else eig)
% and draw, in base MATLAB; eigenvalues below zero are dropped where mvnrnd would
% stop. A non-finite spark count falls to the minimum and a non-finite coordinate is
% redrawn like an out-of-box one. The last iteration is cut to the budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dynfwacm(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    n_fw     = 5;
    m_coef   = 150;
    max_spk  = 0.8 * 150;
    min_spk  = 0.04 * 150;
    n_gauss  = D;
    Ca       = 1.2;
    Cr       = 0.9;
    A_coef   = 0.2 * span;
    A_cf     = span;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;
    bsf = inf;
    bsf_solution = lb + 0.5 * span;

    FW = repmat(lb, n_fw, 1) + rand(n_fw, D) .* repmat(span, n_fw, 1);
    n0 = min(n_fw, maxFE);
    FWf = inf(1, n_fw);
    FWf(1:n0) = evaluate(FW(1:n0, :));
    [population_history, fitness_history, history_index] = record_history( ...
        FE, FW(1:n0, :), FWf(1:n0), population_history, fitness_history, history_index, maxFE);

    iter = 0;
    fit_prev = inf;
    while FE < maxFE
        iter = iter + 1;
        % The first firework is the elite after the first iteration
        fit_iter = FWf(1);
        if iter > 1
            if fit_prev - fit_iter > 0
                A_cf = A_cf * Ca;
            else
                A_cf = A_cf * Cr;
            end
        end
        fit_prev = fit_iter;

        fitness_max = max(FWf);
        fitness_sub_max = abs(fitness_max - FWf);
        fitness_sub_max_sum = sum(fitness_sub_max);
        n_spk = zeros(1, n_fw);
        for i = 1:n_fw
            s = (fitness_sub_max(i) + eps) / (fitness_sub_max_sum + eps);
            s = round(s * m_coef);
            if s > max_spk
                s = max_spk;
            elseif s < min_spk || ~isfinite(s)
                s = min_spk;
            end
            n_spk(i) = s;
        end
        n_spk_sum = sum(n_spk);

        fitness_best = min(FWf);
        fitness_sub_best = abs(fitness_best - FWf);
        fitness_sub_best_sum = sum(fitness_sub_best);
        scope = zeros(n_fw, D);
        for i = 1:n_fw
            scope(i, :) = A_coef * (fitness_sub_best(i) + eps) / (fitness_sub_best_sum + eps);
        end
        [~, min_index] = min(FWf);
        scope(min_index, :) = A_cf;

        E = zeros(n_spk_sum, D);
        e = 0;
        for j = 1:n_fw
            for i = 1:n_spk(j)
                e = e + 1;
                E(e, :) = FW(j, :);
                for k = 1:D
                    if rand > 0.5
                        offset = (rand*2 - 1) * scope(j, k);
                        E(e, k) = E(e, k) + offset;
                        if ~(E(e, k) <= ub(k) && E(e, k) >= lb(k))
                            E(e, k) = (rand*2 - 1) * span(k) / 2 + (ub(k) + lb(k)) / 2;
                        end
                    end
                end
            end
        end
        n_e = min(n_spk_sum, maxFE - FE);
        Ef = inf(1, n_spk_sum);
        Ef(1:n_e) = evaluate(E(1:n_e, :));
        [population_history, fitness_history, history_index] = record_history( ...
            FE, [FW; E(1:n_e, :)], [FWf, Ef(1:n_e)], population_history, fitness_history, history_index, maxFE);
        if FE >= maxFE
            break;
        end

        % Covariance mutation on the core firework's cluster
        [~, best_fw] = min(FWf);
        first = sum(n_spk(1:best_fw-1)) + 1;
        last  = sum(n_spk(1:best_fw));
        patch = [E(first:last, :); FW(best_fw, :)];
        pfit  = [Ef(first:last), FWf(best_fw)];
        lambda = numel(pfit);
        mu = floor(lambda / 2);
        [~, fi] = sort(pfit);
        meanV = mean(patch(fi(1:lambda), :));
        Sigma = zeros(D, D);
        for i = 1:mu
            Sigma = Sigma + (patch(fi(i), :) - meanV)' * (patch(fi(i), :) - meanV);
        end
        Sigma = Sigma / mu;
        meanV = mean(patch(fi(1:mu), :), 1);
        G = mvn_draw(meanV, Sigma, n_gauss);
        for i = 1:n_gauss
            for j = 1:D
                if ~(G(i, j) >= lb(j) && G(i, j) <= ub(j))
                    G(i, j) = lb(j) + rand * (ub(j) - lb(j));
                end
            end
        end
        n_g = min(n_gauss, maxFE - FE);
        Gf = evaluate(G(1:n_g, :));
        [population_history, fitness_history, history_index] = record_history( ...
            FE, [FW; E; G(1:n_g, :)], [FWf, Ef, Gf], population_history, fitness_history, history_index, maxFE);
        if FE >= maxFE
            break;
        end

        % Elite first, the rest uniformly from the explosion sparks and old fireworks
        S  = [E; FW; G];
        Sf = [Ef, FWf, Gf];
        [min_val, min_idx] = min(Sf);
        FW_new  = zeros(n_fw, D);
        FWf_new = zeros(1, n_fw);
        FW_new(1, :) = S(min_idx, :);
        FWf_new(1)   = min_val;
        for i = 2:n_fw
            pick = ceil(rand * (n_spk_sum + n_fw));
            FW_new(i, :) = S(pick, :);
            FWf_new(i)   = Sf(pick);
        end
        FW  = FW_new;
        FWf = FWf_new;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of Xr (already cut to the budget), tracking the best per evaluation
    function fr = evaluate(Xr)
        [fv, FE_new] = calculate_fitness(Xr', problem, FE);
        fr = fv(:)';
        for q = 1:numel(fr)
            if fr(q) < bsf
                bsf = fr(q);
                bsf_solution = Xr(q, :);
            end
            curve(FE + q) = bsf;
        end
        FE = FE_new;
    end
end

% mvnrnd(mu, Sigma, n) with cholcov's factor: chol, else the sign-fixed eigenbasis
function G = mvn_draw(mu, Sigma, n)
    d = numel(mu);
    if ~all(isfinite(Sigma(:)))
        G = repmat(mu, n, 1);
        return;
    end
    [T, p] = chol(Sigma);
    if p > 0
        [U, Dm] = eig(full((Sigma + Sigma') / 2));
        [~, maxind] = max(abs(U), [], 1);
        negloc = (U(maxind + (0:d:(d-1)*d)) < 0);
        U(:, negloc) = -U(:, negloc);
        Dm = diag(Dm);
        tol = eps(max(Dm)) * length(Dm);
        keep = Dm > tol;
        T = diag(sqrt(Dm(keep))) * U(:, keep)';
    end
    G = randn(n, size(T, 1)) * T + repmat(mu, n, 1);
    flat = diag(Sigma) == 0;
    G(:, flat) = repmat(mu(flat), n, 1);
end

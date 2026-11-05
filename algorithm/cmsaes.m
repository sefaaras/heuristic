% ----------------------------------------------------------------------- %
% Covariance Matrix Self-Adaptation Evolution Strategy (CMSA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4*D                   % Offspring per generation (the paper's linear sizing)
%   mu     = floor(lambda/4)       % Parents, truncation ratio 1/4, equal weights
%   tau    = 1/sqrt(2*D)           % Learning rate of the log-normal sigma mutation, Eq. (1)
%   tau_c  = 1 + D*(D+1)/(2*mu)    % Time constant of the covariance update, Eq. (2)
%   sigma0 = 0.3, C0 = I           % Initial step in box widths per coordinate
%
% Algorithm Concept:
%   - Each offspring draws its own step size sigma_l = <sigma>*exp(tau*N(0,1))
%     and a direction s_l = sqrt(C)*N(0, I), sqrt(C) the transposed Cholesky factor
%   - The parent moves to the mean of the mu best offspring, y + <sigma_l*s_l>
%   - sigma is self-adapted: <sigma> is the mean of the mu best sigma_l
%   - C <- (1 - 1/tau_c)*C + (1/tau_c)*<s*s'> over the mu best directions; no
%     evolution paths and no cumulative step-size adaptation
%   - Independent restarts from a uniform point once the run has converged
%
% Reference:
% Hans-Georg Beyer, Bernhard Sendhoff,
% Covariance Matrix Adaptation Revisited -- The CMSA Evolution Strategy,
% Parallel Problem Solving from Nature -- PPSN X, Lecture Notes in Computer
% Science 5199 (2008) 123-132.
% https://doi.org/10.1007/978-3-540-87700-4_13
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Fig. 2, Eqs. (1)-(2)); no released code was found. The paper
% tests lambda = 8, 4*D and 4*D^2 and fixes none; 4*D is used. It has no
% restart, sigma0 or bound rule: the run restarts from x ~ U(lb,ub), sigma0 =
% 0.3, C = I when the largest coordinate std falls below 1e-12 box widths, when
% the best of the last 10 + ceil(30*D/lambda) generations and the current
% generation span 1e-12 or less (cmaes.m TolFun), or when chol(C) fails. Steps
% are in box widths; offspring are clamped to the box and their s and z
% recomputed from the clamped step, so the update uses the steps taken; the
% largest coordinate std is capped at half the box.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cmsaes(problem)

    D     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    w     = ub - lb;

    lambda = 4 * D;
    mu     = floor(lambda / 4);
    tau    = 1 / sqrt(2 * D);
    tau_c  = 1 + D * (D + 1) / (2 * mu);
    sigma0 = 0.3;

    tol_x    = 1e-12;                       % restart: coordinate std, in box widths
    tol_fun  = 1e-12;                       % restart: fitness span, as cmaes.m's TolFun
    hist_len = 10 + ceil(30 * D / lambda);
    std_hi   = 0.5;                         % coordinate std cap: half the box

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        y     = lb + rand(D, 1) .* w;
        sigma = sigma0;
        C     = eye(D);
        R     = eye(D);
        hist  = [];

        while FE < maxFE
            n = min(lambda, maxFE - FE);
            sig = sigma * exp(tau * randn(1, n));    % (R1)
            S   = R' * randn(D, n);                  % (R2), R'*R = C
            Xr  = y + w .* (sig .* S);               % (R3)-(R4)
            bad = ~isfinite(Xr);
            if any(bad(:))
                draw = lb + rand(D, n) .* w;
                Xr(bad) = draw(bad);
            end
            X = min(max(Xr, lb), ub);
            moved = any(X ~= Xr, 1);
            if any(moved)
                % The step actually taken: its direction is what C learns
                S(:, moved) = ((X(:, moved) - y) ./ w) ./ sig(moved);
            end

            [fit, FE] = calculate_fitness(X, problem, FE);   % (R5)
            fit = fit(:)';

            popRows = X';
            for k = 1:n
                if isempty(bsfx) || fit(k) < bsf
                    bsf  = fit(k);
                    bsfx = popRows(k, :);
                end
                ec = FE - n + k;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, popRows, fit', population_history, fitness_history, history_index, maxFE);
            end

            % A truncated final generation cannot drive the update
            if n < lambda
                break;
            end

            fs = fit;
            fs(isnan(fs)) = inf;
            [fs, order] = sort(fs);
            sel = order(1:mu);

            y = y + w .* ((S(:, sel) .* sig(sel)) * ones(mu, 1) / mu);     % (R6)
            C = (1 - 1 / tau_c) * C + (1 / tau_c) * (S(:, sel) * S(:, sel)') / mu;   % (R7)
            C = (C + C') / 2;
            sigma = mean(sig(sel));

            cstd = sqrt(max(diag(C)));
            if sigma * cstd > std_hi
                sigma = std_hi / cstd;
            end

            hist = [fs(1), hist(1:min(end, hist_len - 1))]; %#ok<AGROW>
            span = max(max(hist), fs(end)) - min(min(hist), fs(1));
            [R, p] = chol(C);
            if p > 0 || ~isfinite(sigma) || sigma * cstd < tol_x || ...
                    (numel(hist) >= hist_len && span <= tol_fun)
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

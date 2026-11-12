% ----------------------------------------------------------------------- %
% (mu/mu_I, lambda) Evolution Strategy with sigma Self-Adaptation (SA-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   mu = 3, lambda = 12      % Parents and offspring, truncation ratio 1/4
%   tau = 1/sqrt(2*D)        % Learning parameter of the log-normal sigma mutation
%   sigma0 = 0.3             % Initial mutation strength, in box widths
%   sigma_min = 1e-10        % Restart once the best offspring's sigma is below this
%   sigma_max = 0.5          % Cap on the recombined sigma, in box widths
%
% Algorithm Concept:
%   - Intermediate recombination of all mu parents: one recombinant y and one
%     sigma, the means of the parents' object vectors and mutation strengths
%   - Each offspring first mutates sigma log-normally, sigma_l = sigma*exp(tau*N(0,1)),
%     then uses it: y_l = y + sigma_l*N(0, I), the step scaled to each box width
%   - Comma selection: the mu best of the lambda offspring are the next parents,
%     each carrying the sigma that produced it, so good step sizes are inherited
%   - Independent restarts from a uniform point once sigma has collapsed
%
% Reference:
% Hans-Georg Beyer, Hans-Paul Schwefel,
% Evolution strategies -- A comprehensive introduction,
% Natural Computing 1 (1) (2002) 3-52.
% https://doi.org/10.1023/A:1015059928466
% ----------------------------------------------------------------------- %
% Implementation Note:
% Algorithm from the paper (Fig. 1, Sec. 4.2, Eqs. (23)-(24)); mu, lambda, tau,
% sigma_min and the evaluated initial parent are those of Beyer's own release
% mu_mu_I_lambda-ES.oct (homepages.fhv.at/hgb, Wayback 2023-12-13), which the
% port reproduces bit for bit. The paper fixes neither mu nor lambda (it gives
% mu/lambda ~ 0.27 as optimal) and no box rule: steps are in box widths,
% offspring are clamped to the box, sigma0 = 0.3 as cmsaes, and the recombined
% sigma is capped at half the box. The release stops at sigma_min; here the run
% restarts from x ~ U(lb, ub) until maxFe.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = saes(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    w     = ub - lb;

    mu        = 3;
    lambda    = 12;
    tau       = 1 / sqrt(2 * D);
    sigma0    = 0.3;
    sigma_min = 1e-10;
    sigma_max = 0.5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        y = lb + rand(1, D) .* w;
        % The release evaluates its initial parent once; all mu parents are copies of it
        [fy, FE] = calculate_fitness(y', problem, FE);
        fy = fy(1);
        if isempty(bsfx) || fy < bsf || (isnan(bsf) && ~isnan(fy))
            bsf  = fy;
            bsfx = y;
        end
        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history( ...
            FE, y, fy, population_history, fitness_history, history_index, maxFE);

        Y = repmat(y, mu, 1);
        S = sigma0 * ones(mu, 1);

        while FE < maxFE
            yr = Y(1, :);
            sr = S(1);
            for i = 2:mu
                yr = yr + Y(i, :);
                sr = sr + S(i);
            end
            yr = yr / mu;
            sr = min(sr / mu, sigma_max);

            n   = min(lambda, maxFE - FE);
            Xr  = zeros(n, D);
            sig = zeros(n, 1);
            % sigma is mutated before the object vector, Fig. 1 lines 9-10
            for l = 1:n
                sig(l) = sr * exp(tau * randn);
                Xr(l, :) = yr + sig(l) * (w .* randn(1, D));
            end
            bad = ~isfinite(Xr);
            if any(bad(:))
                draw = lb + rand(n, D) .* w;
                Xr(bad) = draw(bad);
            end
            X = min(max(Xr, lb), ub);

            [fit, FE] = calculate_fitness(X', problem, FE);
            fit = fit(:);

            for k = 1:n
                if fit(k) < bsf || (isnan(bsf) && ~isnan(fit(k)))
                    bsf  = fit(k);
                    bsfx = X(k, :);
                end
                ec = FE - n + k;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, X, fit, population_history, fitness_history, history_index, maxFE);
            end

            % A truncated final generation cannot drive the update
            if n < lambda
                break;
            end

            % MATLAB's sort puts NaN last, as the release's SortPop
            [~, order] = sort(fit);
            Y = X(order(1:mu), :);
            S = sig(order(1:mu));

            if S(1) < sigma_min
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

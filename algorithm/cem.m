% ----------------------------------------------------------------------- %
% Cross-Entropy Method for continuous optimization (CEM)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 100*D                       % Samples per iteration
%   N_el = 10 (D < 50), 20 (D >= 50)  % Elite samples that refit the model
%   alpha = 0.8                     % Fixed smoothing of the mean
%   beta = 0.7, q = 5 (D < 50), 6 (D >= 50)  % Dynamic smoothing of sigma, Eq. (13)
%   mu0 ~ U(lb, ub), sigma0 = 5*(ub - lb)    % Initial sampling distribution
%   eps = 2.5e-4                    % Restart once an elite std is below it, box widths
%
% Algorithm Concept:
%   - Samples x ~ N(mu, diag(sigma.^2)) with independent coordinates, truncated
%     to the box by acceptance-rejection
%   - The elite, the N_el best samples, gives mu_el and sigma_el, their mean and
%     standard deviation (the maximum-likelihood update of the CE program)
%   - Smoothing: mu <- alpha*mu_el + (1-alpha)*mu, sigma <- b_t*sigma_el +
%     (1-b_t)*sigma, b_t = beta - beta*(1-1/t)^q, so sigma shrinks polynomially
%   - Independent restarts once one elite standard deviation has collapsed
%
% Reference:
% Dirk P. Kroese, Sergey Porotsky, Reuven Y. Rubinstein,
% The Cross-Entropy Method for Continuous Multi-Extremal Optimization,
% Methodology and Computing in Applied Probability 8 (3) (2006) 383-407.
% https://doi.org/10.1007/s11009-006-9753-0
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper's own MATLAB program (Appendix B, dynamic smoothing)
% with the Sec. 4 settings; bit-identical to it while the box is inactive. Kept
% as released: std is the 1/(N_el-1) sample std, not the 1/N_el of Eq. (11), and
% 'while sigma > eps' ends a run when ANY elite std reaches eps, where Algorithm
% 2 says max_j (Table 5's unconverged final mean is consistent with the code).
% eps is Appendix B's 1e-3 relative to its initial-mean interval [-2, 2]; the
% paper fixes nothing for D > 100, which keeps the 50..100 setting. The box:
% acceptance-rejection of Sec. 3.3.1 per coordinate (exact for independent
% coordinates), clamping a coordinate still rejected after 1000 redraws. A run
% restarts with new mu0, sigma0 and t = 0 until maxFe.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cem(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    w     = ub - lb;

    N     = 100 * D;
    alpha = 0.8;
    beta  = 0.7;
    if D < 50
        Nel = 10;
        q   = 5;
    else
        Nel = 20;
        q   = 6;
    end
    tol        = 1e-3 / 4 * w;               % Appendix B: eps = 1e-3 on a mu0 range of width 4
    max_redraw = 1000;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        mu    = lb + rand(1, D) .* w;
        sigma = 5 * w;
        mu_last    = mu;
        sigma_last = sigma;
        t = 0;

        while FE < maxFE
            % Appendix B order: smooth the previous elite fit, then sample
            t  = t + 1;
            mu = alpha * mu + (1 - alpha) * mu_last;
            B_mod = beta - beta * (1 - 1 / t) ^ q;
            sigma = B_mod * sigma + (1 - B_mod) * sigma_last;

            n = min(N, maxFE - FE);
            X = mu + randn(n, D) .* sigma;
            out = X < lb | X > ub;
            tries = 0;
            while any(out(:)) && tries < max_redraw
                [~, ci] = find(out);
                X(out) = mu(ci)' + randn(numel(ci), 1) .* sigma(ci)';
                out = X < lb | X > ub;
                tries = tries + 1;
            end
            if any(out(:))
                X = min(max(X, lb), ub);
            end

            [fit, FE] = calculate_fitness(X', problem, FE);
            fit = fit(:);

            for k = 1:n
                if isempty(bsfx) || fit(k) < bsf || (isnan(bsf) && ~isnan(fit(k)))
                    bsf  = fit(k);
                    bsfx = X(k, :);
                end
                ec = FE - n + k;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, X, fit, population_history, fitness_history, history_index, maxFE);
            end

            % A truncated final iteration cannot drive the update
            if n < N
                break;
            end

            % MATLAB's sort puts NaN last, so NaN samples join the elite only if they must
            [~, I_sort] = sort(fit);
            mu_last    = mu;
            sigma_last = sigma;
            Xel   = X(I_sort(1:Nel), :);
            mu    = mean(Xel, 1);
            sigma = std(Xel, 0, 1);
            if ~all(sigma > tol)
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

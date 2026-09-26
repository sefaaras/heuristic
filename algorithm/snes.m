% ----------------------------------------------------------------------- %
% Separable Natural Evolution Strategies (SNES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   lambda = 4 + floor(3*log(D))          % Offspring per generation
%   eta_mu = 1                            % Learning rate of the mean
%   eta_sigma = (3 + log(D))/(5*sqrt(D))  % Learning rate of the per-coordinate std
%   u = (h:-1:1)/h on the best h = floor(lambda/2), normalised  % Utilities
%   sigma0 = 0.3*(ub - lb)                % Initial per-coordinate std
%
% Algorithm Concept:
%   - Search distribution N(mu, diag(sigma.^2)): one std per coordinate, so
%     sampling and update cost O(D) per offspring
%   - Fitness shaping: linear utilities on the better half by rank, zero on
%     the worse half, so only ranks enter the update
%   - Mean moves by sigma times the utility-weighted standard samples z
%   - Each std is multiplied by exp(eta_sigma/2 * sum_k u_k*(z_k.^2 - 1)), the
%     natural gradient of the separable Gaussian in log-std coordinates
%
% Reference:
% Tom Schaul, Tobias Glasmachers, Juergen Schmidhuber,
% High dimensions and heavy tails for natural evolution strategies,
% Proceedings of the 13th Annual Conference on Genetic and Evolutionary
% Computation (GECCO 2011), 2011, pp. 845-852.
% https://doi.org/10.1145/2001576.2001692
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' snes.m (by Matt Luciw, distributed by Tom Schaul),
% whose linear utilities on the better half replace the paper's log-rank ones
% with the -1/lambda baseline. The reference maximises; ranks are reversed
% here. Its per-iteration evaluation of the mean is for plotting only and is
% dropped, so every FE goes to offspring; its fixed iteration count becomes
% the budget. Samples are clamped to the box and z re-derived from the clamped
% point (a clamped coordinate keeps its share of the step). Each std is kept in
% [eps, 0.5] box widths, so it never exceeds half the box nor underflows.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = snes(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    lambda    = 4 + floor(3 * log(D));
    eta_mu    = 1;
    eta_sigma = (3 + log(D)) / (5 * sqrt(D));

    threshold = floor(lambda / 2);
    utility = zeros(1, lambda);
    utility(1:threshold) = (threshold:-1:1) / threshold;   % best rank first
    utility = utility / sum(utility);

    sig_lo = eps * span;
    sig_hi = 0.5 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    mu    = lb + rand(1, D) .* span;
    sigma = 0.3 * span;       % the reference's unit start, scaled to the box

    bsf          = inf;
    bsf_solution = mu;

    while FE < maxFE
        n = min(lambda, maxFE - FE);
        S = randn(n, D);
        [X, S] = into_box(mu + S .* sigma, S, mu, lb, ub, span);

        [fit, FE] = calculate_fitness(X', problem, FE);
        fit = fit(:);

        for k = 1:n
            if fit(k) < bsf
                bsf = fit(k);
                bsf_solution = X(k, :);
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

        % Ascending sort puts NaN last, so a NaN fitness never earns utility
        [~, order] = sort(fit);
        U = zeros(1, lambda);
        U(order) = utility;

        mean_grad = U * S;
        var_grad  = U * (S .^ 2 - 1);

        mu    = mu + eta_mu * sigma .* mean_grad;
        sigma = sigma .* exp(eta_sigma / 2 * var_grad);
        sigma = min(max(sigma, sig_lo), sig_hi);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [X, S] = into_box(Xraw, S, m, lb, ub, span)
% Clamp to the box; a clamped coordinate keeps the share t of its step that stays inside
    bad = ~isfinite(Xraw);
    if any(bad(:))
        draw = lb + rand(size(Xraw)) .* span;
        Xraw(bad) = draw(bad);
        S(bad) = 0;
    end
    X = min(max(Xraw, lb), ub);
    hit = X ~= Xraw;
    if any(hit(:))
        M = repmat(m, size(Xraw, 1), 1);
        t = (X(hit) - M(hit)) ./ (Xraw(hit) - M(hit));
        t(~(t >= 0)) = 0;                  % the mean sits an ulp outside, or 0/0
        S(hit) = S(hit) .* min(t, 1);
    end
end

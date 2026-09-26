% ----------------------------------------------------------------------- %
% Exponential Natural Evolution Strategies (xNES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   L = 4 + 3*floor(log(D))        % Offspring per generation (reference code)
%   etax = 1                       % Learning rate of the mean
%   etaA = 0.5*min(1/D, 0.25)      % Learning rate of the factor A, scale and shape
%   u_k = max(0, log(L/2+1) - log(k)), normalised  % Utility of rank k
%   A0 = 0.3*I                     % Initial std 0.3 of the box width
%
% Algorithm Concept:
%   - Samples x + A*z with z ~ N(0, I); A is a square root of the covariance
%   - Fitness shaping: the k-th best sample gets the fixed utility u_k, the
%     worse half gets zero, so only ranks enter the update
%   - Natural gradient in the local coordinates: G = sum_k u_k*(z_k*z_k' - I)
%   - The mean moves by A times the utility-weighted z and A <- A*expm(etaA*G),
%     so scale and shape adapt together and A stays invertible by construction
%
% Reference:
% Tobias Glasmachers, Tom Schaul, Yi Sun, Daan Wierstra, Juergen Schmidhuber,
% Exponential natural evolution strategies,
% Proceedings of the 12th Annual Conference on Genetic and Evolutionary
% Computation (GECCO 2010), 2010, pp. 393-400.
% https://doi.org/10.1145/1830483.1830557
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' xnes.m (people.idsia.ch/~tom/code/xnes.m): its L,
% one rate etaA for scale and shape, no -1/L utility baseline. Its additive
% log update A += etaA*G (x + expm(A)*z) equals the paper's A*expm(etaA*G) only
% when the two commute; on a rotated ellipsoid (cond 1e6, 1e5 FE, 5 seeds, no
% box) it reached 49..3e4 at D=10 against 1e-126, so the paper's form is ported,
% as in the authors' PyBrain xnes.py. Its cputime and trace(A) stops are removed.
% Steps are in box widths. Samples are clamped, the mean takes the clamped steps
% and z = inv(A)*step is capped at the drawn |z|: uncapped it reached 1e15 and
% overflowed A in 28 of 57 CEC2020RW runs. Coordinate std <= half the box, and
% an SVD floors principal stds at eps box widths so inv(A) stays finite.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = xnes(problem)

    D     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    span  = ub - lb;

    L     = 4 + 3 * floor(log(D));
    etax  = 1;
    etaA  = 0.5 * min(1.0 / D, 0.25);
    shape = max(0.0, log(L / 2 + 1.0) - log(1:L));
    shape = shape / sum(shape);

    std_lo = eps;             % principal std floor, in box widths
    std_hi = 0.5;             % coordinate std cap: half the box, Hansen's maxdx rule

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x    = lb + rand(D, 1) .* span;
    A    = 0.3 * eye(D);      % the reference's unit start, in box widths
    invA = eye(D) / 0.3;
    weights = zeros(1, L);

    bsf          = inf;
    bsf_solution = x';

    % The reference evaluates its starting point once
    [f0, FE] = calculate_fitness(x, problem, FE);
    if f0(1) < bsf
        bsf = f0(1);
        bsf_solution = x';
    end
    curve(FE) = bsf;
    [population_history, fitness_history, history_index] = record_history( ...
        FE, x', f0(1), population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        n = min(L, maxFE - FE);
        Z = randn(D, n);
        step = A * Z;
        [X, step, moved] = into_box(x + span .* step, step, x, lb, ub, span);
        if any(moved)
            Zc = invA * step(:, moved);
            % Capped at the drawn |z|: clamping a thin, oblique distribution inflates it without bound
            Z(:, moved) = Zc .* min(1, sqrt(sum(Z(:, moved) .^ 2, 1)) ./ sqrt(sum(Zc .^ 2, 1)));
        end

        [fit, FE] = calculate_fitness(X, problem, FE);
        fit = fit(:)';

        popRows = X';
        for k = 1:n
            if fit(k) < bsf
                bsf = fit(k);
                bsf_solution = popRows(k, :);
            end
            ec = FE - n + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                ec, popRows, fit', population_history, fitness_history, history_index, maxFE);
        end

        % A truncated final generation cannot drive the update
        if n < L
            break;
        end

        [~, idx] = sort(fit);
        weights(idx) = shape;

        G  = (repmat(weights, D, 1) .* Z) * Z' - sum(weights) * eye(D);
        dx = etax * (step * weights');     % A*z for unclamped samples, the clamped step otherwise
        x  = x + span .* dx;

        % expm of the symmetric G through its eigendecomposition
        [Q, Eg] = eig((G + G') / 2);
        g = real(diag(Eg));
        Q = real(Q);
        A    = A * (Q * diag(exp(etaA * g)) * Q');
        invA = (Q * diag(exp(-etaA * g)) * Q') * invA;

        rmax = max(sqrt(sum(A .^ 2, 2)));
        if rmax > std_hi
            A    = A * (std_hi / rmax);
            invA = invA * (rmax / std_hi);
        end
        if ~all(isfinite(A(:)))
            A    = 0.3 * eye(D);
            invA = eye(D) / 0.3;
        elseif ~all(isfinite(invA(:))) || max(sqrt(sum(invA .^ 2, 2))) > 1 / std_lo
            [U, S, W] = svd(A);
            s = min(max(diag(S), std_lo), std_hi);
            A    = U * diag(s) * W';
            invA = W * diag(1 ./ s) * U';
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [X, step, moved] = into_box(Xraw, step, m, lb, ub, span)
% Clamp to the box; a clamped coordinate keeps the share t of its step that stays inside
    bad = ~isfinite(Xraw);
    if any(bad(:))
        draw = lb + rand(size(Xraw)) .* span;
        Xraw(bad) = draw(bad);
        step(bad) = 0;
    end
    X = min(max(Xraw, lb), ub);
    hit = X ~= Xraw;
    moved = any(hit | bad, 1);
    if any(hit(:))
        M = repmat(m, 1, size(Xraw, 2));
        t = (X(hit) - M(hit)) ./ (Xraw(hit) - M(hit));
        t(~(t >= 0)) = 0;                  % the mean sits an ulp outside, or 0/0
        step(hit) = step(hit) .* min(t, 1);
    end
end

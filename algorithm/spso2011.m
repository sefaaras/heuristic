% ----------------------------------------------------------------------- %
% Standard Particle Swarm Optimisation 2011 (SPSO-2011)
% CEC 2013 competition entry (baseline)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                      % Swarm size, the release's default
%   w = 1/(2*ln 2)              % Inertia weight, about 0.721
%   c = 0.5 + ln 2              % Acceleration towards p and l, about 1.193
%   K = 3                       % Informants per particle: link probability 1-(1-1/N)^K
%   radius = U(0,1)*|G - x|     % Non-uniform radius in the hypersphere (release unif = 0)
%   bounce = -0.5               % Velocity factor on a coordinate clamped to the box
%
% Algorithm Concept:
%   - Random topology: every particle informs itself and, with probability
%     1-(1-1/N)^K, each other particle; l is the best pbest among its informants
%   - Centre of gravity G of x, p' = x + c(p - x) and l' = x + c(l - x); when the
%     particle is its own best informant, G is the midpoint of x and p'
%   - The new point is drawn in the hypersphere H(G, |G - x|), which makes the
%     move rotation invariant; v = w*v + x' - x and x = x + v
%   - A coordinate that leaves the box is put on the bound and its velocity is
%     reversed and halved
%   - pbests are updated synchronously once the whole swarm has moved
%
% Reference:
% Mauricio Zambrano-Bigiarini, Maurice Clerc, Rodrigo Rojas,
% Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future
% PSO improvements,
% 2013 IEEE Congress on Evolutionary Computation (2013) 2337-2344.
% https://doi.org/10.1109/CEC.2013.6557848
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the Omran/Clerc MATLAB release (SPSO2011_matlab.zip from
% particleswarm.info: SPSO2011.m, alea_sphere.m). Its stagnation flag is never
% cleared, so the topology is redrawn EVERY iteration, not only after one without
% improvement as the paper and the C code describe; kept as released.
% The release searches [0,1]^D when normalize = 1 and recommends that whenever the
% box is not a hypercube, so the switch is set from the box: off on the CEC
% suites, on for CEC2020RW's per-dimension ranges.
% Its known-optimum stop (err) is removed; FE >= maxFe is the only terminator.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = spso2011(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N = 40;
    w = 1 / (2 * log(2));
    c = 0.5 + log(2);
    K = 3;
    p_link = 1 - (1 - 1 / N)^K;

    normalize = any(span ~= span(1));
    if normalize
        xMin = zeros(1, D);
        xMax = ones(1, D);
    else
        xMin = lb;
        xMax = ub;
    end
    XMIN = repmat(xMin, N, 1);
    XMAX = repmat(xMax, N, 1);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = XMIN + rand(N, D) .* (XMAX - XMIN);
    % U(xMin - x, xMax - x) per coordinate, as alea() draws it
    v = (XMIN - x) + rand(N, D) .* (XMAX - XMIN);

    XE = to_problem(x, normalize, lb, ub, span);
    [f, FE] = calculate_fitness(XE', problem, FE);
    f = f(:);

    bsf          = inf;
    bsf_solution = XE(1, :);
    for i = 1:N
        if f(i) < bsf
            bsf          = f(i);
            bsf_solution = XE(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, XE, f, population_history, fitness_history, history_index, maxFE);
        end
    end

    p_x = x;
    p_f = f;

    while FE < maxFE
        % Column i lists the informants of particle i
        L = rand(N, N) < p_link;
        L(1:N+1:end) = true;

        % Best informant; the release keeps the LAST index among ties (p_f(s) <= MIN)
        PF = repmat(p_f, 1, N);
        PF(~L) = NaN;
        [mv, gr] = min(flipud(PF), [], 1);
        g = N + 1 - gr(:);
        nan_all = isnan(mv(:));
        g(nan_all) = find(nan_all);

        n  = min(N, maxFE - FE);
        mi = 1:n;
        xi = x(mi, :);

        pp = xi + c * (p_x(mi, :) - xi);
        pl = xi + c * (p_x(g(mi), :) - xi);
        G  = (1 / 3) * (xi + pp + pl);
        own = g(mi) == mi(:);
        G(own, :) = 0.5 * (xi(own, :) + pp(own, :));

        rad = sqrt(sum((G - xi).^2, 2));
        dir = randn(n, D);
        dir = dir ./ sqrt(sum(dir.^2, 2));
        xp  = G + (rand(n, 1) .* rad) .* dir;

        vi = w * v(mi, :) + xp - xi;
        xi = xi + vi;

        xmax_i = XMAX(mi, :);
        xmin_i = XMIN(mi, :);
        hi = xi > xmax_i;
        xi(hi) = xmax_i(hi);
        vi(hi) = -0.5 * vi(hi);
        lo = xi < xmin_i;
        xi(lo) = xmin_i(lo);
        vi(lo) = -0.5 * vi(lo);

        bad = ~isfinite(xi) | ~isfinite(vi);
        if any(bad(:))
            XR = xmin_i + rand(n, D) .* (xmax_i - xmin_i);
            xi(bad) = XR(bad);
            vi(bad) = 0;
        end

        x(mi, :) = xi;
        v(mi, :) = vi;
        XE(mi, :) = to_problem(xi, normalize, lb, ub, span);

        [fv, FE] = calculate_fitness(XE(mi, :)', problem, FE);
        f(mi) = fv(:);

        for k = 1:n
            if f(k) < bsf
                bsf          = f(k);
                bsf_solution = XE(k, :);
            end
            ec = FE - n + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, XE, f, population_history, fitness_history, history_index, maxFE);
            end
        end

        % A NaN pbest could never be replaced by `f <= p_f`, so it is overwritten
        upd = f <= p_f | isnan(p_f);
        p_x(upd, :) = x(upd, :);
        p_f(upd)    = f(upd);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Problem-space coordinates; the clamp only absorbs rounding of lb + x.*span
function xe = to_problem(x, normalize, lb, ub, span)
    if normalize
        xe = min(max(lb + x .* span, lb), ub);
    else
        xe = x;
    end
end

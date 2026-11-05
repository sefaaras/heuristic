% ----------------------------------------------------------------------- %
% Hybrid Particle Swarm Optimization and Gravitational Search Algorithm (PSOGSA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                      % Swarm size, the release's default
%   G0 = 1                      % Initial gravitational constant
%   G = G0*exp(-23*t/T)         % t = iteration, T = planned iterations ceil(maxFe/N)
%   C1 = 0.5                    % Weight of the gravitational acceleration
%   C2 = 1.5                    % Weight of the gbest attraction
%   w = U(0,1)                  % Inertia, redrawn per agent and dimension
%   v0 = 0.3*N(0,1)             % Initial velocity, in absolute units
%
% Algorithm Concept:
%   - Masses from fitness, m_i = (f_i - 0.99*f_worst)/(f_best - f_worst), then
%     rescaled one agent at a time against the running sum so they total about 5
%   - Per-dimension gravity: every agent j pulls agent i with rand*G*M_i*M_j
%     towards x_j's coordinate, whatever the distance between them
%   - PSO-style velocity, v = w*v + C1*rand*a + C2*rand*(gbest - x), with a the
%     force divided by the agent's own mass; x = x + v
%   - A coordinate that leaves the box is put on the bound before evaluation
%
% Reference:
% Seyedali Mirjalili, Siti Zaiton Mohd Hashim,
% A new hybrid PSOGSA algorithm for function optimization,
% 2010 International Conference on Computer and Information Application
% (ICCIA 2010) 374-377.
% https://doi.org/10.1109/ICCIA.2010.6141614
% Components:
%   PSO - James Kennedy, Russell Eberhart, Particle swarm optimization,
%     Proceedings of ICNN'95 - International Conference on Neural Networks
%     (1995) 1942-1948, https://doi.org/10.1109/ICNN.1995.488968
%   GSA - Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A
%     gravitational search algorithm, Information Sciences 179 (2009)
%     2232-2248, https://doi.org/10.1016/j.ins.2009.03.004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Mirjalili's release (FX 35939, PSOGSA source code v3.0, PSOGSA.m);
% seeded runs reproduce its evaluated fitness sequence bit for bit. As released,
% and unlike the paper: alpha = 23 (paper 20), the force uses the per-dimension
% |x_kj - x_ij| as distance so only its sign survives, the masses are rescaled to
% 5 inside a loop that re-sums the partly rescaled vector, and v0 = 0.3*randn
% ignores the box; all kept. Harness guards that never fire on a finite spread:
% non-finite fitness is ranked as the worst (+Inf, NaN) or best (-Inf) finite one,
% a zero spread or a non-finite rescale gives equal masses (the release divides
% by zero there), and a NaN coordinate is redrawn in the box.
% The last generation evaluates only the agents the budget still pays for.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = psogsa(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    n  = 30;
    G0 = 1;
    C1 = 0.5;
    C2 = 1.5;
    iteration = ceil(maxFE / n);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Column by column, the same stream order as the release's per-dimension loop
    X = rand(n, D) .* (ub - lb) + lb;
    V = 0.3 * randn(n, D);

    gBest      = zeros(1, D);
    gBestScore = inf;
    fit  = zeros(n, 1);
    Xrec = X;
    frec = inf(n, 1);

    iter = 0;
    while FE < maxFE
        G = G0 * exp(-23 * iter / iteration);  % Eq. (4) with the release's 23
        iter = iter + 1;

        X = min(max(X, lb), ub);
        nanx = isnan(X);
        if any(nanx(:))
            XR = lb + rand(n, D) .* (ub - lb);
            X(nanx) = XR(nanx);
        end

        m = min(n, maxFE - FE);
        [fv, FE] = calculate_fitness(X(1:m, :)', problem, FE);
        fit(1:m) = fv(:);
        Xrec(1:m, :) = X(1:m, :);
        frec(1:m)    = fit(1:m);

        for k = 1:m
            if gBestScore > fit(k)
                gBestScore = fit(k);
                gBest      = X(k, :);
            end
            ec = FE - m + k;
            curve(ec) = gBestScore;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, history_index, maxFE);
        end
        if FE >= maxFE
            break;
        end

        mass = gravity_mass(fit);

        % Mask(k,j,i): the release adds a force term (and draws rand) only where x_kj ~= x_ij
        Xk   = repmat(X, 1, 1, n);
        Xi   = permute(repmat(X, 1, 1, n), [3 2 1]);
        Mask = Xk ~= Xi;
        R    = zeros(n, D, n);
        R(Mask) = rand(nnz(Mask), 1);
        dX   = Xk - Xi;
        mk   = repmat(mass(:), 1, D, n);
        mi   = repmat(reshape(mass(:), 1, 1, n), n, D, 1);
        T    = ((((R * G) .* mk) .* mi) .* dX) ./ abs(dX);
        T(~Mask) = 0;
        % Eq. (3) summed over k in the release's order
        force = zeros(1, D, n);
        for k = 1:n
            force = force + T(k, :, :);
        end
        force = permute(force, [3 2 1]);

        % Eq. (6); an agent of zero mass keeps zero acceleration
        acc = zeros(n, D);
        nz  = mass(:) ~= 0;
        acc(nz, :) = force(nz, :) ./ mass(nz);

        % Eq. (9): three draws per (agent, dimension), agent-major as in the release
        R3 = rand(3, D, n);
        r1 = permute(R3(1, :, :), [3 2 1]);
        r2 = permute(R3(2, :, :), [3 2 1]);
        r3 = permute(R3(3, :, :), [3 2 1]);
        V  = r1 .* V + (C1 * r2) .* acc + (C2 * r3) .* (gBest - X);

        X = X + V;  % Eq. (10)
    end

    curve(min(max(FE, 1), maxFE):end) = gBestScore;

    best_fitness  = gBestScore;
    best_solution = gBest;
end

% Release masses; the guards only replace what would be Inf/NaN there
function mass = gravity_mass(fit)
    n = numel(fit);
    ok = isfinite(fit);
    if ~any(ok)
        mass = ones(n, 1) * (5 / n);
        return;
    end
    best  = min(fit(ok));
    worst = max(fit(ok));
    ranked = fit;
    ranked(isnan(fit) | fit == inf) = worst;
    ranked(fit == -inf) = best;
    if best == worst
        mass = ones(n, 1) * (5 / n);
        return;
    end
    mass = (ranked - 0.99 * worst) / (best - worst);
    % Sequential rescale exactly as released: sum(mass) already holds the rescaled entries
    for i = 1:n
        mass(i) = mass(i) * 5 / sum(mass);
    end
    if any(~isfinite(mass))
        mass = ones(n, 1) * (5 / n);
    end
end

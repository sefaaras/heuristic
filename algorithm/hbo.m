% ----------------------------------------------------------------------- %
% Heavy Ball Optimizer (HBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50    % Population size (balls)
%
% Algorithm Concept:
%   - Heavy Ball Search Rule (HBSR): a gradient-like operator GSR is built
%     from the current, previous, best and a random position; the Lipschitz
%     estimates L and mu give the Polyak step alpha and momentum beta, and
%     the adaptive momentum term HBM carries the previous displacement
%   - Random Heavy Ball Strategy (RHBS): three branches selected by the mean
%     positions of the current and previous swarms, mixing Brownian motion
%     with differential steps
%   - Accelerated Convergence Mechanism (ACM): a rational (secant-like)
%     acceleration built from the HBSR and RHBS candidates
%
% Reference:
% Mohammed Jameel, Ahmed R. El-Saeed, Anis Elgabli et al.,
% Heavy Ball Optimizer: A Momentum-Driven Metaheuristic with Superior
% Scalability and Real-World Applications,
% Communications in Nonlinear Science and Numerical Simulation (2026) 110465.
% https://doi.org/10.1016/j.cnsns.2026.110465
%
% Implementation Note:
%   The RHBS candidate X_EBSR is clamped to the box where it is built, as the
%   ACM candidate Xnew already was. Unclamped it was evaluated outside the box,
%   which the numeric CEC mex accepts, and the reported solution was stored
%   clamped while its fitness came from the unclamped point.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hbo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N        = 50;
    max_iter = max(1, ceil(maxFE / (2 + 2 * N)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation phase
    X    = initialization(N, dim, ub, lb);
    GSR  = zeros(N, dim);
    GSRP = zeros(N, dim);
    Xnew = zeros(N, dim);

    [Fitness, FE] = calculate_fitness(X', problem, FE);
    Fitness = Fitness(:);

    [~, Ind] = sort(Fitness);
    f_best = Fitness(Ind(1));
    Xbest  = X(Ind(1), :);
    bsf    = f_best;
    bsx    = Xbest;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Independent initialisation for the previous positions (t-1)
    Xprev = initialization(N, dim, ub, lb);
    [FitnessP, FE] = calculate_fitness(Xprev', problem, FE);
    FitnessP = FitnessP(:);
    [~, Ind] = sort(FitnessP);
    XprevBest = Xprev(Ind(1), :);

    [mp, ip] = min(FitnessP);
    if mp < bsf
        bsf = mp;
        bsx = Xprev(ip, :);
    end
    for k = 1:N
        ec = FE - N + k;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, Fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main optimisation loop
    iter = 0;
    while iter < max_iter && FE < maxFE
        X_M     = mean(X);
        Xprev_M = mean(Xprev);
        [F_XM, FE]     = calculate_fitness(X_M', problem, FE);
        if F_XM < bsf
            bsf = F_XM;
            bsx = X_M;
        end
        [bsf, curve, population_history, fitness_history, history_index] = ...
            stamp(FE, maxFE, F_XM, bsf, curve, X, Fitness, population_history, ...
                  fitness_history, history_index);
        if FE >= maxFE, break; end
        [F_XprevM, FE] = calculate_fitness(Xprev_M', problem, FE);
        if F_XprevM < bsf
            bsf = F_XprevM;
            bsx = Xprev_M;
        end
        [bsf, curve, population_history, fitness_history, history_index] = ...
            stamp(FE, maxFE, F_XprevM, bsf, curve, X, Fitness, population_history, ...
                  fitness_history, history_index);

        for i = 1:N
            if FE >= maxFE, break; end

            % Heavy Ball Search Rule (HBSR)
            if iter == 0
                GSRP(i, :) = initialization(1, dim, ub, lb);
            else
                GSRP(i, :) = GSR(i, :);
            end

            k = Xprev(i, :);
            y1 = randi(dim);
            k(y1) = rand * k(y1);
            Xprev(i, :) = k;
            XR = initialization(1, dim, ub, lb);
            GSR(i, :) = gradientOperator(X(i, :), Xbest, Xprev(i, :), XR);
            gamma_c = 1 + rand;
            h = (GSR(i, :) - rand .* GSRP(i, :));
            DeltXX = (X(i, :) - rand .* Xprev(i, :));
            L  = gamma_c * (norm(h) / (norm(DeltXX) + eps));
            mu = norm(h - L .* DeltXX) / (norm(DeltXX) + eps);
            alpha = (rand * (sqrt(L) - sqrt(mu)) .^ 2);
            beta  = (rand * (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)) + eps) ^ 2;

            HBM = beta .* (rand .* X(i, :) - Xprev(i, :));
            X_HBSR = X(i, :) - alpha .* (2 .* GSR(i, :) - rand .* GSRP(i, :)) + HBM;   % Eq. (5)

            % Random Heavy Ball Strategy (RHBS)
            omega1 = -1 + 2 * rand();
            omega2 = -0.5 + rand();
            omega3 = 0.5 * rand;

            a = randi(N);
            b = randi(N);
            while a == i || b == i || a == b
                a = randi(N);
                b = randi(N);
            end
            RB = Brownian(dim);
            if F_XprevM < F_XM                                                    % Eq. (21)
                X_p = XprevBest + omega1 .* (X_HBSR - rand .* Xprev_M) + rand .* (X(a, :) - X(b, :));
                X_EBSR = X_p + omega1 .* (XprevBest - rand .* Xprev(i, :)) + omega2 .* RB .* (Xprev_M - X(a, :));
            else
                if Fitness(i) < F_XM
                    X_c = Xbest + omega1 .* (X_HBSR - rand .* X_M) + rand .* (X(a, :) - X(b, :));
                    X_EBSR = X_c + omega1 .* (Xbest - X(i, :)) + omega2 .* RB .* (X_M - X(a, :));
                else
                    X_EBSR = Xbest - rand * X_M + omega3 .* (X(i, :) - X(a, :)) + omega2 .* (X(i, :) - X(b, :));
                end
            end
            X_EBSR = min(max(X_EBSR, lb), ub);

            % Accelerated Convergence Mechanism (ACM)
            if rand < rand
                phi1 = rand;
            elseif rand < rand
                phi1 = RB(1);
            else
                phi1 = 0.05 * levy(1, 1, 1.5);
            end

            if rand < rand
                phi2 = rand;
            else
                phi2 = 1;
            end
            rho1 = 0.5 - 0.5 * rand();

            [f_ebsr, FE] = calculate_fitness(X_EBSR', problem, FE);
            if f_ebsr < bsf
                bsf = f_ebsr;
                bsx = X_EBSR;
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, f_ebsr, bsf, curve, X, Fitness, population_history, ...
                      fitness_history, history_index);

            if f_best < f_ebsr
                S3 = phi2 .* X_HBSR - rand .* X_M; S2 = phi1 .* X_EBSR; S1 = Xbest;
                X_ACM = S1 - rand .* (S2 - rho1 .* S1) .^ 2 ./ ...
                        (rand .* S3 - 2 .* rand(1, dim) .* S2 + rand(1, dim) .* S1 + eps);
            else
                S3 = (phi2 .* X_HBSR - rand .* X_M); S2 = phi1 .* Xbest; S1 = X_EBSR;
                X_ACM = S1 - rand .* (S2 - rho1 .* S1) .^ 2 ./ ...
                        (rand .* S3 - 2 .* rand(1, dim) .* S2 + rand(1, dim) .* S1 + eps);
            end

            if rand < 0.5                                                          % Eq. (27)
                Xnew(i, :) = X_ACM;
            else
                if i == N
                    Xnew(i, :) = (lb + rand * (ub - lb));
                else
                    if mod(iter, 2) == 0
                        RR = 0.05 * levy(1, dim, 1.5);
                    else
                        RR = zeros(1, dim);
                    end
                    omega22 = -1 + 2 * rand();
                    a1 = randperm(N, 1);
                    b1 = randperm(N, 1);
                    c1 = randperm(N, 1);
                    XX = (X(a1, :) + X(b1, :) + X(c1, :)) ./ 3;
                    Xnew(i, :) = X(i, :) + rand .* (Xbest - XX) + omega22 .* (Xbest - X_M) + RR;
                end
            end

            Xnew(i, :) = min(max(Xnew(i, :), lb), ub);

            Xprev(i, :) = X(i, :);      % current becomes previous
            XprevBest   = Xbest;        % current best becomes previous best

            if FE >= maxFE, break; end
            [Xnew_Cost, FE] = calculate_fitness(Xnew(i, :)', problem, FE);

            if Xnew_Cost < Fitness(i)
                X(i, :)    = Xnew(i, :);
                Fitness(i) = Xnew_Cost;
                if Fitness(i) < f_best
                    f_best = Fitness(i);
                    Xbest  = X(i, :);
                end
            end
            if Xnew_Cost < bsf
                bsf = Xnew_Cost;
                bsx = Xnew(i, :);
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Xnew_Cost, bsf, curve, X, Fitness, population_history, ...
                      fitness_history, history_index);
        end

        iter = iter + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Gradient-like search-rule operator
function GSR = gradientOperator(X, Xbest, Xprev, XR)
    dim = size(X, 2);
    S0 = rand(1, dim) .* (X - Xbest) ./ (((Xprev - rand .* X) .* (Xprev - rand .* Xbest)) + eps);
    S1 = rand(1, dim) .* (2 .* X - Xprev - Xbest) ./ (((X - rand .* Xprev) .* (X - rand .* Xbest)) + eps);
    S2 = rand(1, dim) .* (X - Xprev) ./ (((Xbest - rand .* Xprev) .* (Xbest - rand .* X)) + eps);
    Y0 = rand(1, dim) .* (Xbest - rand .* Xprev);
    Y1 = rand(1, dim) .* (Xbest - rand .* X);
    Y2 = rand(1, dim) .* (Xbest - rand .* XR);
    GSR = randn .* (Y0 .* S0 + Y1 .* S1 + Y2 .* S2);
end

% Brownian increment
function o = Brownian(dim)
    T = 1;
    r = T / dim;
    dw = sqrt(r) * randn(1, dim);
    o = cumsum(dw);
end

% Levy random numbers
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
    sigma_u = (num / den) ^ (1 / beta);
    u = random('Normal', 0, sigma_u, n, m);
    v = random('Normal', 0, 1, n, m);
    z = u ./ (abs(v) .^ (1 / beta));
end

% Curve / history stamp for a single evaluation
function [bsf, curve, ph, fh, hi] = stamp(FE, maxFE, f, bsf, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

% Initialization
function X = initialization(nP, dim, ub, lb)
    X = zeros(nP, dim);
    for i = 1:dim
        X(:, i) = rand(nP, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

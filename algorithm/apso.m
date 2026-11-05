% ----------------------------------------------------------------------- %
% Adaptive Particle Swarm Optimization (APSO)
% Not X.-S. Yang's Accelerated PSO, which shares the acronym
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20                      % Swarm size
%   w = 1/(1 + 1.5*exp(-2.6 f)) % Inertia from the evolutionary factor f, in [0.4, 0.9]
%   c1, c2 = 2.0 -> adaptive    % Clamped to [1.5, 2.5], sum renormalised to 4 when above
%   delta = U(0.05, 0.1)        % Acceleration rate per generation (0.5*delta when "slight")
%   sigma = 1.0 -> 0.1 (linear) % Elitist learning rate, SD of the gbest perturbation
%   Vmax = 0.2*(ub - lb)        % Velocity clamp, per dimension
%
% Algorithm Concept:
%   - Global-best PSO; each generation first estimates the evolutionary state from
%     f = (d_g - d_min)/(d_max - d_min), d_i the mean distance of x_i to the others
%   - Fuzzy memberships classify f as exploration, exploitation, convergence or
%     jumping out; an overlap is resolved by the S1->S2->S3->S4 rule base
%   - c1/c2 move by delta per state (S1 +/-, S2 +/- half, S3 +/+ half, S4 -/+)
%   - In the convergence state, elitist learning perturbs one random dimension of
%     gbest by (ub - lb)*N(0, sigma^2); a loser replaces the worst particle
%   - Particles move one at a time; one outside the box is not evaluated
%
% Reference:
% Zhi-Hui Zhan, Jun Zhang, Yun Li, Henry Shu-Hung Chung,
% Adaptive Particle Swarm Optimization,
% IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39(6)
% (2009) 1362-1381.
% https://doi.org/10.1109/TSMCB.2009.2015956
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Eqs. 1, 7-14, Table II, Figs. 6-8); the authors released no code.
% Choices it leaves open: S1 before the first estimate; an overlap goes to the
% previous state if it is one of the two, else to the one nearest it on the cycle
% (this gives every case the text spells out); one delta for both coefficients;
% d_g of the particle whose pbest is gbest; the ELS loser replaces the worst
% current fitness and updates that pbest if better; v0 ~ U(-Vmax, Vmax); sigma on
% FE/maxFe for g/G. Harness guard: d_max = d_min (collapsed swarm) gives f = 0.
% Paper Table VI (D = 30, 2e5 FE) against 10 seeded runs here, mean paper / port:
%      Sphere 1.5e-150 / 1.3e-151   Quadric 1.0e-10 / 2.8e-10   Rosenbrock 2.84 / 3.94
%      Rastrigin 5.8e-15 / 0.10 (9 of 10 at 1e-14)   Ackley 1.1e-14 / 1.8e-14
%      Griewank 0.017 / 0.041   Schwefel f - f* 0 / 781: NOT reproduced
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = apso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N  = 20;
    c1 = 2.0;
    c2 = 2.0;
    sigma_max = 1.0;
    sigma_min = 0.1;
    vmax  = 0.2 * span;
    state = 1;
    % Table II, rows S1..S4: sign and size of the c1 and c2 steps in units of delta
    rate  = [1 -1; 0.5 -0.5; 0.5 0.5; -1 1];

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;
    v = -vmax + 2 * vmax .* rand(N, D);

    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    f = inf(N, 1);
    f(1:m) = fv(:);

    gfit = inf;
    gbest = x(1, :);
    for k = 1:m
        if f(k) < gfit
            gfit  = f(k);
            gbest = x(k, :);
        end
        curve(k) = gfit;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:m, :), f(1:m), population_history, fitness_history, history_index, maxFE);
    end

    p  = x;
    fp = f;
    % Mirror for the recorder: each particle at the position its fitness was measured
    xrec = x;
    frec = f;

    while FE < maxFE
        % Eq. (7): mean Euclidean distance of each particle to the others
        dm = sqrt(sum((permute(x, [1 3 2]) - permute(x, [3 1 2])).^2, 3));
        d  = sum(dm, 2) / (N - 1);
        [~, g] = min(fp);
        dmin = min(d);
        dmax = max(d);
        if dmax > dmin
            fevo = (d(g) - dmin) / (dmax - dmin);  % Eq. (8)
        else
            fevo = 0;
        end
        fevo = min(max(fevo, 0), 1);

        state = classify_state(fevo, state);

        % Table II and Eq. (11); "slightly" is 0.5*delta
        delta = 0.05 + 0.05 * rand;
        c1 = c1 + rate(state, 1) * delta;
        c2 = c2 + rate(state, 2) * delta;
        c1 = min(max(c1, 1.5), 2.5);
        c2 = min(max(c2, 1.5), 2.5);
        s = c1 + c2;
        if s > 4.0
            c1 = c1 / s * 4.0;  % Eq. (12)
            c2 = c2 / s * 4.0;
        end
        w = 1 / (1 + 1.5 * exp(-2.6 * fevo));  % Eq. (10)

        % Elitist learning (Fig. 7) in the convergence state
        if state == 3
            sigma = sigma_max - (sigma_max - sigma_min) * min(FE / maxFE, 1);  % Eq. (14)
            P  = gbest;
            dd = randi(D);
            P(dd) = P(dd) + span(dd) * sigma * randn;  % Eq. (13)
            P  = min(max(P, lb), ub);
            [fP, FE] = calculate_fitness(P', problem, FE);
            if fP < gfit
                gfit  = fP;
                gbest = P;
            else
                [~, wst] = max(f);
                x(wst, :) = P;
                f(wst)    = fP;
                xrec(wst, :) = P;
                frec(wst)    = fP;
                if fP < fp(wst)
                    fp(wst)   = fP;
                    p(wst, :) = P;
                end
            end
            curve(FE) = gfit;
            [population_history, fitness_history, history_index] = record_history(...
                FE, xrec, frec, population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE
                break;
            end
        end

        for i = 1:N
            r1 = rand(1, D);
            r2 = rand(1, D);
            vi = w * v(i, :) + c1 * r1 .* (p(i, :) - x(i, :)) + c2 * r2 .* (gbest - x(i, :));  % Eq. (1)
            vi = min(max(vi, -vmax), vmax);
            v(i, :) = vi;
            x(i, :) = x(i, :) + vi;  % Eq. (2)

            if all(x(i, :) >= lb & x(i, :) <= ub)
                [fi, FE] = calculate_fitness(x(i, :)', problem, FE);
                f(i) = fi;
                xrec(i, :) = x(i, :);
                frec(i)    = fi;
                % A NaN personal best could never be beaten, so it is overwritten
                if fi < fp(i) || isnan(fp(i))
                    fp(i)   = fi;
                    p(i, :) = x(i, :);
                end
                if fi < gfit
                    gfit  = fi;
                    gbest = x(i, :);
                end
                curve(FE) = gfit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xrec, frec, population_history, fitness_history, history_index, maxFE);
                if FE >= maxFE
                    break;
                end
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = gfit;

    best_fitness  = gfit;
    best_solution = gbest;
end

% Eq. (9a-d) memberships, singleton defuzzification with the state-sequence rule base
function state = classify_state(f, prev)
    mu = zeros(1, 4);
    if f > 0.4 && f <= 0.6
        mu(1) = 5 * f - 2;
    elseif f > 0.6 && f <= 0.7
        mu(1) = 1;
    elseif f > 0.7 && f <= 0.8
        mu(1) = -10 * f + 8;
    end
    if f > 0.2 && f <= 0.3
        mu(2) = 10 * f - 2;
    elseif f > 0.3 && f <= 0.4
        mu(2) = 1;
    elseif f > 0.4 && f <= 0.6
        mu(2) = -5 * f + 3;
    end
    if f <= 0.1
        mu(3) = 1;
    elseif f <= 0.3
        mu(3) = -5 * f + 1.5;
    end
    if f > 0.7 && f <= 0.9
        mu(4) = 5 * f - 3.5;
    elseif f > 0.9
        mu(4) = 1;
    end
    cand = find(mu > 0);
    if isempty(cand)
        [~, cand] = max(mu);
    end
    if isscalar(cand)
        state = cand;
    elseif any(cand == prev)
        state = prev;
    else
        % Nearest to the previous state on the cycle S1 -> S2 -> S3 -> S4 -> S1
        step = min(mod(cand - prev, 4), mod(prev - cand, 4));
        [~, k] = min(step);
        state = cand(k);
    end
end

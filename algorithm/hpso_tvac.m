% ----------------------------------------------------------------------- %
% Self-Organizing Hierarchical Particle Swarm Optimizer with Time-Varying Acceleration Coefficients (HPSO-TVAC)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                          % Swarm size
%   c1 = 2.5 -> 0.5 (linear)        % Cognitive coefficient over the iterations (TVAC)
%   c2 = 0.5 -> 2.5 (linear)        % Social coefficient over the iterations (TVAC)
%   Vmax = (ub - lb)/2              % Velocity clamp, the paper's Vmax = Xmax
%   v = Vmax -> 0.1*Vmax (linear)   % Reinitialization velocity
%
% Algorithm Concept:
%   - No momentum term: v = c1*r1.*(p_i - x) + c2*r2.*(p_g - x), then x = x + v,
%     with r1, r2 ~ U(0,1) per dimension
%   - TVAC: a strong cognitive pull early (wide search) hands over to a strong
%     social pull late (convergence on the global best)
%   - A velocity component that is exactly zero (stagnation on pbest and gbest) is
%     re-drawn as +U(0,1)*v or -U(0,1)*v with equal odds; v shrinks over the run
%   - Particles are processed one at a time: evaluated, pbest and gbest updated,
%     then moved, so later particles already follow an improved gbest
%
% Reference:
% Asanga Ratnaweera, Saman K. Halgamuge, Harry C. Watson,
% Self-Organizing Hierarchical Particle Swarm Optimizer With Time-Varying
% Acceleration Coefficients,
% IEEE Transactions on Evolutionary Computation, vol. 8, no. 3, pp. 240-255, 2004.
% https://doi.org/10.1109/TEVC.2004.826071
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Sec. III-C pseudocode, Sec. IV-V settings; the authors' own copy,
% read via the Wayback Machine). Its asymmetric initial range is a testing device, so
% x0 ~ U(lb, ub). It bounds only the velocity; positions are clamped to the box here
% (unbounded and not-evaluated-outside variants measured the same on Rastrigin).
% MAXITR = ceil(maxFe/N), one evaluation per particle per iteration. The decay of v is
% taken as linear (the paper says "decay"; its mutation step decays linearly). Lynn and
% Suganthan's HPSO_TVAC.m (P-N-Suganthan/CODES, 2017-ASOC-EPSO.zip) moves all particles
% before evaluating any and leaves positions unbounded. Not the jumping-coefficient
% HPSO-TVAC of Ghasemi et al. 2017 that MEALPY ships under this name.
% Paper Table VIII (time-varying v, 50 trials) against 20 seeded runs here (paper / here):
%      D = 10: Sphere 0.01 / 0.01 (20 of 20 hit)   Rosenbrock 12.97 / 11.74   Griewank 0.057 / 0.080
%      Rastrigin D = 10: 0.01 (all hit) / 0.45 (13 of 20); D = 30: 0.044 / 8.9 (0 of 20)
%      Rastrigin not reproduced; HPSO_TVAC.m on the same set-up: 12 of 20, and 9.4 at D = 30
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hpso_tvac(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N    = 40;
    c1i  = 2.5;
    c1f  = 0.5;
    c2i  = 0.5;
    c2f  = 2.5;
    vmax = span / 2;
    maxit = ceil(maxFE / N);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x  = lb + rand(N, D) .* span;
    p  = x;
    fp = inf(N, 1);
    gb = x(1, :);
    fg = inf;

    bsf          = inf;
    bsf_solution = x(1, :);
    % Mirror for the recorder: each particle at the position its fitness was measured
    xrec = x;
    frec = inf(N, 1);
    seen = false(N, 1);

    for it = 1:maxit
        % Eqs. (6)-(7), and the reinitialization velocity decaying Vmax -> 0.1*Vmax
        c1 = (c1f - c1i) * it / maxit + c1i;
        c2 = (c2f - c2i) * it / maxit + c2i;
        vr = vmax - 0.9 * vmax * it / maxit;

        % Positions are fixed for the whole sweep, so one batch equals the per-particle loop
        m = min(N, maxFE - FE);
        fe0 = FE;
        [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
        fv = fv(:);

        for i = 1:m
            fi = fv(i);
            xrec(i, :) = x(i, :);
            frec(i)    = fi;
            seen(i)    = true;
            if fi < fp(i)
                fp(i)   = fi;
                p(i, :) = x(i, :);
            end
            if fp(i) < fg
                fg = fp(i);
                gb = p(i, :);
            end
            if fi < bsf
                bsf          = fi;
                bsf_solution = x(i, :);
            end
            curve(fe0 + i) = bsf;
            if seen(N)
                [population_history, fitness_history, history_index] = record_history(...
                    fe0 + i, xrec, frec, population_history, fitness_history, history_index, maxFE);
            else
                [population_history, fitness_history, history_index] = record_history(...
                    fe0 + i, xrec(seen, :), frec(seen), population_history, fitness_history, history_index, maxFE);
            end

            v = c1 * rand(1, D) .* (p(i, :) - x(i, :)) + c2 * rand(1, D) .* (gb - x(i, :));
            for d = find(v == 0)
                if rand < 0.5
                    v(d) = rand * vr(d);
                else
                    v(d) = -rand * vr(d);
                end
            end
            v = sign(v) .* min(abs(v), vmax);
            x(i, :) = min(max(x(i, :) + v, lb), ub);
        end
        if FE >= maxFE
            break;
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

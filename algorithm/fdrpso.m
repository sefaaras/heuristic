% ----------------------------------------------------------------------- %
% Fitness-Distance-Ratio based Particle Swarm Optimization (FDR-PSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 10                          % Swarm size (Table II)
%   psi1, psi2, psi3 = 1, 1, 2      % Weights of the pbest, gbest and nbest terms (Table I)
%   w = 0.9 -> 0.4 (linear)         % Inertia weight over the generations, Eq. (11)
%   Vmax = 0.2*(ub - lb)            % Velocity clamp
%
% Algorithm Concept:
%   - Velocity v = w*v + psi1*r1.*(p_i - x) + psi2*r2.*(p_g - x) + psi3*r3.*(p_n - x),
%     r ~ U(0,1) per dimension; the position is clamped to the box
%   - p_n is chosen PER DIMENSION d: the pbest of the other particle j maximising
%     the fitness-distance ratio (f(x_i) - f(p_j)) / |p_jd - x_id|
%   - So each coordinate is pulled toward a nearby particle that has done better,
%     not only toward the single global best
%   - Particles move one at a time; pbest and gbest are updated at once and are
%     already seen by the next particle in the same generation
%
% Reference:
% Thanmaya Peram, Kalyan Veeramachaneni, Chilukuri K. Mohan,
% Fitness-distance-ratio based particle swarm optimization,
% Proceedings of the 2003 IEEE Swarm Intelligence Symposium (SIS'03), 2003, 174-181.
% https://doi.org/10.1109/SIS.2003.1202264
% ----------------------------------------------------------------------- %
% Implementation Note:
% Port follows the paper's pseudocode (Sec. II) with its Table I and II settings. Not
% stated there and taken from FDR.m (P-N-Suganthan/CODES, 2017-ASOC-EPSO.zip): Vmax =
% 0.2*(ub - lb), v0 ~ U(-Vmax, Vmax), ties replace the pbest, a zero distance scores 0.
% Read as misprints: Eq. (4) and the pseudocode carry no U(0,1) factors; they are
% restored, as without them Table III is missed by 5 to 28 orders of magnitude (below).
% The minimisation numerator Cost(P_j) - Cost(X_i) contradicts the paper's "higher
% fitness" criterion, so f(x_i) - f(p_j) is maximised. Eq. (11), recursive and with +0.4
% in the denominator as printed, turns w negative after three updates from 0.9; it is read
% as the linear w = (0.9 - 0.4)*(gsize - i)/gsize + 0.4 of the Shi code it cites (the
% 0.9 start is not stated). Table III, Table II settings (N = 10, 1000 generations), mean:
%      30 runs     De Jong  axis     powers   rot.ell. Rosenbr. Griewank
%      paper       2.2e-6   1.9e-18  9.4e-34  3.8e-7   4.1e-12  1.2e-16
%      here        7.0e-12  1.7e-33  2.0e-69  1.4e-7   1.4e-15  0.079 (not reproduced)
%      no U(0,1)   0.78     0.67     6.0e-6   22.9     2.9e-6   1.41
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdrpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N    = 10;
    psi  = [1 1 2];
    wmax = 0.9;
    wmin = 0.4;
    vmax = 0.2 * span;
    gsize = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;
    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    f = inf(N, 1);
    f(1:m) = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for k = 1:m
        if f(k) < bsf
            bsf          = f(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:m, :), f(1:m), population_history, fitness_history, history_index, maxFE);
    end

    v  = -vmax + 2 .* vmax .* rand(N, D);
    p  = x;
    fp = f;
    [fg, g] = min(fp);
    gb = p(g, :);

    t = 0;
    while FE < maxFE
        t = t + 1;
        % Eq. (11) with gsize the generations the budget allows after initialisation
        w = (wmax - wmin) * (gsize - min(t, gsize)) / gsize + wmin;
        for i = 1:N
            % Eq. (3) for every other particle j and dimension d at once
            num = f(i) - fp;
            dis = abs(p - x(i, :));
            fdr = num ./ dis;
            fdr(dis == 0) = 0;
            fdr(isnan(num), :) = -inf;
            fdr(isnan(fdr)) = -inf;
            fdr(i, :) = -inf;
            [~, nb] = max(fdr, [], 1);
            pn = p(sub2ind([N, D], nb, 1:D));

            vi = w * v(i, :) + psi(1) * rand(1, D) .* (p(i, :) - x(i, :)) ...
                 + psi(2) * rand(1, D) .* (gb - x(i, :)) + psi(3) * rand(1, D) .* (pn - x(i, :));
            vi = min(vmax, max(-vmax, vi));
            v(i, :) = vi;
            x(i, :) = min(ub, max(lb, x(i, :) + vi));

            [fi, FE] = calculate_fitness(x(i, :)', problem, FE);
            f(i) = fi;
            % Ties replace the pbest as in FDR.m; a NaN pbest is overwritten, a NaN never kept
            if fi <= fp(i) || isnan(fp(i))
                fp(i)   = fi;
                p(i, :) = x(i, :);
            end
            if fp(i) < fg || isnan(fg)
                fg = fp(i);
                gb = p(i, :);
            end
            if fi < bsf
                bsf          = fi;
                bsf_solution = x(i, :);
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, x, f, population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

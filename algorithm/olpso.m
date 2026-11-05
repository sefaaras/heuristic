% ----------------------------------------------------------------------- %
% Orthogonal Learning Particle Swarm Optimization (OLPSO)
% Local version OLPSO-L (ring topology), the variant the paper recommends
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                      % Swarm size
%   w = 0.9 -> 0.4 (linear)     % Inertia weight
%   c = 2.0                     % Acceleration towards the exemplar Po
%   G = 5                       % Reconstruction gap: Po rebuilt after more than G failures
%   Vmax = 0.2*(ub - lb)        % Velocity clamp, per dimension
%   M = 2^ceil(log2(D + 1))     % Rows of the two-level orthogonal array L_M(2^(M-1))
%
% Algorithm Concept:
%   - Each particle follows one exemplar, v = w*v + c*r.*(Po - x), r ~ U(0,1)
%     per dimension; Po takes every dimension from P_i or from the ring best P_l
%   - Po by orthogonal design: M array rows mix P_i and P_l per dimension, factor
%     analysis predicts one more mix, the best of the M + 1 evaluated ones is kept
%   - Po stores the source particle of each dimension, not its value, so improved
%     pbests reach the particle at once; P_l = P_i is replaced by a random P_r
%   - Particles move one at a time; one outside the box is not evaluated, and
%     G + 1 evaluated non-improving moves in a row trigger a new Po
%
% Reference:
% Zhi-Hui Zhan, Jun Zhang, Yun Li, Yu-Hui Shi,
% Orthogonal Learning Particle Swarm Optimization,
% IEEE Transactions on Evolutionary Computation 15(6) (2011) 832-847.
% https://doi.org/10.1109/TEVC.2010.2052054
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Sec. III-D/E, Fig. 2, Appendix); no author code is public, and the
% third-party ports (SakuraAyase C++, ALFRAME MATLAB) depart from it. Choices it
% leaves open: the first D array columns; P_l = best pbest of {i-1, i, i+1} at build
% time, kept by particle index until the next rebuild (re-resolved at every use it
% gave Sphere 3e-33); factor-analysis ties take P_i, an X_p tie keeps X_b; v0 ~
% U(-Vmax, Vmax) as in the authors' APSO paper; design points never update a pbest
% but count as FE and in best-so-far; w on FE/maxFe for gen/GENERATION.
% Paper Table V (D = 30, 2e5 FE, its biased init) against 10 seeded runs here:
%      Rastrigin 0 / 0   Griewank 0 / 0   Schwefel 3.8e-4 / 3.8e-4 (9 of 10 runs)
%      Ackley 4.1e-15 / 7.2e-15   Sphere 1.1e-38 / 1.8e-37   Rosenbrock 1.26 / 12.5
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = olpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N    = 40;
    c    = 2.0;
    Gap  = 5;
    vmax = 0.2 * span;
    OA   = orthogonal_array(D);

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

    p  = x;
    fp = f;
    % Mirror for the recorder: each particle at the position its fitness was measured
    xrec = x;
    frec = f;

    stag    = zeros(N, 1);
    lev     = ones(N, D);   % 1 = dimension from P_i, 2 = from the partner
    partner = (1:N)';       % particle that level-2 dimensions read their pbest from
    ring    = [[N, 1:N-1]', (1:N)', [2:N, 1]'];

    % Fig. 2: every particle builds its first exemplar after initialisation
    for i = 1:N
        if FE >= maxFE
            break;
        end
        FE0 = FE;
        [lev(i, :), partner(i), FE, xs, fs] = build_exemplar(i, p, fp, ring, OA, ...
                                                              lev(i, :), partner(i), problem, FE, maxFE);
        for k = 1:numel(fs)
            if fs(k) < bsf
                bsf          = fs(k);
                bsf_solution = xs(k, :);
            end
            curve(FE0 + k) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE0 + k, xrec, frec, population_history, fitness_history, history_index, maxFE);
        end
    end

    while FE < maxFE
        w = 0.9 - 0.5 * min(FE / maxFE, 1);
        for i = 1:N
            q  = partner(i);
            po = p(i, :);
            from_q = lev(i, :) == 2;
            po(from_q) = p(q, from_q);

            r  = rand(1, D);
            vi = w * v(i, :) + c * r .* (po - x(i, :));  % Eq. (8)
            vi = min(max(vi, -vmax), vmax);
            v(i, :) = vi;
            x(i, :) = x(i, :) + vi;

            if all(x(i, :) >= lb & x(i, :) <= ub)
                [fi, FE] = calculate_fitness(x(i, :)', problem, FE);
                xrec(i, :) = x(i, :);
                frec(i)    = fi;
                if fi < bsf
                    bsf          = fi;
                    bsf_solution = x(i, :);
                end
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xrec, frec, population_history, fitness_history, history_index, maxFE);

                % A NaN personal best could never be beaten, so it is overwritten
                if fi < fp(i) || isnan(fp(i))
                    fp(i)   = fi;
                    p(i, :) = x(i, :);
                    stag(i) = 0;
                else
                    stag(i) = stag(i) + 1;
                    if stag(i) > Gap && FE < maxFE
                        FE0 = FE;
                        [lev(i, :), partner(i), FE, xs, fs] = build_exemplar(i, p, fp, ring, OA, ...
                                                                              lev(i, :), partner(i), problem, FE, maxFE);
                        for k = 1:numel(fs)
                            if fs(k) < bsf
                                bsf          = fs(k);
                                bsf_solution = xs(k, :);
                            end
                            curve(FE0 + k) = bsf;
                            [population_history, fitness_history, history_index] = record_history(...
                                FE0 + k, xrec, frec, population_history, fitness_history, history_index, maxFE);
                        end
                        stag(i) = 0;
                    end
                end
            end
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Steps 1-6 of Sec. III-D; returns every evaluated point so the caller can book it
function [lev, partner, FE, xs, fs] = build_exemplar(i, p, fp, ring, OA, lev, partner, problem, FE, maxFE)
    N = size(p, 1);
    [M, D] = size(OA);
    [~, b] = min(fp(ring(i, :)));
    n = ring(i, b);
    if n == i || isequal(p(n, :), p(i, :))
        % Sec. III-E: P_i = P_n teaches nothing, so a random other particle stands in
        r = randi(N - 1);
        r = r + (r >= i);
        pn = p(r, :);
        new_partner = r;
    else
        pn = p(n, :);
        new_partner = n;
    end

    % Step 2: level 1 takes P_i, level 2 the partner
    Pi = repmat(p(i, :), M, 1);
    Pn = repmat(pn, M, 1);
    xs = Pi;
    xs(OA == 2) = Pn(OA == 2);

    % Step 3; a budget that ends inside the design leaves the old exemplar in place
    nt = min(M, maxFE - FE);
    [ft, FE] = calculate_fitness(xs(1:nt, :)', problem, FE);
    fs = ft(:);
    xs = xs(1:nt, :);
    if nt < M
        return;
    end
    [fb, jb] = min(fs);
    lev_b = OA(jb, :);

    % Step 4, Eq. (7): mean result per level and factor; a tie keeps P_i
    F1 = repmat(fs, 1, D);
    F2 = F1;
    F1(OA ~= 1) = 0;
    F2(OA ~= 2) = 0;
    S1 = sum(F1, 1) ./ sum(OA == 1, 1);
    S2 = sum(F2, 1) ./ sum(OA == 2, 1);
    lev_p = 1 + (S2 < S1);

    lev     = lev_b;
    partner = new_partner;
    if FE >= maxFE
        return;
    end

    % Steps 5-6: evaluate the predicted combination X_p and keep the better level set
    xp = p(i, :);
    xp(lev_p == 2) = pn(lev_p == 2);
    [fpred, FE] = calculate_fitness(xp', problem, FE);
    xs = [xs; xp];
    fs = [fs; fpred(1)];
    if fpred(1) < fb
        lev = lev_p;
    end
end

% Appendix, Eqs. (A1)-(A3): two-level array L_M(2^(M-1)), first D columns, levels 1/2
function L = orthogonal_array(D)
    u = ceil(log2(D + 1));
    M = 2^u;
    L = zeros(M, M - 1);
    a = (1:M)';
    for k = 1:u
        L(:, 2^(k - 1)) = mod(floor((a - 1) / 2^(u - k)), 2);
    end
    for k = 2:u
        bcol = 2^(k - 1);
        for s = 1:bcol - 1
            L(:, bcol + s) = mod(L(:, s) + L(:, bcol), 2);
        end
    end
    L = L(:, 1:D) + 1;
end

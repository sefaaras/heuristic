% ----------------------------------------------------------------------- %
% Cooperative Particle Swarm Optimizer, hybrid variant (CPSO-H_K)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   s = 10                          % Particles in every sub-swarm and in Q
%   K = min(6, D)                   % Split factor: K1 = D mod K blocks of ceil(D/K), rest floor
%   c1 = c2 = 1.49                  % Acceleration coefficients
%   w = 1.0 -> 0.72 (linear)        % Inertia weight over the first 35 % of maxFe, then 0.72
%   Vmax = (ub - lb)/2              % Velocity clamp, the paper's domain
%   exchange targets = first s/2    % Particles that may be overwritten, never a swarm's best
%
% Algorithm Concept:
%   - CPSO-S_K half: the variables are cut into K blocks, each searched by its own
%     swarm; a particle is scored by inserting its block into the context vector b
%   - b is the concatenation of the K swarm bests; a swarm best that improves is
%     written into b at once, so f(b) never rises
%   - PSO half: a plain gbest swarm Q over all D variables
%   - The halves alternate one iteration each: b overwrites a random Q particle, and
%     Q's best, cut into blocks, overwrites one random particle in every sub-swarm
%
% Reference:
% Frans van den Bergh, Andries P. Engelbrecht,
% A Cooperative Approach to Particle Swarm Optimization,
% IEEE Transactions on Evolutionary Computation, vol. 8, no. 3, pp. 225-239, 2004.
% https://doi.org/10.1109/TEVC.2004.826069
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Fig. 5, Sec. V-A) and van den Bergh's thesis (Pretoria, Sec. 4.3,
% 5.5.1), which fixes w = 1.0 -> 0.72 over the first 1000 iterations (35 % of the
% paper's 2e5 FE at s = 10, K = 6, hence 35 % of maxFe) and v0 ~ U(-Vmax, Vmax). The
% paper runs s = 10, 15, 20 with no default; 10 is used. D < 6 takes K = D. Each
% context or Q evaluation is one FE, as the paper counts: stored pbest and context
% values are compared, never re-scored. Positions are unbounded in the paper; here a
% particle outside the box is not evaluated and costs no FE (clamping froze runs:
% coordinates pinned on a bound became x = y = y^, quadric stuck at 1.5e4). Hang
% guard: after 1000 iterations with no evaluation all particles are clamped.
% TEVC Tables II-V (unrotated, D = 30, 2e5 FE, s = 10, PSO column) / 10 seeded runs here:
%      Rosenbrock 0.194 / 0.033   Ackley 9.4e-11 / 9.0e-14   Quadric 2.6e-66 / 1.6e-14   Rastrigin 1.47 / 10.5
%      Quadric and Rastrigin not reproduced; an iteration-count ramp changes nothing, and
%      re-scoring pbests in the current context (uncounted FEs) gives 1e-27 and 8.6
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cpso_h(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    s    = 10;
    K    = min(6, D);
    c1   = 1.49;
    c2   = 1.49;
    w0   = 1.0;
    w1   = 0.72;
    ramp = 0.35;
    vmax = span / 2;
    h    = floor(s / 2);
    stall_max = 1000;

    % Fig. 5: K1 = D mod K swarms of ceil(D/K) variables, then K - K1 of floor(D/K)
    K1 = mod(D, K);
    sizes = [repmat(ceil(D / K), 1, K1), repmat(floor(D / K), 1, K - K1)];
    edges = [0, cumsum(sizes)];

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % The K sub-swarms side by side: columns edges(j)+1 : edges(j+1) belong to swarm j
    X  = lb + rand(s, D) .* span;
    V  = -vmax + 2 .* vmax .* rand(s, D);
    Y  = X;
    fY = inf(s, K);
    ctx  = X(1, :);
    fctx = inf;

    QX  = lb + rand(s, D) .* span;
    QV  = -vmax + 2 .* vmax .* rand(s, D);
    QY  = QX;
    fQY = inf(s, 1);
    qg  = QX(1, :);
    fqg = inf;

    bsf          = inf;
    bsf_solution = ctx;
    % Recorder mirror: the last full vector evaluated in each slot (sub-swarm particle or Q particle)
    xrec   = zeros(s * (K + 1), D);
    frec   = inf(s * (K + 1), 1);
    filled = false(s * (K + 1), 1);
    allfilled = false;

    stall = 0;
    while FE < maxFE
        w = w0 - (w0 - w1) * min(1, FE / (ramp * maxFE));
        n_before = FE;

        for j = 1:K
            cols = edges(j) + 1:edges(j + 1);
            idx = find(all(X(:, cols) >= lb(cols) & X(:, cols) <= ub(cols), 2));
            idx = idx(1:min(numel(idx), maxFE - FE));
            if ~isempty(idx)
                B = repmat(ctx, numel(idx), 1);
                B(:, cols) = X(idx, cols);
                fv = evaluate(B, (j - 1) * s + idx);
                for q = 1:numel(idx)
                    i = idx(q);
                    if fv(q) < fY(i, j)
                        Y(i, cols) = X(i, cols);
                        fY(i, j)   = fv(q);
                    end
                    if fY(i, j) < fctx
                        ctx(cols) = Y(i, cols);
                        fctx      = fY(i, j);
                    end
                end
            end
            if FE >= maxFE
                break;
            end
            V(:, cols) = w * V(:, cols) + c1 * rand(s, sizes(j)) .* (Y(:, cols) - X(:, cols)) ...
                         + c2 * rand(s, sizes(j)) .* (ctx(cols) - X(:, cols));
            V(:, cols) = min(max(V(:, cols), -vmax(cols)), vmax(cols));
            X(:, cols) = X(:, cols) + V(:, cols);
        end
        if FE >= maxFE
            break;
        end

        k = pick(QY(1:h, :), qg);
        if k > 0
            QX(k, :) = ctx;
        end
        idx = find(all(QX >= lb & QX <= ub, 2));
        idx = idx(1:min(numel(idx), maxFE - FE));
        if ~isempty(idx)
            fq = evaluate(QX(idx, :), K * s + idx);
            for q = 1:numel(idx)
                i = idx(q);
                if fq(q) < fQY(i)
                    QY(i, :) = QX(i, :);
                    fQY(i)   = fq(q);
                end
                if fQY(i) < fqg
                    qg  = QY(i, :);
                    fqg = fQY(i);
                end
            end
        end
        if FE >= maxFE
            break;
        end
        QV = w * QV + c1 * rand(s, D) .* (QY - QX) + c2 * rand(s, D) .* (qg - QX);
        QV = min(max(QV, -vmax), vmax);
        QX = QX + QV;

        for j = 1:K
            cols = edges(j) + 1:edges(j + 1);
            k = pick(Y(1:h, cols), ctx(cols));
            if k > 0
                X(k, cols) = qg(cols);
            end
        end

        % Hang guard: after stall_max idle iterations every particle is clamped onto the box
        if FE == n_before
            stall = stall + 1;
            if stall >= stall_max
                X  = min(max(X, lb), ub);
                QX = min(max(QX, lb), ub);
                stall = 0;
            end
        else
            stall = 0;
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of Xe into recorder slots sl(1..), tracking the best per evaluation
    function f = evaluate(Xe, sl)
        fe0 = FE;
        [fr, FE] = calculate_fitness(Xe', problem, FE);
        f = fr(:);
        for r = 1:numel(f)
            xrec(sl(r), :) = Xe(r, :);
            frec(sl(r))    = f(r);
            filled(sl(r))  = true;
            if f(r) < bsf
                bsf          = f(r);
                bsf_solution = Xe(r, :);
            end
            curve(fe0 + r) = bsf;
            if allfilled
                [population_history, fitness_history, history_index] = record_history( ...
                    fe0 + r, xrec, frec, population_history, fitness_history, history_index, maxFE);
            else
                [population_history, fitness_history, history_index] = record_history( ...
                    fe0 + r, xrec(filled, :), frec(filled), population_history, fitness_history, ...
                    history_index, maxFE);
                allfilled = all(filled);
            end
        end
    end

    % Fig. 5 exchange target: k ~ U(1, s/2) among the particles whose pbest is not the swarm best
    function k = pick(Ycand, best)
        cand = find(any(Ycand ~= best, 2));
        if isempty(cand)
            k = 0;
        else
            k = cand(randi(numel(cand)));
        end
    end
end

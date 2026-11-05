% ----------------------------------------------------------------------- %
% Fireworks Algorithm (FWA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 5                          % Fireworks
%   m = 50, a = 0.04, b = 0.8      % Explosion sparks shared by fitness, each in [2, 40]
%   A_hat = 0.2*(ub - lb)          % Maximum explosion amplitude, 40 on [-100, 100]
%   m_hat = 5                      % Gaussian sparks per generation
%
% Algorithm Concept:
%   - Better fireworks get more sparks in a smaller amplitude, Eqs. (2)-(4):
%     s_i ~ y_max - f_i, bounded to [a*m, b*m]; A_i ~ f_i - y_min
%   - Explosion spark: z = round(D*rand) random dimensions all move by one
%     shared displacement h = A_i*U(-1, 1), Alg. 1
%   - Gaussian spark: z random dimensions of a random firework are multiplied
%     by one g ~ N(1, 1), Alg. 2, which pulls sparks toward the origin
%   - A coordinate leaving the box is mapped to x_min + |x| mod (x_max - x_min)
%   - The best location survives; the other n - 1 are drawn with probability
%     proportional to their summed distance to all locations, Eqs. (6)-(7)
%
% Reference:
% Ying Tan, Yuanchun Zhu,
% Fireworks Algorithm for Optimization,
% Advances in Swarm Intelligence (ICSI 2010), Lecture Notes in Computer Science,
% vol. 6145, Springer, 2010, pp. 355-364.
% https://doi.org/10.1007/978-3-642-13495-1_44
% ----------------------------------------------------------------------- %
% Implementation Note:
% No code release is reachable (the Tan group's site archived only dead AFWA/EFWA/
% dynFWA links): ported from the paper, the authors' copy on cil.pku.edu.cn. Kept as
% published, although EFWA later removed them as a centre bias: the |x| mapping and
% the multiplicative Gaussian spark. A_hat, absolute in the paper, scales to each
% range (exact on [-100, 100]); the box is the initialisation range (paper: [15, 30]
% or [30, 50]). Chosen, the paper being silent: the n - 1 drawn survivors are
% distinct and exclude the best; a non-finite spark count falls to round(a*m), a
% non-finite amplitude ratio to 1. Paper Table 3 rerun (D = 30, 1e4 FE, 20 runs, its
% init boxes; drawing with replacement or keeping the best in the draw moves neither gap):
%     Sphere, Rastrigin, Griewank, Ellipse, Cigar, Tablet, Ackley  0 (paper 0)
%     Rosenbrock 27.4 +/- 6.4 (19.4 +/- 11.9)   Schwefel 10.8 +/- 3.9 (4.35 +/- 1.48)
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fwa(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    n     = 5;
    m     = 50;
    a     = 0.04;
    b     = 0.8;
    A_hat = 0.2 * span;
    m_hat = 5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    FW  = lb + rand(n, D) .* span;
    n0  = min(n, maxFE);
    [f0, FE] = calculate_fitness(FW(1:n0, :)', problem, FE);
    FWf = inf(1, n);
    FWf(1:n0) = f0(:)';

    best_fitness  = inf;
    best_solution = FW(1, :);
    for e = 1:n0
        if FWf(e) < best_fitness
            best_fitness  = FWf(e);
            best_solution = FW(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, FW(1:n0, :), FWf(1:n0), population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE
        y_max = max(FWf);
        y_min = min(FWf);

        % Eqs. (2)-(3)
        s = m * (y_max - FWf + eps) / (sum(y_max - FWf) + eps);
        for i = 1:n
            if ~isfinite(s(i)) || s(i) < a * m
                s(i) = round(a * m);
            elseif s(i) > b * m
                s(i) = round(b * m);
            else
                s(i) = round(s(i));
            end
        end

        % Eq. (4) as a share of A_hat, which carries the per-dimension range
        r = (FWf - y_min + eps) / (sum(FWf - y_min) + eps);
        r(~isfinite(r)) = 1;

        % Explosion sparks, Alg. 1
        S = zeros(sum(s) + m_hat, D);
        e = 0;
        for i = 1:n
            for j = 1:s(i)
                e = e + 1;
                x = FW(i, :);
                z = round(D * rand);
                k = randperm(D, z);
                h = r(i) * (2 * rand - 1);
                x(k) = x(k) + h * A_hat(k);
                S(e, :) = map_into_box(x, k, lb, ub, span);
            end
        end

        % Gaussian sparks, Alg. 2
        for j = 1:m_hat
            e = e + 1;
            x = FW(randi(n), :);
            z = round(D * rand);
            k = randperm(D, z);
            g = 1 + randn;
            x(k) = x(k) * g;
            S(e, :) = map_into_box(x, k, lb, ub, span);
        end

        nS = min(e, maxFE - FE);
        Sf = evaluate(S(1:nS, :), FW, FWf);
        if FE >= maxFE, break; end

        % Selection, Eqs. (6)-(7), over fireworks and both spark types
        C  = [FW; S];
        Cf = [FWf, Sf];
        K  = size(C, 1);
        R  = zeros(1, K);
        for q = 1:K
            R(q) = sum(sqrt(sum((C - C(q, :)) .^ 2, 2)));
        end
        [~, ib] = min(Cf);
        pick = zeros(1, n);
        pick(1) = ib;
        R(ib) = 0;
        for q = 2:n
            pick(q) = draw(R, pick(1:q-1));
            R(pick(q)) = 0;
        end
        FW  = C(pick, :);
        FWf = Cf(pick);
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Evaluates the rows of Xb; history gets the fireworks plus the sparks so far
    function fb = evaluate(Xb, Xprev, fprev)
        [fv, FE_new] = calculate_fitness(Xb', problem, FE);
        fb = fv(:)';
        for q2 = 1:numel(fb)
            if fb(q2) < best_fitness
                best_fitness  = fb(q2);
                best_solution = Xb(q2, :);
            end
            curve(FE + q2) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE + q2, [Xprev; Xb(1:q2, :)], [fprev, fb(1:q2)], population_history, ...
                fitness_history, history_index, maxFE);
        end
        FE = FE_new;
    end
end

% The paper's mapping of out-of-box coordinates, applied to the moved dimensions k
function x = map_into_box(x, k, lb, ub, span)
    out = k(x(k) < lb(k) | x(k) > ub(k));
    x(out) = lb(out) + mod(abs(x(out)), span(out));
end

% Roulette over the weights w, never returning an index in taken
function idx = draw(w, taken)
    c = cumsum(w);
    if c(end) > 0
        idx = find(rand * c(end) <= c, 1, 'first');
        if isempty(idx) || w(idx) == 0
            idx = find(w > 0, 1, 'last');   % floating-point rounding past the last edge
        end
    else
        free = setdiff(1:numel(w), taken);  % every location coincides: uniform draw
        idx = free(randi(numel(free)));
    end
end

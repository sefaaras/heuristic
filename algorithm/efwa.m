% ----------------------------------------------------------------------- %
% Enhanced Fireworks Algorithm (EFWA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 5                          % Fireworks
%   Me = 50, a = 0.04, b = 0.8     % Explosion sparks shared by fitness, each in [2, 40]
%   A_hat = 0.2*(ub - lb)          % Amplitude budget, FWA's 40 on [-100, 100]
%   Mg = 5                         % Gaussian sparks per iteration
%   A_init = 0.02*(ub - lb)        % Minimal explosion amplitude at the start
%   A_final = 0.001*(ub - lb)      % Minimal explosion amplitude at the budget
%
% Algorithm Concept:
%   - Better fireworks get more sparks in a smaller amplitude, Eqs. (1)-(3):
%     s_i ~ y_max - f_i, bounded to [a*Me, b*Me]; A_i ~ f_i - y_min
%   - Minimal amplitude check per dimension, Eqs. (6), (8): A_min = A_init -
%     (A_init - A_final)*sqrt((2E - t)t)/E after t of E evaluations
%   - Explosion spark: each dimension moves with chance 1/2, by its own U(-A_i, A_i)
%   - Gaussian spark: a random firework moves such dimensions toward the best
%     firework or explosion spark so far, x + (x_B - x)*e, one e ~ N(0, 1)
%   - A coordinate leaving the box is redrawn uniformly in its range
%   - Elitism-random selection: the best of all, plus N - 1 others at random
%
% Reference:
% Shaoqiu Zheng, Andreas Janecek, Ying Tan,
% Enhanced Fireworks Algorithm,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 2069-2077.
% https://doi.org/10.1109/CEC.2013.6557813
% ----------------------------------------------------------------------- %
% Implementation Note:
% No code release is reachable (matlab_efwa.zip, once on the CIL site, was never
% archived): ported from the paper, which takes N, Me, a, b, A_hat, Mg from FWA
% (Tan & Zhu 2010): 5, 50, 0.04, 0.8, 40, 5. Kept as published: z = round(rand)
% per dimension (a spark may move none and is still evaluated), though the authors
% later wrote that the paper's own runs used FWA's random dimension count. Chosen
% here: the N - 1 random survivors are distinct. A_hat, absolute in FWA, scales to
% each range (exact on [-100, 100]); the box is the initialisation range (paper:
% [ub/2, ub]); a non-finite spark count falls to the minimum. Mean error against
% the authors' EFWA from this pseudo-code (doi 10.1109/CEC.2014.6900485 Table II,
% CEC2013 D=30, 3e5 FE, 51 runs; 10 here), F14 unexplained:
%     F1 0.076 (0.1)  F2 7.0e5 (6.9e5)  F4 0.95 (1.1)  F5 0.082 (0.08)  F6 52 (49)
%     F7 122 (174)  F10 0.84 (0.84)  F11 453 (406)  F12 623 (699)  F15 4268 (4360)
%     F3 1.3e8 (7.8e7, heavy-tailed)  F14 4129 (2624)
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = efwa(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N       = 5;
    Me      = 50;
    a       = 0.04;
    b       = 0.8;
    A_hat   = 0.2 * span;
    Mg      = 5;
    A_init  = 0.02 * span;
    A_final = 0.001 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    FW  = lb + rand(N, D) .* span;
    n0  = min(N, maxFE);
    [f0, FE] = calculate_fitness(FW(1:n0, :)', problem, FE);
    FWf = inf(1, N);
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
        t = FE;   % evaluations spent at the start of this iteration
        y_max = max(FWf);
        y_min = min(FWf);

        % Eqs. (2)-(3)
        s = Me * (y_max - FWf + eps) / (sum(y_max - FWf) + eps);
        for i = 1:N
            if s(i) < a * Me || ~isfinite(s(i))
                s(i) = round(a * Me);
            elseif s(i) > b * Me
                s(i) = round(b * Me);
            else
                s(i) = round(s(i));
            end
        end

        % Eq. (1) per dimension, then the minimal amplitude check, Eqs. (6) and (8)
        A = (FWf(:) - y_min + eps) / (sum(FWf - y_min) + eps) .* A_hat;
        A_min = A_init - (A_init - A_final) / maxFE * sqrt((2 * maxFE - t) * t);
        A = max(A, A_min);   % max drops a NaN ratio in favour of A_min

        % Explosion sparks (Alg. 3), out-of-box coordinates mapped uniformly
        E = zeros(sum(s), D);
        e = 0;
        for i = 1:N
            for j = 1:s(i)
                e = e + 1;
                x = FW(i, :);
                z = round(rand(1, D));
                for k = 1:D
                    if z(k) == 1
                        x(k) = x(k) + A(i, k) * (2 * rand - 1);
                        if ~(x(k) >= lb(k) && x(k) <= ub(k))
                            x(k) = lb(k) + rand * span(k);
                        end
                    end
                end
                E(e, :) = x;
            end
        end
        nE = min(e, maxFE - FE);
        Ef = evaluate(E(1:nE, :), FW, FWf);
        if FE >= maxFE, break; end

        % Gaussian sparks (Alg. 4), toward the best firework/explosion spark so far
        [~, ib] = min([FWf, Ef]);
        C   = [FW; E];
        x_B = C(ib, :);
        G = zeros(Mg, D);
        for j = 1:Mg
            x = FW(randi(N), :);
            z = round(rand(1, D));
            g = randn;
            for k = 1:D
                if z(k) == 1
                    x(k) = x(k) + (x_B(k) - x(k)) * g;
                    if ~(x(k) >= lb(k) && x(k) <= ub(k))
                        x(k) = lb(k) + rand * span(k);
                    end
                end
            end
            G(j, :) = x;
        end
        nG = min(Mg, maxFE - FE);
        Gf = evaluate(G(1:nG, :), [FW; E], [FWf, Ef]);
        if FE >= maxFE, break; end

        % Elitism-random selection (ERP) over fireworks, explosion and Gaussian sparks
        C  = [FW; E; G];
        Cf = [FWf, Ef, Gf];
        [~, ib] = min(Cf);
        rest = [1:ib-1, ib+1:numel(Cf)];
        pick = [ib, rest(randperm(numel(rest), N - 1))];
        FW  = C(pick, :);
        FWf = Cf(pick);
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Evaluates the rows of Xb (cut to the budget); history gets the candidates so far
    function fb = evaluate(Xb, Xprev, fprev)
        [fv, FE_new] = calculate_fitness(Xb', problem, FE);
        fb = fv(:)';
        for q = 1:numel(fb)
            if fb(q) < best_fitness
                best_fitness  = fb(q);
                best_solution = Xb(q, :);
            end
            curve(FE + q) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE + q, [Xprev; Xb(1:q, :)], [fprev, fb(1:q)], population_history, ...
                fitness_history, history_index, maxFE);
        end
        FE = FE_new;
    end
end

% ----------------------------------------------------------------------- %
% Quantum-behaved Particle Swarm Optimization (QPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   M = 20                      % Swarm size (the paper's smallest of 20, 40, 80)
%   beta = 1.0 -> 0.5 (linear)  % Creativity coefficient over the budget (RQPSO1)
%
% Algorithm Concept:
%   - No velocity: each coordinate is drawn from a double exponential centred on a
%     local attractor between the particle's best P_i and the swarm best G
%   - Attractor per dimension p = (phi1*P_i + phi2*G)/(phi1 + phi2), phi1, phi2 ~ U(0,1)
%   - Spread set by the mean best position: x = p +- beta*|mbest - x|*ln(1/u),
%     mbest = mean of all personal bests, u ~ U(0,1), the sign a fair coin
%   - Particles take turns: each refreshes its own best from the position measured
%     last, then G and mbest are recomputed before it moves
%
% Reference:
% Jun Sun, Wenbo Xu, Bin Feng,
% A global search strategy of quantum-behaved particle swarm optimization,
% IEEE Conference on Cybernetics and Intelligent Systems (2004) vol. 1, 111-116.
% https://doi.org/10.1109/ICCIS.2004.1460396
% ----------------------------------------------------------------------- %
% Implementation Note:
% Port follows the paper's Sec. IV pseudocode and Eqs. (5), (11)-(13), with beta
% 1.0 -> 0.5 of its RQPSO1; RQPSO2 (beta 1.2 -> 0.5 with L capped by the initial
% max|mbest - G|) is the paper's second experiment and is not implemented. The paper
% gives no position bound, only asymmetric initial ranges (Table 1); a new position is
% clamped to the box here. Positions are measured in one batch once all M have moved,
% the values the pseudocode measures at each particle's next turn. Tables 2-5 and 8
% (M = 20, G = 1000/1500/2000 at D = 10/20/30, 50 runs) against 50 seeded runs here with
% the Table 1 initial ranges and no bound, mean (median), paper / here:
%      Sphere            1e-31 1e-20 1e-11     /  8.3e-36 1.7e-17 2.3e-12
%      Rosenbrock        13.84 116.05 187.18   /  91.7 202 2641  (6.2 80 110)
%      Rastrigin         4.571 16.02 35.21     /  5.41 17.6 33.8
%      Griewank, Eq. 17  0.0078 0.0002 0.0011  /  0.066 0.022 0.011       (not reproduced)
%      Schaffer f6, M = 20/40/80  7.9e-4 1.5e-5 8.5e-7  /  1.6e-3 1.6e-3 5.8e-4  (not reproduced)
%      Griewank, D = 10, M = 80: 0.0000 / 0.051; mbest once per iteration: 0.072 0.024 0.0099
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = qpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    M        = 20;
    beta_max = 1.0;
    beta_min = 0.5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(M, D) .* span;

    m = min(M, maxFE);
    [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    fx = inf(M, 1);
    fx(1:m) = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for k = 1:m
        if fx(k) < bsf
            bsf          = fx(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:m, :), fx(1:m), population_history, fitness_history, history_index, maxFE);
    end

    p  = x;
    fp = fx;
    % Mirror for the recorder: each particle at the position its fitness was measured
    xrec = x;
    frec = fx;

    while FE < maxFE
        % Linear in the iteration count, written with FE; M evaluations per iteration
        beta  = beta_min + (beta_max - beta_min) * (maxFE - FE) / maxFE;

        for i = 1:M
            % A NaN personal best could never be beaten, so it is overwritten
            if fx(i) < fp(i) || isnan(fp(i))
                p(i, :) = x(i, :);
                fp(i)   = fx(i);
            end
            % Sec. IV pseudocode: G and mbest (Eq. 11) recomputed at each particle's turn
            [~, g] = min(fp);
            mbest  = mean(p, 1);

            phi1 = rand(1, D);
            phi2 = rand(1, D);
            att  = (phi1 .* p(i, :) + phi2 .* p(g, :)) ./ (phi1 + phi2);
            u    = rand(1, D);
            step = beta * abs(mbest - x(i, :)) .* log(1 ./ u);
            % Eq. (13): the pseudocode subtracts when rand > 0.5 and adds otherwise
            sgn  = 1 - 2 * (rand(1, D) > 0.5);
            x(i, :) = min(max(att + sgn .* step, lb), ub);
        end

        n = min(M, maxFE - FE);
        [fv, FE] = calculate_fitness(x(1:n, :)', problem, FE);
        fx(1:n) = fv(:);
        xrec(1:n, :) = x(1:n, :);
        frec(1:n)    = fx(1:n);
        for k = 1:n
            if fx(k) < bsf
                bsf          = fx(k);
                bsf_solution = x(k, :);
            end
            ec = FE - n + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, xrec, frec, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

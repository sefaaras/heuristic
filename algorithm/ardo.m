% ----------------------------------------------------------------------- %
% Adaptive Role Division Optimizer (ARDO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 50                 % Population size
%   alpha = (ub-lb)*0.02       % Levy step scale of the pioneers
%   beta  = 1.5                % Levy exponent
%   sigma = 0.01               % Crowding-kernel width
%   tr    = 0.9                % Per-dimension activation threshold
%   c_up  = 0.1, c_down = 0.5  % Role-transition rates
%   c1    = 0.3, c2 = 0.4      % Initial pioneer / coordinator proportions
%
% Algorithm Concept:
%   - The population is divided into three dynamically assigned roles:
%       pioneers     -- Levy flights for global exploration
%       coordinators -- rank-bounded interpolation between better and worse
%                       peers, or a random peer, gated by the crowding speed
%       executors    -- local exploitation around better peers or the best
%   - The crowding velocity v_crowd is the spectral radius of the fitness
%     similarity matrix exp(-(F_m - F_n)^2/(sigma*best)^2) divided by N
%   - A social mobility mechanism (p12, p21, p23, p32) adaptively rebalances
%     c1/c2 from the per-role success counts and the forward velocity
%
% Reference:
% Pengfei Liu, Yue Deng, Jianan Wang,
% Adaptive role division optimizer: An efficient socio-inspired meta-heuristic
% algorithm with dynamic proportion adjustment,
% Swarm and Evolutionary Computation 107 (2026) 102438.
% https://doi.org/10.1016/j.swevo.2026.102438
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ardo(problem)

    D     = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N        = 50;
    Max_iter = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    alpha = (ub - lb) * 0.02;
    beta  = 1.5;
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / ...
               (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    sigma_v = 1;
    sigma = 0.01;
    tr = 0.9;
    c_up = 0.1;
    c_down = 0.5;
    c1 = 0.3;
    c2 = 0.4;

    % Initialise the population
    X = lb + (ub - lb) .* rand(N, D);
    [F, FE] = calculate_fitness(X', problem, FE);
    F = F(:);

    [best_score, ibest] = min(F);
    best_pos = X(ibest, :);
    past_f   = best_score;
    bsf      = best_score;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, F, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    for iter = 1:Max_iter
        if FE >= maxFE, break; end

        count_a = 0; count_b = 0; count_c = 0;
        [~, ind] = sort(F);
        fr = (1 - iter / Max_iter) ^ 3 * exp(-5 * (iter / Max_iter));

        % Crowding velocity from the fitness similarity matrix
        adjacency_matrix = exp(-(F - F.') .^ 2 / (sigma * best_score) ^ 2);
        eigenvalues = eig(adjacency_matrix);
        spectral_radius = max(abs(eigenvalues));
        v_crowd = spectral_radius / N;

        for i = 1:N
            if FE >= maxFE, break; end
            rank_i = find(i == ind);

            if rank_i <= c1 * N
                % Pioneers -- Levy flight
                u = normrnd(0, sigma_u, 1, D);
                v = normrnd(0, sigma_v, 1, D);
                step = u ./ (abs(v) .^ (1 / beta));
                newE = X(i, :) + alpha .* step;
                role = 1;

            elseif rank_i <= (c1 + c2) * N
                % Coordinators
                r_ae = rank_i;
                R  = X(ind(randi([1, r_ae])), :);
                L  = X(ind(randi([r_ae, N])), :);
                E1 = X(ind(randi([1, N])), :);
                u  = rand(1, D) > tr;
                ri = u .* random('Normal', zeros(1, D), fr * (ub - lb));
                if rand > v_crowd
                    newE = X(i, :) + rand(1, D) .* (E1 - X(i, :)) + ri;
                else
                    newE = X(i, :) + rand(1, D) .* (R - L) + ri;
                end
                role = 2;

            else
                % Executors
                E1 = X(ind(randi([1, rank_i])), :);
                E2 = X(ind(randi([1, rank_i])), :);
                u  = rand(1, D) > tr;
                ri = u .* random('Normal', zeros(1, D), fr * (ub - lb));
                if rand > v_crowd
                    newE = E1 + rand(1, D) .* (E2 - X(i, :)) + ri;
                else
                    newE = best_pos + rand(1, D) .* (E2 - X(i, :)) + ri;
                end
                role = 3;
            end

            % Midpoint bound repair
            flagub = newE > ub;
            flaglb = newE < lb;
            newE(flagub) = (X(i, flagub) + ub(flagub)) / 2;
            newE(flaglb) = (X(i, flaglb) + lb(flaglb)) / 2;

            [newf, FE] = calculate_fitness(newE', problem, FE);
            if newf <= F(i)
                F(i)    = newf;
                X(i, :) = newE;
                if F(i) <= best_score
                    best_score = F(i);
                    best_pos   = X(i, :);
                    switch role
                        case 1, count_a = count_a + 1;
                        case 2, count_b = count_b + 1;
                        otherwise, count_c = count_c + 1;
                    end
                end
            end

            if newf < bsf
                bsf = newf;
            end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, F, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Social mobility: rebalance the role proportions
        t0  = 0.5 * (1 - iter / Max_iter);
        t00 = 1   * (1 - iter / Max_iter);
        epsilon = 1e-6;
        t1 = count_a / (count_a + count_b + count_c + epsilon);
        t2 = count_b / (count_a + count_b + count_c + epsilon);
        v_forward = tanh((past_f - best_score) / (0.0001 * best_score));
        v_c = 1 - v_crowd;
        v_s = v_c * v_forward;
        lambda = max(0.1, v_s);
        p12 = lambda * c_down * max(0, 1 - t1 - t0);
        p21 = lambda * c_up   * min(1, t1 + t0);
        p23 = lambda * c_down * max(0, 1 - t1 - t2 - t00);
        p32 = lambda * c_up   * min(1, t1 + t2 + t00);
        c1_new = c1 - c1 * p12 + c2 * p21;
        c2_new = c2 - c2 * p21 - c2 * p23 + c1 * p12 + (1 - c1 - c2) * p32;
        c1 = max(0, c1_new);
        c2 = max(0, c2_new);

        past_f = best_score;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = best_score;
    best_solution = best_pos;
end

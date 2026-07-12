% ----------------------------------------------------------------------- %
% Hybrid Firefly and Particle Swarm Optimization (HFPSO) for unconstrained
% benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   swarm_size = 30       % Swarm size
%   c1 = c2 = 1.49445     % PSO cognitive/social coefficients
%   vmax_coef = 0.1       % Velocity clamp coefficient (fraction of range)
%   alpha = 0.2; beta0 = 2; gamma = 1; m = 2   % Firefly parameters
%
% Algorithm Concept:
%   - Hybridises the Firefly Algorithm with PSO
%   - A particle uses the firefly attraction move when it is at least as good
%     as the global best two iterations earlier; otherwise it uses the
%     standard PSO velocity update with a linearly decreasing inertia weight
%
% Reference:
% Ibrahim Berkan Aydilek,
% A hybrid firefly and particle swarm optimization algorithm for
% computationally expensive numerical problems,
% Applied Soft Computing 66 (2018) 232-249
% (Elsevier, ScienceDirect PII: S156849461830084X)
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hfpso(problem)

    % Extract problem parameters
    dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    swarm_size = 30;
    c1 = 1.49445;
    c2 = 1.49445;
    vmax_coef = 0.1;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, swarm_size, dim);
    fitness_history = zeros(history_size, swarm_size);
    history_index = 1;

    % Velocity bounds (per dimension, as row vectors)
    v_max = vmax_coef * (UB - LB);
    v_min = -v_max;

    % Initialize particles and velocities
    particles_x = zeros(swarm_size, dim);
    particles_v = zeros(swarm_size, dim);
    for piiz = 1:swarm_size
        particles_x(piiz, :) = LB + rand(1, dim) .* (UB - LB);
        particles_v(piiz, :) = v_min + rand(1, dim) .* (v_max - v_min);
    end

    % Evaluate the initial swarm
    [f_val, FE] = calculate_fitness(particles_x', problem, FE);
    f_val = f_val(:);   % swarm_size x 1

    p_best = particles_x;
    p_best_val = f_val;
    [g_best_val, index] = min(f_val);
    g_best = particles_x(index, :);
    dmax = (UB - LB) * sqrt(dim);

    % Record initial evaluations
    for eval_count = 1:swarm_size
        curve(eval_count) = g_best_val;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, particles_x, f_val, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    % Global-best trajectory (used by the firefly branch, two iterations back)
    iter_total = maxFE / swarm_size;
    max_iters = ceil(maxFE / swarm_size) + 2;
    g_best_t = zeros(max_iters, dim);
    g_best_val_t = inf(max_iters, 1);

    it = 0;
    while FE < maxFE
        FE_before = FE;
        it = it + 1;

        % Linearly decreasing inertia weight
        w = 0.9 - ((0.9 - 0.5) / iter_total) * it;

        for j = 1:swarm_size
            if (it > 2) && (f_val(j) <= g_best_val_t(it - 2))
                % --- Firefly attraction move ---
                rij = norm(particles_x(j, :) - g_best_t(it - 2, :) / dmax);
                alpha = 0.2;
                beta0 = 2;
                m = 2;
                gamma = 1;
                beta = beta0 * exp(-gamma * rij.^m);
                e = rand(1, dim) - 1/2;
                prev_pos = particles_x(j, :);
                particles_x(j, :) = particles_x(j, :) + beta .* (particles_x(j, :) - g_best_t(it - 2, :)) + alpha .* e;
                particles_x(j, :) = min(max(particles_x(j, :), LB), UB);
                particles_v(j, :) = particles_x(j, :) - prev_pos;
                particles_v(j, :) = min(max(particles_v(j, :), v_min), v_max);
            else
                % --- Standard PSO velocity/position update ---
                r1 = rand(1, dim);
                r2 = rand(1, dim);
                particles_v(j, :) = w * particles_v(j, :) ...
                    + c1 * r1 .* (p_best(j, :) - particles_x(j, :)) ...
                    + c2 * r2 .* (g_best - particles_x(j, :));
                particles_v(j, :) = min(max(particles_v(j, :), v_min), v_max);
                particles_x(j, :) = particles_x(j, :) + particles_v(j, :);
                particles_x(j, :) = min(max(particles_x(j, :), LB), UB);
            end
        end

        % Evaluate the swarm
        [f_val, FE] = calculate_fitness(particles_x', problem, FE);
        f_val = f_val(:);

        % Update personal and global bests
        for j = 1:swarm_size
            if f_val(j) < p_best_val(j)
                p_best(j, :) = particles_x(j, :);
                p_best_val(j) = f_val(j);
            end
            if p_best_val(j) < g_best_val
                g_best = particles_x(j, :);
                g_best_val = p_best_val(j);
            end
        end

        g_best_t(it, :) = g_best;
        g_best_val_t(it) = g_best_val;

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = g_best_val;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, particles_x, f_val, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness = g_best_val;
    best_solution = g_best;

end

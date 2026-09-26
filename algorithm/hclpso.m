% ----------------------------------------------------------------------- %
% Heterogeneous Comprehensive Learning Particle Swarm Optimization (HCLPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40 (g1 = 15, g2 = 25)   % Swarm size; g1 explores, g2 exploits
%   w = 0.99 -> 0.2 (linear)    % Inertia weight
%   c = 3 -> 1.5 (linear)       % Acceleration of g1, comprehensive-learning term only
%   c1 = 2.5 -> 0.5 (linear)    % Comprehensive-learning acceleration of g2
%   c2 = 0.5 -> 2.5 (linear)    % gbest acceleration of g2
%   Pc_i = 0 -> 0.25            % Learning probability, exponential in particle index i
%   Vmax = 0.2*(ub - lb)        % Velocity clamp, per dimension
%   refreshing gap = 5          % Exemplars redrawn after more than 5 stalled generations
%
% Algorithm Concept:
%   - Comprehensive learning as in CLPSO: every dimension follows the pbest of a
%     binary-tournament winner with probability Pc_i, otherwise the particle's own
%   - Exploration group g1 draws its exemplars from g1's pbests only and has no
%     gbest term, so it is never pulled towards the swarm's best
%   - Exploitation group g2 draws exemplars from the whole swarm and adds a
%     gbest term whose weight c2 grows while c1 shrinks
%   - A particle's exemplars are redrawn once its pbest has not improved for
%     more than 5 consecutive generations
%   - Positions are not clamped; a particle outside the box is not evaluated
%     and cannot update its pbest
%
% Reference:
% Nandar Lynn, Ponnuthurai Nagaratnam Suganthan,
% Heterogeneous comprehensive learning particle swarm optimization with enhanced
% exploration and exploitation,
% Swarm and Evolutionary Computation 24 (2015) 11-24.
% https://doi.org/10.1016/j.swevo.2015.05.002
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (HCLPSO.zip, HCLPSO_PS_15_35.m, run by its
% test script with N = 40 and max_iteration = max_FES/40). As released, the
% exemplar positions are copied when drawn and are not refreshed when those
% pbests (the particle's own included) later improve; kept.
% BUDGET: the release schedules w, c, c1 and c2 on the iteration index against
% that estimate and freezes them at the end; out-of-box particles cost no FE, so
% the estimate runs out before the budget. The schedules run on FE/maxFe here,
% which is the same clock when every particle is inside the box.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hclpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    num_g  = 40;
    num_g1 = 15;
    num_g2 = num_g - num_g1;
    g1 = 1:num_g1;
    g2 = num_g1 + 1:num_g;

    j  = (0:(1 / (num_g - 1)):1) * 10;
    Pc = 0.25 .* (exp(j) - exp(j(1))) ./ (exp(j(num_g)) - exp(j(1)));

    range_min = repmat(lb, num_g, 1);
    range_max = repmat(ub, num_g, 1);
    interval  = range_max - range_min;
    v_max = interval * 0.2;
    v_min = -v_max;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    pos = range_min + interval .* rand(num_g, D);
    vel = v_min + (v_max - v_min) .* rand(num_g, D);

    [result, FE] = calculate_fitness(pos', problem, FE);
    result = result(:);

    bsf          = inf;
    bsf_solution = pos(1, :);
    for i = 1:num_g
        if result(i) < bsf
            bsf          = result(i);
            bsf_solution = pos(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pos, result, population_history, fitness_history, history_index, maxFE);
        end
    end

    [gbest_val, g_index] = min(result);
    gbest_pos = pos(g_index, :);

    pbest_pos = pos;
    pbest_val = result;

    obj_func_slope = zeros(num_g, 1);
    fri_best_pos   = zeros(num_g, D);
    for i = 1:num_g
        fri_best_pos(i, :) = draw_exemplar(i, pool_size(i, num_g1, num_g), D, Pc(i), ...
                                           pbest_val, pbest_pos);
    end

    while FE < maxFE
        % Release clock k/max_iteration with max_iteration = max_FES/N, on the budget
        t  = min(FE / maxFE, 1);
        W  = 0.99 - t * 0.79;
        K  = 3 - t * 1.5;
        c1 = 2.5 - t * 2;
        c2 = 0.5 + t * 2;

        delta_g1 = K .* rand(num_g1, D) .* (fri_best_pos(g1, :) - pos(g1, :));
        vel_g1   = clamp_vel(W * vel(g1, :) + delta_g1, v_min(g1, :), v_max(g1, :));
        pos_g1   = pos(g1, :) + vel_g1;

        gbest_pos_temp = repmat(gbest_pos, num_g2, 1);
        delta_g2 = c1 .* rand(num_g2, D) .* (fri_best_pos(g2, :) - pos(g2, :)) ...
                 + c2 .* rand(num_g2, D) .* (gbest_pos_temp - pos(g2, :));
        vel_g2   = clamp_vel(W * vel(g2, :) + delta_g2, v_min(g2, :), v_max(g2, :));
        pos_g2   = pos(g2, :) + vel_g2;

        pos = [pos_g1; pos_g2];
        vel = [vel_g1; vel_g2];

        bad = ~isfinite(pos);
        if any(bad(:))
            PR = range_min + interval .* rand(num_g, D);
            pos(bad) = PR(bad);
        end

        % Only particles inside the box are evaluated, in index order until the budget ends
        inbox = find(all(pos <= range_max & pos >= range_min, 2));
        ne = min(numel(inbox), maxFE - FE);
        ev_idx = inbox(1:ne);
        evaluated = false(num_g, 1);
        evaluated(ev_idx) = true;

        if ne > 0
            [fv, FE] = calculate_fitness(pos(ev_idx, :)', problem, FE);
            result(ev_idx) = fv(:);

            for k = 1:ne
                i = ev_idx(k);
                if result(i) < bsf
                    bsf          = result(i);
                    bsf_solution = pos(i, :);
                end
                ec = FE - ne + k;
                if ec >= 1 && ec <= maxFE
                    curve(ec) = bsf;
                    % An unevaluated particle has no fitness at its current position
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, pos(ev_idx, :), result(ev_idx), population_history, ...
                        fitness_history, history_index, maxFE);
                end
            end
        end

        % Out-of-box particles still hold their previous result, which cannot beat pbest
        for i = 1:num_g
            if evaluated(i) && result(i) < pbest_val(i)
                pbest_pos(i, :) = pos(i, :);
                pbest_val(i)    = result(i);
                obj_func_slope(i) = 0;
            else
                obj_func_slope(i) = obj_func_slope(i) + 1;
            end
            if pbest_val(i) < gbest_val
                gbest_pos = pbest_pos(i, :);
                gbest_val = pbest_val(i);
            end
        end

        for i = 1:num_g
            if obj_func_slope(i) > 5
                fri_best_pos(i, :) = draw_exemplar(i, pool_size(i, num_g1, num_g), D, Pc(i), ...
                                                   pbest_val, pbest_pos);
                obj_func_slope(i) = 0;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% g1 learns from g1's pbests only, g2 from the whole swarm
function n = pool_size(i, num_g1, num_g)
    if i <= num_g1
        n = num_g1;
    else
        n = num_g;
    end
end

% Per dimension: own pbest unless rand <= Pc_i, then a binary pbest tournament winner
function fbpos = draw_exemplar(i, pool, D, Pci, pbest_val, pbest_pos)
    friend1 = ceil(pool * rand(1, D));
    friend2 = ceil(pool * rand(1, D));
    % `<` and its complement, so a NaN pbest cannot leave the index at 0
    better = reshape(pbest_val(friend1) < pbest_val(friend2), 1, D);
    friend = better .* friend1 + ~better .* friend2;
    toss = ceil(rand(1, D) - Pci);
    if all(toss == 1)
        toss(randi(D)) = 0;
    end
    fb = (1 - toss) .* friend + toss .* i;
    fbpos = pbest_pos(sub2ind(size(pbest_pos), fb, 1:D));
end

% The release's clamp: a component exactly on +-Vmax (or NaN) comes out as 0
function vel = clamp_vel(vel, v_min, v_max)
    vel = (vel < v_min) .* v_min + (vel > v_max) .* v_max ...
        + ((vel < v_max) & (vel > v_min)) .* vel;
end

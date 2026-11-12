% ----------------------------------------------------------------------- %
% Dynamic Multi-Swarm Particle Swarm Optimizer (DMS-PSO)
% The original without the local search of pool dmslpso
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 10, m = 3                  % Sub-swarms and particles per sub-swarm (30 in all)
%   w = 0.9 -> 0.2 (linear)        % Inertia weight over the budget
%   c1 = c2 = 1.49445              % Acceleration constants
%   Vmax = 0.2*(ub - lb)           % Velocity clamp
%   R = 5                          % Regrouping period, generations
%   phases = 0.9 / 0.1 of maxFe    % Dynamic multi-swarm search, then one global swarm
%
% Algorithm Concept:
%   - The swarm is split into small sub-swarms of 3; each particle follows its own
%     pbest and the best pbest of its sub-swarm (lbest)
%   - Every R generations the sub-swarms are re-drawn at random, so information
%     travels between them while each stays small and diverse
%   - A particle outside the box is not evaluated and its pbest is not updated
%   - For the last 10 % of the search all particles form one global-best swarm
%     led by the best lbest
%
% Reference:
% J. J. Liang, P. N. Suganthan,
% Dynamic Multi-Swarm Particle Swarm Optimizer,
% Proceedings 2005 IEEE Swarm Intelligence Symposium (SIS 2005), 2005, 124-129.
% https://doi.org/10.1109/SIS.2005.1501611
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the SIS 2005 paper (Fig. 2, Sec. 3) with the mechanics of the authors' release
% DMS_PSO_func.m (P-N-Suganthan/CODES, 2005-IEEE-SIS-DMS_PSO_func.zip), which is the
% CEC 2005 DMS-L-PSO: its quasi-Newton local search and its keep-half-of-pbest move,
% both introduced by the CEC 2005 paper, are removed. The SIS paper fixes w, Vmax, m,
% n, R and the 0.9 split but not c1, c2, v0 or the bound rule; those are the release's
% (its commented-out iwt line is the paper's w). The paper counts generations; here w
% and the split run on FE/maxFe, identical while every particle is in the box. Against
% the release with those two parts cut and the same w, 10 seeded runs (D = 2, 10; 2e4
% FE) evaluate bit-identical sequences. Hang guard: after 1000 generations with no
% particle in the box, the outside ones are clamped onto it.
%      SIS Table 3 (D = 10, 6e4 FE, 20 runs) / here with the CEC 2005 shift and rotation
%      data, so only approximate: rotated Rastrigin 3.28 / 4.97, rotated Griewank 0.023 / 0.063
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dmspso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    group_num = 10;
    group_ps  = 3;
    ps        = group_num * group_ps;
    c1        = 1.49445;
    c2        = 1.49445;
    R         = 5;
    mv        = 0.2 * span;
    stall_max = 1000;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = (lb + ub) / 2;

    pos = lb + span .* rand(ps, D);
    inb = true(ps, 1);
    n0 = min(ps, maxFE);
    e = inf(ps, 1);
    e(1:n0) = evaluate(pos(1:n0, :));
    inb(n0 + 1:end) = false;
    for q = 1:n0
        record(q);
    end

    vel = -mv + 2 .* mv .* rand(ps, D);
    pbest = pos;
    pbestval = e;
    group_id = reshape(1:ps, group_ps, group_num)';
    pos_group = zeros(1, ps);
    gbest = zeros(group_num, D);
    gbestval = zeros(1, group_num);
    regroup_leaders();

    it = 0;
    stall = 0;
    while FE < 0.9 * maxFE && FE < maxFE
        it = it + 1;
        % Paper: w falls 0.9 -> 0.2 over the run; FE/maxFe stands in for gen/Max_gen
        w = 0.9 - 0.7 * FE / maxFE;
        n_before = FE;
        for k = 1:ps
            if FE >= maxFE
                break;
            end
            g = pos_group(k);
            aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + c2 .* rand(1, D) .* (gbest(g, :) - pos(k, :));
            vel(k, :) = min(max(w .* vel(k, :) + aa, -mv), mv);
            pos(k, :) = pos(k, :) + vel(k, :);
            move_particle(k);
            if inb(k) && pbestval(k) < gbestval(g)
                gbest(g, :) = pbest(k, :);
                gbestval(g) = pbestval(k);
            end
        end
        stall = check_stall(FE == n_before, stall);

        if mod(it, R) == 0
            group_id = reshape(randperm(ps), group_ps, group_num)';
            regroup_leaders();
        end
    end

    [~, tmpid] = sort(gbestval);
    gb = gbest(tmpid(1), :);
    gbv = gbestval(tmpid(1));

    while FE < maxFE
        w = 0.9 - 0.7 * FE / maxFE;
        n_before = FE;
        for k = 1:ps
            if FE >= maxFE
                break;
            end
            aa = c1 .* rand(1, D) .* (pbest(k, :) - pos(k, :)) + c2 .* rand(1, D) .* (gb - pos(k, :));
            vel(k, :) = min(max(w .* vel(k, :) + aa, -mv), mv);
            pos(k, :) = pos(k, :) + vel(k, :);
            move_particle(k);
            if inb(k) && pbestval(k) < gbv
                gb = pbest(k, :);
                gbv = pbestval(k);
            end
        end
        stall = check_stall(FE == n_before, stall);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of X (in the box, within the budget), tracking the best per evaluation
    function f = evaluate(X)
        [fr, FE_new] = calculate_fitness(X', problem, FE);
        f = fr(:);
        for q2 = 1:numel(f)
            if f(q2) < bsf
                bsf = f(q2);
                bsf_solution = X(q2, :);
            end
            curve(FE + q2) = bsf;
        end
        FE = FE_new;
    end

    % Population = the particles inside the box, each at the value of its current position
    function record(ec)
        if any(inb)
            [population_history, fitness_history, history_index] = record_history( ...
                ec, pos(inb, :), e(inb), population_history, fitness_history, history_index, maxFE);
        else
            [population_history, fitness_history, history_index] = record_history( ...
                ec, pbest, pbestval, population_history, fitness_history, history_index, maxFE);
        end
    end

    % Evaluates particle k only if it lies in the box; pbest takes ties, as in the release
    function move_particle(k)
        inb(k) = all(pos(k, :) <= ub) && all(pos(k, :) >= lb);
        if inb(k)
            e(k) = evaluate(pos(k, :));
            if ~(pbestval(k) < e(k))
                pbest(k, :) = pos(k, :);
                pbestval(k) = e(k);
            end
            record(FE);
        end
    end

    % Sub-swarm membership and leaders from the current group_id
    function regroup_leaders()
        for gi = 1:group_num
            pos_group(group_id(gi, :)) = gi;
            [gbestval(gi), gid2] = min(pbestval(group_id(gi, :)));
            gbest(gi, :) = pbest(group_id(gi, gid2), :);
        end
    end

    % Hang guard: a swarm that stays wholly outside the box is clamped onto it
    function stall = check_stall(idle, stall)
        if ~idle
            stall = 0;
            return;
        end
        stall = stall + 1;
        if stall >= stall_max
            out = find(~inb)';
            for kk = out
                if FE >= maxFE
                    break;
                end
                pos(kk, :) = min(max(pos(kk, :), lb), ub);
                move_particle(kk);
            end
            stall = 0;
        end
    end
end

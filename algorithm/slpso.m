% ----------------------------------------------------------------------- %
% Social Learning Particle Swarm Optimization (SL-PSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   M = 100                     % Base swarm size
%   m = M + floor(D/10)         % Swarm size, grows with the dimension
%   epsilon = 0.01*D/M          % Social influence of the swarm mean (c3 in the code)
%   PL_i = (1-(i-1)/m)^ln(sqrt(ceil(D/M)))  % Learning probability, i = 1 the worst
%
% Algorithm Concept:
%   - The swarm is sorted by fitness each generation, and every particle but the
%     best learns, with probability PL_i, from particles better than itself
%   - Each dimension imitates its own demonstrator, drawn uniformly from all the
%     particles ranked above the learner
%   - v = r1*v + r2*(x_demo - x) + epsilon*r3*(mean(x) - x), then x = x + v
%   - PL_i falls with rank, so the worse a particle the surer it learns; it is 1
%     for every particle while D <= 100
%   - No pbest or gbest memory; positions are clamped to the box
%
% Reference:
% Ran Cheng, Yaochu Jin,
% A social learning particle swarm optimization algorithm for scalable
% optimization,
% Information Sciences 291 (2015) 43-60.
% https://doi.org/10.1016/j.ins.2014.08.039
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (SL_PSO_Matlab.zip, SLPSO.m; the main.asv
% autosave beside it is an unfinished draft and was ignored). As released, every
% particle but the best is re-evaluated each generation, including those that
% did not learn and so did not move (possible only at D > 100, where PL_i < 1);
% that FE cost is kept.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = slpso(problem)

    d     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    M  = 100;
    m  = M + floor(d / 10);
    c3 = d / M * 0.01;
    PL = (1 - ((1:m)' - 1) / m) .^ log(sqrt(ceil(d / M)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    p = repmat(lb, m, 1) + repmat(ub - lb, m, 1) .* rand(m, d);
    [fitness, FE] = calculate_fitness(p', problem, FE);
    fitness = fitness(:);
    v = zeros(m, d);

    bsf          = inf;
    bsf_solution = p(1, :);
    for i = 1:m
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = p(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, p, fitness, population_history, fitness_history, history_index, maxFE);
        end
    end

    winidxmask = repmat((1:m)', 1, d);
    colidx     = repmat(1:d, m, 1);

    while FE < maxFE
        % Descending, so row m holds the best and row 1 the worst
        [fitness, order] = sort(fitness, 'descend');
        p = p(order, :);
        v = v(order, :);

        center = mean(p, 1);

        randco1 = rand(m, d);
        randco2 = rand(m, d);
        randco3 = rand(m, d);
        % Per dimension, a demonstrator from the rows above (better than) the learner
        winidx = winidxmask + ceil(rand(m, d) .* (m - winidxmask));
        pwin   = p(sub2ind([m d], winidx, colidx));

        lp = rand(m, 1) < PL;
        lp(m) = false;
        % Rows 1..ne are the ones the budget can still evaluate; the rest stay put
        ne = min(m - 1, maxFE - FE);
        lp(ne + 1:end) = false;

        v1 = randco1 .* v + randco2 .* (pwin - p) + c3 * randco3 .* (center - p);
        p1 = p + v1;

        v(lp, :) = v1(lp, :);
        p(lp, :) = p1(lp, :);

        rows = 1:ne;
        pr = p(rows, :);
        % Checked before the clamp, which would turn a NaN into lb and hide it
        bad = ~isfinite(pr);
        if any(bad(:))
            PR = repmat(lb, ne, 1) + repmat(ub - lb, ne, 1) .* rand(ne, d);
            pr(bad) = PR(bad);
            vr = v(rows, :);
            vr(bad) = 0;
            v(rows, :) = vr;
        end
        pr = min(max(pr, lb), ub);
        p(rows, :) = pr;

        [fr, FE] = calculate_fitness(pr', problem, FE);
        fitness(rows) = fr(:);

        for k = 1:ne
            if fitness(k) < bsf
                bsf          = fitness(k);
                bsf_solution = pr(k, :);
            end
            ec = FE - ne + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, p, fitness, population_history, fitness_history, history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

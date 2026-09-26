% ----------------------------------------------------------------------- %
% Competitive Swarm Optimizer (CSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   m = 100 (D < 1000)          % Swarm size; 500, 1000, 1500 from D = 1000, 2000, 5000
%   phi = 0 (D < 500)           % Swarm-mean pull; 0.05, 0.1, 0.2 from D = 500, 1000, 2000
%
% Algorithm Concept:
%   - No pbest and no gbest: every generation the swarm is split into random
%     pairs and the two members of each pair compete on fitness
%   - The winner passes to the next generation untouched; only the loser moves,
%     v = r1*v + r2*(x_w - x_l) + phi*r3*(mean(x) - x_l), x_l = x_l + v
%   - So half the swarm is evaluated per generation, and a particle keeps moving
%     only while it keeps losing
%   - Positions are clamped to the box; velocities are not limited
%
% Reference:
% Ran Cheng, Yaochu Jin,
% A Competitive Swarm Optimizer for Large Scale Optimization,
% IEEE Transactions on Cybernetics 45(2) (2015) 191-204.
% https://doi.org/10.1109/TCYB.2014.2322602
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (CSO_Matlab.zip, CSO.m). The release sets m
% and phi from D in bands and gives phi a separable and a non-separable column;
% both columns are 0 below D = 500, and the harness cannot tell the two classes
% apart, so the non-separable column is used. It leaves m undefined below
% D = 100: the lowest band (m = 100) is extended down, so this grid (D = 2..118)
% always runs m = 100 with phi = 0, i.e. with the swarm-mean term inactive.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cso(problem)

    d     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    if d >= 2000
        phi = 0.2;
    elseif d >= 1000
        phi = 0.1;
    elseif d >= 500
        phi = 0.05;
    else
        phi = 0;
    end

    if d >= 5000
        m = 1500;
    elseif d >= 2000
        m = 1000;
    elseif d >= 1000
        m = 500;
    else
        m = 100;
    end
    half = ceil(m / 2);

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

    while FE < maxFE
        rlist  = randperm(m);
        rpairs = [rlist(1:half); rlist(floor(m / 2) + 1:m)]';

        center = mean(p, 1);

        % The larger fitness loses; a tie makes the second member the loser
        mask    = fitness(rpairs(:, 1)) > fitness(rpairs(:, 2));
        losers  = mask .* rpairs(:, 1) + ~mask .* rpairs(:, 2);
        winners = ~mask .* rpairs(:, 1) + mask .* rpairs(:, 2);

        % Only the pairs the budget can still evaluate are played out
        nl      = min(half, maxFE - FE);
        losers  = losers(1:nl);
        winners = winners(1:nl);

        randco1 = rand(nl, d);
        randco2 = rand(nl, d);
        randco3 = rand(nl, d);

        vl = randco1 .* v(losers, :) ...
           + randco2 .* (p(winners, :) - p(losers, :)) ...
           + phi * randco3 .* (center - p(losers, :));
        pl = p(losers, :) + vl;

        % Checked before the clamp, which would turn a NaN into lb and hide it
        bad = ~isfinite(pl) | ~isfinite(vl);
        if any(bad(:))
            PR = repmat(lb, nl, 1) + repmat(ub - lb, nl, 1) .* rand(nl, d);
            pl(bad) = PR(bad);
            vl(bad) = 0;
        end
        pl = min(max(pl, lb), ub);

        v(losers, :) = vl;
        p(losers, :) = pl;

        [fl, FE] = calculate_fitness(pl', problem, FE);
        fitness(losers) = fl(:);

        for k = 1:nl
            if fitness(losers(k)) < bsf
                bsf          = fitness(losers(k));
                bsf_solution = pl(k, :);
            end
            ec = FE - nl + k;
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

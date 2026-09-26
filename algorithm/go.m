% ----------------------------------------------------------------------- %
% Growth Optimizer (GO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 40                % Population size
%   P1 = 5                      % Leading group: better is drawn from ranks 2..P1
%   P2 = 0.001                  % Chance a worse trial is accepted anyway
%   P3 = 0.3                    % Per-coordinate chance of reflection
%   AF = 0.01 + 0.09*(1 - t)    % Reflection's reinitialisation chance, t the spent budget ratio
%
% Algorithm Concept:
%   - Learning phase: each member moves by four gaps (best-better, best-worst,
%     better-worst, two random peers), each weighted by its share of their total length
%   - That step is scaled by the member's fitness over the population's largest
%   - Better is a random member of ranks 2..P1, worst a random one of the last P1
%   - Reflection phase: each coordinate, with probability P3, moves a random fraction
%     towards one of the P1 best, and with probability AF is then redrawn uniformly
%   - Each trial replaces its parent if better, and otherwise still with
%     probability P2; coordinates outside the box are clamped
%
% Reference:
% Qingke Zhang, Hao Gao, Zhi-Hui Zhan, Junqing Li, Huaxiang Zhang,
% Growth Optimizer: A powerful metaheuristic algorithm for solving continuous
% and discrete global optimization problems,
% Knowledge-Based Systems 261 (2023) 110206.
% https://doi.org/10.1016/j.knosys.2022.110206
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/tsingke/Growth-Optimizer, GO.m,
% with popsize = 40 from its test.m). It already counts FEs but tests the budget
% only after a whole phase; here every evaluation is guarded, so a run stops at
% maxFe exactly. The worse-trial acceptance is skipped when ind(i) == 1, i.e. for
% the loop slot whose rank entry is individual 1 rather than for the current best;
% kept as released (the reported best is tracked apart, so it is never lost).
% When all four gaps vanish their weights are 0/0, and the reference clamps the
% resulting NaN step to lb; here a non-finite coordinate is redrawn uniformly in
% its own box, as in avoa, gto and nrbo.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = go(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    popsize = 40;
    P1 = 5;
    P2 = 0.001;
    P3 = 0.3;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = repmat(lb, popsize, 1) + rand(popsize, dim) .* repmat(ub - lb, popsize, 1);
    [fitness, FE] = calculate_fitness(x', problem, FE);
    fitness = fitness(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for i = 1:popsize
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = x(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, x, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    while FE < maxFE
        % Ranks are taken once per iteration and serve both phases, as in the reference
        [~, ind] = sort(fitness);
        Best_X = x(ind(1), :);

        for i = 1:popsize
            if FE >= maxFE
                break;
            end
            Worst_X  = x(ind(randi([popsize - P1 + 1, popsize], 1)), :);
            Better_X = x(ind(randi([2, P1], 1)), :);
            random = selectID(popsize, i, 2);
            L1 = random(1);
            L2 = random(2);

            Gap1 = Best_X - Better_X;
            Gap2 = Best_X - Worst_X;
            Gap3 = Better_X - Worst_X;
            Gap4 = x(L1, :) - x(L2, :);
            Distance1 = norm(Gap1);
            Distance2 = norm(Gap2);
            Distance3 = norm(Gap3);
            Distance4 = norm(Gap4);
            SumDistance = Distance1 + Distance2 + Distance3 + Distance4;
            LF1 = Distance1 / SumDistance;
            LF2 = Distance2 / SumDistance;
            LF3 = Distance3 / SumDistance;
            LF4 = Distance4 / SumDistance;
            SF = fitness(i) / max(fitness);

            newx = x(i, :) + LF1 * SF * Gap1 + LF2 * SF * Gap2 + LF3 * SF * Gap3 + LF4 * SF * Gap4;
            NF = ~isfinite(newx);
            newx(NF) = lb(NF) + rand(1, sum(NF)) .* (ub(NF) - lb(NF));
            newx = min(max(newx, lb), ub);

            [x, fitness, bsf, bsf_solution, FE] = trial(newx, i, ind, x, fitness, ...
                bsf, bsf_solution, FE, problem, P2);
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, x, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end

        for i = 1:popsize
            if FE >= maxFE
                break;
            end
            newx = x(i, :);
            for j = 1:dim
                if rand < P3
                    R = x(ind(randi(P1)), :);
                    newx(j) = x(i, j) + (R(j) - x(i, j)) * rand;
                    AF = 0.01 + (0.1 - 0.01) * (1 - FE / maxFE);
                    if rand < AF
                        newx(j) = lb(j) + (ub(j) - lb(j)) * rand;
                    end
                end
            end
            newx = min(max(newx, lb), ub);

            [x, fitness, bsf, bsf_solution, FE] = trial(newx, i, ind, x, fitness, ...
                bsf, bsf_solution, FE, problem, P2);
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, x, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [x, fitness, bsf, bsf_solution, FE] = trial(newx, i, ind, x, fitness, ...
                                                      bsf, bsf_solution, FE, problem, P2)
% Evaluate one trial, replace greedily or with probability P2, keep the reported best
    [fv, FE] = calculate_fitness(newx', problem, FE);
    newfitness = fv(1);
    if fitness(i) > newfitness
        fitness(i) = newfitness;
        x(i, :)    = newx;
    elseif rand < P2 && ind(i) ~= 1
        fitness(i) = newfitness;
        x(i, :)    = newx;
    end
    if newfitness < bsf
        bsf          = newfitness;
        bsf_solution = newx;
    end
end

function r = selectID(popsize, i, k)
% k distinct indices from 1..popsize, none equal to i
    vecc = [1:i-1, i+1:popsize];
    r = zeros(1, k);
    for kkk = 1:k
        n = popsize - kkk;
        t = randi(n, 1, 1);
        r(kkk) = vecc(t);
        vecc(t) = [];
    end
end

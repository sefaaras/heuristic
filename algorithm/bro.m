% ----------------------------------------------------------------------- %
% Battle Royale Optimization (BRO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N         = 200                       % Population size (soldiers)
%   MaxFault  = 4                         % Damage count at which a soldier respawns
%   CrossType = 1                         % Move toward the best: x + r.*(best - x)
%   Step      = round(T/ceil(log10 T)) -> Step + round(Step/2)   % Safe-zone shrink iterations
%
% Algorithm Concept:
%   - Each soldier fights its nearest neighbour (Euclidean); the worse one is
%     damaged (fault + 1), the winner's fault count is reset to 0
%   - A damaged soldier moves toward the best: x + r.*(best - x), r ~ U(0,1)^D
%   - A soldier damaged MaxFault times in a row respawns uniformly in the
%     safe zone and its fault count restarts
%   - At the shrink iterations the safe zone becomes best +- the per-dimension
%     standard deviation of the soldiers, clipped to the box
%   - Both fighters are clamped to the safe zone; the damaged one is evaluated
%
% Reference:
% Taymaz Rahkar Farshi,
% Battle royale optimization algorithm,
% Neural Computing and Applications 33(4) (2021) 1139-1157.
% https://doi.org/10.1007/s00521-020-05004-4
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from github.com/cominsys/BRO (BRO_Fun.m and helpers, each headed by
% the author's name, last commit 2022-01-14); seeded runs are bit-identical.
% N, MaxFault, CrossType are its Main.m values (the paper is closed access;
% the author's 2022 follow-up, arXiv 2203.09889, ran BRO at N = 100, threshold 3).
% Kept as released: a winner outside the safe zone is re-evaluated after each
% coordinate it clamps, intermediate points included, and those evaluations
% never update the best that guides the moves (the reported best counts them).
% T = ceil((maxFe - N)/N) assumes N FEs per iteration and fixes the shrink
% schedule; the re-evaluations take 0.06-0.27 % of the FEs on cec14 D = 2-30,
% so the run ends one or two iterations before T.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bro(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N         = 200;
    MaxFault  = 4;
    maxiter   = max(1, ceil((maxFE - N) / N));   % N damaged-soldier evaluations per iteration
    Shrink    = ceil(log10(maxiter));
    Step      = round(maxiter / Shrink);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Fault = zeros(1, N);
    ShrinkLo = lb;   % ShrinkMinMax(:, 1) of the release
    ShrinkHi = ub;   % ShrinkMinMax(:, 2)

    % Soldiers are rows here, columns in the release; rand(dim, N)' keeps its stream order
    X = rand(dim, N)' .* (ub - lb) + lb;
    nev = min(N, maxFE);
    [Fit, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
    Fit = Fit(:)';

    best_fitness  = Inf;
    best_solution = X(1, :);
    for e = 1:nev
        if Fit(e) < best_fitness
            best_fitness  = Fit(e);
            best_solution = X(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, X(1:nev, :), Fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    [flagFit, indx] = min(Fit);
    flagX = X(indx, :);

    iter = 0;
    while FE < maxFE
        iter = iter + 1;
        for i = 1:N
            Dist = sqrt(sum((X - X(i, :)) .^ 2, 2));   % edistance.m
            Dist(i) = inf;
            [~, Cn] = min(Dist);
            TempXY = rand(dim, 1)' .* (ShrinkHi - ShrinkLo) + ShrinkLo;
            dam = i;
            Vic = Cn;
            if Fit(Cn) > Fit(i)
                dam = Cn;
                Vic = i;
            end
            Fault(dam) = Fault(dam) + 1;
            Fault(Vic) = 0;
            damOld = X(dam, :);   % the position Fit(dam) still belongs to
            if Fault(dam) < MaxFault
                X(dam, :) = X(dam, :) + rand(dim, 1)' .* (flagX - X(dam, :));   % CrossType 1
            else
                X(dam, :) = TempXY;
                Fault(dam) = 0;
            end

            for d = 1:dim
                if X(dam, d) < ShrinkLo(d)
                    X(dam, d) = ShrinkLo(d);
                end
                if X(Vic, d) < ShrinkLo(d)
                    X(Vic, d) = ShrinkLo(d);
                    [Fit(Vic), FE] = calculate_fitness(X(Vic, :)', problem, FE);   % as released, inside the loop
                    [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                        sample(X, Fit, dam, damOld, Vic, FE, maxFE, best_fitness, best_solution, ...
                        curve, population_history, fitness_history, history_index);
                    if FE >= maxFE, break; end
                end
                if X(dam, d) > ShrinkHi(d)
                    X(dam, d) = ShrinkHi(d);
                end
                if X(Vic, d) > ShrinkHi(d)
                    X(Vic, d) = ShrinkHi(d);
                    [Fit(Vic), FE] = calculate_fitness(X(Vic, :)', problem, FE);
                    [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                        sample(X, Fit, dam, damOld, Vic, FE, maxFE, best_fitness, best_solution, ...
                        curve, population_history, fitness_history, history_index);
                    if FE >= maxFE, break; end
                end
            end
            if FE >= maxFE
                X(dam, :) = damOld;   % the budget ran out before the damaged soldier was evaluated
                break;
            end

            [Fit(dam), FE] = calculate_fitness(X(dam, :)', problem, FE);
            if Fit(dam) < flagFit
                flagFit = Fit(dam);
                flagX = X(dam, :);
            end
            [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                sample(X, Fit, dam, X(dam, :), dam, FE, maxFE, best_fitness, best_solution, ...
                curve, population_history, fitness_history, history_index);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        if iter >= Step
            STDdim = std(X, 0, 1);   % std of each dimension over the soldiers
            ShrinkLo = flagX - STDdim;
            ShrinkHi = flagX + STDdim;
            Step = Step + round(Step / 2);
        end
        ShrinkLo = max(ShrinkLo, lb);
        ShrinkHi = min(ShrinkHi, ub);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Reported best, curve and history after one evaluation; the damaged soldier sits where Fit(dam) was measured
function [best_fitness, best_solution, curve, ph, fh, hi] = sample(X, Fit, dam, damPos, who, FE, maxFE, ...
        best_fitness, best_solution, curve, ph, fh, hi)
    if Fit(who) < best_fitness
        best_fitness  = Fit(who);
        best_solution = X(who, :);
    end
    if FE <= maxFE
        curve(FE) = best_fitness;
        X(dam, :) = damPos;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

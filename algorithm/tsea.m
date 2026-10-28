% ----------------------------------------------------------------------- %
% Tree-Seed Algorithm (TSA)
% Stored as tsea; the acronym TSA collides with tsa (Tunicate Swarm)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 50                % Trees in the stand
%   ST   = 0.1               % Search tendency, per-coordinate chance of Eq. (1)
%   ns   = 6..13 per tree    % fix(low + (high - low)*rand) + 1, capped at high
%   low  = ceil(0.1*N)       % Seed-count floor, 5
%   high = ceil(0.25*N)      % Seed-count cap, 13
%
% Algorithm Concept:
%   - Each tree in turn sows ns seeds; T_r is a random other tree per seed and
%     alpha ~ U(-1, 1) is redrawn for every coordinate
%   - With probability ST a coordinate follows Eq. (1), s = T_i + alpha*(B - T_r),
%     B the best tree; otherwise Eq. (2), s = T_i + alpha*(T_i - T_r)
%   - The best seed replaces its tree only when strictly better
%
% Reference:
% Mustafa Servet Kiran,
% TSA: Tree-seed algorithm for continuous optimization,
% Expert Systems with Applications 42(19) (2015) 6686-6698.
% https://doi.org/10.1016/j.eswa.2015.04.055
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from File Exchange 62646 v1.1.0.0 (TSA.m, 11 May 2018), coded by
% A. C. Cinar for Kiran; Kiran's own tsa.rar (mskiran.kisisel.selcuk.edu.tr/tsa)
% is offline and was never archived. The release's scalar dmin/dmax become
% per-dimension lb/ub, clamped coordinate by coordinate as released. It runs
% whole iterations while FEs <= maxFEs and overshoots; here the run stops at
% maxFe, mid-tree if need be. Kept as released: ns spans low+1..high, B is the
% best tree re-read before each tree, and the reported best is the stand's best
% (the best seed always enters it), here updated at every evaluation. The dead
% seeds(j,:) = trees(j,:) copy (every coordinate is overwritten) is dropped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tsea(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    N    = 50;
    ST   = 0.1;
    low  = ceil(N * 0.1);
    high = ceil(N * 0.25);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % The release fills the stand tree by tree, coordinate by coordinate
    trees = lb + (ub - lb) .* rand(D, N)';
    nev = min(N, maxFE);
    trees = trees(1:nev, :);
    [obj_trees, FE] = calculate_fitness(trees', problem, FE);
    obj_trees = obj_trees(:)';

    best_fitness  = Inf;
    best_solution = trees(1, :);
    for e = 1:nev
        if obj_trees(e) < best_fitness
            best_fitness  = obj_trees(e);
            best_solution = trees(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, trees, obj_trees, population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE
        for i = 1:N
            ns = fix(low + (high - low) * rand) + 1;
            if ns > high
                ns = high;
            end

            [~, min_indis] = min(obj_trees);
            bestParams = trees(min_indis, :);

            seeds = zeros(ns, D);
            for j = 1:ns
                komsu = fix(rand * N) + 1;
                while i == komsu
                    komsu = fix(rand * N) + 1;
                end
                % Row 1 decides the equation, row 2 is alpha; the release draws them in this order per coordinate
                r = rand(2, D);
                step = trees(i, :) - trees(komsu, :);   % Eq. (2)
                useBest = r(1, :) < ST;
                step(useBest) = bestParams(useBest) - trees(komsu, useBest);   % Eq. (1)
                s = trees(i, :) + step .* (r(2, :) - 0.5) * 2;
                hi = s > ub;
                s(hi) = ub(hi);
                lo = s < lb;
                s(lo) = lb(lo);
                seeds(j, :) = s;
            end

            nev = min(ns, maxFE - FE);
            [obj_seeds, FE] = calculate_fitness(seeds(1:nev, :)', problem, FE);
            obj_seeds = obj_seeds(:)';

            for e = 1:nev
                if obj_seeds(e) < best_fitness
                    best_fitness  = obj_seeds(e);
                    best_solution = seeds(e, :);
                end
                % The last seed completes the tree's turn, so its sample shows the stand after replacement
                if e == ns
                    [mintohum, mintohum_indis] = min(obj_seeds);
                    if mintohum < obj_trees(i)
                        trees(i, :) = seeds(mintohum_indis, :);
                        obj_trees(i) = mintohum;
                    end
                end
                ec = FE - nev + e;
                curve(ec) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, trees, obj_trees, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

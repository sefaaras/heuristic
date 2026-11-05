% ----------------------------------------------------------------------- %
% Cuckoo Optimization Algorithm (COA)
% Stored as cuoa; the acronym COA collides with coa (Coyote Optimization Algorithm)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   numCuckooS = 5               % Initial cuckoos (habitats)
%   minNumberOfEggs = 2, maxNumberOfEggs = 4   % floor(U(2, 4)): 2 or 3 eggs each
%   radiusCoeff = 5              % Egg laying radius = eggs/all eggs * 5*(ub - lb)
%   maxNumOfCuckoos = 10         % Cap on the living cuckoos
%   knnClusterNum = 1            % k-means groups of the grown eggs
%   motionCoeff = 9              % Immigration step x + 9*U(0, 1) .* (goal - x)
%
% Algorithm Concept:
%   - Each cuckoo lays its eggs at radius U(0, 1)*ELR, on angles spread over
%     [0, 2*pi]: every dimension moves +/- r*cos(angle) + r*sin(angle)
%   - Duplicate eggs are removed; cuckoos and eggs are evaluated together
%   - Above the cap only the best maxNumOfCuckoos habitats live on, otherwise
%     the eggs grow into the next cuckoos and their parents die
%   - The grown eggs are grouped by k-means; the group with the best mean profit
%     holds the goal point, its best egg, and every egg immigrates toward it
%   - The best cuckoo so far and a copy U(0, 1) .* x_best replace the last two
%
% Reference:
% Ramin Rajabioun,
% Cuckoo Optimization Algorithm,
% Applied Soft Computing 11(8) (2011) 5508-5518.
% https://doi.org/10.1016/j.asoc.2011.05.008
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's cuckooMain.m, File Exchange 35635 v5 (14 Sep 2012), and its
% parameters (v1: 50 cuckoos, cap 200, 2 groups, radius 50 decaying); seeded runs are
% bit-identical to it. Kept as released: four extra evaluations per generation (the
% generation's best twice, the global best, the U(0, 1) .* x_best copy; the last three
% discarded), the duplicate filter skipping the row after each removal, the global best
% seeded unevaluated at profit -1e20. Changed: per-dimension radius (exact on a cube);
% rounded eggs clamped into the box; the 1e-13 variance-sum stop dropped for the budget.
% Neither v1 nor v5 uses the paper's settings (20 cuckoos, 5-10 eggs, k = 3-5 groups);
% the release also lacks its 10 % egg kill and deviation phi ~ U(-pi/6, pi/6), and steps
% 9*U(0, 1) per dimension where the paper has lambda ~ U(0, 1) (Eq. 5).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cuoa(problem)

    npar  = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    numCuckooS      = 5;
    minNumberOfEggs = 2;
    maxNumberOfEggs = 4;
    knnClusterNum   = 1;
    motionCoeff     = 9;
    maxNumOfCuckoos = 10;
    radiusCoeff     = 5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    center = zeros(numCuckooS, npar);
    for c = 1:numCuckooS
        center(c, :) = span .* rand(1, npar) + lb;
    end
    goalPoint = span .* rand(1, npar) + lb;
    globalBestCuckoo = goalPoint;
    globalMaxProfit  = -1e20;

    best_fitness  = inf;
    best_solution = center(1, :);
    Hpos = center;           % the last evaluated habitats, for the recorder
    Hfit = inf(1, numCuckooS);

    while FE < maxFE
        numberOfEggs = floor((maxNumberOfEggs - minNumberOfEggs) * rand(numCuckooS, 1) + minNumberOfEggs);
        summ = sum(numberOfEggs);
        radii = cell(numCuckooS, 1);
        for c = 1:numCuckooS
            radii{c} = rand(numberOfEggs(c), 1);
        end

        % Egg laying, then the release's rounding and duplicate filter
        eggs = cell(numCuckooS, 1);
        for c = 1:numCuckooS
            r = radii{c} .* (numberOfEggs(c) / summ * (radiusCoeff * span));
            angles = linspace(0, 2 * pi, numberOfEggs(c))';
            sgn = (-1) .^ (floor(2 * rand(npar, numberOfEggs(c)))' + 1);
            newParams = center(c, :) + ((sgn .* r) .* cos(angles) + r .* sin(angles));
            newParams = max(min(newParams, ub), lb);

            tmpPopulation = floor(newParams * 1e20) / 1e20;
            ii = 2;
            cntt = 1;
            while ii <= size(tmpPopulation, 1) || cntt <= size(tmpPopulation, 1)
                if all((tmpPopulation(cntt, :) == tmpPopulation(ii, :)))
                    tmpPopulation(ii, :) = [];
                end
                ii = ii + 1;
                if ii > size(tmpPopulation, 1) && cntt <= size(tmpPopulation, 1)
                    cntt = cntt + 1;
                    ii = cntt + 1;
                    if ii > size(tmpPopulation, 1)
                        break
                    end
                end
            end
            eggs{c} = max(min(tmpPopulation, ub), lb);
        end

        % Every cuckoo with its eggs, cuckoo by cuckoo as the release evaluates them
        allPositions = zeros(numCuckooS + sum(cellfun(@(E) size(E, 1), eggs)), npar);
        isEgg = true(size(allPositions, 1), 1);
        row = 0;
        for c = 1:numCuckooS
            isEgg(row + 1) = false;
            allPositions(row + 1:row + 1 + size(eggs{c}, 1), :) = [center(c, :); eggs{c}];
            row = row + 1 + size(eggs{c}, 1);
        end
        nb = min(row, maxFE - FE);
        [cost, FE_new] = calculate_fitness(allPositions(1:nb, :)', problem, FE);
        cost = cost(:);
        Hpos = allPositions(1:nb, :);
        Hfit = cost';
        for q = 1:nb
            note_best(cost(q), allPositions(q, :), FE + q);
        end
        FE = FE_new;
        if nb < row, break; end
        profits = -cost;

        [sortedProfits, sortingIndex] = sort(profits, 'descend');
        bestCuckooCenter = allPositions(sortingIndex(1), :);
        if numCuckooS > maxNumOfCuckoos
            eggPos    = allPositions(sortingIndex(1:maxNumOfCuckoos), :);
            eggProfit = sortedProfits(1:maxNumOfCuckoos);
        else
            eggPos    = allPositions(isEgg, :);
            eggProfit = profits(isEgg);
        end

        [f, ok] = evaluate_one(bestCuckooCenter);
        if ~ok, break; end
        if -f > globalMaxProfit
            globalBestCuckoo = bestCuckooCenter;
            globalMaxProfit  = -f;
        end

        % Grouping of the grown eggs and immigration toward the best group's best egg
        clusterNumbers = kmeans(eggPos, knnClusterNum);
        f_mean = -inf(knnClusterNum, 1);
        for cnt = 1:knnClusterNum
            if any(clusterNumbers == cnt)
                f_mean(cnt) = mean(eggProfit(clusterNumbers == cnt));
            end
        end
        [~, sortingIndex_f_mean] = sort(f_mean, 'descend');
        mine = find(clusterNumbers == sortingIndex_f_mean(1));
        [~, indexOfBestEggPosition] = max(eggProfit(mine));
        goalPoint = eggPos(mine(indexOfBestEggPosition), :);

        center = zeros(0, npar);
        for cntClstr = 1:knnClusterNum
            tmpPositions = eggPos(clusterNumbers == cntClstr, :);
            for cntPosition = 1:size(tmpPositions, 1)
                tmpPositions(cntPosition, :) = tmpPositions(cntPosition, :) + ...
                    motionCoeff * rand(1, npar) .* (goalPoint - tmpPositions(cntPosition, :));
            end
            tmpPositions = max(min(tmpPositions, ub), lb);
            center = [center; tmpPositions]; %#ok<AGROW>
        end
        numCuckooS = size(center, 1);

        [f, ok] = evaluate_one(bestCuckooCenter);
        if ~ok, break; end
        if -f > globalMaxProfit
            globalBestCuckoo = bestCuckooCenter;
            globalMaxProfit  = -f;
        end
        center(end, :) = globalBestCuckoo;
        [~, ok] = evaluate_one(center(end, :));
        if ~ok, break; end

        tmp = rand(1, npar) .* globalBestCuckoo;
        tmp = max(min(tmp, ub), lb);
        center(end - 1, :) = tmp;
        [~, ok] = evaluate_one(center(end - 1, :));
        if ~ok, break; end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % One evaluation of x; ok is false once the budget is spent
    function [fx, ok] = evaluate_one(x)
        ok = FE < maxFE;
        fx = inf;
        if ~ok, return; end
        [fx, FE] = calculate_fitness(x', problem, FE);
        fx = fx(1);
        note_best(fx, x, FE);
    end

    % Best-so-far, curve and history for the evaluation that brought the count to fe
    function note_best(fx, x, fe)
        if fx < best_fitness
            best_fitness  = fx;
            best_solution = x;
        end
        curve(fe) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            fe, Hpos, Hfit, population_history, fitness_history, history_index, maxFE);
    end
end

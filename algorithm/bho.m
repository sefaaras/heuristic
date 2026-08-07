% ----------------------------------------------------------------------- %
% Bounty Hunter Optimizer (BHO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popSize          = dim   % Population size (hunters), as in the driver
%   nGroups          = 4     % Hunter groups (k-means over the initial swarm)
%   eliteRatioSwitch = 0.3   % Elite / explorer role split
%   cCoeff           = 0.5   % Global / group / personal attraction weights
%   nTransfer        = 5, rho = 0.25
%   totalElitePool   = 10*popSize
%   TAU              = 0.30*sqrt(dim)   % Explorpolis domination radius
%
% Algorithm Concept:
%   - Explorpolis rule: a trial that fails the greedy test is still accepted
%     unless it is dominated -- i.e. unless some archived solution is both
%     better and closer than the normalised radius TAU. This filters inferior
%     solutions without collapsing the population onto a centroid
%   - Elite update: PSO-like pull towards the global, group and personal best,
%     carrying the previous displacement deltaX
%   - Explorer update: a linearly shrinking differential step towards or away
%     from a random peer depending on which was better last iteration
%   - Guided-centre network (Xnet): a preliminary covariance-guided DE run
%     supplies restart centres for the worst hunter
%   - Back-off: rejected trials that survive a nearest-neighbour thinning are
%     re-injected into the population
%
% Reference:
% Mingyang Yu, Haorui Yang, Jiaqi Zhang, Kaichen Ouyang, Shengwei Fu,
% Panlong Tan, Frank Jiang, Jing Xu,
% Bounty hunter optimizer: A novel metaheuristic with an application to
% multi-UAV mobile edge computing and path planning,
% Knowledge-Based Systems 341 (2026) 115836.
% https://doi.org/10.1016/j.knosys.2026.115836
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bho(problem)

    dim     = problem.dimension;
    lb      = problem.lb;
    ub      = problem.ub;
    maxEval = problem.maxFe;

    popSize = dim;
    if popSize < 4, popSize = 4; end

    nGroups          = 4;
    eliteRatioSwitch = 0.3;
    cCoeff           = repmat([0.5 0.5 0.5], popSize, 1);
    nTransfer        = 5;
    rho              = 0.25;
    totalElitePool   = popSize * 10;
    TAU = 0.30 * sqrt(dim);

    evalCount = 0;
    curve = zeros(1, maxEval);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsx = lb + 0.5 * (ub - lb);

    % Initialisation
    randX = rand(popSize * 10, dim);
    [~, randCenter] = kmeans(randX, popSize);
    X = lb + (ub - lb) .* randCenter;

    [Xnet, evalCount, bsf, bsx, curve, population_history, fitness_history, history_index] = ...
        build_guided_centers(popSize, maxEval, lb, ub, dim, problem, evalCount, maxEval, ...
                             bsf, bsx, curve, population_history, fitness_history, history_index);

    groupIdx = kmeans(X, nGroups);
    F = zeros(popSize, 1);

    for i = 1:popSize
        if evalCount >= maxEval, break; end
        [F(i), evalCount] = calculate_fitness(X(i, :)', problem, evalCount);
        [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
            stamp(evalCount, maxEval, F(i), X(i, :), bsf, bsx, curve, X, F, ...
                  population_history, fitness_history, history_index);
    end

    iter = 1;
    tab = zeros(popSize, 9);
    tab(:, 1) = (1:popSize)';
    tab(:, 2) = F;

    [~, ord] = sort(F, 'ascend');
    rankPos = zeros(popSize, 1); rankPos(ord) = 1:popSize;
    tab(:, 3) = rankPos / popSize;
    tab(:, 4) = groupIdx;
    tab(tab(:, 3) <= eliteRatioSwitch, 5) = 1;
    tab(tab(:, 3) >  eliteRatioSwitch, 5) = 2;

    Fi_sum = max(sum(F), eps);
    tab(:, 6) = 1 - (iter / (2 * maxEval)) * (F / Fi_sum);

    F_eps = F + 1e-200;
    manValue = max(F_eps) - F_eps;
    groupValue = zeros(nGroups, 1);
    for g = 1:nGroups
        groupValue(g) = (1 - rho) * groupValue(g) + sum(manValue(tab(:, 4) == g));
    end

    pBestX = X;
    pBestF = F;

    gBestX = zeros(nGroups, dim);
    gBestF = inf(nGroups, 1);
    for g = 1:nGroups
        rows = (tab(:, 4) == g);
        if any(rows)
            [gBestF(g), idx] = min(F(rows));
            gBestX(g, :) = X(find(rows, 1, 'first') - 1 + idx, :);
        end
    end

    [globBestF, idx] = min(F);
    globBestX = X(idx, :);

    deltaX = zeros(size(X));
    archiveX = X;
    archiveF = F;

    % Main loop
    while evalCount < maxEval
        iter = iter + 1;

        oldX = X;
        oldF = F;

        backList = ones(5 * popSize, dim + 2) * 1e9;
        backPtr  = 0;

        for i = 1:popSize
            if evalCount >= maxEval, break; end

            if tab(i, 5) == 1
                % ELITE update
                r = rand(1, dim);
                trial = X(i, :) + deltaX(i, :) + tab(i, 6) .* ( ...
                    cCoeff(i, 1) * r .* (globBestX - X(i, :)) + ...
                    cCoeff(i, 2) * r .* (gBestX(tab(i, 4), :) - X(i, :)) + ...
                    cCoeff(i, 3) * r .* (pBestX(i, :) - X(i, :)));
            else
                % EXPLORER update
                j = randperm(popSize, 1);
                if oldF(j) > F(i)
                    trial = X(i, :) + (2 - iter / maxEval * 2) .* rand .* (oldX(i, :) - oldX(j, :));
                else
                    trial = X(i, :) + (2 - iter / maxEval * 2) .* rand .* (oldX(j, :) - X(i, :));
                end
            end

            trial = min(max(trial, lb), ub);
            [fTrial, evalCount] = calculate_fitness(trial', problem, evalCount);

            if fTrial < globBestF
                globBestF = fTrial;
                globBestX = trial;
            end
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(evalCount, maxEval, fTrial, trial, bsf, bsx, curve, X, F, ...
                      population_history, fitness_history, history_index);

            if fTrial < F(i)
                F(i) = fTrial; X(i, :) = trial;
            else
                if explorpolis_accept(trial, fTrial, archiveX, archiveF, lb, ub, TAU, dim)
                    F(i) = fTrial; X(i, :) = trial;
                else
                    backPtr = backPtr + 1;
                    backList(backPtr, :) = [i, fTrial, trial];
                end
            end
        end

        % back-off attempts
        backList(backList(:, 1) == 1e9, :) = [];
        if ~isempty(backList)
            [X, F, evalCount, bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                backoff_try(backList, X, F, Xnet, iter, ub, lb, maxEval, problem, evalCount, ...
                            bsf, bsx, curve, population_history, fitness_history, history_index, ...
                            maxEval);
        end

        deltaX = X - oldX;

        % update the archive
        tmp = sortrows([[archiveX; X], [archiveF; F]], dim + 1);
        k = min(size(tmp, 1), totalElitePool);
        archiveX = tmp(1:k, 1:end-1);
        archiveF = tmp(1:k, end);

        % refresh bests
        improv = find(pBestF > F);
        if ~isempty(improv)
            pBestX(improv, :) = X(improv, :);
            pBestF(improv)    = F(improv);
        end
        uGroups = unique(tab(:, 4));
        for kk = 1:numel(uGroups)
            g = uGroups(kk);
            rows = find(tab(:, 4) == g);
            [minVal, loc] = min(F(rows));
            if minVal < gBestF(g)
                gBestF(g) = minVal;
                gBestX(g, :) = X(rows(loc), :);
            end
        end
        [minVal, loc] = min(F);
        if minVal < globBestF
            globBestF = minVal;
            globBestX = X(loc, :);
        end

        % update the group values
        F_eps = F + 1e-200;
        manValue = max(F_eps) - F_eps;
        groupValue = (1 - rho) * groupValue;
        for g = 1:nGroups
            groupValue(g) = groupValue(g) + sum(manValue(tab(:, 4) == g));
        end

        % refresh the table ranks
        tab(:, 7) = F;
        [~, ord] = sort(tab(:, 7)); tab(ord, 8) = (1:popSize).';
        tab = sortrows(tab, 1);
        tab(:, 8) = tab(:, 8) / popSize;

        % group transfer bookkeeping
        trIdx = find(tab(:, 9) == nTransfer);
        for t = 1:numel(trIdx)
            gList = unique(tab(:, 4));
            gScore = [gList, groupValue(gList), zeros(numel(gList), 1), zeros(numel(gList), 1)];
            gScore(:, 3) = gScore(:, 2) / sum(gScore(:, 2));
            gScore(:, 4) = cumsum(gScore(:, 3));
        end
    end

    curve(min(evalCount, maxEval):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Explorpolis rule
function ok = explorpolis_accept(cand, fCand, EliteX, EliteF, lb, ub, tau, dim)
    if isempty(EliteX)
        ok = true; return;
    end
    range = max(ub - lb, eps);
    dif = (EliteX - cand) ./ range;
    dn  = sqrt(sum(dif .^ 2, 2)) / dim;
    dominated = any((EliteF <= fCand) & (dn <= tau));
    ok = ~dominated;
end

% Guided-centre network
function [centers, FE, bsf, bsx, curve, ph, fh, hi] = ...
        build_guided_centers(popSize, maxIter, lb, ub, dim, problem, FE, maxFE, ...
                             bsf, bsx, curve, ph, fh, hi)

    N = popSize;
    halfN = round(0.5 * N);
    Old = lb + (ub - lb) .* rand(N, dim);
    New = Old;

    centers = Old(1, :);
    if FE >= maxFE, return; end

    [Fy, FE] = calculate_fitness(New', problem, FE);
    Fy = Fy(:);
    [mv, mi] = min(Fy);
    bestF = mv; bestX = New(mi, :);
    [bsf, bsx, curve, ph, fh, hi] = stampN(FE, maxFE, numel(Fy), min(bsf, mv), ...
        pick(bsf, mv, bsx, New(mi, :)), curve, New, Fy, ph, fh, hi);

    try1 = 0.5 * ones(5, 1);
    try2 = 0.5 * ones(5, 1);
    try3 = 0.5 * ones(5, 1);
    cPtr = 1;

    archiveCap = round(1.4 * N);
    archX = zeros(0, dim);
    archF = zeros(0, 1);

    counter = 0;
    sel11 = []; sel12 = []; sel21 = []; sel22 = [];
    centers = zeros(0, dim);

    for k = 1:min(N * 20, maxIter)
        if FE >= maxFE, break; end
        counter = counter + 1;
        New = Old;

        [tx2, tx1, sortIdx, rr] = rand_place(Fy, N, try1, try2, try3);
        [tx1, flag1, flag2] = show_place(tx1, 0, maxIter, counter, N, dim, tx2, sel11, sel21, sel12, sel22);

        idx1 = 1:N;
        bag  = [New; archX];
        R1 = randi(N, 1, numel(idx1));
        guard = 0;
        while any(R1 == idx1) && guard < 1000
            R1(R1 == idx1) = randi(N, 1, sum(R1 == idx1));
            guard = guard + 1;
        end
        R2 = randi(size(bag, 1), 1, numel(idx1));
        guard = 0;
        while any(R2 == R1 | R2 == idx1) && guard < 1000
            msk = (R2 == R1 | R2 == idx1);
            R2(msk) = randi(size(bag, 1), 1, nnz(msk));
            guard = guard + 1;
        end

        rndIndex = max(1, ceil(rand(1, N) * max(round(0.11 * N), 2)));
        pbest = New(sortIdx(rndIndex), :);
        T1 = New + tx1(:, ones(1, dim)) .* (pbest - New + New(R1, :) - bag(R2, :));
        T1 = bound_mid(New, lb, ub, T1);

        pick_d = mod(floor(rand(N, 1) * dim), dim) + 1;
        pickLin = (pick_d - 1) * N + (1:N)';
        mask = rand(N, dim) < rr(:, ones(1, dim));

        [KX, newF, FE] = call_place(New, sortIdx, halfN, dim, pickLin, mask, problem, T1, FE, maxFE, lb, ub);
        if isempty(newF), break; end

        [minF, minI] = min(newF);
        if minF < bestF, bestF = minF; bestX = KX(minI, :); end

        [bsf, bsx, curve, ph, fh, hi] = stampN(FE, maxFE, numel(newF), min(bsf, minF), ...
            pick(bsf, minF, bsx, KX(minI, :)), curve, KX, newF, ph, fh, hi);

        dF = abs(Fy - newF);
        improved = (Fy > newF);

        cx = rr(improved == 1);
        cy = tx1(improved == 1);
        cb = tx2(improved == 1);
        round_d = dF(improved == 1);

        if flag1
            sel11 = [sel11 numel(cy)]; %#ok<AGROW>
            sel21 = [sel21 numel(tx1(improved == 0))]; %#ok<AGROW>
            sel12 = [sel12 1]; sel22 = [sel22 1]; %#ok<AGROW>
        end
        if flag2
            sel12 = [sel12 numel(tx1(improved == 1))]; %#ok<AGROW>
            sel22 = [sel22 numel(tx1(improved == 0))]; %#ok<AGROW>
            sel11 = [sel11 1]; sel21 = [sel21 1]; %#ok<AGROW>
        end

        addX = Old(improved == 1, :);
        addF = Fy(improved == 1);
        tmpX = [archX; addX]; tmpF = [archF; addF];
        [~, uq] = unique(tmpX, 'rows');
        tmpX = tmpX(uq, :); tmpF = tmpF(uq, :);
        if size(tmpX, 1) <= archiveCap
            archX = tmpX; archF = tmpF;
        else
            rp = randperm(size(tmpX, 1)); rp = rp(1:archiveCap);
            archX = tmpX(rp, :); archF = tmpF(rp, :);
        end

        Fy = min([Fy, newF], [], 2);
        Old(improved == 1, :) = KX(improved == 1, :);
        centers(k, :) = bestX;

        if ~isempty(cx)
            w = round_d / sum(round_d);
            try1(cPtr) = (w' * (cy .^ 2)) / (w' * cy);
            try2(cPtr) = (w' * (cx .^ 2)) / (w' * cx);
            try3(cPtr) = (w' * (cb .^ 2)) / (w' * cb);
            cPtr = cPtr + 1; if cPtr > 5, cPtr = 1; end
        end
    end

    if isempty(centers)
        centers = bestX;
    end
end

function [KX, newF, FE] = call_place(New, sortIdx, halfN, dim, pickLin, mask, problem, T1, FE, maxFE, lb, ub)
    if rand < 0.4 && halfN >= 2
        d = pdist2(New, New(sortIdx(1), :), 'euclidean');
        [~, ord] = sort(d, 'ascend');
        nb = New(ord(1:halfN), :);
        mx = mean(nb);
        C = 1 / (halfN - 1) * (nb - mx(ones(halfN, 1), :))' * (nb - mx(ones(halfN, 1), :));
        C = triu(C) + triu(C, 1)';
        [R, D] = eig(C);
        if max(diag(D)) > 1e20 * min(diag(D))
            C = C + (max(diag(D)) / 1e20 - min(diag(D))) * eye(dim);
            [R, ~] = eig(C);
        end
        R = real(R);
        Xr  = New * R;
        T1r = T1 * R;
        Q = Xr;
        Q(pickLin) = T1r(pickLin);
        Q(mask)    = T1r(mask);
        KX = Q * R';
    else
        KX = New;
        KX(pickLin) = T1(pickLin);
        KX(mask)    = T1(mask);
    end
    KX = min(max(KX, lb), ub);
    n = size(KX, 1);
    if FE >= maxFE
        newF = [];
        return;
    end
    nEval = min(n, maxFE - FE);
    [f, FE] = calculate_fitness(KX(1:nEval, :)', problem, FE);
    newF = inf(n, 1);
    newF(1:nEval) = f(:);
end

function [tx1, flag1, flag2] = show_place(tx1, temp_f, maxIter, counter, N, dim, tx2, s11, s21, s12, s22)
    flag1 = false; flag2 = false;
    if temp_f <= maxIter / 2
        if counter <= 20
            if rand < 0.5
                tx1 = 0.5 .* (sin(2 * pi * 0.5 * counter + pi) .* ((2745 - counter) / 2745) + 1) .* ones(N, dim);
                flag1 = true;
            else
                tx1 = 0.5 * (sin(2 * pi .* tx2(:, ones(1, dim)) .* counter) .* (counter / 2745) + 1) .* ones(N, dim);
                flag2 = true;
            end
        else
            n2 = sum(s11(max(counter - 20, 1):counter - 1));
            n3 = sum(s21(max(counter - 20, 1):counter - 1));
            r1 = n2 / (n2 + n3 + eps) + 0.01;

            c1 = sum(s12(max(counter - 20, 1):counter - 1));
            c2 = sum(s22(max(counter - 20, 1):counter - 1));
            r2 = c1 / (c1 + c2 + eps) + 0.01;

            p1 = r1 / (r1 + r2); p2 = r2 / (r1 + r2);
            if p1 > p2
                tx1 = 0.5 .* (sin(2 * pi * 0.5 * counter + pi) .* ((2745 - counter) / 2745) + 1) .* ones(N, dim);
                flag1 = true;
            else
                tx1 = 0.5 * (sin(2 * pi .* tx2(:, ones(1, dim)) .* counter) .* (counter / 2745) + 1) .* ones(N, dim);
                flag2 = true;
            end
        end
    end
end

function [tx2, tx1, sortIdx, rr] = rand_place(F, N, try1, try2, try3)
    [~, sortIdx] = sort(F, 'ascend');
    k = ceil(5 * rand(N, 1));
    x1 = try1(k); x2 = try2(k); x3 = try3(k);

    rr = normrnd(x2, 0.1);
    rr(x2 == -1) = 0; rr = min(max(rr, 0), 1);

    tx1 = x1 + 0.1 * tan(pi * (rand(N, 1) - 0.5));
    w = find(tx1 <= 0);
    guard = 0;
    while ~isempty(w) && guard < 1000
        tx1(w) = x1(w) + 0.1 * tan(pi * (rand(numel(w), 1) - 0.5));
        w = find(tx1 <= 0);
        guard = guard + 1;
    end
    tx1(tx1 <= 0) = eps;

    tx2 = x3 + 0.1 * tan(pi * (rand(N, 1) - 0.5));
    w = find(tx2 <= 0);
    guard = 0;
    while ~isempty(w) && guard < 1000
        tx2(w) = x3(w) + 0.1 * tan(pi * (rand(numel(w), 1) - 0.5));
        w = find(tx2 <= 0);
        guard = guard + 1;
    end
    tx2(tx2 <= 0) = eps;

    tx1 = min(tx1, 1);
    tx2 = min(tx2, 1);
end

function T1 = bound_mid(New, lb, ub, T1)
    [n, d] = size(New);
    for i = 1:n
        for j = 1:d
            if T1(i, j) < lb(j), T1(i, j) = (T1(i, j) + lb(j)) / 2; end
            if T1(i, j) > ub(j), T1(i, j) = (T1(i, j) + ub(j)) / 2; end
        end
    end
end

% Back-off
function [X, F, FE, bsf, bsx, curve, ph, fh, hi] = ...
        backoff_try(backList, X, F, Xnet, iter, ub, lb, maxIter, problem, FE, ...
                    bsf, bsx, curve, ph, fh, hi, maxFE)

    pool = [[X F]; [backList(:, 3:end) backList(:, 2)]];
    pool = sortrows(pool, size(pool, 2));

    [~, worstIdx] = max(F);
    if rand <= iter / maxIter * 2 && FE < maxIter
        idx = min(iter, size(Xnet, 1));
        X(worstIdx, :) = Xnet(idx, :);
        [F(worstIdx), FE] = calculate_fitness(X(worstIdx, :)', problem, FE);
        [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxIter, F(worstIdx), X(worstIdx, :), ...
            bsf, bsx, curve, X, F, ph, fh, hi);
    end

    newLink = pool(1, :);
    p = 1;
    while p < size(pool, 1)
        p = p + 1;
        for i = 1:size(newLink, 1)
            if p > size(pool, 1), break; end
            d1 = sqrt(sum(abs(pool(p, 1:end-1) - newLink(i, 1:end-1)) ./ (ub - lb), 2));
            d2 = sqrt(sum(abs(pool(p+1:end, 1:end-1) - newLink(i, 1:end-1)) ./ (ub - lb), 2));
            e = find(d2 <= d1);
            if ~isempty(e)
                e = e + p;
                pool(e, :) = [];
            end
        end
        if p > size(pool, 1), break; end
        newLink(end + 1, :) = pool(p, :); %#ok<AGROW>
    end

    for i = size(newLink, 1):-1:1
        d = sqrt(sum(abs(newLink(i, 1:end-1) - backList(:, 3:end)) .^ 2, 2));
        t = find(d == 0);
        if ~isempty(t)
            if all(backList(t, 1) ~= worstIdx)
                X(backList(t, 1), :) = backList(t, 3:end);
                F(backList(t, 1))    = backList(t, 2);
            end
        end
    end
end

% small helpers
function xs = pick(bsf, f, bsx, x)
    if f < bsf, xs = x; else, xs = bsx; end
end

function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, x, bsf, bsx, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

function [bsf, bsx, curve, ph, fh, hi] = stampN(FE, maxFE, n, bsf, bsx, curve, X, Fit, ph, fh, hi)
    for k = 1:n
        ec = FE - n + k;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

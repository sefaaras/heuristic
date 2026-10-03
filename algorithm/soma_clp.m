% ----------------------------------------------------------------------- %
% SOMA with CLustering-aided migration and adaptive Perturbation vector control (SOMA-CLP)
% CEC 2021 competition -- 8th provisional overall, 7th in the shifted rankings
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 100                          % Population size
%   NPL = 10                          % k-means clusters, so at most 10 cluster leaders
%   step = 0.33, pathLength = 3.0     % Exploration path t = 0, 0.33, ..., 2.97 (10 points)
%   stepL = 0.11, pathLengthL = 2.0   % Exploitation path t = 0.11, ..., 1.98 (18 points)
%   prt = 0.08 + 0.9*FE/maxFe         % Perturbation rate, recomputed before every jump
%   kmeans_iter = 100                 % Cap on the Lloyd iterations
%
% Algorithm Concept:
%   - Exploration (All-To-Random): every individual walks towards a random other
%     individual, and every evaluated point is stored in a memory M
%   - M is split into NPL clusters by k-means; the best point of each cluster
%     becomes a cluster leader
%   - Exploitation (All-To-Leader): every individual walks towards a cluster
%     leader drawn with linear rank weights
%   - A path point replaces its individual only if strictly better; the population
%     is updated at the end of each phase
%   - A coordinate moves only if a uniform draw is below prt, which rises with the
%     spent budget; coordinates that leave the box are redrawn uniformly
%
% Reference:
% Tomas Kadavy, Michal Pluhacek, Adam Viktorin, Roman Senkerik,
% SOMA-CLP for competition on bound constrained single objective numerical
% optimization benchmark, Proceedings of the Genetic and Evolutionary Computation
% Conference Companion (GECCO '21), 2021, pp. 11-12.
% https://doi.org/10.1145/3449726.3463286
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the official C++ release, SOMA_CLP_CEC21.cpp and kmeans.cpp in
% github.com/TBU-AILab/SOMA_CLP (the URL named in the paper). Release quirks kept:
% the exploration path starts at t = 0 and so re-evaluates the individual itself;
% its leader index is floor(U*(NP-1)), so individual NP is never an exploration
% leader; the rank weights NPL..1 follow cluster-id order, not fitness (leaderRank
% never sorts the leaders), and sum to floor(n/2)*(n+1), which starves the last
% leaders when n is odd. Harness: per-dimension bounds replace the fixed [-100,100]
% box and the k-means seeding (C rand() there) uses MATLAB's stream. k-means is
% skipped once the budget is spent: on fewer than NPL points the release's seeding
% loop never terminates, and nothing is evaluated after it anyway.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = soma_clp(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NP          = 100;
    NPL         = 10;
    step        = 0.33;
    pathLength  = 3.0;
    stepL       = 0.11;
    pathLengthL = 2.0;
    kmeans_iter = 100;

    % The release accumulates t += step in double, so the path lengths come from the same sums
    tA = zeros(1, 0);
    t = 0;
    while t <= pathLength
        tA(end + 1) = t; %#ok<AGROW>
        t = t + step;
    end
    tL = zeros(1, 0);
    t = stepL;
    while t <= pathLengthL
        tL(end + 1) = t; %#ok<AGROW>
        t = t + stepL;
    end

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    NP = max(3, min(NP, maxFE));

    % Drawn individual by individual, the release's order
    pop = lb + rand(dim, NP)' .* span;
    [fv, FE] = calculate_fitness(pop', problem, FE);
    fit = fv(:);
    bsf = inf;
    bsf_solution = pop(1, :);
    for q = 1:NP
        if fit(q) < bsf
            bsf = fit(q);
            bsf_solution = pop(q, :);
        end
        curve(q) = bsf;
    end
    [population_history, fitness_history, history_index] = record_history( ...
        FE, pop, fit, population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        % Exploration (All-To-Random): leaders are read from pop, improvements go to popT
        popT = pop;
        fitT = fit;
        memX = zeros(NP * numel(tA), dim);
        memF = zeros(NP * numel(tA), 1);
        nMem = 0;
        for i = 1:NP
            if FE >= maxFE
                break;
            end
            leader = i;
            while leader == i
                leader = floor(rand * (NP - 1)) + 1;  % (int)doubleRand(0, NP-1): never NP
            end
            n = min(numel(tA), maxFE - FE);
            P = make_path(pop(i, :), pop(leader, :), tA(1:n), FE, maxFE, lb, ub, span, dim);
            [f, popT, fitT, FE, bsf, bsf_solution, curve, population_history, fitness_history, history_index] = ...
                walk(i, P, popT, fitT, problem, FE, bsf, bsf_solution, curve, ...
                     population_history, fitness_history, history_index, maxFE);
            memX(nMem + 1:nMem + n, :) = P;
            memF(nMem + 1:nMem + n)    = f;
            nMem = nMem + n;
        end
        pop = popT;
        fit = fitT;
        if FE >= maxFE
            break;
        end

        lead = cluster_leaders(memX(1:nMem, :), memF(1:nMem), NPL, kmeans_iter);
        nL = numel(lead);
        % Rank weights nL..1 in cluster-id order; the release's integer total (n/2)*(n+1)
        cw = cumsum(nL:-1:1);
        rank_total = floor(nL / 2) * (nL + 1);

        % Exploitation (All-To-Leader)
        popT = pop;
        fitT = fit;
        for i = 1:NP
            if FE >= maxFE
                break;
            end
            r = floor(rand * rank_total);
            li = find(r < cw, 1);
            if isempty(li)
                li = nL;
            end
            n = min(numel(tL), maxFE - FE);
            P = make_path(pop(i, :), memX(lead(li), :), tL(1:n), FE, maxFE, lb, ub, span, dim);
            [~, popT, fitT, FE, bsf, bsf_solution, curve, population_history, fitness_history, history_index] = ...
                walk(i, P, popT, fitT, problem, FE, bsf, bsf_solution, curve, ...
                     population_history, fitness_history, history_index, maxFE);
        end
        pop = popT;
        fit = fitT;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Rows are the path points x + (xl - x)*t*prtVector; jump s sees prt at FE + s - 1, as in the release
function P = make_path(x, xl, tt, FE, maxFE, lb, ub, span, dim)
    n = numel(tt);
    prt = 0.08 + 0.9 * ((FE + (0:n - 1)) / maxFE);
    PV = rand(dim, n) <= prt;
    P = x' + (xl' - x') .* tt .* PV;
    out = P > ub' | P < lb' | ~isfinite(P);
    [rw, ~] = find(out);
    P(out) = lb(rw)' + rand(numel(rw), 1) .* span(rw)';
    P = P';
end

% Evaluates a path; each point updates the best-so-far and the release's popTemp row i if strictly better
function [f, popT, fitT, FE, bsf, bsf_solution, curve, ph, fh, hi] = walk(i, P, popT, fitT, problem, FE, ...
        bsf, bsf_solution, curve, ph, fh, hi, maxFE)
    fv = calculate_fitness(P', problem, FE);
    f = fv(:);
    for q = 1:numel(f)
        FE = FE + 1;
        if f(q) < bsf
            bsf = f(q);
            bsf_solution = P(q, :);
        end
        curve(FE) = bsf;
        if f(q) < fitT(i)
            popT(i, :) = P(q, :);
            fitT(i)    = f(q);
        end
        [ph, fh, hi] = record_history(FE, popT, fitT, ph, fh, hi, maxFE);
    end
end

% kmeans.cpp's Lloyd loop, then the best member of each non-empty cluster in cluster-id order
function lead = cluster_leaders(X, f, K, iters)
    n = size(X, 1);
    K = min(K, n);
    seeds = randperm(n, K);
    C = X(seeds, :);
    assign = zeros(n, 1);
    assign(seeds) = 1:K;
    d = zeros(n, K);
    for it = 1:iters
        for k = 1:K
            d(:, k) = sum((X - C(k, :)) .^ 2, 2);
        end
        [~, near] = min(d, [], 2);
        changed = any(near ~= assign);
        assign = near;
        for k = 1:K
            m = assign == k;
            if any(m)
                C(k, :) = mean(X(m, :), 1);
            end
        end
        if ~changed
            break;
        end
    end
    lead = zeros(1, 0);
    for k = 1:K
        m = find(assign == k);
        if ~isempty(m)
            [~, b] = min(f(m));
            lead(end + 1) = m(b); %#ok<AGROW>
        end
    end
end

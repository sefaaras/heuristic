% ----------------------------------------------------------------------- %
% Evolutionary Population Management based Particle Swarm Optimization (EPM-PSO)
% Case-3 of the paper; its most frequent Proposed Case on CSBP-2 (5 of 12)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 200                  % Parents per epoch; as many children
%   w = 1, x0.99 per epoch      % Inertia weight, damped geometrically
%   c1 = 1.5, c2 = 2.0          % Pulls toward pbest - x_R1 and gbest - x_i
%   VelMax = 0.1*(ub - lb)      % Velocity clamp; velocity mirrored at the bounds
%   phase switch = 0.5*maxFE    % R1: paired mate before, random row after
%
% Algorithm Concept:
%   - Eq. (11): v = w*v + c1*r1.*(pbest_i - x_R1) + c2*r2.*(gbest - x_i); parents
%     survive and their children form a separate swarm (Hypothesis-1)
%   - R1 (Hypothesis-3, Case-3): mate i +- N/2 in the first half of the budget,
%     then random in rows 1..N/4 (i <= N/2) or N/4+1..N/2 (i > N/2)
%   - Epoch end: parents and children merged into U (2N); gbest = best of U
%   - Next swarm (Hypothesis-2, Method-2): N/2 fitness elites with no
%     parent-child pair, then N/2 more in FDB-score order relative to gbest
%   - pbest and velocity travel with each selected individual
%
% Reference:
% Furkan Ustunsoy, Hamdi Tolga Kahraman, H. Huseyin Sayan, Yusuf Sonmez,
% Evolutionary population management for the design of metaheuristic search
% algorithms: Three improved algorithms, real-time charge scheduling problems,
% optimal solutions and stability analysis,
% Knowledge-Based Systems 328 (2025) 114221.
% https://doi.org/10.1016/j.knosys.2025.114221
% Components:
%   PSO - R. Eberhart, J. Kennedy, A new optimizer using particle swarm theory,
%     MHS'95, Proceedings of the Sixth International Symposium on Micro Machine
%     and Human Science, 1995, 39-43, https://doi.org/10.1109/MHS.1995.494215
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' package (FX 181760, epm_ps-1.0.2, EPM_PSO_V3.m +
% pop_productions_2.m) on the release's own PSO, not pso.m (N 30, w 0.9->0.4,
% c 2/2, Vmax 20 %). The paper names no single case: Table 3 defines Cases 1-4,
% Table 15 a Proposed Case per CSBP-2 problem; Case-3 is the user's choice as the
% most frequent there. Harness: the 5000*D switch is read as 0.5*maxFE (equal at
% maxFE = 10000*D) over main-loop evaluations, as the release counts; the literal
% rows [1,50]/[51,100] are written N/4, N/2 (equal at N = 200); the release's N
% extra evaluations are not spent, a truncated last epoch evaluates only what is
% left. Children are evaluated as one batch, exact since none reads another child.
% The production re-uses stored fitness and spends no evaluations.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = epm_pso(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    nPop = 200;
    w = 1;
    wdamp = 0.99;
    c1 = 1.5;
    c2 = 2.0;
    VelMax = 0.1 * (ub - lb);
    VelMin = -VelMax;

    FE = 0;
    curve = nan(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Release order: position then velocity, particle by particle, both uniform over the box
    pos = zeros(nPop, dim);
    vel = zeros(nPop, dim);
    for i = 1:nPop
        pos(i, :) = unifrndBox(lb, ub);
        vel(i, :) = unifrndBox(lb, ub);
    end
    [cost, FE] = calculate_fitness(pos', problem, FE);
    cost = cost(:);
    pbest = pos;
    pbestCost = cost;

    [gCost, gi] = min(cost);
    gPos = pos(gi, :);
    bestF = gCost;
    bestX = gPos;

    for e = 1:min(nPop, maxFE)
        curve(e) = bestF;
        [population_history, fitness_history, history_index] = record_history(...
            e, pos, cost, population_history, fitness_history, history_index, maxFE);
    end

    nfeval = 0;                 % main-loop evaluations, as the release counts them
    while FE < maxFE
        nb = min(nPop, maxFE - FE);
        childPos = zeros(nb, dim);
        childVel = zeros(nb, dim);
        for i = 1:nb
            if nfeval + i - 1 <= 0.5 * maxFE
                if i <= nPop / 2
                    r1 = i + nPop / 2;
                else
                    r1 = i - nPop / 2;
                end
            else
                if i <= nPop / 2
                    r1 = randi([1, nPop / 4]);
                else
                    r1 = randi([nPop / 4 + 1, nPop / 2]);
                end
            end
            v = w * vel(i, :) + c1 * rand(1, dim) .* (pbest(i, :) - pos(r1, :)) ...
                + c2 * rand(1, dim) .* (gPos - pos(i, :));    % Eq. (11)
            v = max(v, VelMin);
            v = min(v, VelMax);
            x = pos(i, :) + v;
            out = x < lb | x > ub;
            v(out) = -v(out);
            x = max(x, lb);
            x = min(x, ub);
            childPos(i, :) = x;
            childVel(i, :) = v;
        end

        [childCost, FE] = calculate_fitness(childPos', problem, FE);
        childCost = childCost(:);
        nfeval = nfeval + nb;

        % Children start from the parent's pbest and improve it on their own
        childPbest = pbest(1:nb, :);
        childPbestCost = pbestCost(1:nb);
        for i = 1:nb
            if childCost(i) < childPbestCost(i)
                childPbest(i, :) = childPos(i, :);
                childPbestCost(i) = childCost(i);
            end
        end

        for i = 1:nb
            if childCost(i) < bestF
                bestF = childCost(i);
                bestX = childPos(i, :);
            end
            ec = FE - nb + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bestF;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pos, cost, population_history, fitness_history, history_index, maxFE);
            end
        end

        if nb < nPop || FE >= maxFE
            break;
        end

        U = [pos; childPos];
        FU = [cost; childCost];
        Upbest = [pbest; childPbest];
        UpbestCost = [pbestCost; childPbestCost];
        Uvel = [vel; childVel];

        [gCost, gi] = min(FU);
        gPos = U(gi, :);

        % Hypothesis-2: next swarm from the merged parents + children
        idx = epmProduceMethod2(U, FU);
        pos = U(idx, :);
        cost = FU(idx);
        pbest = Upbest(idx, :);
        pbestCost = UpbestCost(idx);
        vel = Uvel(idx, :);

        w = w * wdamp;
    end

    % NaN marks a slot no evaluation reached; 0 is a legal fitness value
    for k = 2:maxFE
        if isnan(curve(k))
            curve(k) = curve(k - 1);
        end
    end

    best_fitness = bestF;
    best_solution = bestX;
end

% unifrnd(lb, ub, [1 D]) without the Statistics Toolbox, same formula and draws
function r = unifrndBox(lb, ub)
    mu = lb / 2 + ub / 2;
    sig = ub / 2 - lb / 2;
    r = mu + sig .* (2 * rand(1, numel(lb)) - 1);
end

% EPM Hypothesis-2, Method-2 (paper Algorithm-2 line 17; released pop_productions_2.m)
function idx = epmProduceMethod2(U, FU)
    n2 = size(U, 1);
    [~, order] = sort(FU);
    FL = zeros(1, n2);
    idx = zeros(1, n2 / 2);
    c = 0;
    for i = 1:n2 / 2
        if ~any(FL == order(i))
            [idx, FL, c] = admit(idx, FL, c, order(i), n2);
        end
        if c == n2 / 4, break; end
    end
    [~, bestIndex] = min(FU);
    mates = fdbRanking(U, FU, U(bestIndex, :));
    for i = 1:n2 / 4
        for j = 1:n2
            if ~any(FL == mates(j))
                [idx, FL, c] = admit(idx, FL, c, mates(j), n2);
                break;
            end
        end
    end
end

% Takes row k into the next population and forbids it and its parent-child twin
function [idx, FL, c] = admit(idx, FL, c, k, n2)
    idx(c + 1) = k;
    FL(2 * c + 1) = k;
    if k > n2 / 2
        FL(2 * c + 2) = k - n2 / 2;
    else
        FL(2 * c + 2) = k + n2 / 2;
    end
    c = c + 1;
end

% Rows ordered by FDB score (normalised fitness + Manhattan distance to ref), best first
function order = fdbRanking(U, FU, ref)
    n = size(U, 1);
    if min(FU) == max(FU)
        order = randperm(n);
        return;
    end
    dist = zeros(n, 1);
    for j = 1:size(U, 2)
        dist = dist + abs(ref(j) - U(:, j));   % accumulated in the release's dimension order
    end
    minF = min(FU); rangeF = max(FU) - minF;
    minD = min(dist); rangeD = max(dist) - minD;
    score = (1 - (FU - minF) / rangeF) + (dist - minD) / rangeD;
    [~, order] = sort(score, 'descend');
end

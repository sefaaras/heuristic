% ----------------------------------------------------------------------- %
% Water Cycle Algorithm (WCA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Npop   = 50       % Raindrops: the sea, Nsr - 1 rivers and Npop - Nsr streams
%   Nsr    = 4        % Rivers plus the sea
%   C      = 2        % Flow step x + C*rand*(target - x), fixed in the release
%   dmax   = 1e-16    % Evaporation distance, times (1 - 1/max_it) each iteration
%   pRain  = 0.1      % Chance a river's streams rain anew whatever its distance
%   max_it = ceil((maxFe - Npop)/(Npop - 1))   % Iterations the dmax decay spans
%
% Algorithm Concept:
%   - The best raindrop is the sea, the next Nsr - 1 the rivers; the streams are
%     shared out once, in proportion to the gap to the (Nsr + 1)-th best cost
%   - Streams flow toward their river or the sea, x + 2*rand*(target - x): one
%     scalar rand per sea stream, one per coordinate for a river's streams
%   - Every move is kept; a stream better than its river, or a river better than
%     the sea, swaps places with it, so the sea is the best point evaluated
%   - After the streams, each river flows toward the sea the same way
%   - Evaporation: a river within dmax of the sea, or with chance pRain, rains all
%     its streams anew uniformly in the box; a sea stream within dmax likewise
%
% Reference:
% Hadi Eskandar, Ali Sadollah, Ardeshir Bahreininejad, Mohd Hamdi,
% Water cycle algorithm -- A novel metaheuristic optimization method for solving
% constrained engineering optimization problems,
% Computers & Structures 110-111 (2012) 151-166.
% https://doi.org/10.1016/j.compstruc.2012.07.010
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from co-author Ali Sadollah's release, File Exchange 56339 v1.3.0.0 of
% 1 Sep 2016 (WCA.m "Standard Version", revised 4 Apr 2016); seeded runs are
% bit-identical to it. Kept as released: NS is set once from the initial costs;
% rain is uniform in the box and a rained stream keeps its old cost until it moves.
% Changed: max_it counts the Npop - 1 evaluations an iteration costs (the release
% reports Npop*max_it); the closing re-evaluation of the sea is dropped; where the
% NS repair's round(NS(i)/6) is 0 the release spins forever (2.7 % of CEC2020RW
% starts, F13 35 %), here group 1 gives one stream instead; non-finite initial
% costs (a stream(NaN) index crash) take the release's all-costs-equal split; the
% best is updated at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wca(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    Npop = 50;
    Nsr  = 4;
    dmax = 1e-16;
    N_stream = Npop - Nsr;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Raindrop by raindrop, as released
    X = lb + (ub - lb) .* rand(dim, Npop)';
    m = min(Npop, maxFE);
    [fit, FE] = calculate_fitness(X(1:m, :)', problem, FE);
    fit = fit(:)';

    best_fitness  = inf;
    best_solution = X(1, :);
    for e = 1:m
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = X(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, X(1:m, :), fit, population_history, fitness_history, ...
            history_index, maxFE);
    end
    if FE >= maxFE
        curve(min(FE, maxFE):end) = best_fitness;
        return;
    end

    % Row 1 is the sea, rows 2..Nsr the rivers, row Nsr + j stream j
    [~, index] = sort(fit);
    P = X(index, :);
    cost = fit(index);
    Xrec = P;   % where each cost(r) was measured; rain moves P(r, :) ahead of it

    % Streams per river and sea, in proportion to the cost gap to stream 1
    cs = [cost(1); cost(2:Nsr)'; cost(Nsr + 1)];
    f = 0;
    if length(unique(cs)) ~= 1
        CN = cs - max(cs);
    else
        CN = cs;
        f = 1;
    end
    NS = round(abs(CN / sum(CN)) * N_stream);
    if ~all(isfinite(NS))
        % The release indexes stream(NaN) and stops; this is its all-equal branch
        NS = round(abs(ones(Nsr + 1, 1) / (Nsr + 1)) * N_stream);
        f = 1;
    end
    if f ~= 1
        NS(end) = [];
    end
    NS = sort(NS, 'descend');
    i = Nsr;
    while sum(NS) > N_stream
        if NS(i) > 1
            NS(i) = NS(i) - 1;
        else
            i = i - 1;
        end
    end
    while sum(NS) < N_stream
        NS(1) = NS(1) + 1;
    end
    zi = find(NS == 0);
    for i = 1:size(zi, 1)
        while NS(zi(i)) == 0
            give = round(NS(i) / 6);   % group i donates, as released
            if give == 0
                NS(zi(i)) = 1;   % the release loops forever here
                NS(1) = NS(1) - 1;
            else
                NS(zi(i)) = NS(zi(i)) + give;
                NS(i) = NS(i) - give;
            end
        end
    end
    NS = sort(NS, 'descend');
    NB = NS(2:end);
    first = Nsr + cumsum(NS);   % stream j of river k sits in row first(k) + j

    perIter = NS(1) + sum(NB(1:Nsr - 1)) + Nsr - 1;
    max_it = ceil((maxFE - m) / perIter);

    while FE < maxFE
        % Streams flow to the sea, one scalar rand per stream
        for j = 1:NS(1)
            r = Nsr + j;
            P(r, :) = P(r, :) + 2 .* rand(1) .* (P(1, :) - P(r, :));
            P(r, :) = min(P(r, :), ub);
            P(r, :) = max(P(r, :), lb);
            [cost(r), FE] = calculate_fitness(P(r, :)', problem, FE);
            Xrec(r, :) = P(r, :);
            x = P(r, :);
            fx = cost(r);
            if cost(r) < cost(1)
                swap(r, 1);
            end
            stamp(x, fx);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Streams flow to their rivers, one rand per coordinate
        for k = 1:Nsr - 1
            for j = 1:NB(k)
                r = first(k) + j;
                P(r, :) = P(r, :) + 2 .* rand(1, dim) .* (P(1 + k, :) - P(r, :));
                P(r, :) = min(P(r, :), ub);
                P(r, :) = max(P(r, :), lb);
                [cost(r), FE] = calculate_fitness(P(r, :)', problem, FE);
                Xrec(r, :) = P(r, :);
                x = P(r, :);
                fx = cost(r);
                if cost(r) < cost(1 + k)
                    swap(r, 1 + k);
                    if cost(1 + k) < cost(1)
                        swap(1 + k, 1);
                    end
                end
                stamp(x, fx);
                if FE >= maxFE, break; end
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Rivers flow to the sea
        for j = 1:Nsr - 1
            r = 1 + j;
            P(r, :) = P(r, :) + 2 .* rand(1, dim) .* (P(1, :) - P(r, :));
            P(r, :) = min(P(r, :), ub);
            P(r, :) = max(P(r, :), lb);
            [cost(r), FE] = calculate_fitness(P(r, :)', problem, FE);
            Xrec(r, :) = P(r, :);
            x = P(r, :);
            fx = cost(r);
            if cost(r) < cost(1)
                swap(r, 1);
            end
            stamp(x, fx);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Evaporation and raining; rand is drawn only when the distance test fails
        for k = 1:Nsr - 1
            if (norm(P(1 + k, :) - P(1, :)) < dmax) || rand < 0.1
                for j = 1:NB(k)
                    r = first(k) + j;
                    P(r, :) = lb + rand(1, dim) .* (ub - lb);
                end
            end
        end
        for j = 1:NS(1)
            r = Nsr + j;
            if norm(P(r, :) - P(1, :)) < dmax
                P(r, :) = lb + rand(1, dim) .* (ub - lb);
            end
        end
        dmax = dmax - (dmax / max_it);
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Exchange two raindrops' roles: position, cost and recorded point travel together
    function swap(a, b)
        P([a b], :) = P([b a], :);
        cost([a b]) = cost([b a]);
        Xrec([a b], :) = Xrec([b a], :);
    end

    % Best-so-far at this evaluation, then one history sample of (Xrec, cost)
    function stamp(x, fx)
        if fx < best_fitness
            best_fitness  = fx;
            best_solution = x;
        end
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            FE, Xrec, cost, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

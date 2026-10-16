% ----------------------------------------------------------------------- %
% Dwarf Mongoose Optimization Algorithm (DMO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop        = 20               % Family size, babysitters included
%   nBabysitter = 3                % Babysitters; the other 17 mongooses are stored
%   L           = round(0.6*D*3)   % Babysitter exchange parameter (failure count)
%   peep        = 2                % Alpha vocalisation, phi ~ U(-peep/2, peep/2)
%
% Algorithm Concept:
%   - Alpha group: an alpha female drawn by roulette on exp(-f/mean f) moves to
%     x_a + phi.*(x_a - x_k), k a random other mongoose; 17 draws per iteration
%   - Scout group: every mongoose i makes the same move against a random k
%   - Greedy replacement, in place; a rejected move adds 1 to the mover's counter
%   - Babysitter exchange: mongooses 1..3 are redrawn uniformly once their
%     counter reaches L, which resets it; success never resets a counter
%
% Reference:
% Jeffrey O. Agushaka, Absalom E. Ezugwu, Laith Abualigah,
% Dwarf Mongoose Optimization Algorithm,
% Computer Methods in Applied Mechanics and Engineering 391 (2022) 114570.
% https://doi.org/10.1016/j.cma.2022.114570
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from File Exchange 105125 (v1.0.0, DMOA.m, 13 Jan 2022); seeded runs
% are bit-identical to it given the same clamp: the release has no bound
% handling, here every move is clamped to [lb, ub]. Kept as released, against
% the paper: the move is ABC's x + phi.*(x - x_k) with phi = (peep/2)*U(-1,1),
% and the paper's scout move (sleeping mound, CF, M) is computed and thrown
% away, so scouts repeat the foraging move (its rand draws are kept; CF, frozen
% at it = 1, was the only use of MaxIt, so the loop just runs to maxFe). The
% weights favour the worse mongooses when mean f < 0 (kept); when they are not
% a finite distribution (mean 0, overflow, Inf/NaN) the alpha is drawn
% uniformly. The best is tracked per evaluation, not after the exchange.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dmo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    nPop        = 20;
    nBabysitter = 3;
    nAlpha      = nPop - nBabysitter;   % babysitters are not separate individuals in the release
    nScout      = nAlpha;
    L           = round(0.6 * dim * nBabysitter);
    peep        = 2;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = zeros(nAlpha, dim);
    for i = 1:nAlpha
        X(i, :) = unifrnd_row(lb, ub, dim);   % one row of draws per mongoose, as in the release
    end
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [fbest, bidx] = min(fit);
    Xbest = X(bidx, :);
    for e = 1:min(nAlpha, maxFE)
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    C = zeros(1, nAlpha);   % abandonment counter

    while FE < maxFE
        % Alpha probabilities via Yarpiz-ABC's cost-to-fitness map, fixed for the whole pass
        F = exp(-fit / mean(fit));
        P = F / sum(F);
        if ~all(isfinite(P))
            P = ones(1, nAlpha) / nAlpha;
        end
        Pcum = cumsum(P);

        % Foraging led by the alpha female: the selected alpha is the one that moves
        for m = 1:nAlpha
            i = find(rand <= Pcum, 1, 'first');
            if isempty(i), i = nAlpha; end   % cumsum can end just below 1
            K = [1:i-1, i+1:nAlpha];
            k = K(randi([1 numel(K)]));
            phi = (peep / 2) * unifrnd_row(-1, 1, dim);
            xn = X(i, :) + phi .* (X(i, :) - X(k, :));
            xn = min(max(xn, lb), ub);
            [fn, FE] = calculate_fitness(xn', problem, FE);
            if fn <= fit(i)
                X(i, :) = xn;
                fit(i)  = fn;
                if fn < fbest
                    fbest = fn;
                    Xbest = xn;
                end
            else
                C(i) = C(i) + 1;
            end
            if FE <= maxFE
                curve(FE) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Scout group: in the release the same move again, mongoose by mongoose
        for i = 1:nScout
            K = [1:i-1, i+1:nAlpha];
            k = K(randi([1 numel(K)]));
            phi = (peep / 2) * unifrnd_row(-1, 1, dim);
            xn = X(i, :) + phi .* (X(i, :) - X(k, :));
            xn = min(max(xn, lb), ub);
            [fn, FE] = calculate_fitness(xn', problem, FE);
            if fn <= fit(i)
                X(i, :) = xn;
                fit(i)  = fn;
                if fn < fbest
                    fbest = fn;
                    Xbest = xn;
                end
            else
                C(i) = C(i) + 1;
            end
            if FE <= maxFE
                curve(FE) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Babysitter exchange: only mongooses 1..nBabysitter are ever redrawn
        for i = 1:nBabysitter
            if C(i) >= L
                X(i, :) = unifrnd_row(lb, ub, dim);
                [fit(i), FE] = calculate_fitness(X(i, :)', problem, FE);
                C(i) = 0;
                if fit(i) < fbest
                    fbest = fit(i);
                    Xbest = X(i, :);
                end
                if FE <= maxFE
                    curve(FE) = fbest;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
            end
        end

        % The release's discarded sleeping-mound move draws one rand per scout
        rand(1, nScout);
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% unifrnd(a, b, [1 d]) with the toolbox's own arithmetic, so seeded runs match the release
function r = unifrnd_row(a, b, d)
    a2 = a / 2;
    b2 = b / 2;
    r = (a2 + b2) + (b2 - a2) .* (2 * rand(1, d) - 1);
end

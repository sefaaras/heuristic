% ----------------------------------------------------------------------- %
% Parrot Optimizer (PO)
% Stored as parrot; the acronym PO collides with po (Political Optimizer)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 30          % Population size (parrots)
%   alpha = rand / 5    % Communication step, redrawn every iteration
%   sita  = rand * pi   % Fear-of-strangers angle, redrawn every iteration
%   beta  = 1.5         % Levy-flight exponent of foraging and staying
%
% Algorithm Concept:
%   - Every parrot draws one of four behaviours uniformly at random per iteration
%   - Foraging: Levy-scaled offset from the best plus a decaying pull on the swarm mean
%   - Staying: Levy step proportional to the best position plus a decaying normal shift
%   - Communicating: step along the offset from the swarm mean, or a decaying
%     exponential nudge, each with probability 1/2
%   - Fear of strangers: cosine-weighted moves towards the best position
%   - No selection: new positions replace the old; the best is updated per parrot
%     and steers the parrots after it in the same iteration
%
% Reference:
% Junbo Lian, Guohua Hui, Ling Ma, Ting Zhu, Xincan Wu, Ali Asghar Heidari, Yi Chen,
% Huiling Chen,
% Parrot optimizer: Algorithm and applications to medical problems,
% Computers in Biology and Medicine 172 (2024) 108064.
% https://doi.org/10.1016/j.compbiomed.2024.108064
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' PO.m (version 2.0, last update Apr 07 2024) in the
% Heidari/Chen group zip at aliasgharheidari.com. The release evaluates every new
% position two or three times (the in-loop best check, its repeat on improvement,
% and the end-of-iteration sweep); the objective is deterministic, so here each
% position is evaluated once and Max_iter = ceil((maxFE - N) / N), about twice the
% iterations the release reaches on the same budget. With the same seed and
% iteration count the release's best is reproduced exactly (9/9). min/max clamping also
% sends a NaN coordinate (0 * Inf in a Levy step) to lb; the release would keep it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = parrot(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    Max_iter = max(1, ceil((maxFE - N) / N));   % one evaluation per parrot per iteration

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';
    [fit, order] = sort(fit);
    X = X(order, :);
    GBestF = fit(1);
    GBestX = X(1, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = GBestF;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    it = 0;
    while FE < maxFE
        it = it + 1;
        tr = min(it, Max_iter) / Max_iter;   % i / Max_iter of the release
        alpha = rand / 5;
        sita  = rand * pi;
        mX = mean(X, 1);                     % the release reads mean(X) of the iteration-start swarm
        % Xnew/fnew mirror the swarm: row j moves only once parrot j has been evaluated
        Xnew = X;
        fnew = fit;
        for j = 1:N
            St = randi([1, 4]);
            if St == 1
                xn = (X(j, :) - GBestX) .* levy(dim) + rand * mX * (1 - tr) ^ (2 * tr);
            elseif St == 2
                xn = X(j, :) + GBestX .* levy(dim) + randn * (1 - tr) * ones(1, dim);
            elseif St == 3
                if rand < 0.5
                    xn = X(j, :) + alpha * (1 - tr) * (X(j, :) - mX);
                else
                    xn = X(j, :) + alpha * (1 - tr) * exp(-j / (rand * Max_iter));
                end
            else
                xn = X(j, :) + rand * cos((pi * it) / (2 * Max_iter)) * (GBestX - X(j, :)) ...
                     - cos(sita) * tr ^ (2 / Max_iter) * (X(j, :) - GBestX);
            end
            xn = min(max(xn, lb), ub);

            [fn, FE] = calculate_fitness(xn', problem, FE);
            Xnew(j, :) = xn;
            fnew(j) = fn;
            if fn < GBestF
                GBestF = fn;
                GBestX = xn;
            end
            if FE <= maxFE
                curve(FE) = GBestF;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Xnew, fnew, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        % No greedy selection in the release: the whole swarm is replaced, then sorted
        [fit, order] = sort(fnew);
        X = Xnew(order, :);
    end

    curve(min(FE, maxFE):end) = GBestF;
    best_fitness  = GBestF;
    best_solution = GBestX;
end

% Levy flight step (Mantegna, beta = 1.5)
function o = levy(d)
    beta  = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    o = u ./ abs(v) .^ (1 / beta);
end

% Initialization, column by column as in the release
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

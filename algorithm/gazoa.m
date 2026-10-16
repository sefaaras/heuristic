% ----------------------------------------------------------------------- %
% Gazelle Optimization Algorithm (GOA)
% Stored as gazoa; the acronym GOA collides with goa (Grasshopper)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 50                   % Population size (gazelles), from the release's main.m
%   PSRs = 0.34                 % Predator success rate: escape-move branch and U density
%   S    = 0.88                 % Speed factor of the two escape moves
%   s    = rand                 % Grazing speed, drawn once per run
%   CF   = (1-t/T)^(2*t/T)      % Cumulative effect of the predator, 1 -> 0
%   RL   = 0.05*Levy, beta=1.5  % Mantegna Levy steps; RB standard normal (Brownian)
%
% Algorithm Concept:
%   - Grazing (Brownian) on each coordinate with r > 0.5: x + s*R*RB.*(Elite - RB.*x),
%     Elite the best gazelle ever evaluated
%   - Otherwise escape, mu = +-1 alternating by iteration: the first half of the herd
%     Levy, x + S*mu*R*RL.*(Elite - RL.*x); the second Elite + S*mu*CF*RB.*(RL.*Elite - x)
%   - Herd memory: after each evaluated batch a gazelle returns to its previous position
%     when that one was strictly better
%   - Predator success, probability PSRs: x + CF*U.*(random box point), else x plus a
%     scaled difference of two shuffled herds; evaluated at the next iteration's start
%
% Reference:
% Jeffrey O. Agushaka, Absalom E. Ezugwu, Laith Abualigah,
% Gazelle optimization algorithm: a novel nature-inspired metaheuristic
% optimizer,
% Neural Computing and Applications 35(5) (2023) 4099-4131.
% https://doi.org/10.1007/s00521-022-07854-6
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from File Exchange 119363 v1.0.0 (GOA2.m and levy.m, 22 Oct 2022, by
% co-author Ezugwu); seeded runs are bit-identical to it. Kept as released: s is
% one rand per run, the grazing/escape switch is drawn per coordinate, and the
% memory is reset after the first move batch, so iteration 0 accepts every move
% (inherited from MPA). Harness changes: T = ceil(maxFe/(2N)) iterations of two
% batches, the last batch cut at maxFe; the memory selects by logical index, not
% the release's mask.*old + ~mask.*new, which turns an Inf fitness into NaN (as in
% mpa); a min/max clamp, so a non-finite coordinate lands in the box where the
% release's clamp makes it NaN; Levy |v| floored at realmin.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gazoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N    = 50;
    PSRs = 0.34;
    S    = 0.88;
    T    = max(1, ceil(maxFE / (2 * N)));   % two evaluation batches per iteration

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    s = rand;   % drawn once per run, after the herd, as in the release
    half = (1:N)' > N / 2;

    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:);
    [fbest, bidx] = min(fit);
    Xbest = X(bidx, :);
    for e = 1:min(N, maxFE)
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Herd memory (fit_old/Prey_old of the release): the accepted position and fitness per gazelle
    Xold = X;
    fold = fit;
    Iter = 0;
    while FE < maxFE
        if Iter > 0
            % Evaluate the PSRs-moved herd, top of the release's loop
            X = min(max(X, lb), ub);
            nb = min(N, maxFE - FE);
            [fnew, FE] = calculate_fitness(X(1:nb, :)', problem, FE);
            fnew = fnew(:);
            for i = 1:nb
                if fnew(i) < fbest
                    fbest = fnew(i);
                    Xbest = X(i, :);
                end
                if ~(fold(i) < fnew(i))
                    Xold(i, :) = X(i, :);
                    fold(i) = fnew(i);
                end
                ec = FE - nb + i;
                curve(ec) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Xold, fold, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
            X = Xold;
        end

        Elite = repmat(Xbest, N, 1);   % Eq. (3) per the release's comment
        CF = (1 - Iter / T)^(2 * Iter / T);
        RL = 0.05 * levy(N, dim, 1.5);
        RB = randn(N, dim);
        % R and r are drawn per (i, j), i outer, j inner, R before r, as in the release loop
        Rr = rand(2, dim, N);
        R  = reshape(Rr(1, :, :), dim, N)';
        r  = reshape(Rr(2, :, :), dim, N)';
        if mod(Iter, 2) == 0
            mu = -1;
        else
            mu = 1;
        end
        graze = r > 0.5;
        lev   = ~graze & ~half;
        bro   = ~graze & half;
        Xn = X;
        st = RB .* (Elite - RB .* X);
        Xn(graze) = X(graze) + s * R(graze) .* st(graze);
        st = RL .* (Elite - RL .* X);
        Xn(lev) = X(lev) + S * mu * R(lev) .* st(lev);
        st = RB .* (RL .* Elite - X);
        Xn(bro) = Elite(bro) + S * mu * CF * st(bro);
        X = min(max(Xn, lb), ub);

        nb = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(X(1:nb, :)', problem, FE);
        fnew = fnew(:);
        for i = 1:nb
            if fnew(i) < fbest
                fbest = fnew(i);
                Xbest = X(i, :);
            end
            % Iteration 0 resets the memory to the moved herd first, so every move is kept
            if Iter == 0 || ~(fold(i) < fnew(i))
                Xold(i, :) = X(i, :);
                fold(i) = fnew(i);
            end
            ec = FE - nb + i;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xold, fold, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end
        X = Xold;

        % Predator success: a masked random box point, else a shuffled-herd difference
        if rand < PSRs
            U = rand(N, dim) < PSRs;
            X = X + CF * ((lb + rand(N, dim) .* (ub - lb)) .* U);
        else
            r2 = rand;
            X = X + (PSRs * (1 - r2) + r2) * (X(randperm(N), :) - X(randperm(N), :));
        end
        Iter = Iter + 1;
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Mantegna Levy steps, the release's levy.m
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = randn(n, m) * sigma_u;   % the same stream as the release's random('Normal', 0, sigma_u, n, m)
    v = randn(n, m);
    z = u ./ max(abs(v), realmin).^(1 / beta);   % realmin floor: v == 0 would give Inf
end

% Initialization, one column per dimension as in the release
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

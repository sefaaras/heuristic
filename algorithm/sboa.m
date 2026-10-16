% ----------------------------------------------------------------------- %
% Secretary Bird Optimization Algorithm (SBOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30                       % Population size (secretary birds)
%   CF = (1-t/T)^(2*t/T)          % Attack-stage perturbation factor, 1 -> 0
%   RL = 0.5*Levy(D), beta = 1.5  % Weighted Mantegna Levy step
%
% Algorithm Concept:
%   - Hunting, t < T/3: x + (x_r1 - x_r2).*rand, a differential move
%   - Hunting, T/3 < t < 2T/3: x_best + exp((t/T)^4)*(randn - 0.5).*(x_best - x)
%   - Hunting, later: x_best + CF*x.*RL, a Levy kick around the best
%   - Escape, one branch per iteration: camouflage x_best + (1-t/T)^2*(2*rand-1).*x
%     or flight x + rand.*(x_rand - K*x) with K in {1, 2}
%   - Greedy replacement after each move; x_best refreshed once per iteration
%
% Reference:
% Youfa Fu, Dan Liu, Jiadui Chen, Ling He,
% Secretary bird optimization algorithm: a new metaheuristic for solving global
% optimization problems,
% Artificial Intelligence Review 57(5) (2024) 123.
% https://doi.org/10.1007/s10462-024-10729-y
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the release on File Exchange 164456 (v1.0.0, SBOA.m, whose
% header names the four authors as programmers); seeded runs are bit-identical
% to it. The release departs from the paper's text four ways: Levy scale 1 not
% s = 0.01, uniform R2 and camouflage RB not normal, one r per iteration not
% per bird, '<=' acceptance. The release is followed: at the paper's CEC2017
% D=30 setting (N = 30, T = 500, mean of 10 seeds) it tracks the paper's Table
% 2 and an implementation of the text does not; F7 is off for both:
%    F1   reported 1.79e4   release 1.95e4   paper text 4.28e6
%    F3   reported 2.56e4   release 2.32e4   paper text 5.79e4
%    F7   reported 819      release 904      paper text 861
%    F10  reported 3.98e3   release 4.55e3   paper text 5.49e3
% T = ceil((maxFe - N)/(2*N)); best tracked per evaluation, not x_best at the end.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sboa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    T = max(1, ceil((maxFE - N) / (2 * N)));   % hunting + escape: two evaluations per bird

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [fbest, bidx] = min(fit);
    Xbest = X(bidx, :);
    for e = 1:min(N, maxFE)
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % x_best of the reference: moved only by a strict improvement seen at the top of an iteration
    Bast_P = Xbest;
    fB     = fbest;

    t = 0;
    while FE < maxFE
        t = t + 1;
        CF = (1 - t / T)^(2 * t / T);   % Eq. (9)
        [best, location] = min(fit);
        if best < fB
            fB     = best;
            Bast_P = X(location, :);
        end

        for i = 1:N
            if t < T / 3
                % Searching for prey, Eq. (4)
                r1 = randi(N);
                r2 = randi(N);
                X1 = X(i, :) + (X(r1, :) - X(r2, :)) .* rand(1, dim);
            elseif t > T / 3 && t < 2 * T / 3
                % Consuming prey, Eq. (6)-(7)
                RB = randn(1, dim);
                X1 = Bast_P + exp((t / T)^4) * (RB - 0.5) .* (Bast_P - X(i, :));
            else
                % Attacking prey, Eq. (9)-(11)
                RL = 0.5 * levy(dim);
                X1 = Bast_P + CF * X(i, :) .* RL;
            end
            X1 = min(max(X1, lb), ub);
            [f1, FE] = calculate_fitness(X1', problem, FE);
            if f1 <= fit(i)
                X(i, :) = X1;
                fit(i) = f1;
                if f1 < fbest
                    fbest = f1;
                    Xbest = X1;
                end
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

        % Escape strategy, Eq. (14): one branch and one x_random for the whole iteration
        r = rand;
        Xrandom = X(randperm(N, 1), :);   % randperm as in the release, so seeded runs match it
        for i = 1:N
            if r < 0.5
                % C1, camouflage
                RB = rand(1, dim);
                X2 = Bast_P + (1 - t / T) * (1 - t / T) * (2 * RB - 1) .* X(i, :);
            else
                % C2, fly or run away; K from Eq. (16)
                K  = round(1 + rand);
                R2 = rand(1, dim);
                X2 = X(i, :) + R2 .* (Xrandom - K .* X(i, :));
            end
            X2 = min(max(X2, lb), ub);
            [f2, FE] = calculate_fitness(X2', problem, FE);
            if f2 <= fit(i)
                X(i, :) = X2;
                fit(i) = f2;
                if f2 < fbest
                    fbest = f2;
                    Xbest = X2;
                end
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

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Mantegna Levy step, Eq. (12)-(13) without the paper's s = 0.01 (see the note)
function o = levy(d)
    beta  = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    o = u ./ max(abs(v), realmin).^(1 / beta);   % realmin floor: v == 0 would give Inf
end

% Initialization, one column per dimension
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

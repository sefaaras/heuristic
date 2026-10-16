% ----------------------------------------------------------------------- %
% Prairie Dog Optimization Algorithm (PDO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 100                           % Population size (prairie dogs), from mainPDO.m
%   rho   = 0.005                         % Individual-difference term of eCBest
%   epsPD = 0.1                           % Food-source alarm, scales eCBest in phase 1
%   eps   = 2.2e-16                       % MATLAB eps: food quality, divisor offset
%   DS    = 1.5*randn*(1-t/T)^(2*t/T)*mu  % Digging strength, one randn per iteration
%   PE    = 1.5*(1-t/T)^(2*t/T)*mu        % Predator effect; mu = +-1 by iteration parity
%   Levy beta = 1.5                       % Mantegna steps, unscaled
%
% Algorithm Concept:
%   - Four phases by t/T, each coordinate j built on the live best G and on G0, the
%     best at the iteration start; x_r is coordinate j of a random dog
%   - t < T/4, foraging: G - eCBest*epsPD - CPD*Levy, CPD = rand*(G0 - x_r)/(G0 + eps),
%     eCBest = G*(rho + (x - mean(x))/(G0*(ub - lb) + eps))
%   - T/4 <= t < T/2, burrowing: G*x_r*DS*Levy
%   - T/2 <= t < 3T/4, predator response: G*PE*rand
%   - t >= 3T/4, alarm response: G - eCBest*eps - CPD*rand
%   - Dogs move one at a time with strict greedy acceptance; G follows every improvement
%
% Reference:
% Absalom E. Ezugwu, Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili,
% Amir H. Gandomi,
% Prairie Dog Optimization Algorithm,
% Neural Computing and Applications 34(22) (2022) 20017-20065.
% https://doi.org/10.1007/s00521-022-07530-9
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from File Exchange 110980 v1.2.0 (PDO1.m and levym.m, 3 Jul 2022, by the
% authors); seeded runs are bit-identical to it. Kept as released: CPD and eCBest
% divide by G0 (the best at the iteration start) while the moves use the live best;
% CPD's rand and x_r are drawn in every phase but used only in phases 1 and 4; the
% phase 2 and 3 products scale G toward the origin. Harness changes: ub(j) - lb(j)
% per dimension, since the release's scalar / (G0*(UB - LB) + eps) is a matrix
% division that errors on row-vector bounds; T = ceil((maxFe - N)/N); a min/max
% clamp, so a non-finite coordinate (an overflowed product, G0 + eps == 0) lands in
% the box where the release's clamp makes it NaN; Levy |v| floored at realmin.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = pdo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N     = 100;
    rho   = 0.005;
    epsPD = 0.1;
    T     = max(1, ceil((maxFE - N) / N));   % one evaluation per dog per iteration

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

    Xnew = zeros(1, dim);
    t = 0;
    while FE < maxFE
        t = t + 1;
        if mod(t, 2) == 0
            mu = -1;
        else
            mu = 1;
        end
        DS = 1.5 * randn * (1 - t / T)^(2 * t / T) * mu;
        PE = 1.5 * (1 - t / T)^(2 * t / T) * mu;
        RL = levy(N, dim, 1.5);
        G0 = Xbest;   % TPD of the release, fixed for the iteration

        % Sequential: x_r reads dogs already moved this iteration and G moves with every improvement
        for i = 1:N
            mX = mean(X(i, :));
            for j = 1:dim
                cpd = rand * (G0(j) - X(randi([1 N]), j)) / (G0(j) + eps);
                P   = rho + (X(i, j) - mX) / (G0(j) * (ub(j) - lb(j)) + eps);
                eCB = Xbest(j) * P;
                if t < T / 4
                    Xnew(j) = Xbest(j) - eCB * epsPD - cpd * RL(i, j);
                elseif t < 2 * T / 4
                    Xnew(j) = Xbest(j) * X(randi([1 N]), j) * DS * RL(i, j);
                elseif t < 3 * T / 4
                    Xnew(j) = Xbest(j) * PE * rand;
                else
                    Xnew(j) = Xbest(j) - eCB * eps - cpd * rand;
                end
            end
            Xnew = min(max(Xnew, lb), ub);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < fit(i)
                X(i, :) = Xnew;
                fit(i) = fnew;
                if fnew < fbest
                    fbest = fnew;
                    Xbest = Xnew;
                end
            end
            curve(FE) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Mantegna Levy steps, the release's levym.m
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = randn(n, m) * sigma_u;   % the same stream as the release's random('Normal', 0, sigma_u, n, m)
    v = randn(n, m);
    z = u ./ max(abs(v), realmin).^(1 / beta);   % realmin floor: v == 0 would give Inf
end

% Initialization, one column per dimension
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

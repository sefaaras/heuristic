% ----------------------------------------------------------------------- %
% Sooty Tern Optimization Algorithm (STOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30            % Population size (sooty terns)
%   Sa           = 2 -> 0        % Collision-avoidance factor, linear over the budget
%   b            = 1             % Spiral constant
%
% Algorithm Concept:
%   - Every coordinate of every tern is redrawn each iteration; no selection,
%     the best ever evaluated position P_best is the only memory
%   - Migration: D = Sa*x + X1*(P_best - x), X1 = Sa*(2*r1 - 1) in [-Sa, 0]
%     with r1 ~ U(0, 0.5); the Sa*x term pulls every tern toward the origin
%   - Attack: spiral x = D*e^(b*l)*cos(2*pi*l) + P_best, l ~ U between 1 and Sa
%   - Terns are clamped to the box just before they are evaluated
%
% Reference:
% Gaurav Dhiman, Amandeep Kaur,
% STOA: A bio-inspired based optimization algorithm for industrial
% engineering problems,
% Engineering Applications of Artificial Intelligence 82 (2019) 148-174.
% https://doi.org/10.1016/j.engappai.2019.03.021
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 76667 v1.0.0 (stoa.m,
% 6 Jun 2020); seeded runs are bit-identical to it. The release departs from
% the paper, all kept: the pull toward the best has X1 in [-Sa, 0] where
% Eqs. (3)-(4) have C_B = 0.5*rand, the spiral is WOA's additive one where
% Eq. (10) multiplies by P_best, and a second number r2 is drawn and never
% used. N = 30 as in the release's main.m (the paper's study favours 100).
% T = ceil(maxFe/N) evaluation rounds stands for Max_iter, so the last
% evaluated move has Sa = 4/T as in the release, and the last round evaluates
% only the terns the budget covers.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = stoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    T = max(1, ceil(maxFE / N));   % evaluation rounds of N terns, the release's Max_iterations

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    nev = min(N, maxFE);
    [fit, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
    fit = fit(:)';

    fbest = Inf;
    Xbest = X(1, :);   % the release starts at the origin; differs only if the whole round is Inf or NaN
    for e = 1:nev
        if fit(e) < fbest
            fbest = fit(e);
            Xbest = X(e, :);
        end
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X(1:nev, :), fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    Xrec = X;   % position at which each entry of fit was measured
    l = 0;
    while FE < maxFE
        Sa = 2 - l * (2 / T);
        for i = 1:N
            for j = 1:dim
                r1 = 0.5 * rand();
                r2 = 0.5 * rand(); %#ok<NASGU> drawn and unused in the release; keeps the stream aligned
                X1 = 2 * Sa * r1 - Sa;
                b = 1;
                ll = (Sa - 1) * rand() + 1;
                D_alphs = Sa * X(i, j) + X1 * (Xbest(j) - X(i, j));
                X(i, j) = D_alphs * exp(b .* ll) .* cos(ll .* 2 * pi) + Xbest(j);
            end
        end
        l = l + 1;

        % The release clamps each tern just before evaluating it at the top of the next iteration
        X = min(max(X, lb), ub);
        nev = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(X(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for i = 1:nev
            Xrec(i, :) = X(i, :);
            fit(i) = fnew(i);
            if fit(i) < fbest
                fbest = fit(i);
                Xbest = X(i, :);
            end
            ec = FE - nev + i;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Initialization, one column per dimension as in the release
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

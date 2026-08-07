% ----------------------------------------------------------------------- %
% Jaya
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                      % Population size
%
% Algorithm Concept:
%   - Jaya is TLBO stripped down to a single equation. There are no phases, no
%     teacher, no teaching factor -- one update, applied to everybody:
%         x_new = x + r1 .* (x_best - |x|) - r2 .* (x_worst - |x|)
%     with r1 and r2 drawn independently per dimension
%   - The two terms move TOWARDS the best and AWAY from the worst at once;
%     attraction alone converges prematurely, repulsion alone diverges
%   - The absolute value |x| is not a typo and is what separates Jaya from an
%     ordinary attraction/repulsion rule: step magnitude also depends on the
%     distance from the ORIGIN, so a variable far from zero takes larger steps
%   - Selection is greedy per individual, so the population never degrades and
%     the best and worst of the next generation are recomputed from scratch
%   - Like TLBO, the only settings are the population size and the budget
%
% Reference:
% R. V. Rao,
% Jaya: A simple and new optimization algorithm for solving constrained and
% unconstrained optimization problems,
% International Journal of Industrial Engineering Computations, vol. 7, no. 1,
% pp. 19-34, 2016.
% https://doi.org/10.5267/j.ijiec.2015.8.004
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author MATLAB release could be located, but the algorithm is one update
% equation (Eq. 1) plus greedy selection, so it is implemented from the paper.
% The two random numbers are drawn per dimension as the paper specifies, and
% best and worst are recomputed at the start of every generation.
% POPULATION SIZE: the paper uses a different size per test problem; NP = 50
% matches tlbo.m in this folder so the two compare directly, Jaya being Rao's
% own simplification of TLBO. A Jaya generation costs NP evaluations to
% TLBO's 2*NP.
% The paper specifies no bound handling; violating components are clamped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jaya(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    NP = 50;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LBm = repmat(lb, NP, 1);
    UBm = repmat(ub, NP, 1);

    % Initialisation
    X = LBm + rand(NP, D) .* (UBm - LBm);

    [f, FE] = calculate_fitness(X', problem, FE);
    f = f(:);

    bsf  = inf;
    bsfx = X(1, :);
    for i = 1:NP
        if f(i) < bsf
            bsf  = f(i);
            bsfx = X(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        [~, ib] = min(f);
        [~, iw] = max(f);
        xbest  = X(ib, :);
        xworst = X(iw, :);

        r1 = rand(NP, D);
        r2 = rand(NP, D);
        absX = abs(X);

        % Eq. (1)
        Xnew = X + r1 .* (repmat(xbest, NP, 1) - absX) ...
                 - r2 .* (repmat(xworst, NP, 1) - absX);
        Xnew = min(max(Xnew, LBm), UBm);

        [fn, FE] = calculate_fitness(Xnew', problem, FE);
        fn = fn(:);

        for i = 1:NP
            if fn(i) < bsf
                bsf  = fn(i);
                bsfx = Xnew(i, :);
            end
            ec = FE - NP + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, f, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        better = fn < f;
        X(better, :) = Xnew(better, :);
        f(better)    = fn(better);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

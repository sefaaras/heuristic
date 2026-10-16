% ----------------------------------------------------------------------- %
% Sand Cat Swarm Optimization (SCSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 30           % Population size (sand cats)
%   S     = 2            % Maximum sensitivity range
%   rG    = S - S*t/T    % General sensitivity range, 2 -> 0 over the budget
%   theta = 1..360       % Roulette-wheel angle, drawn with weight proportional to theta
%
% Algorithm Concept:
%   - Each cat draws r = rG*rand and R = 2*rG*rand - rG once per move
%   - |R| <= 1, attack: x_j = b_j - r*|rand*b_j - x_j|*cos(theta), b the best
%   - |R| > 1, search: x_j = r*(x_cp,j - rand*x_j), cp a random cat per dimension
%   - No selection: every cat takes its move; once rG <= 1 only the attack is left
%   - Moves are made in place, so a later cat reads an earlier cat's new position
%
% Reference:
% Amir Seyyedabbasi, Farzad Kiani,
% Sand Cat swarm optimization: a nature-inspired algorithm to solve global
% optimization problems,
% Engineering with Computers 39(4) (2023) 2627-2651.
% https://doi.org/10.1007/s00366-022-01604-x
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release on File Exchange 110185 (v1.0.0, SCSO.m,
% 15 Apr 2022); seeded runs are bit-identical to it. Kept as released, where it
% departs from the paper's text: the search move uses a random cat per
% dimension instead of the best-candidate position, and the roulette's integer
% angle 1..360 goes into cos as radians, not degrees. T = ceil(maxFe/N) rounds
% stands for Max_iter, so the last evaluated move has rG = 4/T as in the
% release, and the last round evaluates only the cats the budget covers. The
% release starts the best at the origin with fitness Inf; here it starts at the
% best of the first round, which is the same unless all of it is Inf or NaN.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = scso(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    S = 2;
    T = max(1, ceil(maxFE / N));   % evaluation rounds of N cats, the release's Max_iter
    p = 1:360;
    Pcum = cumsum(p ./ sum(p));    % roulette over theta, the release's RouletteWheelSelection(p)

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [fbest, bidx] = min(fit);   % first minimum, as the release's strict-< scan finds it
    Xbest = X(bidx, :);
    for e = 1:min(N, maxFE)
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    Xrec = X;   % position at which each entry of fit was measured
    t = 0;
    while FE < maxFE
        rg = S - (S * t / T);   % 4/T at the last evaluated move
        for i = 1:N
            r = rand * rg;
            R = ((2 * rg) * rand) - rg;     % |R| <= 1 selects the attack
            for j = 1:dim
                teta = find(rand <= Pcum, 1, 'first');
                if isempty(teta), teta = 360; end   % cumsum can end just below 1
                if (-1 <= R) && (R <= 1)
                    % Attack; teta enters cos as radians, as released
                    Rand_position = abs(rand * Xbest(j) - X(i, j));
                    X(i, j) = Xbest(j) - r * Rand_position * cos(teta);
                else
                    % Search, with a random cat in place of the paper's best candidate
                    cp = floor(N * rand() + 1);
                    X(i, j) = r * (X(cp, j) - rand * X(i, j));
                end
            end
        end
        t = t + 1;

        % The release clamps each cat just before evaluating it at the top of the next iteration
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

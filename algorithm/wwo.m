% ----------------------------------------------------------------------- %
% Water Wave Optimization (WWO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n       = 10                   % Waves, as in the release's example
%   hMax    = 12                   % Wave height: failed propagations before refraction
%   lambda  = 0.5                  % Initial wavelength, a fraction of each range ub - lb
%   alpha   = 1.0026               % Wavelength reduction coefficient
%   beta    = 0.25 -> 0.001        % Breaking coefficient, linear in evaluations spent
%   kmax    = min(12, floor(D/2))  % Solitary waves per breaking
%   epsilon = 1e-7                 % Guard of the wavelength exponent
%
% Algorithm Concept:
%   - Propagation: x + U(-1,1)*lambda*(ub - lb) per coordinate, kept only if it
%     improves, which restores the wave height to hMax; a failure lowers it by 1
%   - Refraction: a wave of height 0 is redrawn per coordinate from
%     N((x* + x)/2, |x* - x|/2) around the best wave x*, whatever its fitness
%   - Breaking: a propagation that beats the best spawns kmax solitary waves, each
%     moving one random coordinate by N(0,1)*beta*(ub - lb)
%   - Wavelength: lambda*alpha^(-(f_max - f + eps)/(f_max - f_min + eps)) after
%     each generation; refraction and an improving solitary wave scale it by f'/f
%
% Reference:
% Yu-Jun Zheng,
% Water wave optimization: A new nature-inspired metaheuristic,
% Computers & Operations Research 55 (2015) 1-11.
% https://doi.org/10.1016/j.cor.2014.10.008
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release, WWO.m of 10 Dec 2014 (WWO.rar on Zheng's
% group page compintell.cn/en/dataAndCode.html); seeded runs are bit-identical
% to it. Kept as released: k = kmax solitary waves (randi(kmax) is commented
% out), the last generation's best wave never breaks, refraction is always
% accepted, and breaking reads and writes x(d), a linear index into the
% population: a solitary wave starts from another wave's coordinate, an improving
% one overwrites that cell, and the broken wave takes its fitness but keeps the
% propagated position. Changed: floor(D/2), as D/2 crashes randperm at odd D < 24;
% beta runs over the budget net of initialisation; NaN coordinates are redrawn;
% the best is the best point evaluated; history holds each wave where measured.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wwo(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    n = 10;
    hMax = 12;
    alpha = 1.0026;
    betaMax = 0.25;
    betaMin = 0.001;
    beta = betaMax;
    kmax = min(12, floor(dim / 2));
    epsilon = 0.0000001;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Wave by wave, coordinate by coordinate, as released
    X = lb + rand(dim, n)' .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';
    h = hMax * ones(1, n);
    lambda = 0.5 * ones(1, n);
    Xrec = X;   % the point each fit(i) was measured at; breaking lets X drift from it

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);
    for e = 1:min(n, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, Xrec, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    minIndex = min_max_index(fit);
    optValue = fit(minIndex);
    optVector = X(minIndex, :);
    u = zeros(1, dim);

    while FE < maxFE
        for i = 1:n
            % Propagation
            for d = 1:dim
                u(d) = X(i, d) + (-1 + 2 * rand) * lambda(i) * (ub(d) - lb(d));
                if ~(u(d) >= lb(d) && u(d) <= ub(d))
                    u(d) = lb(d) + rand * (ub(d) - lb(d));
                end
            end
            [newValue, FE] = calculate_fitness(u', problem, FE);
            improved = newValue < fit(i);
            if improved
                X(i, :) = u;
                fit(i) = newValue;
                h(i) = hMax;
                Xrec(i, :) = u;
            end
            stamp(u, newValue);
            if FE >= maxFE, break; end

            if improved
                if newValue < optValue && i ~= minIndex
                    % Breaking
                    optValue = newValue;
                    temp = randperm(dim, kmax);
                    for l = 1:kmax
                        tempX = u;
                        d = temp(l);
                        tempX(d) = X(d) + randn * beta * (ub(d) - lb(d));   % linear index, as released
                        if ~(tempX(d) >= lb(d) && tempX(d) <= ub(d))
                            tempX(d) = lb(d) + rand * (ub(d) - lb(d));
                        end
                        [newValueTemp, FE] = calculate_fitness(tempX', problem, FE);
                        if newValueTemp < newValue
                            X(d) = tempX(d);
                            lambda(i) = lambda(i) * newValueTemp / fit(i);
                            fit(i) = newValueTemp;
                            Xrec(i, :) = tempX;
                            newValue = newValueTemp;
                            if newValueTemp < optValue
                                optValue = newValueTemp;
                                optVector = X(i, :);
                            end
                        end
                        stamp(tempX, newValueTemp);
                        if FE >= maxFE, break; end
                    end
                end
            else
                h(i) = h(i) - 1;
                if h(i) == 0
                    % Refraction
                    for d = 1:dim
                        mu = (optVector(d) + X(i, d)) / 2;
                        sigma = abs(optVector(d) - X(i, d)) / 2;
                        u(d) = randn * sigma + mu;
                        if ~(u(d) >= lb(d) && u(d) <= ub(d))
                            u(d) = lb(d) + rand * (ub(d) - lb(d));
                        end
                    end
                    oldValue = fit(i);
                    X(i, :) = u;
                    [fit(i), FE] = calculate_fitness(u', problem, FE);
                    h(i) = hMax;
                    lambda(i) = lambda(i) * fit(i) / oldValue;
                    Xrec(i, :) = u;
                    stamp(u, fit(i));
                end
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        [minIndex, maxIndex] = min_max_index(fit);
        worstValue = fit(maxIndex);
        optValue = fit(minIndex);
        optVector = X(minIndex, :);
        den = worstValue - optValue + epsilon;
        for i = 1:n
            lambda(i) = lambda(i) * power(alpha, -(worstValue - fit(i) + epsilon) / den);
        end
        beta = betaMax - (betaMax - betaMin) * (FE - n) / (maxFE - n);   % the release's nfe/nfes
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Best-so-far at this evaluation, then one history sample of (Xrec, fit)
    function stamp(x, f)
        if f < best_fitness
            best_fitness = f;
            best_solution = x;
        end
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            FE, Xrec, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

% The release's MinIndexMaxIndex: first strict extremes, the worst tested only off a new best
function [minIndex, maxIndex] = min_max_index(f)
    minIndex = 1;
    maxIndex = 1;
    bestValue = f(1);
    worstValue = bestValue;
    for i = 2:numel(f)
        if f(i) < bestValue
            minIndex = i;
            bestValue = f(i);
        elseif f(i) > worstValue
            maxIndex = i;
            worstValue = f(i);
        end
    end
end

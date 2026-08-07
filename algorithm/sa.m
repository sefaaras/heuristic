% ----------------------------------------------------------------------- %
% Simulated Annealing (SA)
% Continuous variant
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Mmax = 100                   % Number of temperature levels
%   T    = m / Mmax              % INVERSE temperature, rises 0 -> 1
%   mu   = 10^(100*T)            % Neighbourhood sharpness, 1 -> 1e100
%   TolFun = 1e-4                % Scaling constant of the Metropolis test
%   k_max = floor(maxFe/(Mmax+1))% Trials per temperature (thermal equilibrium)
%
% Algorithm Concept:
%   - A single point wanders the space: a downhill move is always taken, an
%     uphill move of size df with probability exp(-T*df/(|f|*TolFun))
%   - The CONTINUOUS variant is defined by its proposal distribution,
%         dx = ((1+mu)^|y| - 1)/mu * sign(y) * (ub - lb),   y ~ U(-1, 1)
%     the inverse mu-law companding curve: nearly uniform over the box at
%     mu = 1, concentrated at zero with a heavy tail as mu grows to 1e100
%   - So the NEIGHBOURHOOD shrinks with temperature just as the acceptance
%     probability does -- the search anneals in position and acceptance at once
%   - Dividing the Metropolis threshold by |f(x)| makes acceptance scale-free:
%     what matters is the RELATIVE worsening
%   - Best-so-far is tracked separately, so the walk never loses the incumbent
%
% Reference:
% S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi,
% Optimization by Simulated Annealing,
% Science, vol. 220, no. 4598, pp. 671-680, 1983.
% https://doi.org/10.1126/science.220.4598.671
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from sim_anl.m of the Nature-Inspired-Algorithms collection. There is
% no canonical author release for SA on continuous variables -- Kirkpatrick's
% paper is combinatorial -- so this widely used implementation is the reference
% and its structure, schedules and constants are reproduced exactly.
% BUDGET: the reference nests a fixed 501-point inner loop inside 101
% temperature levels and takes neither from the caller. The annealing SCHEDULE
% is what matters, so Mmax = 100 is kept and the thermal-equilibrium length is
% derived from maxFe, which at the budgets used here gives roughly the
% reference's own 500 trials per level.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sa(problem)

    d     = problem.dimension;
    l     = problem.lb(:)';
    u     = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    Mmax   = 100;
    TolFun = 1e-4;
    % ceil, not floor: floor would leave up to Mmax evaluations unspent
    kmax   = max(1, ceil((maxFE - 1) / (Mmax + 1)));

    FE    = 0;
    curve = zeros(1, maxFE);

    % SA is single-point, so the recorded "population" is that one point
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    x0 = l + rand(1, d) .* (u - l);
    x  = x0;

    [fx, FE] = calculate_fitness(x', problem, FE);
    fx = fx(1);
    f0 = fx;

    if FE >= 1 && FE <= maxFE
        curve(FE) = f0;
        [population_history, fitness_history, history_index] = record_history(...
            FE, x, fx, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Annealing loop
    for m = 0:Mmax
        if FE >= maxFE
            break;
        end

        T  = m / Mmax;              % inverse temperature, 0 -> 1
        mu = 10 ^ (T * 100);

        for k = 1:kmax
            if FE >= maxFE
                break;
            end

            % mu-law inverse proposal, scaled by the box width
            dx = muInv(2 * rand(1, d) - 1, mu) .* (u - l);
            x1 = min(max(x + dx, l), u);

            [fx1, FE] = calculate_fitness(x1', problem, FE);
            fx1 = fx1(1);
            df  = fx1 - fx;

            if df < 0 || rand < exp(-T * df / (abs(fx) + eps) / TolFun)
                x  = x1;
                fx = fx1;
            end

            if fx1 < f0
                x0 = x1;
                f0 = fx1;
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = f0;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, x, fx, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = f0;

    best_fitness  = f0;
    best_solution = x0;
end

% Helper Functions

function x = muInv(y, mu)
% Inverse mu-law companding: near-uniform at mu = 1, heavy-tailed at zero as mu grows
    x = (((1 + mu) .^ abs(y) - 1) / mu) .* sign(y);
end

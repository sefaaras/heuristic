% ----------------------------------------------------------------------- %
% Wind Driven Optimization (WDO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 20     % Air parcels
%   RT      = 3      % Pull toward the global best (gas constant times temperature)
%   g       = 0.2    % Gravitational constant, a pull toward the origin
%   alp     = 0.4    % Friction: the kept share of the velocity is 1 - alp
%   c       = 0.4    % Coriolis coefficient
%   maxV    = 0.3    % Velocity limit per coordinate, in the normalised space
%
% Algorithm Concept:
%   - Parcels move in [-1, 1]^D, mapped linearly onto the box for evaluation
%   - Ranked by pressure (fitness), parcel i updates u = (1-alp)u - g*x
%     + |1 - 1/i| RT (x_best - x) + c u_perm/i
%   - u_perm is the parcel's own velocity under a random permutation of its
%     coordinates, the Coriolis term
%   - Velocity is clipped to [-maxV, maxV], position to [-1, 1]; the population
%     is re-sorted by pressure after every evaluation
%
% Reference:
% Zikri Bayraktar, Muge Komurcu, Jeremy A. Bossard, Douglas H. Werner,
% The Wind Driven Optimization Technique and its Application in Electromagnetics,
% IEEE Transactions on Antennas and Propagation 61(5) (2013) 2745-2757.
% https://doi.org/10.1109/TAP.2013.2238654
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, wdo_matlab_03.m in
% github.com/zikribayraktar/Wind_Driven_Optimization (commit 9cafcb7, 2 Jun
% 2014; WDOcode.R there is the same algorithm); seeded runs are bit-identical to
% it. Parameters are the release's; the paper itself was not checked. Kept as
% released: -g*x pulls every parcel toward the box centre; the best-ranked parcel
% gets no pull toward the global best; the initial velocities are not re-sorted
% with the parcels; clipping, not reflection. Changed: per-dimension bounds via
% x = (ub - lb)(pos + 1)/2 + lb, the release's scalar mapping; maxit becomes the
% FE budget with the last batch truncated; the best is tracked at every FE.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wdo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    popsize = 20;
    RT      = 3;
    g       = 0.2;
    alp     = 0.4;
    c       = 0.4;
    maxV    = 0.3;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    pos = 2 * (rand(popsize, dim) - 0.5);
    vel = maxV * 2 * (rand(popsize, dim) - 0.5);

    x = (ub - lb) .* ((pos + 1) ./ 2) + lb;
    m = min(popsize, maxFE);
    [pres, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    pres = pres(:)';
    Xrec = x(1:m, :);   % the points each frec entry was measured at
    frec = pres;

    best_fitness  = inf;
    best_solution = x(1, :);
    for k = 1:m
        stamp(k, x(k, :), pres(k));
    end

    if m == popsize
        [globalpres, indx] = min(pres);
        globalpos = pos(indx, :);
        [~, rank_ind] = sort(pres);
        pos  = pos(rank_ind, :);    % vel is left in evaluation order here, as released
        Xrec = Xrec(rank_ind, :);
        frec = frec(rank_ind);
    end

    while FE < maxFE
        for i = 1:popsize
            a = randperm(dim);
            velot = vel(i, a);
            % i is the pressure rank, standing in for the pressure ratio P_opt/P_cur
            vel(i, :) = (1 - alp) * vel(i, :) - (g * pos(i, :)) + ...
                        abs(1 - 1 / i) * ((globalpos - pos(i, :)) .* RT) + ...
                        (c * velot / i);
        end

        vel = min(vel, maxV);
        vel = max(vel, -maxV);
        pos = pos + vel;
        pos = min(pos, 1.0);
        pos = max(pos, -1.0);

        x = (ub - lb) .* ((pos + 1) ./ 2) + lb;
        m = min(popsize, maxFE - FE);
        [pres, FE] = calculate_fitness(x(1:m, :)', problem, FE);
        pres = pres(:)';
        for k = 1:m
            Xrec(k, :) = x(k, :);
            frec(k)    = pres(k);
            stamp(FE - m + k, x(k, :), pres(k));
        end
        if m < popsize
            break;
        end

        [minpres, indx] = min(pres);
        minpos = pos(indx, :);
        [~, rank_ind] = sort(pres);
        pos  = pos(rank_ind, :);
        vel  = vel(rank_ind, :);
        Xrec = Xrec(rank_ind, :);
        frec = frec(rank_ind);

        if minpres < globalpres
            globalpres = minpres;
            globalpos  = minpos;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Best-so-far at evaluation fe, then one history sample of (Xrec, frec)
    function stamp(fe, xk, fk)
        if fk < best_fitness
            best_fitness  = fk;
            best_solution = xk;
        end
        curve(fe) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            fe, Xrec, frec, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

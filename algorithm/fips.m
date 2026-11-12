% ----------------------------------------------------------------------- %
% Fully Informed Particle Swarm (FIPS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20                      % Swarm size, laid out as a 4 x 5 torus
%   topology = USquare          % Four lattice neighbours, the particle itself excluded
%   chi = 0.7298                % Constriction coefficient
%   phi = 4.1                   % Total acceleration, shared equally by the K = 4 neighbours
%   Vmax = (ub - lb)/2          % Velocity cap per dimension (Xmax)
%   v0 = U(-Vmax, Vmax)         % Initial velocity
%
% Algorithm Concept:
%   - Every neighbour's personal best pulls the particle, none is singled out:
%     v = chi*(v + sum_k U(0, phi/K) .* (p_k - x)), x = x + v
%   - The particle's own personal best is not one of the K informers (the
%     "U" topologies of the paper)
%   - Particles move one at a time, so a neighbour's improved personal best is
%     seen by the next particle in the same sweep
%   - A particle outside the box keeps flying but is not evaluated and cannot
%     update its personal best
%
% Reference:
% Rui Mendes, James Kennedy, Jose Neves,
% The fully informed particle swarm: simpler, maybe better,
% IEEE Transactions on Evolutionary Computation 8(3) (2004) 204-210.
% https://doi.org/10.1109/TEVC.2004.826074
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Eqs. 5-8, Secs. V-VI); no author code is known. Of the weightings
% and nine topologies tested, Sec. VI-D recommends unweighted FIPS on USquare: 98.9 %
% success (Table V) and the best 1000-iteration score (Table III); the one 100 % cell
% is the fitness-weighted wFIPS on URing. With constant W, Eqs. 5 and 8 reduce to the
% sum above. The paper leaves Vmax, v0, update order and the box open: Vmax = Xmax,
% Kennedy's per-particle loop, and particles fly out of the box as in lpso.
% Paper Tables IV/V (USquare FIPS, symmetric init, 10000 iterations) against 10 seeded
% runs per problem here:
%      median iterations to criterion, mean over the 6 problems   367.5 / 351.7
%      success proportion                                         0.988 / 0.983
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fips(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    rows = 4;
    cols = 5;
    N    = rows * cols;
    chi  = 0.7298;
    phi  = 4.1;
    vmax = span / 2;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;
    v = (2 * rand(N, D) - 1) .* vmax;

    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    f = inf(N, 1);
    f(1:m) = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for k = 1:m
        if f(k) < bsf
            bsf          = f(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:m, :), f(1:m), population_history, fitness_history, history_index, maxFE);
    end

    p  = x;
    fp = f;
    % Mirror for the recorder: each particle at the position its fitness was measured
    xrec = x;
    frec = f;

    % USquare: up, down, left and right on the torus, self excluded
    nb = zeros(N, 4);
    for i = 1:N
        r = floor((i - 1) / cols);
        c = mod(i - 1, cols);
        nb(i, :) = [mod(r - 1, rows) * cols + c + 1, mod(r + 1, rows) * cols + c + 1, ...
                    r * cols + mod(c - 1, cols) + 1, r * cols + mod(c + 1, cols) + 1];
    end
    K = size(nb, 2);

    while FE < maxFE
        for i = 1:N
            % Eqs. 5-8 with constant W: one U(0, phi/K) vector per informer
            pull = sum((phi / K) * rand(K, D) .* (p(nb(i, :), :) - x(i, :)), 1);
            vi = chi * (v(i, :) + pull);
            vi = min(max(vi, -vmax), vmax);
            v(i, :) = vi;
            x(i, :) = x(i, :) + vi;

            if all(x(i, :) >= lb & x(i, :) <= ub)
                [fi, FE] = calculate_fitness(x(i, :)', problem, FE);
                xrec(i, :) = x(i, :);
                frec(i)    = fi;
                % A NaN personal best could never be beaten, so it is overwritten
                if fi < fp(i) || isnan(fp(i))
                    fp(i)   = fi;
                    p(i, :) = x(i, :);
                end
                if fi < bsf
                    bsf          = fi;
                    bsf_solution = x(i, :);
                end
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xrec, frec, population_history, fitness_history, history_index, maxFE);
                if FE >= maxFE
                    break;
                end
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

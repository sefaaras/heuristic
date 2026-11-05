% ----------------------------------------------------------------------- %
% Standard Particle Swarm Optimization of Bratton and Kennedy (SPSO-BK07)
% Stored as lpso: the 2007 constricted lbest-ring standard, not pso or spso2011
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                      % Swarm size
%   chi = 0.72984               % Constriction factor, phi = c1 + c2 = 4.1
%   c1 = c2 = 2.05              % Cognitive and social acceleration
%   ring K = 2                  % Neighbourhood: the particle and its two index neighbours
%   Vmax = 10*(ub - lb)         % Generous velocity cap, per dimension
%   v0 = U(-(ub-lb)/2, (ub-lb)/2)  % Initial velocity, drawn like a centred position
%
% Algorithm Concept:
%   - Constricted update v = chi*(v + c1*e1.*(p_i - x) + c2*e2.*(p_l - x)),
%     x = x + v, with e1, e2 ~ U(0,1) per dimension
%   - p_l is the best personal best on the ring {i-1, i, i+1}
%   - Particles move one at a time, so a neighbour's improved personal best is
%     seen by the next particle in the same sweep
%   - Particles may leave the box: they keep position and velocity, are not
%     evaluated and cannot update their personal best ("letting them fly")
%
% Reference:
% Daniel Bratton, James Kennedy,
% Defining a Standard for Particle Swarm Optimization,
% 2007 IEEE Swarm Intelligence Symposium (SIS 2007) 120-127.
% https://doi.org/10.1109/SIS.2007.368035
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the SIS 2007 paper (Algorithm 1 and Sec. III-B..D) with the definition in
% Bratton's thesis (Goldsmiths 2010, Ch. 3.4), the open restatement. Neither
% gives code. Region-scaled initialisation is a testing device that needs the
% optimum's location, so x0 ~ U(lb, ub); the shifted CEC optima serve the same
% purpose. The paper leaves v0 and the Vmax value open: the thesis says v0 is
% drawn the way x0 is (on a centred box) and uses Vmax = 10*range. The thesis
% pseudocode's gbest term contradicts its own ring definition; the paper's
% neighbourhood best is used. An out-of-box particle costs no FE.
% Paper Table III (D = 30, 3e5 FE, region-scaled init) against 8 seeded runs here:
%      Sphere 0 / 1.7e-56   Quadric 0.126 / 0.071   Rosenbrock 12.7 / 7.3
%      Schwefel 3360 / 3339   Rastrigin 145 / 163   Ackley 17.6 / 19.6   Griewank 9e-4 / 2.5e-13
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N   = 50;
    chi = 0.72984;
    c1  = 2.05;
    c2  = 2.05;
    vmax = 10 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;
    v = (rand(N, D) - 0.5) .* span;

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

    left  = [N, 1:N-1];
    right = [2:N, 1];

    while FE < maxFE
        for i = 1:N
            nb = [left(i), i, right(i)];
            [~, b] = min(fp(nb));
            pl = p(nb(b), :);

            e1 = rand(1, D);
            e2 = rand(1, D);
            vi = chi * (v(i, :) + c1 * e1 .* (p(i, :) - x(i, :)) + c2 * e2 .* (pl - x(i, :)));
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

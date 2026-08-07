% ----------------------------------------------------------------------- %
% Phasor Particle Swarm Optimization (PPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   npop = 40             % Swarm size
%
% Algorithm Concept:
%   - Parameter-free variant of PSO in which the control coefficients are
%     modelled with a phase angle (theta) using trigonometric functions
%   - The velocity limit is adapted through the phase angle of each particle
%
% Reference:
% Mojtaba Ghasemi, Ebrahim Akbari, Abolfazl Rahimnejad, Seyed Ehsan Razavi,
% Sahand Ghavidel, Li Li,
% Phasor particle swarm optimization: a simple and efficient variant of PSO,
% Soft Computing 23 (2019) 9701-9718
% https://doi.org/10.1007/s00500-018-3536-8
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ppso(problem)

    % Extract problem parameters
    nvar  = problem.dimension;
    xmin  = problem.lb;
    xmax  = problem.ub;
    maxFE = problem.maxFe;

    dx = xmax - xmin;
    vmax = 0.5 * dx;
    npop = 40;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    delta    = zeros(npop, 1);
    velocity = zeros(npop, nvar);
    position = zeros(npop, nvar);
    pbest    = zeros(npop, nvar);
    pbestcost = zeros(npop, 1);
    gbestcost = inf;
    gbest = zeros(1, nvar);

    % Initialisation
    for i = 1:npop
        velocity(i, :) = zeros(1, nvar);
        delta(i) = unifrnd(0, 2 * pi);
        position(i, :) = xmin + (xmax - xmin) .* rand(1, nvar);
    end

    % An unevaluated particle keeps Inf, so it cannot pose as the best of the swarm
    cost = inf(npop, 1);
    for i = 1:npop
        [cost(i), FE] = calculate_fitness(position(i, :)', problem, FE);

        pbest(i, :) = position(i, :);
        pbestcost(i) = cost(i);

        if pbestcost(i) < gbestcost
            gbest = pbest(i, 1:nvar);
            gbestcost = pbestcost(i);
        end

        if FE >= 1 && FE <= maxFE
            curve(FE) = gbestcost;
        end
        % cost is still Inf past i, so the swarm is only worth recording once
        % every particle has been evaluated
        if i == npop && FE >= 1 && FE <= maxFE
            [population_history, fitness_history, history_index] = record_history(...
                FE, position, cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        for i = 1:npop
            if FE >= maxFE
                break;
            end

            aa = 2 * (sin(delta(i)));
            bb = 2 * (cos(delta(i)));
            ee = abs(cos(delta(i)))^aa;
            tt = abs(sin(delta(i)))^bb;

            velocity(i, :) = (ee) * (pbest(i, :) - position(i, :)) + (tt) * (gbest - position(i, :));
            velocity(i, :) = min(max(velocity(i, :), -vmax), vmax);

            position(i, :) = position(i, :) + velocity(i, :);
            position(i, :) = min(max(position(i, :), xmin), xmax);

            [cost(i), FE] = calculate_fitness(position(i, :)', problem, FE);

            delta(i) = delta(i) + (abs(aa + bb) * (2 * pi));
            vmax = (abs(cos(delta(i)))^2) * dx;

            if cost(i) < pbestcost(i)
                pbest(i, :) = position(i, :);
                pbestcost(i) = cost(i);
                if pbestcost(i) < gbestcost
                    gbest = pbest(i, :);
                    gbestcost = pbestcost(i);
                end
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = gbestcost;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, position, cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    best_solution = gbest;
    best_fitness = gbestcost;

end

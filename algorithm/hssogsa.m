% ----------------------------------------------------------------------- %
% Hybrid Sperm Swarm Optimization and Gravitational Search Algorithm (HSSOGSA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 30                % Population size
%   G0 = 1                % Initial gravitational constant
%
% Algorithm Concept:
%   - Combines the exploitation capability of Sperm Swarm Optimization (SSO)
%     with the exploration capability of the Gravitational Search Algorithm
%     (GSA)
%   - Masses interact through gravitational forces while a sperm-swarm style
%     velocity update (with log-based velocity/temperature terms) drives the
%     motion towards the global best
%
% Reference:
% Hisham A. Shehadeh,
% A hybrid sperm swarm optimization and gravitational search algorithm
% (HSSOGSA) for global optimization,
% Neural Computing and Applications 33 (2021) 11739-11752
% https://doi.org/10.1007/s00521-021-05880-4
% ----------------------------------------------------------------------- %
% Implementation Note:
% The GSA mass divides by (best - worst). CEC2020RW RC25 really returns Inf near
% its interior pole, and one Inf individual makes that Inf/Inf, so every mass and
% then every position turns NaN. The range is therefore taken over the finite
% fitnesses and a non-finite individual is ranked as the worst finite one, which
% is the mass it would have carried had the objective merely been large. A
% collapsed range, which the reference divides by directly, now gives every
% individual the same mass instead of Inf.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hssogsa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    n = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    iteration = (maxFE / n) + 1;
    low = lb';
    up = ub';

    current_fitness = zeros(n, 1);
    sgBest = zeros(1, dim);
    sgBestScore = inf;
    G0 = 1;   % gravitational constant

    Boundary_no = size(up, 1);
    if Boundary_no == 1
        current_position = rand(n, dim) .* (up - low) + low;
    end
    if Boundary_no > 1
        for i = 1:dim
            up_i = up(i, 1);
            low_i = low(i, 1);
            current_position(:, i) = rand(n, 1) .* (up_i - low_i) + low_i;
        end
    end

    velocity = .3 * randn(n, dim);
    mass(n) = 0;

    iter = 0;
    while FE < maxFE
        FE_before = FE;

        G = G0 * exp(-23 * iter / iteration);   % Equation (4)
        iter = iter + 1;
        force = zeros(n, dim);
        mass(n) = 0;
        acceleration = zeros(n, dim);

        % Boundary control (positions static during evaluation)
        for i = 1:n
            Flag4up = current_position(i, :) > up';
            Flag4low = current_position(i, :) < low';
            current_position(i, :) = (current_position(i, :) .* (~(Flag4up + Flag4low))) + up' .* Flag4up + low' .* Flag4low;
        end

        % Evaluate the population
        [current_fitness, FE] = calculate_fitness(current_position', problem, FE);
        current_fitness = current_fitness(:);

        for i = 1:n
            if (sgBestScore > current_fitness(i))
                sgBestScore = current_fitness(i);
                sgBest = current_position(i, :);
            end
        end

        % An Inf objective must not set the range; it ranks as the worst finite one
        finite_fit = current_fitness(isfinite(current_fitness));
        if isempty(finite_fit)
            best = 0;
            worst = 0;
        else
            best = min(finite_fit);
            worst = max(finite_fit);
        end
        ranked_fitness = current_fitness;
        ranked_fitness(~isfinite(current_fitness)) = worst;

        for pp = 1:n
            if current_fitness(pp) == best
                break;
            end
        end

        for i = 1:n
            if best == worst
                mass(i) = 1;   % no spread to rank against, so every mass is equal
            else
                mass(i) = (ranked_fitness(i) - 0.99 * worst) / (best - worst);
            end
        end
        for i = 1:n
            mass(i) = mass(i) * 5 / sum(mass);
        end

        % Force update - Equation (3)
        for i = 1:n
            for j = 1:dim
                for k = 1:n
                    if (current_position(k, j) ~= current_position(i, j))
                        force(i, j) = force(i, j) + rand() * G * mass(k) * mass(i) * (current_position(k, j) - current_position(i, j)) / abs(current_position(k, j) - current_position(i, j));
                    end
                end
            end
        end

        % Acceleration - Equation (6)
        for i = 1:n
            for j = 1:dim
                if (mass(i) ~= 0)
                    acceleration(i, j) = force(i, j) / mass(i);
                end
            end
        end

        % Velocity update - Equation (9)
        for i = 1:n
            for j = 1:dim
                velocity(i, j) = rand() * (log10((7 - 14) * rand(1, 1) + 7)) * velocity(i, j) + rand() * acceleration(i, j) + (log10((7 - 14) * rand(1, 1) + 7)) * (log10((35.5 - 38.5) * rand(1, 1) + 35.5)) * (sgBest(j) - current_position(i, j));
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = sgBestScore;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, current_position, current_fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE
            break;
        end

        % Positions update - Equation (10)
        current_position = current_position + velocity;
    end

    best_solution = sgBest;
    best_fitness = sgBestScore;

end

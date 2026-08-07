% ----------------------------------------------------------------------- %
% Water Uptake and Transport in Plants (WUTP)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   no_particles = 50        % Population size (water particles)
%   p    = 0.5               % Pore-size alternation probability
%   rho  = 1000              % Water density
%   eta  = 0.0018            % Dynamic viscosity
%   g    = 9.81              % Gravitational acceleration
%   Lp = D = K = 1e-9        % Radial/hydraulic conductivity and diffusivity
%   a    = 1, chi = 0.5, rr = 0.1
%
% Algorithm Concept:
%   - Five transport stages per iteration, each derived from a hydrodynamic law
%   - Water in motion: velocity update J (inertia lambda, radial Lp pull towards
%     a random personal best, buoyancy rho*g term)
%   - Horizontal soil flow: Darcy's law (large pores) or Fick's diffusion
%   - Vertical soil flow: Richards / Darcy with the buoyancy correction
%   - Soil-to-root-surface uptake: Darcy or Richards with a personal-best pull
%   - Root-surface-to-xylem and xylem-to-leaves: Poiseuille's law
%   - Step scale nu decays on a logistic schedule over the run
%
% Reference:
% Malik Braik, Heba Al-Hiary,
% A novel meta-heuristic optimization algorithm inspired by water uptake and
% transport in plants,
% Neural Computing and Applications (2025).
% https://doi.org/10.1007/s00521-025-11228-z
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wutp(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    no_particles = 50;
    MaxIt = max(1, ceil((maxFE - no_particles) / no_particles));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    y = repmat(lb, no_particles, 1) + repmat(ub - lb, no_particles, 1) .* rand(no_particles, dim);
    J = 0.1 * y;      % initial velocity

    [cost, FE] = calculate_fitness(y', problem, FE);
    cost = cost(:);

    fitness = cost;
    [fmin0, index] = min(cost);
    yo    = y;
    pbest = y;
    gbest = y(index, :);
    bsf   = fmin0;

    for eval_count = 1:no_particles
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, y, cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Controlling parameters
    p   = 0.5;      % alternate parameter
    rho = 1000;     % water density
    eta = 0.0018;
    g   = 9.81;     % gravitational force
    Lp  = 1.0e-9;   % radial hydraulic conductivity
    Dc  = 1e-9;     % diffusivity coefficient
    K   = 1e-9;     % hydraulic conductivity
    a   = 1;
    rr  = 0.1;
    chi = 0.5;

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        lambda = 1.0 - rand / 2;
        nu = (1e-7) / (1 + exp((It - MaxIt / 2) / 100));
        sigma = 1;

        % 1) Water in motion: drivers of flow
        for i = 1:size(y, 1)
            rand_index = ceil(no_particles * rand());
            pbeta = pbest(rand_index, :);
            c1 = rand;
            c2 = rand;
            J(i, :) = lambda * J(i, :) + c1 * Lp * (y(i, :) - sigma * pbeta) + ...
                      c2 * rho * g * (y(i, :) - yo(i, :));
        end

        % 2) Horizontal flow through soil (Darcy / Fick)
        for i = 1:size(y, 1)
            for j = 1:dim
                c3 = rand(); c4 = rand(); c5 = rand();
                if c3 < p            % large pores -- Darcy
                    if c4 > 0.1
                        y(i, j) = y(i, j) - J(i, j) ./ K;
                    else
                        y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                    end
                else                 % Fick's equation
                    if c5 > 0.1
                        y(i, j) = y(i, j) - J(i, j) ./ (2 * Dc * pi * a ^ 2);
                    else
                        y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                    end
                end
            end
        end

        % 3) Vertical flow through soil (Richards / Fick)
        for i = 1:size(y, 1)
            for j = 1:dim
                if rand() > p        % large pores -- Richards' law
                    if rand() > 0.1
                        y(i, j) = y(i, j) - J(i, j) ./ (K) + K * rho * g * (y(i, j) - yo(i, j)) * rand;
                    else
                        y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                    end
                else                 % Fick's equation
                    if rand() > 0.1
                        y(i, j) = y(i, j) - J(i, j) ./ (2 * Dc * pi * a ^ 2) + Dc * rho * g * (y(i, j) - yo(i, j)) * rand;
                    else
                        y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                    end
                end
            end
        end

        % 4) Water movement from soil to root surface
        for j = 1:dim
            for i = 1:size(y, 1)
                rand_index = ceil(no_particles * rand());
                pbeta = pbest(rand_index, :);
                if rand() > p        % Darcy's law
                    y(i, j) = y(i, j) - J(i, j) ./ (K) + K * rho * g * (y(i, j) - yo(i, j)) * rand;
                else                 % Richards' law
                    y(i, j) = y(i, j) - J(i, j) ./ (K * pi * a ^ 2) + rand * (pbeta(j) - yo(i, j));
                end
            end
        end

        % 5) Movement from root surface to xylem
        for j = 1:dim
            for i = 1:size(y, 1)
                rand_index = ceil(no_particles * rand());
                pbeta = pbest(rand_index, :);
                if rand() < 0.1
                    y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                else
                    y(i, j) = y(i, j) - J(i, j) ./ (Lp) - 1 * chi * (y(i, j) - yo(i, j)) * rand + ...
                              1 * 1 * 1 * (pbeta(j) - y(i, j)) * rand;
                end
            end
        end

        % 6) Water movement through xylem to the leaves (Poiseuille)
        for i = 1:size(y, 1)
            if rand < rr
                for j = 1:dim
                    y(i, j) = y(i, j) - nu * (lb(j) - ((lb(j) - ub(j)) * rand));
                end
            else
                y(i, :) = y(i, :) + 1 * J(i, :) * (8 * eta / pi * a ^ 4);
            end
        end

        % Handle boundary violations
        for i = 1:size(y, 1)
            y(i, :) = min(max(y(i, :), lb), ub);
        end

        % Update global, best and new positions
        [cost, FE] = calculate_fitness(y', problem, FE);
        cost = cost(:);

        for ii = 1:no_particles
            if all(lb - y(ii, :) <= 0) && all(ub - y(ii, :) >= 0)
                yo(ii, :) = y(ii, :);
                if cost(ii) < fitness(ii)
                    pbest(ii, :) = y(ii, :);
                    fitness(ii)  = cost(ii);
                end
            end
        end

        [fmin, index] = min(fitness);
        if fmin < fmin0
            gbest = pbest(index, :);
            fmin0 = fmin;
        end

        if fmin0 < bsf
            bsf = fmin0;
        end
        for k = 1:no_particles
            ec = FE - no_particles + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, y, cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = fmin0;
    best_solution = gbest;
end

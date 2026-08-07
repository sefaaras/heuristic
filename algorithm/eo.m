% ----------------------------------------------------------------------- %
% Equilibrium Optimizer (EO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Particles_no = 30   % Population size
%   a1 = 2; a2 = 1;     % Exploration / exploitation constants
%   GP = 0.5;           % Generation probability
%
% Algorithm Concept:
%   - Physics-based, inspired by control-volume mass balance
%   - Four best equilibrium candidates plus their average form a pool
%   - Concentrations updated with an exponential term and adaptive time
%   - Memory saving retains better previous solutions (elitism)
%
% Reference:
% Afshin Faramarzi, Mohammad Heidarinejad, Brent Stephens, Seyedali Mirjalili,
% Equilibrium optimizer: A novel optimization algorithm,
% Knowledge-Based Systems 191 (2020) 105190
% https://doi.org/10.1016/j.knosys.2019.105190
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = eo(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    Particles_no = 30;
    a1 = 2; a2 = 1; GP = 0.5; V = 1;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Equilibrium candidates
    Ceq1 = zeros(1, dim); Ceq1_fit = inf;
    Ceq2 = zeros(1, dim); Ceq2_fit = inf;
    Ceq3 = zeros(1, dim); Ceq3_fit = inf;
    Ceq4 = zeros(1, dim); Ceq4_fit = inf;

    C = initialization(Particles_no, dim, ub, lb);

    fit_old = [];
    C_old   = [];
    best_so_far = inf;
    first_iter = true;

    while FE < maxFE

        % Boundary handling
        for i = 1:Particles_no
            Flag4ub = C(i, :) > ub;
            Flag4lb = C(i, :) < lb;
            C(i, :) = (C(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end

        % Evaluate the whole population
        [fitness, FE] = calculate_fitness(C', problem, FE);
        fitness = fitness(:)';

        % Update equilibrium candidates
        for i = 1:Particles_no
            if fitness(i) < Ceq1_fit
                Ceq1_fit = fitness(i); Ceq1 = C(i, :);
            elseif fitness(i) > Ceq1_fit && fitness(i) < Ceq2_fit
                Ceq2_fit = fitness(i); Ceq2 = C(i, :);
            elseif fitness(i) > Ceq1_fit && fitness(i) > Ceq2_fit && fitness(i) < Ceq3_fit
                Ceq3_fit = fitness(i); Ceq3 = C(i, :);
            elseif fitness(i) > Ceq1_fit && fitness(i) > Ceq2_fit && fitness(i) > Ceq3_fit && fitness(i) < Ceq4_fit
                Ceq4_fit = fitness(i); Ceq4 = C(i, :);
            end
        end

        % Memory saving (elitism vs previous generation)
        if first_iter
            fit_old = fitness; C_old = C;
            first_iter = false;
        end
        for i = 1:Particles_no
            if fit_old(i) < fitness(i)
                fitness(i) = fit_old(i);
                C(i, :)    = C_old(i, :);
            end
        end
        C_old = C; fit_old = fitness;

        best_so_far = min(best_so_far, Ceq1_fit);

        % Record convergence curve and history (population matches fitness)
        for eval_idx = 1:Particles_no
            eval_count = FE - Particles_no + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_so_far;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, C, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Build equilibrium pool
        Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4;
        C_pool  = [Ceq1; Ceq2; Ceq3; Ceq4; Ceq_ave];

        t = (1 - FE / maxFE) ^ (a2 * FE / maxFE);   % Eq. (9)

        % Concentration update
        for i = 1:Particles_no
            lambda = rand(1, dim);
            r = rand(1, dim);
            Ceq = C_pool(randi(size(C_pool, 1)), :);
            F = a1 * sign(r - 0.5) .* (exp(-lambda .* t) - 1);   % Eq. (11)
            r1 = rand(); r2 = rand();
            GCP = 0.5 * r1 * ones(1, dim) * (r2 >= GP);          % Eq. (15)
            G0 = GCP .* (Ceq - lambda .* C(i, :));               % Eq. (14)
            G = G0 .* F;                                         % Eq. (13)
            C(i, :) = Ceq + (C(i, :) - Ceq) .* F + (G ./ lambda * V) .* (1 - F);   % Eq. (16)
        end
    end

    best_fitness  = Ceq1_fit;
    best_solution = Ceq1;
end

% Initialization Function
function Cin = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Cin = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        Cin = zeros(SearchAgents_no, dim);
        for i = 1:dim
            Cin(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

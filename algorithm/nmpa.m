% ----------------------------------------------------------------------- %
% Nonlinear Marine Predator Algorithm (NMPA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 25    % Population size (prey)
%   FADs = 0.2   % Fish Aggregating Devices effect probability
%   P    = 0.5   % Constant
%
% Algorithm Concept:
%   - Three-phase MPA (Brownian high-velocity, unit-velocity, Levy
%     low-velocity) with a nonlinear weight w1 = 2*exp(-(6*Iter/Max)^2)
%     replacing MPA's linear controls, plus FADs/eddy formation.
%
% Reference:
% Ali Safaa Sadiq, Amin Abdollahi Dehkordi, Seyedali Mirjalili,
% Quoc-Viet Pham,
% Nonlinear marine predator algorithm: A cost-effective optimizer for
% fair power allocation in NOMA-VLC-B5G networks,
% Expert Systems with Applications 203 (2022) 117395.
% https://doi.org/10.1016/j.eswa.2022.117395
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nmpa(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 25;
    Max_iter = ceil(maxFE / (N * 2));   % two evaluations per prey per iteration
    FADs = 0.2;
    P = 0.5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Top_predator_pos = zeros(1, dim);
    Top_predator_fit = inf;
    best_pos = Top_predator_pos;

    Prey = initialization(N, dim, ub, lb);
    Xmin = repmat(lb, N, 1);
    Xmax = repmat(ub, N, 1);

    fitness = inf(N, 1);
    fit_old = fitness;
    Prey_old = Prey;

    Iter = 0;
    while FE < maxFE
        % Detecting top predator (evaluate current prey)
        for i = 1:N
            Prey(i, :) = bound(Prey(i, :), ub, lb);
        end
        [fitness, FE] = calculate_fitness(Prey', problem, FE);
        fitness = fitness(:);
        for i = 1:N
            if fitness(i) < Top_predator_fit
                Top_predator_fit = fitness(i);
                Top_predator_pos = Prey(i, :);
                best_pos = Top_predator_pos;
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Top_predator_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Prey, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        % Marine memory saving
        if Iter == 0
            fit_old = fitness; Prey_old = Prey;
        end
        Inx = (fit_old < fitness);
        Indx = repmat(Inx, 1, dim);
        Prey = Indx .* Prey_old + ~Indx .* Prey;
        fitness = Inx .* fit_old + ~Inx .* fitness;
        fit_old = fitness; Prey_old = Prey;

        % Movement
        Elite = repmat(Top_predator_pos, N, 1);           % Eq.(10)
        CF = abs(2 * (1 - (Iter / Max_iter)) - 2);
        RL = 0.05 * levy(N, dim, 1.5);
        RB = randn(N, dim);
        w1 = 2 * exp(-(6 * Iter / Max_iter)^2);           % nonlinear weight

        if Iter < Max_iter / 3
            Rmat = rand(N, dim);
            stepsize = RB .* (Elite - RB .* Prey);
            Prey = Prey + P * Rmat .* stepsize;
        elseif Iter > Max_iter / 3 && Iter < 2 * Max_iter / 3
            topM = repmat(((1:N)' > N / 2), 1, dim);
            Rmat = rand(N, dim);
            step_top = RB .* (RB .* Elite - Prey);
            Prey_top = w1 .* Elite + P * CF * step_top;
            step_bot = RL .* (Elite - RL .* Prey);
            Prey_bot = w1 .* Prey + P * Rmat .* step_bot;
            Prey = topM .* Prey_top + (~topM) .* Prey_bot;
        else
            stepsize = RL .* (RL .* Elite - Prey);
            Prey = Elite + P * CF * stepsize;
        end

        % Detecting top predator (evaluate moved prey)
        for i = 1:N
            Prey(i, :) = bound(Prey(i, :), ub, lb);
        end
        [fitness, FE] = calculate_fitness(Prey', problem, FE);
        fitness = fitness(:);
        for i = 1:N
            if fitness(i) < Top_predator_fit
                Top_predator_fit = fitness(i);
                Top_predator_pos = Prey(i, :);
                best_pos = Top_predator_pos;
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Top_predator_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Prey, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        % Marine memory saving
        Inx = (fit_old < fitness);
        Indx = repmat(Inx, 1, dim);
        Prey = Indx .* Prey_old + ~Indx .* Prey;
        fitness = Inx .* fit_old + ~Inx .* fitness;
        fit_old = fitness; Prey_old = Prey;

        % Eddy formation and FADs effect (Eq.16)
        if rand() < FADs
            U = rand(N, dim) < FADs;
            Prey = Prey + CF * ((Xmin + rand(N, dim) .* (Xmax - Xmin)) .* U);
        else
            r = rand();
            stepsize = (FADs * (1 - r) + r) * (Prey(randperm(N), :) - Prey(randperm(N), :));
            Prey = Prey + stepsize;
        end

        Iter = Iter + 1;
    end

    curve(min(FE, maxFE):end) = Top_predator_fit;
    best_fitness = Top_predator_fit;
    best_solution = best_pos;
end

% Levy flight (n x m)
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = randn(n, m) * sigma_u;
    v = randn(n, m);
    z = u ./ (abs(v).^(1 / beta));
end

% Initialization
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

% Boundary Handling
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

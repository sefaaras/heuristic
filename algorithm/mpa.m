% ----------------------------------------------------------------------- %
% Marine Predators Algorithm (MPA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 25  % Population size (number of predators/prey)
%   FADs = 0.2            % Fish Aggregating Devices effect probability
%   P = 0.5               % Constant number
%
% Algorithm Concept:
%   - Inspired by the foraging strategy (Levy vs Brownian movement) of
%     ocean predators and the optimal encounter-rate policy
%   - Three phases over the iteration budget (high/unit/low velocity ratio)
%     plus eddy formation / FADs effect and a marine-memory (elitism) step
%
% Reference:
% Afshin Faramarzi, Mohammad Heidari, Seyedali Mirjalili, Amir H. Gandomi,
% Marine Predators Algorithm: A nature-inspired metaheuristic,
% Expert Systems with Applications 152 (2020) 113377
% https://doi.org/10.1016/j.eswa.2020.113377
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mpa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 25;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    Max_iter = ceil(maxFE / (SearchAgents_no * 2));

    Top_predator_pos = zeros(1, dim);
    Top_predator_fit = inf;

    stepsize = zeros(SearchAgents_no, dim);
    fitness = inf(SearchAgents_no, 1);
    Prey = initialization(SearchAgents_no, dim, ub, lb);

    Xmin = repmat(ones(1, dim) .* lb, SearchAgents_no, 1);
    Xmax = repmat(ones(1, dim) .* ub, SearchAgents_no, 1);

    Iter = 0;
    FADs = 0.2;
    P = 0.5;

    fit_old = fitness;
    Prey_old = Prey;

    while FE < maxFE
        % ------------------- Detecting top predator -----------------
        for i = 1:size(Prey, 1)
            Flag4ub = Prey(i, :) > ub;
            Flag4lb = Prey(i, :) < lb;
            Prey(i, :) = (Prey(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        FE_before = FE;
        [fitness, FE] = calculate_fitness(Prey', problem, FE);
        fitness = fitness(:);
        for i = 1:size(Prey, 1)
            if fitness(i, 1) < Top_predator_fit
                Top_predator_fit = fitness(i, 1);
                Top_predator_pos = Prey(i, :);
            end
        end

        % ------------------- Marine Memory saving -------------------
        if Iter == 0
            fit_old = fitness;    Prey_old = Prey;
        end
        Inx = (fit_old < fitness);
        Indx = repmat(Inx, 1, dim);
        Prey = Indx .* Prey_old + ~Indx .* Prey;
        fitness = Inx .* fit_old + ~Inx .* fitness;
        fit_old = fitness;    Prey_old = Prey;

        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Top_predator_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Prey, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE
            break;
        end

        % ------------------------------------------------------------
        Elite = repmat(Top_predator_pos, SearchAgents_no, 1);   % (Eq. 10)
        CF = (1 - Iter / Max_iter)^(2 * Iter / Max_iter);

        RL = 0.05 * levy(SearchAgents_no, dim, 1.5);   % Levy random number vector
        RB = randn(SearchAgents_no, dim);              % Brownian random number vector

        for i = 1:size(Prey, 1)
            for j = 1:size(Prey, 2)
                R = rand();
                % ------------------ Phase 1 (Eq.12) -------------------
                if Iter < Max_iter / 3
                    stepsize(i, j) = RB(i, j) * (Elite(i, j) - RB(i, j) * Prey(i, j));
                    Prey(i, j) = Prey(i, j) + P * R * stepsize(i, j);
                    % --------------- Phase 2 (Eqs. 13 & 14)----------------
                elseif Iter > Max_iter / 3 && Iter < 2 * Max_iter / 3
                    if i > size(Prey, 1) / 2
                        stepsize(i, j) = RB(i, j) * (RB(i, j) * Elite(i, j) - Prey(i, j));
                        Prey(i, j) = Elite(i, j) + P * CF * stepsize(i, j);
                    else
                        stepsize(i, j) = RL(i, j) * (Elite(i, j) - RL(i, j) * Prey(i, j));
                        Prey(i, j) = Prey(i, j) + P * R * stepsize(i, j);
                    end
                    % ----------------- Phase 3 (Eq. 15)-------------------
                else
                    stepsize(i, j) = RL(i, j) * (RL(i, j) * Elite(i, j) - Prey(i, j));
                    Prey(i, j) = Elite(i, j) + P * CF * stepsize(i, j);
                end
            end
        end

        % ------------------ Detecting top predator ------------------
        for i = 1:size(Prey, 1)
            Flag4ub = Prey(i, :) > ub;
            Flag4lb = Prey(i, :) < lb;
            Prey(i, :) = (Prey(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        FE_before = FE;
        [fitness, FE] = calculate_fitness(Prey', problem, FE);
        fitness = fitness(:);
        for i = 1:size(Prey, 1)
            if fitness(i, 1) < Top_predator_fit
                Top_predator_fit = fitness(i, 1);
                Top_predator_pos = Prey(i, :);
            end
        end

        % ---------------------- Marine Memory saving ----------------
        if Iter == 0
            fit_old = fitness;    Prey_old = Prey;
        end
        Inx = (fit_old < fitness);
        Indx = repmat(Inx, 1, dim);
        Prey = Indx .* Prey_old + ~Indx .* Prey;
        fitness = Inx .* fit_old + ~Inx .* fitness;
        fit_old = fitness;    Prey_old = Prey;

        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Top_predator_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Prey, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        % ---------- Eddy formation and FADs effect (Eq 16) -----------
        if rand() < FADs
            U = rand(SearchAgents_no, dim) < FADs;
            Prey = Prey + CF * ((Xmin + rand(SearchAgents_no, dim) .* (Xmax - Xmin)) .* U);
        else
            r = rand();  Rs = size(Prey, 1);
            stepsize = (FADs * (1 - r) + r) * (Prey(randperm(Rs), :) - Prey(randperm(Rs), :));
            Prey = Prey + stepsize;
        end

        Iter = Iter + 1;
    end

    best_solution = Top_predator_pos;
    best_fitness = Top_predator_fit;

end

%% --- Initialization Function ---
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Levy flight distribution ---
function [z] = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = random('Normal', 0, sigma_u, n, m);
    v = random('Normal', 0, 1, n, m);
    z = u ./ (abs(v).^(1 / beta));
end

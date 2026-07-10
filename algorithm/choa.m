% ----------------------------------------------------------------------- %
% Chimp Optimization Algorithm (ChOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 30   % Population size
%   m = 0.700              % Chaotic value (fixed, per reference code)
%
% Algorithm Concept:
%   - Inspired by the individual intelligence and sexual motivation of
%     chimps during group hunting
%   - Four chimp roles: attacker, barrier, chaser and driver
%   - Dynamic coefficients drive position updates from the best chimps
%   - Chaotic maps refine the exploitation phase
%
% Reference:
% M. Khishe, M. R. Mosavi,
% Chimp Optimization Algorithm,
% Expert Systems with Applications 149 (2020) 113338
% https://doi.org/10.1016/j.eswa.2020.113338
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = choa(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    % Initialize Attacker, Barrier, Chaser, and Driver
    Attacker_pos = zeros(1, dim); Attacker_score = inf;
    Barrier_pos  = zeros(1, dim); Barrier_score  = inf;
    Chaser_pos   = zeros(1, dim); Chaser_score   = inf;
    Driver_pos   = zeros(1, dim); Driver_score   = inf;

    % Initialize positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb);

    % Main loop
    while FE < maxFE

        % Return agents that go beyond the search space
        for i = 1:SearchAgents_no
            Flag4ub = Positions(i, :) > ub;
            Flag4lb = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end

        % Evaluate the whole population
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        fitness = fitness(:)';

        % Update Attacker, Barrier, Chaser, and Driver
        for i = 1:SearchAgents_no
            if fitness(i) < Attacker_score
                Attacker_score = fitness(i);
                Attacker_pos   = Positions(i, :);
            end
            if fitness(i) > Attacker_score && fitness(i) < Barrier_score
                Barrier_score = fitness(i);
                Barrier_pos   = Positions(i, :);
            end
            if fitness(i) > Attacker_score && fitness(i) > Barrier_score && fitness(i) < Chaser_score
                Chaser_score = fitness(i);
                Chaser_pos   = Positions(i, :);
            end
            if fitness(i) > Attacker_score && fitness(i) > Barrier_score && fitness(i) > Chaser_score && fitness(i) > Driver_score
                Driver_score = fitness(i);
                Driver_pos   = Positions(i, :);
            end
        end

        % Record convergence curve and history (population matches fitness)
        for eval_idx = 1:SearchAgents_no
            eval_count = FE - SearchAgents_no + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = Attacker_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, fitness', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        % f decreases from 2 to 0 over the FE budget
        ratio  = FE / maxFE;
        r3     = ratio ^ (1/3);
        r_cube = ratio ^ 3;
        f = 2 - 2 * ratio;

        % Dynamic coefficients (Table 1)
        C1G1 = 1.95 - 2 * r3;      C2G1 = 2 * r3 + 0.5;
        C1G2 = 1.95 - 2 * r3;      C2G2 = 2 * r_cube + 0.5;
        C1G3 = -2 * r_cube + 2.5;  C2G3 = 2 * r3 + 0.5;
        C1G4 = -2 * r_cube + 2.5;  C2G4 = 2 * r_cube + 0.5;

        m = 0.700;   % chaotic value (Eq. 5)

        % Update the position of search agents
        for i = 1:SearchAgents_no
            for j = 1:dim
                r11 = C1G1 * rand(); r12 = C2G1 * rand();
                r21 = C1G2 * rand(); r22 = C2G2 * rand();
                r31 = C1G3 * rand(); r32 = C2G3 * rand();
                r41 = C1G4 * rand(); r42 = C2G4 * rand();

                A1 = 2 * f * r11 - f; C1 = 2 * r12;
                D_Attacker = abs(C1 * Attacker_pos(j) - m * Positions(i, j));
                X1 = Attacker_pos(j) - A1 * D_Attacker;

                A2 = 2 * f * r21 - f; C2 = 2 * r22;
                D_Barrier = abs(C2 * Barrier_pos(j) - m * Positions(i, j));
                X2 = Barrier_pos(j) - A2 * D_Barrier;

                A3 = 2 * f * r31 - f; C3 = 2 * r32;
                D_Chaser = abs(C3 * Chaser_pos(j) - m * Positions(i, j));
                X3 = Chaser_pos(j) - A3 * D_Chaser;

                A4 = 2 * f * r41 - f; C4 = 2 * r42;
                D_Driver = abs(C4 * Driver_pos(j) - m * Positions(i, j));
                X4 = Chaser_pos(j) - A4 * D_Driver;

                Positions(i, j) = (X1 + X2 + X3 + X4) / 4;   % Eq. (8)
            end
        end
    end

    best_fitness  = Attacker_score;
    best_solution = Attacker_pos;
end

%% --- Initialization Function ---
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(SearchAgents_no, dim);
        for i = 1:dim
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

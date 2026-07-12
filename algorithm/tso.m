% ----------------------------------------------------------------------- %
% Transient Search Optimization (TSO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 30  % Population size (number of search agents)
%   K = 1                 % Real coefficient of the transient response
%
% Algorithm Concept:
%   - Inspired by the transient behaviour of switched electrical circuits
%     (second-order RLC and first-order RC/RL circuits)
%   - Exploration/exploitation controlled by an exponentially damped term
%     exp(-T) combined with oscillatory cos/sin components
%
% Reference:
% Mohammed H. Qais, Hany M. Hasanien, Saad Alghuwainem,
% Transient search optimization: a new meta-heuristic optimization algorithm,
% Applied Intelligence 50 (2020) 3926-3941
% https://doi.org/10.1007/s10489-020-01727-y
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = tso(problem)

    % Extract problem parameters
    dim = problem.dimension;      % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations

    SearchAgents_no = 30;         % Population size

    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    % Initialize the voltages (positions) of search agents
    voltages = initialization(SearchAgents_no, dim, ub, lb);

    % Boundary check then evaluate initial population
    for i = 1:SearchAgents_no
        voltages(i, :) = bound(voltages(i, :), ub, lb);
    end
    [fitness, FE] = calculate_fitness(voltages', problem, FE);

    % Find initial best
    best_score = inf;
    best_voltage = zeros(1, dim);
    for i = 1:SearchAgents_no
        if fitness(i) < best_score
            best_score = fitness(i);
            best_voltage = voltages(i, :);
        end
    end

    % Record initial evaluations
    for eval_count = 1:SearchAgents_no
        curve(eval_count) = best_score;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, voltages, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    % Main loop
    L = (maxFE / SearchAgents_no) + 1;   % analogue of the original iteration budget
    l = 0;
    while FE < maxFE
        FE_before = FE;

        t = 2 - l * (2 / L);   % decreasing control parameter
        K = 1;                 % K is a real number (0, 1, 2, ...)

        % Update the voltage of search agents
        for i = 1:SearchAgents_no
            r1 = rand(); r2 = rand(); r3 = rand();
            T = 2 * t * r1 - t;
            C1 = K * r2 * t + 1;

            if r3 < 0.5
                voltages(i, :) = best_voltage + exp(-T) .* (voltages(i, :) - C1 * best_voltage);
            else
                voltages(i, :) = best_voltage + exp(-T) .* (cos(T * 2 * pi) + sin(T * 2 * pi)) .* abs(voltages(i, :) - C1 * best_voltage);
            end

            voltages(i, :) = bound(voltages(i, :), ub, lb);
        end

        % Evaluate new positions
        [fitness, FE] = calculate_fitness(voltages', problem, FE);

        % Update the best
        for i = 1:SearchAgents_no
            if fitness(i) < best_score
                best_score = fitness(i);
                best_voltage = voltages(i, :);
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = best_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, voltages, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        l = l + 1;
    end

    best_fitness = best_score;
    best_solution = best_voltage;

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

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

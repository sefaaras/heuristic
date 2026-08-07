% ----------------------------------------------------------------------- %
% Exponential-Trigonometric Optimization (ETO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N  = 30                          % Population size
%   b  = 1.55                        % Bounded-search trigger constant
%   CE = floor(1 + Max_Iter/b)       % Iteration where the box is narrowed
%   T  = floor(1.2 + Max_Iter/2.25)  % Phase switch iteration
%
% Algorithm Concept:
%   - Combination of exponential and trigonometric control variables:
%     d1/d2 (exponential-cosine pair), CM (a tangent-power ratio) and the
%     alpha_1..alpha_3 adaptive step amplitudes
%   - Phase 1 (t <= T): sign-symmetric moves around the destination driven by
%     alpha_1 or alpha_3, chosen by CM
%   - Phase 2 (t > T): exponential/linear amplification of the distance to the
%     destination, chosen by CM
%   - A bounded search strategy re-initialises the swarm inside a shrinking
%     box around the destination whenever the trigger iteration is reached
%
% Reference:
% Tran Minh Luan, Samir Khatir, Minh Thi Tran, Bernard De Baets,
% Thanh Cuong-Le,
% Exponential-trigonometric optimization algorithm for solving complicated
% engineering problems,
% Computer Methods in Applied Mechanics and Engineering 432 (2024) 117411.
% https://doi.org/10.1016/j.cma.2024.117411
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = eto(problem)

    Dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    N        = 30;
    Max_Iter = max(2, ceil(maxFE / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Destination_position        = zeros(1, Dim);
    Destination_fitness         = inf;
    Destination_position_second = zeros(1, Dim);
    Position_sort               = zeros(N, Dim);

    % ETO parameters
    b   = 1.55;
    CE  = floor(1 + (Max_Iter / b));
    T   = floor(1.2 + Max_Iter / 2.25);
    CEi = 0;
    CEi_temp = 0;
    UB_2 = UB;
    LB_2 = LB;

    % Initialisation
    X = initialization(N, Dim, UB, LB);
    [Objective_values, FE] = calculate_fitness(X', problem, FE);
    Objective_values = Objective_values(:)';

    for i = 1:N
        if Objective_values(1, i) < Destination_fitness
            Destination_position = X(i, :);
            Destination_fitness  = Objective_values(1, i);
        end
    end
    bsf = Destination_fitness;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Objective_values, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    t = 2;
    while t <= Max_Iter && FE < maxFE

        for i = 1:size(X, 1)
            for j = 1:size(X, 2)

                d1 =  0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (1 - t / Max_Iter));
                d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (1 - t / Max_Iter));

                CM = (sqrt(t / Max_Iter) ^ tan(d1 / (d2))) * rand() * 0.01;

                % Bounded search strategy
                if t == CEi
                    UB_2 = Destination_position(j) + (1 - t / Max_Iter) * ...
                           abs(rand() * Destination_position(j) - Destination_position_second(j)) * rand();
                    LB_2 = Destination_position(j) - (1 - t / Max_Iter) * ...
                           abs(rand() * Destination_position(j) - Destination_position_second(j)) * rand();
                    if UB_2 > UB(j)
                        UB_2 = UB(j);
                    end
                    if LB_2 < LB(j)
                        LB_2 = LB(j);
                    end
                    X = initialization(N, Dim, UB_2, LB_2);
                    CEi_temp = CEi;
                    CEi = 0;
                end

                if t <= T
                    % First phase of exploration / exploitation
                    q1 = rand(); q3 = rand(); q4 = rand(); q5 = rand();
                    if CM > 1
                        d1 =  0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q1));
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q1));
                        alpha_1 = q5 * 3 * (t / Max_Iter - 0.85) * exp(abs(d1 / d2) - 1);
                        if q1 <= 0.5
                            X(i, j) = Destination_position(j) + q5 * alpha_1 * abs(Destination_position(j) - X(i, j));
                        else
                            X(i, j) = Destination_position(j) - q5 * alpha_1 * abs(Destination_position(j) - X(i, j));
                        end
                    else
                        d1 =  0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q3));
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q3));
                        alpha_3 = rand() * 3 * (t / Max_Iter - 0.85) * exp(abs(d1 / d2) - 1.3);
                        if q3 <= 0.5
                            X(i, j) = Destination_position(j) + q4 * alpha_3 * abs(q5 * Destination_position(j) - X(i, j));
                        else
                            X(i, j) = Destination_position(j) - q4 * alpha_3 * abs(q5 * Destination_position(j) - X(i, j));
                        end
                    end
                else
                    % Second phase of exploration / exploitation
                    q2 = rand(); q6 = rand();
                    alpha_2 = q6 * exp(tanh(1.5 * (-t / Max_Iter - 0.75) - q6));
                    if CM < 1
                        d1 =  0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q2));
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * Max_Iter * (q2));
                        X(i, j) = X(i, j) + exp(tan(abs(d1 / d2)) * abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                    else
                        if q2 <= 0.5
                            X(i, j) = X(i, j) + 3 * (abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                        else
                            X(i, j) = X(i, j) - 3 * (abs(q6 * alpha_2 * Destination_position(j) - X(i, j)));
                        end
                    end
                end
            end
            CEi = CEi_temp;
        end

        % Bound handling and evaluation
        for i = 1:size(X, 1)
            Flag4ub = X(i, :) > UB_2;
            Flag4lb = X(i, :) < LB_2;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + (UB_2 + LB_2) / 2 .* Flag4ub + LB_2 .* Flag4lb;
            % Keep the point inside the true (possibly per-dimension) box.
            X(i, :) = min(max(X(i, :), LB), UB);
        end

        [Objective_values, FE] = calculate_fitness(X', problem, FE);
        Objective_values = Objective_values(:)';

        for i = 1:size(X, 1)
            if Objective_values(1, i) < Destination_fitness
                Destination_position = X(i, :);
                Destination_fitness  = Objective_values(1, i);
            end
        end
        if Destination_fitness < bsf
            bsf = Destination_fitness;
        end

        for k = 1:N
            ec = FE - N + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, Objective_values, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Identify the second solution and re-arm the bounded search
        if t == CE
            CEi = CE + 1;
            CE  = CE + floor(2 - t * 2 / (Max_Iter - CE * 4.6) / 1);
            temp  = zeros(1, Dim);
            temp2 = zeros(N, Dim);
            for i = 1:(size(X, 1) - 1)
                for j = 1:(size(X, 1) - 1 - i)
                    if Objective_values(1, j) > Objective_values(1, j + 1)
                        temp(1, j) = Objective_values(1, j);
                        Objective_values(1, j)     = Objective_values(1, j + 1);
                        Objective_values(1, j + 1) = temp(1, j);
                        temp2(j, :) = Position_sort(j, :);
                        Position_sort(j, :)     = Position_sort(j + 1, :);
                        Position_sort(j + 1, :) = temp2(j, :);
                    end
                end
            end
            Destination_position_second = Position_sort(2, :);
        end

        t = t + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = Destination_fitness;
    best_solution = Destination_position;
end

% Initialization (scalar or per-dimension bounds)
function X = initialization(N, Dim, ub, lb)
    X = zeros(N, Dim);
    if isscalar(ub)
        X = rand(N, Dim) .* (ub - lb) + lb;
    else
        for i = 1:Dim
            X(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

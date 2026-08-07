% ----------------------------------------------------------------------- %
% African Vultures Optimization Algorithm (AVOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 30         % Population size (number of vultures)
%   p1 = 0.6, p3 = 0.6    % Behaviour-selection probabilities
%   alpha = 0.8, betha = 0.2   % Roulette weights of the two best vultures
%   gamma = 2.5           % Exploration/exploitation transition exponent
%
% Algorithm Concept:
%   - Inspired by the foraging and navigation behaviour of African vultures
%   - The two best vultures guide the flock; the satiation rate F controls
%     the switch between exploration (|F|>=1) and exploitation (|F|<1)
%
% Reference:
% Benyamin Abdollahzadeh, Farhad Soleimanian Gharehchopogh, Seyedali Mirjalili,
% African vultures optimization algorithm: A new nature-inspired metaheuristic
% algorithm for global optimization problems,
% Computers & Industrial Engineering 158 (2021) 107408
% https://doi.org/10.1016/j.cie.2021.107408
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = avoa(problem)

    % Extract problem parameters
    variables_no = problem.dimension;
    lower_bound = problem.lb;
    upper_bound = problem.ub;
    maxFE = problem.maxFe;

    pop_size = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    max_iter = (maxFE / pop_size) + 1;

    % initialize Best_vulture1, Best_vulture2
    Best_vulture1_X = zeros(1, variables_no);
    Best_vulture1_F = inf;
    Best_vulture2_X = zeros(1, variables_no);
    Best_vulture2_F = inf;

    % Initialize the first random population of vultures
    X = initialization(pop_size, variables_no, upper_bound, lower_bound);

    % Controlling parameters
    p1 = 0.6;
    p3 = 0.6;
    alpha = 0.8;
    betha = 0.2;
    gamma = 2.5;

    current_iter = 0;
    while FE < maxFE
        FE_before = FE;

        % Calculate the fitness of the population
        [fitness, FE] = calculate_fitness(X', problem, FE);
        fitness = fitness(:);

        for i = 1:size(X, 1)
            current_vulture_F = fitness(i);
            if current_vulture_F < Best_vulture1_F
                Best_vulture1_F = current_vulture_F;   % Update the first best vulture
                Best_vulture1_X = X(i, :);
            end
            if current_vulture_F > Best_vulture1_F && current_vulture_F < Best_vulture2_F
                Best_vulture2_F = current_vulture_F;   % Update the second best vulture
                Best_vulture2_X = X(i, :);
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Best_vulture1_F;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE
            break;
        end

        a = unifrnd(-2, 2, 1, 1) * ((sin((pi / 2) * (current_iter / max_iter))^gamma) + cos((pi / 2) * (current_iter / max_iter)) - 1);
        P1 = (2 * rand + 1) * (1 - (current_iter / max_iter)) + a;

        % Update the location
        for i = 1:size(X, 1)
            current_vulture_X = X(i, :);
            F = P1 * (2 * rand() - 1);

            random_vulture_X = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha);

            if abs(F) >= 1   % Exploration
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound);
            elseif abs(F) < 1   % Exploitation
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p1, p3, variables_no, upper_bound, lower_bound);
            end

            X(i, :) = current_vulture_X;
        end

        current_iter = current_iter + 1;
        X = boundaryCheck(X, lower_bound, upper_bound);
    end

    best_solution = Best_vulture1_X;
    best_fitness = Best_vulture1_F;

end

% Exploitation phase
function [current_vulture_X] = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, ...
        random_vulture_X, F, p2, p3, variables_no, ~, ~)
    % phase 1
    if abs(F) < 0.5
        if rand < p2
            A = Best_vulture1_X - ((Best_vulture1_X .* current_vulture_X) ./ (Best_vulture1_X - current_vulture_X.^2)) * F;
            B = Best_vulture2_X - ((Best_vulture2_X .* current_vulture_X) ./ (Best_vulture2_X - current_vulture_X.^2)) * F;
            current_vulture_X = (A + B) / 2;
        else
            current_vulture_X = random_vulture_X - abs(random_vulture_X - current_vulture_X) * F .* levyFlight(variables_no);
        end
    end
    % phase 2
    if abs(F) >= 0.5
        if rand < p3
            current_vulture_X = (abs(2 * rand) * random_vulture_X - current_vulture_X) * (F + rand) - (random_vulture_X - current_vulture_X);
        else
            s1 = random_vulture_X .* (rand() * current_vulture_X / (2 * pi)) .* cos(current_vulture_X);
            s2 = random_vulture_X .* (rand() * current_vulture_X / (2 * pi)) .* sin(current_vulture_X);
            current_vulture_X = random_vulture_X - (s1 + s2);
        end
    end
end

% Exploration phase
function [current_vulture_X] = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
    if rand < p1
        current_vulture_X = random_vulture_X - (abs(2 * rand) * random_vulture_X - current_vulture_X) * F;
    else
        current_vulture_X = (random_vulture_X - (F) + rand() * ((upper_bound - lower_bound) * rand + lower_bound));
    end
end

% Boundary Handling
function [X] = boundaryCheck(X, lb, ub)
    for i = 1:size(X, 1)
        FU = X(i, :) > ub;
        FL = X(i, :) < lb;
        X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
    end
end

% Initialization Function
function [X] = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(N, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(N, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

% Levy Flight
function [o] = levyFlight(d)
    beta = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

% Random selection of one of the two best vultures
function [random_vulture_X] = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)
    probabilities = [alpha, betha];
    if (rouletteWheelSelection(probabilities) == 1)
        random_vulture_X = Best_vulture1_X;
    else
        random_vulture_X = Best_vulture2_X;
    end
end

% Roulette Wheel Selection (standard AVOA helper)
function [index] = rouletteWheelSelection(x)
    index = find(rand() <= cumsum(x), 1, 'first');
    if isempty(index)   % guard against floating-point rounding leaving no hit
        index = numel(x);
    end
end

% ----------------------------------------------------------------------- %
% Artificial Gorilla Troops Optimizer (GTO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 30   % Population size (gorillas)
%   p    = 0.03     % Probability of migration to an unknown location
%   Beta = 3        % Coefficient in the "follow the Silverback" phase
%   w    = 0.8      % Exploration/exploitation switch threshold
%
% Algorithm Concept:
%   - Silverback = best-so-far leader of the troop
%   - Exploration: migration to known/unknown locations and towards other
%     gorillas; Exploitation: follow the Silverback and compete for females
%   - Two group-formation (greedy selection) sweeps per iteration
%
% Reference:
% Benyamin Abdollahzadeh, Farhad Soleimanian Gharehchopogh, Seyedali Mirjalili,
% Artificial gorilla troops optimizer: A new nature-inspired metaheuristic
% algorithm for global optimization problems,
% International Journal of Intelligent Systems 36 (10) (2021) 5887-5958.
% https://doi.org/10.1002/int.22535
% ----------------------------------------------------------------------- %
% Implementation Note:
% The boundary repair also fires on a non-finite coordinate, redrawing it in its
% own box. Exploitation phase 1 builds delta as (abs(mean(GX)).^g).^(1/g), an
% identity whose intermediate power overflows: GX is written in place, so the
% mean is taken over rows already amplified in this sweep, and on CEC2020RW F1
% the dimension-5 column (ub = 2e6) reaches 2.3e109 by It = 1, where .^2.857 is
% Inf. Inf is clamped, but Inf*0 -- a gorilla sitting on the Silverback in that
% dimension -- gives NaN, which passes both x > ub and x < lb and so reached
% best_solution in 9 campaign runs.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gto(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lower_bound = problem.lb;
    upper_bound = problem.ub;
    maxFE = problem.maxFe;

    pop_size = 30;
    variables_no = dim;
    Max_iter = ceil(maxFE / (pop_size * 2));  % two eval sweeps per iteration

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize Silverback
    Silverback = [];
    Silverback_Score = inf;

    % Initialize the first random population of Gorilla
    X = initialization(pop_size, variables_no, upper_bound, lower_bound);
    [Pop_Fit, FE] = calculate_fitness(X', problem, FE);
    Pop_Fit = Pop_Fit(:)';
    for i = 1:pop_size
        if Pop_Fit(i) < Silverback_Score
            Silverback_Score = Pop_Fit(i);
            Silverback = X(i, :);
        end
    end

    for eval_count = 1:pop_size
        if eval_count <= maxFE
            curve(eval_count) = Silverback_Score;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Pop_Fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    GX = X(:, :);
    lb = ones(1, variables_no) .* lower_bound;
    ub = ones(1, variables_no) .* upper_bound;

    % Controlling parameters
    p = 0.03;
    Beta = 3;
    w = 0.8;

    % Main loop
    for It = 1:Max_iter
        if FE >= maxFE, break; end

        a = (cos(2 * rand) + 1) * (1 - It / Max_iter);
        C = a * (2 * rand - 1);

        % Exploration
        for i = 1:pop_size
            if rand < p
                GX(i, :) = (ub - lb) * rand + lb;
            else
                if rand >= 0.5
                    Z = unifrnd(-a, a, 1, variables_no);
                    H = Z .* X(i, :);
                    GX(i, :) = (rand - a) * X(randi([1, pop_size]), :) + C .* H;
                else
                    GX(i, :) = X(i, :) - C .* (C * (X(i, :) - GX(randi([1, pop_size]), :)) + rand * (X(i, :) - GX(randi([1, pop_size]), :)));
                end
            end
        end
        GX = boundaryCheck(GX, lower_bound, upper_bound);

        % Group formation operation
        for i = 1:pop_size
            [New_Fit, FE] = calculate_fitness(GX(i, :)', problem, FE);
            if New_Fit < Pop_Fit(i)
                Pop_Fit(i) = New_Fit;
                X(i, :) = GX(i, :);
            end
            if New_Fit < Silverback_Score
                Silverback_Score = New_Fit;
                Silverback = GX(i, :);
            end
            if FE <= maxFE
                curve(FE) = Silverback_Score;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Pop_Fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Exploitation
        for i = 1:pop_size
            if a >= w
                g = 2^C;
                delta = (abs(mean(GX)).^g).^(1 / g);
                GX(i, :) = C * delta .* (X(i, :) - Silverback) + X(i, :);
            else
                if rand >= 0.5
                    h = randn(1, variables_no);
                else
                    h = randn(1, 1);
                end
                r1 = rand;
                GX(i, :) = Silverback - (Silverback * (2 * r1 - 1) - X(i, :) * (2 * r1 - 1)) .* (Beta * h);
            end
        end
        GX = boundaryCheck(GX, lower_bound, upper_bound);

        % Group formation operation
        for i = 1:pop_size
            [New_Fit, FE] = calculate_fitness(GX(i, :)', problem, FE);
            if New_Fit < Pop_Fit(i)
                Pop_Fit(i) = New_Fit;
                X(i, :) = GX(i, :);
            end
            if New_Fit < Silverback_Score
                Silverback_Score = New_Fit;
                Silverback = GX(i, :);
            end
            if FE <= maxFE
                curve(FE) = Silverback_Score;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Pop_Fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = Silverback_Score;

    best_fitness = Silverback_Score;
    best_solution = Silverback;
end

% Initialization
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

% Boundary handling
function [X] = boundaryCheck(X, lb, ub)
    lb = lb .* ones(1, size(X, 2));
    ub = ub .* ones(1, size(X, 2));
    for i = 1:size(X, 1)
        NF = ~isfinite(X(i, :));
        X(i, NF) = rand(1, sum(NF)) .* (ub(NF) - lb(NF)) + lb(NF);
        FU = X(i, :) > ub;
        FL = X(i, :) < lb;
        X(i, :) = (X(i, :) .* (~(FU + FL))) + ub .* FU + lb .* FL;
    end
end

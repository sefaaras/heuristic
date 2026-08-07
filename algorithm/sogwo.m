% ----------------------------------------------------------------------- %
% Selective Opposition based Grey Wolf Optimization (SOGWO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 50  % Population size (number of wolves)
%
% Algorithm Concept:
%   - Standard GWO social hierarchy (alpha/beta/delta) with a selective
%     opposition operator applied to the weakest wolves
%   - Spearman's correlation selects the dimensions on which opposition
%     based learning is performed, boosting exploration
%
% Reference:
% Souvik Dhargupta, Manosij Ghosh, Seyedali Mirjalili, Ram Sarkar,
% Selective Opposition based Grey Wolf Optimization,
% Expert Systems with Applications 151 (2020) 113389
% https://doi.org/10.1016/j.eswa.2020.113389
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sogwo(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Max_iter = ceil(maxFE / SearchAgents_no);

    lower = lb;
    upper = ub;

    % initialize alpha, beta, and delta_pos
    Alpha_pos = zeros(1, dim);
    Alpha_score = inf;
    Beta_pos = zeros(1, dim);
    Beta_score = inf;
    Delta_pos = zeros(1, dim);
    Delta_score = inf;

    Positions = initialization(SearchAgents_no, dim, ub, lb);

    l = 0;
    fitness = zeros(size(Positions(:, 1)));

    while FE < maxFE
        FE_before = FE;

        % Boundary control
        for i = 1:size(Positions, 1)
            Flag4ub = Positions(i, :) > ub;
            Flag4lb = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end

        % Evaluate the whole population
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        fitness = fitness(:);

        for i = 1:size(Positions, 1)
            % Update Alpha, Beta, and Delta
            if fitness(i) < Alpha_score
                Alpha_score = fitness(i);
                Alpha_pos = Positions(i, :);
            end
            if fitness(i) > Alpha_score && fitness(i) < Beta_score
                Beta_score = fitness(i);
                Beta_pos = Positions(i, :);
            end
            if fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score
                Delta_score = fitness(i);
                Delta_pos = Positions(i, :);
            end
        end

        % updating boundary for opposition (positions are static here)
        for x = 1:size(Positions, 1)
            for y = 1:size(Positions, 2)
                if upper(1, y) < Positions(x, y)
                    upper(1, y) = Positions(x, y);
                end
                if lower(1, y) > Positions(x, y)
                    lower(1, y) = Positions(x, y);
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Alpha_score;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE
            break;
        end

        a = 2 - l * ((2) / Max_iter);   % a decreases linearly from 2 to 0

        % Oppose the least fitness elements
        threshold = a;
        Positions = corOppose(Positions, fitness, ub, lb, upper, lower, dim, threshold);

        % Update the Position of search agents including omegas
        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)
                r1 = rand();
                r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;

                r1 = rand();
                r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;

                r1 = rand();
                r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;

                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
        end

        l = l + 1;
    end

    best_solution = Alpha_pos;
    best_fitness = Alpha_score;

end

% Initialization Function
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

% Selective Opposition Operator
function [Positions] = corOppose(Positions, fitness, ~, ~, upper, lower, ~, threshold)
    n = size(fitness);
    for i = 4:n(1)
        sum = 0;
        greater = [];
        less = [];
        x = 1; z = 1; y = 1;

        for j = 1:size(Positions(1, :), 2)
            d(x) = abs(Positions(1, j) - Positions(i, j));
            if d(x) < threshold
                greater(y) = j;
                y = y + 1;
            else
                less(z) = j;
                z = z + 1;
            end
            sum = sum + d(x) * d(x);
            x = x + 1;
        end
        src = 1 - (double(6 * sum)) / (double(n(1) * (n(1) * n(1) - 1)));
        if src <= 0
            if size(greater) < size(less)
                % opposition on 'less' dimensions is disabled in the reference
            else
                for j = 1:size(greater, 1)   % faithful to reference (resolves to 1:1); size(...,1) silences colon deprecation
                    dim = greater(j);
                    Positions(i, dim) = upper(1, dim) + lower(1, dim) - Positions(i, dim);
                end
            end
        end
    end
end

% ----------------------------------------------------------------------- %
% Dandelion Optimizer (DO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Popsize = 30   % Population size (dandelion seeds)
%
% Algorithm Concept:
%   - Rising stage: seeds fly up (weather-dependent lognormal / decline factor)
%   - Decline stage: seeds descend guided by the population mean
%   - Landing stage: Levy-flight landing around the elite (best-so-far)
%
% Reference:
% Shijie Zhao, Tianran Zhang, Shilin Ma, Miao Chen,
% Dandelion Optimizer: A nature-inspired metaheuristic algorithm for
% engineering applications,
% Engineering Applications of Artificial Intelligence 114 (2022) 105075.
% https://doi.org/10.1016/j.engappai.2022.105075
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = do(problem)

    dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    Popsize = 30;
    Maxiteration = ceil(maxFE / Popsize);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    dandelions = initialization(Popsize, dim, UB, LB);
    [dandelionsFitness, FE] = calculate_fitness(dandelions', problem, FE);
    dandelionsFitness = dandelionsFitness(:)';

    [~, sorted_indexes] = sort(dandelionsFitness);
    Best_position = dandelions(sorted_indexes(1), :);
    Best_fitness = dandelionsFitness(sorted_indexes(1));

    for e = 1:Popsize
        if e <= maxFE
            curve(e) = Best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                e, dandelions, dandelionsFitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    t = 0;
    while FE < maxFE
        t = t + 1;

        % Rising stage
        beta = randn(Popsize, dim);
        alpha = rand() * ((1 / Maxiteration^2) * t^2 - 2 / Maxiteration * t + 1); % Eq.(8)
        a = -1 / (Maxiteration^2 - 2 * Maxiteration + 1);
        b = -2 * a;
        c = 1 - a - b;
        k = 1 - rand() * (c + a * t^2 + b * t); % Eq.(11)
        dandelions_1 = zeros(Popsize, dim);
        if randn() < 1.5
            for i = 1:Popsize
                lamb = abs(randn(1, dim));
                theta = (2 * rand() - 1) * pi;
                row = 1 / exp(theta);
                vx = row * cos(theta);
                vy = row * sin(theta);
                NEW = rand(1, dim) .* (UB - LB) + LB;
                dandelions_1(i, :) = dandelions(i, :) + alpha .* vx .* vy .* lognpdf(lamb, 0, 1) .* (NEW - dandelions(i, :)); % Eq.(5)
            end
        else
            for i = 1:Popsize
                dandelions_1(i, :) = dandelions(i, :) .* k; % Eq.(10)
            end
        end
        dandelions = dandelions_1;
        dandelions = max(dandelions, LB);
        dandelions = min(dandelions, UB);

        % Decline stage
        dandelions_mean = sum(dandelions, 1) / Popsize; % Eq.(14)
        dandelions_2 = zeros(Popsize, dim);
        for i = 1:Popsize
            for j = 1:dim
                dandelions_2(i, j) = dandelions(i, j) - beta(i, j) * alpha * (dandelions_mean(1, j) - beta(i, j) * alpha * dandelions(i, j)); % Eq.(13)
            end
        end
        dandelions = dandelions_2;
        dandelions = max(dandelions, LB);
        dandelions = min(dandelions, UB);

        % Landing stage
        Step_length = levy(Popsize, dim, 1.5);
        Elite = repmat(Best_position, Popsize, 1);
        dandelions_3 = zeros(Popsize, dim);
        for i = 1:Popsize
            for j = 1:dim
                dandelions_3(i, j) = Elite(i, j) + Step_length(i, j) * alpha * (Elite(i, j) - dandelions(i, j) * (2 * t / Maxiteration)); % Eq.(15)
            end
        end
        dandelions = dandelions_3;
        dandelions = max(dandelions, LB);
        dandelions = min(dandelions, UB);

        % Evaluate
        [dandelionsFitness, FE] = calculate_fitness(dandelions', problem, FE);
        dandelionsFitness = dandelionsFitness(:)';

        [~, sorted_indexes] = sort(dandelionsFitness);
        dandelions = dandelions(sorted_indexes, :);
        dandelionsFitness = dandelionsFitness(sorted_indexes);

        if dandelionsFitness(1) < Best_fitness
            Best_position = dandelions(1, :);
            Best_fitness = dandelionsFitness(1);
        end

        for i = 1:Popsize
            ec = FE - Popsize + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, dandelions, dandelionsFitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = Best_fitness;
    best_fitness = Best_fitness;
    best_solution = Best_position;
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

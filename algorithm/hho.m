% ----------------------------------------------------------------------- %
% Harris Hawks Optimization (HHO) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                    % Population size (number of hawks)
%
% Algorithm Concept:
%   - Inspired by cooperative hunting behavior of Harris' hawks
%   - Exploration: Hawks perch randomly based on random tall trees or family members
%   - Exploitation: Surprise pounce with soft/hard besiege and rapid dives
%   - Escaping energy models the prey's energy decrease over iterations
%
% Reference:
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah,
% Majdi Mafarja, Huiling Chen,
% Harris hawks optimization: Algorithm and applications,
% Future Generation Computer Systems 97 (2019) 849-872
% https://doi.org/10.1016/j.future.2019.02.028
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = hho(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    X = initialization(N, dim, ub, lb);
    [Fitness, FE] = calculate_fitness(X', problem, FE);

    [Rabbit_Energy, minIndex] = min(Fitness);
    Rabbit_Location = X(minIndex, :);

    for eval_count = 1:min(N, maxFE)
        curve(eval_count) = Rabbit_Energy;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, Fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    while FE < maxFE

        E1 = 2 * (1 - (FE / maxFE));

        for i = 1:N
            if FE >= maxFE, break; end

            E0 = 2 * rand() - 1;
            Escaping_Energy = E1 * E0;

            if abs(Escaping_Energy) >= 1
                q = rand();
                rand_Hawk_index = floor(N * rand() + 1);
                X_rand = X(rand_Hawk_index, :);
                if q < 0.5
                    X(i,:) = X_rand - rand() * abs(X_rand - 2 * rand() * X(i,:));
                else
                    X(i,:) = (Rabbit_Location - mean(X)) - rand() * ((ub - lb) * rand + lb);
                end
                X(i,:) = bound(X(i,:), ub, lb);
                [Fitness(i), FE] = calculate_fitness(X(i,:)', problem, FE);

                if Fitness(i) < Rabbit_Energy
                    Rabbit_Energy = Fitness(i);
                    Rabbit_Location = X(i,:);
                end
                if FE <= maxFE
                    curve(FE) = Rabbit_Energy;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, Fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end

            else
                r = rand();

                if r >= 0.5
                    if abs(Escaping_Energy) < 0.5
                        X(i,:) = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X(i,:));
                    else
                        Jump_strength = 2 * (1 - rand());
                        X(i,:) = (Rabbit_Location - X(i,:)) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i,:));
                    end
                    X(i,:) = bound(X(i,:), ub, lb);
                    [Fitness(i), FE] = calculate_fitness(X(i,:)', problem, FE);

                    if Fitness(i) < Rabbit_Energy
                        Rabbit_Energy = Fitness(i);
                        Rabbit_Location = X(i,:);
                    end
                    if FE <= maxFE
                        curve(FE) = Rabbit_Energy;
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, X, Fitness, population_history, fitness_history, ...
                            history_index, sampling_interval, history_size);
                    end

                else
                    if abs(Escaping_Energy) >= 0.5
                        Jump_strength = 2 * (1 - rand());
                        X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i,:));
                        X1 = bound(X1, ub, lb);
                        [FitX1, FE] = calculate_fitness(X1', problem, FE);

                        if FitX1 < Rabbit_Energy
                            Rabbit_Energy = FitX1;
                            Rabbit_Location = X1;
                        end
                        if FE <= maxFE
                            curve(FE) = Rabbit_Energy;
                            [population_history, fitness_history, history_index] = record_history(...
                                FE, X, Fitness, population_history, fitness_history, ...
                                history_index, sampling_interval, history_size);
                        end

                        if FitX1 < Fitness(i)
                            X(i,:) = X1;
                            Fitness(i) = FitX1;
                        else
                            if FE >= maxFE, break; end
                            X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X(i,:)) + rand(1, dim) .* Levy(dim);
                            X2 = bound(X2, ub, lb);
                            [FitX2, FE] = calculate_fitness(X2', problem, FE);

                            if FitX2 < Rabbit_Energy
                                Rabbit_Energy = FitX2;
                                Rabbit_Location = X2;
                            end
                            if FE <= maxFE
                                curve(FE) = Rabbit_Energy;
                                [population_history, fitness_history, history_index] = record_history(...
                                    FE, X, Fitness, population_history, fitness_history, ...
                                    history_index, sampling_interval, history_size);
                            end

                            if FitX2 < Fitness(i)
                                X(i,:) = X2;
                                Fitness(i) = FitX2;
                            end
                        end

                    else
                        Jump_strength = 2 * (1 - rand());
                        X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X));
                        X1 = bound(X1, ub, lb);
                        [FitX1, FE] = calculate_fitness(X1', problem, FE);

                        if FitX1 < Rabbit_Energy
                            Rabbit_Energy = FitX1;
                            Rabbit_Location = X1;
                        end
                        if FE <= maxFE
                            curve(FE) = Rabbit_Energy;
                            [population_history, fitness_history, history_index] = record_history(...
                                FE, X, Fitness, population_history, fitness_history, ...
                                history_index, sampling_interval, history_size);
                        end

                        if FitX1 < Fitness(i)
                            X(i,:) = X1;
                            Fitness(i) = FitX1;
                        else
                            if FE >= maxFE, break; end
                            X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - mean(X)) + rand(1, dim) .* Levy(dim);
                            X2 = bound(X2, ub, lb);
                            [FitX2, FE] = calculate_fitness(X2', problem, FE);

                            if FitX2 < Rabbit_Energy
                                Rabbit_Energy = FitX2;
                                Rabbit_Location = X2;
                            end
                            if FE <= maxFE
                                curve(FE) = Rabbit_Energy;
                                [population_history, fitness_history, history_index] = record_history(...
                                    FE, X, Fitness, population_history, fitness_history, ...
                                    history_index, sampling_interval, history_size);
                            end

                            if FitX2 < Fitness(i)
                                X(i,:) = X2;
                                Fitness(i) = FitX2;
                            end
                        end
                    end
                end
            end
        end
    end

    for idx = 2:maxFE
        if curve(idx) == 0
            curve(idx) = curve(idx - 1);
        end
    end

    best_fitness = Rabbit_Energy;
    best_solution = Rabbit_Location;

end

%% --- Levy Flight ---
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

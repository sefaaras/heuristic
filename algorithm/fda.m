% ----------------------------------------------------------------------- %
% Flow Direction Algorithm (FDA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   alpha = 50   % Number of flows (population size)
%   beta  = 1    % Number of neighborhoods per flow
%
% Algorithm Concept:
%   - Flows move toward the lowest-height neighbor (steepest slope), or
%   - toward a fitter flow / the outlet (best-so-far) when no better
%     neighbor exists, mimicking water flowing to a basin outlet.
%
% Reference:
% Hojat Karami, Mahdi Valikhan Anaraki, Saeed Farzin, Seyedali Mirjalili,
% Flow Direction Algorithm (FDA): A novel optimization approach for
% solving optimization problems,
% Computers & Industrial Engineering 156 (2021) 107224.
% https://doi.org/10.1016/j.cie.2021.107224
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fda(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    alpha = 50;   % number of flows
    beta = 1;     % number of neighborhoods
    maxiter = ceil(maxFE / (alpha * (1 + beta)));   % (1 neighbor + 1 flow) evals per flow

    Vmax = 0.1 * (ub - lb);
    Vmin = -0.1 * (ub - lb);

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, alpha, dim);
    fitness_history = zeros(history_size, alpha);
    history_index = 1;

    flow_x = initialization(alpha, dim, ub, lb);
    [fitness_flow, FE] = calculate_fitness(flow_x', problem, FE);
    fitness_flow = fitness_flow(:);

    [~, indx] = sort(fitness_flow);
    flow_x = flow_x(indx, :);
    fitness_flow = fitness_flow(indx);
    Best_fitness = fitness_flow(1);
    BestX = flow_x(1, :);

    for e = 1:alpha
        if e <= maxFE
            curve(e) = Best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                e, flow_x, fitness_flow, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    iter = 0;
    while FE < maxFE
        iter = iter + 1;
        W = (((1 - 1 * iter / maxiter + eps)^(2 * randn)) .* (rand(1, dim) .* iter / maxiter) .* rand(1, dim));

        % --- Stage 1: generate one neighbor per flow, evaluate all ---
        neighbor_x = zeros(alpha, dim);
        for i = 1:alpha
            Xrand = lb + rand(1, dim) .* (ub - lb);
            delta = W .* (rand * Xrand - rand * flow_x(i, :)) .* norm(BestX - flow_x(i, :));
            neighbor_x(i, :) = flow_x(i, :) + randn(1, dim) .* delta;
            neighbor_x(i, :) = max(neighbor_x(i, :), lb);
            neighbor_x(i, :) = min(neighbor_x(i, :), ub);
        end
        [fitness_neighbor, FE] = calculate_fitness(neighbor_x', problem, FE);
        fitness_neighbor = fitness_neighbor(:);
        for i = 1:alpha
            ec = FE - alpha + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, flow_x, fitness_flow, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE, break; end

        % --- Stage 2: move each flow, evaluate all ---
        newflow_x = flow_x;
        for i = 1:alpha
            if fitness_neighbor(i) < fitness_flow(i)
                Sf = (fitness_neighbor(i) - fitness_flow(i)) ./ sqrt(norm(neighbor_x(i, :) - flow_x(i, :)));
                V = randn .* Sf;
                if V < Vmin
                    V = -Vmin;
                elseif V > Vmax
                    V = -Vmax;
                end
                newflow_x(i, :) = flow_x(i, :) + V .* (neighbor_x(i, :) - flow_x(i, :)) ./ sqrt(norm(neighbor_x(i, :) - flow_x(i, :)));
            else
                r = randi([1 alpha]);
                if fitness_flow(r) <= fitness_flow(i)
                    newflow_x(i, :) = flow_x(i, :) + randn(1, dim) .* (flow_x(r, :) - flow_x(i, :));
                else
                    newflow_x(i, :) = flow_x(i, :) + randn * (BestX - flow_x(i, :));
                end
            end
            newflow_x(i, :) = max(newflow_x(i, :), lb);
            newflow_x(i, :) = min(newflow_x(i, :), ub);
        end
        [newfitness_flow, FE] = calculate_fitness(newflow_x', problem, FE);
        newfitness_flow = newfitness_flow(:);

        % --- Stage 3: greedy accept & update outlet ---
        for i = 1:alpha
            if newfitness_flow(i) < fitness_flow(i)
                flow_x(i, :) = newflow_x(i, :);
                fitness_flow(i) = newfitness_flow(i);
            end
            if fitness_flow(i) < Best_fitness
                BestX = flow_x(i, :);
                Best_fitness = fitness_flow(i);
            end
            ec = FE - alpha + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = Best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, flow_x, fitness_flow, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = Best_fitness;
    best_fitness = Best_fitness;
    best_solution = BestX;
end

%% --- Initialization ---
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

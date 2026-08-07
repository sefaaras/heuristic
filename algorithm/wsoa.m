% ----------------------------------------------------------------------- %
% War Strategy Optimization (WSO/WSOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Soldiers_no = 30   % Population size (soldiers)
%   R           = 0.1  % Attack/defense strategy threshold
%
% Algorithm Concept:
%   - King = best soldier, Commander (Co) = second-best soldier
%   - Attack strategy (RR<R) vs defense strategy for the soldiers' moves
%   - Weight W updated by each soldier's success (Wg); weak-soldier relocation
%
% Reference:
% Tummala S. L. V. Ayyarao, N. S. S. Ramakrishna, Rajvikram M. Elavarasan,
% et al., War Strategy Optimization Algorithm: A New Effective
% Metaheuristic Algorithm for Global Optimization,
% IEEE Access, vol. 10, pp. 25073-25105, 2022.
% https://doi.org/10.1109/ACCESS.2022.3153493
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wsoa(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    Soldiers_no = 30;
    Max_iter = maxFE;      % used in the weight-decay term W1*(1-Wg/Max_iter)^2
    R = 0.1;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Positions = initialization(Soldiers_no, dim, ub, lb);
    pop_size = Soldiers_no;

    [fitness_old, FE] = calculate_fitness(Positions', problem, FE);
    fitness_old = fitness_old(:)';

    [King_fit, kidx] = min(fitness_old);
    King = Positions(kidx, :);
    best_pos = King;

    for e = 1:pop_size
        if e <= maxFE
            curve(e) = King_fit;
            [population_history, fitness_history, history_index] = record_history(...
                e, Positions, fitness_old, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    W1 = 2 * ones(1, pop_size);
    Wg = zeros(1, pop_size);

    while FE < maxFE
        [~, tindex] = sort(fitness_old);
        Co = Positions(tindex(2), :);         % commander (second-best)
        com = randperm(pop_size);

        for i = 1:pop_size
            RR = rand;
            if RR < R
                D_V = 2 * RR * (King - Positions(com(i), :)) + 1 * W1(i) * rand * (Co - Positions(i, :));
            else
                D_V = 2 * RR * (Co - King) + 1 * rand * (W1(i) * King - Positions(i, :));
            end
            Positions_new = Positions(i, :) + D_V;
            Positions_new = bound(Positions_new, ub, lb);

            [fitness, FE] = calculate_fitness(Positions_new', problem, FE);

            if fitness < King_fit
                King_fit = fitness;
                King = Positions_new;
                best_pos = King;
            end

            if fitness < fitness_old(i)
                Positions(i, :) = Positions_new;
                fitness_old(i) = fitness;
                Wg(i) = Wg(i) + 1;
                W1(i) = 1 * W1(i) * (1 - Wg(i) / Max_iter)^2;
            end

            if FE <= maxFE
                curve(FE) = King_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Positions, fitness_old, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        % Weak-soldier relocation (early phase, per reference code)
        if FE < 1000
            [~, tindex1] = max(fitness_old);
            Positions(tindex1, :) = lb + rand .* (ub - lb);
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = King_fit;
    best_fitness = King_fit;
    best_solution = best_pos;
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

% Boundary Handling
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

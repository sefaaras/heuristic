% ----------------------------------------------------------------------- %
% Exponential Distribution Optimizer (EDO) for benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30   % Population size
%
% Algorithm Concept:
%   - Exploitation: exponential-distribution guided moves toward the guiding
%     solution X_guide (mean of the three best "winners")
%   - Exploration: mean-difference vectors (Z1, Z2) with a time factor c
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Osama M. Abdel-Baset,
% Karam M. Sallam, Ibrahim M. Hezam,
% Exponential distribution optimizer (EDO): a novel math-inspired
% algorithm for global optimization and engineering problems,
% Artificial Intelligence Review 56 (2023) 9329-9400.
% https://doi.org/10.1007/s10462-023-10403-9
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = edo(problem)

    Dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    Max_iter = ceil(maxFE / N);

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, Dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    Xwinners = initialization(N, Dim, UB, LB);
    [Fitness, FE] = calculate_fitness(Xwinners', problem, FE);
    Fitness = Fitness(:)';

    [BestFitness, bidx] = min(Fitness);
    BestSol = Xwinners(bidx, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = BestFitness;
            [population_history, fitness_history, history_index] = record_history(...
                e, Xwinners, Fitness, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    Memoryless = Xwinners;   % old population
    iter = 0;
    while FE < maxFE
        iter = iter + 1;
        V = zeros(N, Dim);

        % Rank the solutions according to fitness (winners only; Memoryless stays)
        [Fitness, sorted_indices] = sort(Fitness);
        Xwinners = Xwinners(sorted_indices, :);

        d = (1 - iter / Max_iter);   % Eq.(23)
        f = 2 * rand - 1;            % Eq.(17)
        a = f^10;                    % Eq.(15)
        b = f^5;                     % Eq.(16)
        c = d * f;                   % Eq.(22)
        X_guide = (Xwinners(1, :) + Xwinners(2, :) + Xwinners(3, :)) / 3; % Eq.(13)

        for i = 1:N
            alpha = rand;
            if alpha < 0.5
                % Exploitation
                Mu = (X_guide + Memoryless(i, :)) / 2.0;   % Eq.(19)
                ExP_rate = 1 ./ Mu;                        % Eq.(18)
                variance = 1 ./ ExP_rate.^2;               % Eq.(4)
                if isequal(Memoryless(i, :), Xwinners(i, :))
                    V(i, :) = a .* (Memoryless(i, :) - variance) + b .* X_guide;         % Eq.(14) first branch
                else
                    phi = rand;
                    V(i, :) = b .* (Memoryless(i, :) - variance) + log(phi) .* Xwinners(i, :); % Eq.(14) second branch
                end
            else
                % Exploration
                M = mean(Xwinners);
                s = randperm(N);
                D1 = M - Xwinners(s(1), :);   % Eq.(26)
                D2 = M - Xwinners(s(2), :);   % Eq.(27)
                Z1 = M - D1 + D2;             % Eq.(24)
                Z2 = M - D2 + D1;             % Eq.(25)
                V(i, :) = (Xwinners(i, :) + (c .* Z1 + (1 - c) .* Z2) - M); % Eq.(20)
            end
            V(i, :) = bound(V(i, :), UB, LB);
        end

        Memoryless = V;   % new population becomes the memory

        [V_Fitness, FE] = calculate_fitness(V', problem, FE);
        V_Fitness = V_Fitness(:)';

        for i = 1:N
            if V_Fitness(i) < Fitness(i)
                Xwinners(i, :) = V(i, :);
                Fitness(i) = V_Fitness(i);
                if Fitness(i) < BestFitness
                    BestFitness = Fitness(i);
                    BestSol = Xwinners(i, :);
                end
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = BestFitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Xwinners, Fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = BestFitness;
    best_fitness = BestFitness;
    best_solution = BestSol;
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

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

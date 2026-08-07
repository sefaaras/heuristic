% ----------------------------------------------------------------------- %
% Modified Particle Swarm Optimization (MPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Swarm size
%   w_max = 0.9; w_min = 0.4 % Inertia weight range (chaotic adaptive)
%   c1 = c2 = 2             % Learning factors
%
% Algorithm Concept:
%   - PSO with an adaptive, chaos-based non-linear inertia weight
%   - Stochastic and mainstream learning avoid premature convergence
%   - Velocity uses the personal-best mean and a chosen exemplar
%   - A guided trial solution replaces the worst particle each iteration
%
% Reference:
% Hao Liu, XueWei Zhang, LiangPing Tu,
% A modified particle swarm optimization using adaptive strategy,
% Expert Systems with Applications 152 (2020) 113353
% https://doi.org/10.1016/j.eswa.2020.113353
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mpso(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    N = 50;
    Vmax0 = 0.5 * (UB - LB);
    w_max = 0.9; w_min = 0.4;
    c1 = 2; c2 = c1;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize population
    Xmax = repmat(UB, N, 1);
    Xmin = repmat(LB, N, 1);
    X = Xmin + (Xmax - Xmin) .* rand(N, dim);
    Vmax = repmat(Vmax0, N, 1);
    V = -Vmax + 2 * Vmax .* rand(N, dim);

    x = rand;   % chaotic variable

    % Evaluate the swarm
    [fX, FE] = calculate_fitness(X', problem, FE);
    fX = fX(:)';

    % Personal / global bests
    Pbest = X;  fPbest = fX;
    [gbestValue, gbestIndex] = min(fPbest);
    Gbest = Pbest(gbestIndex, :);

    for eval_count = 1:N
        curve(eval_count) = gbestValue;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, fX', population_history, fitness_history, ...
            history_index, maxFE);
    end

    Ubest = zeros(N, dim);

    while FE < maxFE
        FE_before = FE;

        % Chaotic adaptive inertia weight
        x = 4 * x * (1 - x);
        w = x * w_min + (w_max - w_min) * (FE / maxFE);

        u = randperm(N, 2);
        for i = 1:N
            [~, index] = min(fPbest(u));
            if min(fPbest(u)) < fPbest(i)
                Ubest(i, :) = Pbest(index, :);
            else
                Ubest(i, :) = Pbest(i, :);
            end
        end

        V = w * V + c1 * rand(N, dim) .* (Ubest - X) + c2 * rand(N, dim) .* (repmat(mean(Pbest), N, 1) - X);
        V = max(-Vmax, min(Vmax, V));

        meanfX = mean(fX);
        for i = 1:N
            if exp(fX(i)) / exp(meanfX) > rand
                X(i, :) = w * X(i, :) + (1 - w) * V(i, :) + Gbest;
            else
                X(i, :) = X(i, :) + V(i, :);
            end
        end
        X = max(Xmin, min(Xmax, X));

        % Evaluate the swarm
        [fX, FE] = calculate_fitness(X', problem, FE);
        fX = fX(:)';

        % Replace the worst particle with a guided trial solution
        [worst, index] = max(fX);
        z = [1:index-1, index+1:N];
        d = randperm(length(z), 2);
        NewX = Gbest + rand * (Pbest(z(d(2)), :) - Pbest(z(d(1)), :));
        NewX = max(LB, min(UB, NewX));
        [fNewX, FE] = calculate_fitness(NewX', problem, FE);
        fNewX = fNewX(1);
        if fNewX < worst
            X(index, :) = NewX;
            fX(index)   = fNewX;
        end

        % Update personal and global bests
        for i = 1:N
            if fX(i) < fPbest(i)
                Pbest(i, :) = X(i, :);
                fPbest(i)   = fX(i);
            end
            if fPbest(i) < gbestValue
                Gbest = Pbest(i, :);
                gbestValue = fPbest(i);
            end
        end

        % Record convergence curve and history for this generation
        for eval_count = (FE_before + 1):FE
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = gbestValue;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fX', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    best_fitness  = gbestValue;
    best_solution = Gbest;
end

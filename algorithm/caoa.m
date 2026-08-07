% ----------------------------------------------------------------------- %
% Crocodile Ambush Optimization Algorithm (CAOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N              = 30     % Population size (crocodiles)
%   alpha          = 0.5    % Attraction towards the leader
%   beta           = 0.1    % Random ambush perturbation
%   gamma          = 0.1    % Energy consumed per unit distance
%   delta          = 1e-4   % Threshold for the re-initialisation trigger
%   initial_energy = 100    % Starting energy of every crocodile
%
% Algorithm Concept:
%   - Stochastic leader selection through the transformed score 1/(1+f)
%   - Ambush move: drift towards the leader plus a bounded random component
%   - Threshold-based reinitialisation when a move degrades fitness by more
%     than delta
%   - Adaptive energy decay proportional to the travelled distance; depleted
%     crocodiles are relocated at random and recharged
%
% Reference:
% Xinpeng Xu,
% Crocodile Ambush Optimization Algorithm: A new bio-inspired metaheuristic
% algorithm for solving optimization problems,
% Array 28 (2025) 100529.
% https://doi.org/10.1016/j.array.2025.100529
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = caoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N              = 30;
    alpha          = 0.5;
    beta           = 0.1;
    gamma          = 0.1;
    delta          = 1e-4;
    initial_energy = 100.0;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    pos      = repmat(lb, N, 1) + repmat(ub - lb, N, 1) .* rand(N, dim);
    energies = initial_energy * ones(N, 1);

    [fitness, FE] = calculate_fitness(pos', problem, FE);
    fitness = fitness(:);

    [gBestScore, best_idx] = min(fitness);
    gBest = pos(best_idx, :);
    % CAOA accepts every move and may relocate crocodiles, so bsf/bsx track the incumbent itself
    bsf = gBestScore;
    bsx = gBest;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, pos, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        old_positions = pos;
        old_fitness   = fitness;

        % Leader selection
        probs = 1 ./ (1 + fitness);
        [~, leader_idx] = max(probs);
        leader_position = pos(leader_idx, :);

        for i = 1:N
            if i == leader_idx, continue; end
            if FE >= maxFE, break; end

            r = rand(1, dim);
            new_pos = pos(i, :) + alpha * (leader_position - pos(i, :)) + beta * (1 - 2 * r);

            Flag4ub = new_pos > ub;
            Flag4lb = new_pos < lb;
            new_pos = (new_pos .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;

            [new_fit, FE] = calculate_fitness(new_pos', problem, FE);
            if new_fit < bsf
                bsf = new_fit;
                bsx = new_pos;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, pos, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end

            % Threshold-based reinitialisation
            if abs(new_fit - old_fitness(i)) > delta && new_fit > old_fitness(i)
                new_pos = lb + (ub - lb) .* rand(1, dim);
                if FE < maxFE
                    [new_fit, FE] = calculate_fitness(new_pos', problem, FE);
                    if new_fit < bsf
                        bsf = new_fit;
                        bsx = new_pos;
                    end
                    if FE <= maxFE
                        curve(FE) = bsf;
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, pos, fitness, population_history, fitness_history, ...
                            history_index, maxFE);
                    end
                end
            end

            pos(i, :)  = new_pos;
            fitness(i) = new_fit;
        end

        % Energy consumption proportional to the travelled distance
        distances = sqrt(sum((pos - old_positions) .^ 2, 2));
        energies  = energies - gamma * distances;

        % Relocate the depleted crocodiles
        depleted = energies <= 0;
        if any(depleted)
            n_depleted = sum(depleted);
            pos(depleted, :)  = repmat(lb, n_depleted, 1) + ...
                                repmat(ub - lb, n_depleted, 1) .* rand(n_depleted, dim);
            energies(depleted) = initial_energy;
            for i = find(depleted)'
                if FE >= maxFE, break; end
                [fi, FE] = calculate_fitness(pos(i, :)', problem, FE);
                fitness(i) = fi;
                if fi < bsf
                    bsf = fi;
                    bsx = pos(i, :);
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, pos, fitness, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end

        [min_fit, min_idx] = min(fitness);
        if min_fit < gBestScore
            gBestScore = min_fit;
            gBest      = pos(min_idx, :);
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

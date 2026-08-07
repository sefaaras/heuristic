% ----------------------------------------------------------------------- %
% Enhanced Frilled Lizard Optimizer (EFLO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents   = 50      % Population size (frilled lizards)
%   frillCrest     = 5       % Top-K prey pool size
%   frillThreshold = 1e-4    % Diversity threshold for the frill factor
%
% Algorithm Concept:
%   - Phase 1 (hunting / exploration): move towards a prey chosen from the
%     top-K better lizards, scaled by the adaptive "frill factor"
%   - Phase 2 (moving up the tree / exploitation): shrinking climb impulse
%     plus a differential term built from two random lizards
%   - Enhancement: Gaussian local search for the better half of the swarm
%   - Frill factor combines an iteration decay, a sinusoidal modulation and a
%     population-diversity switch
%
% Reference:
% Ali Rodan,
% Enhanced Frilled Lizard Optimizer for Global Optimization and Engineering
% Design Problems,
% International Journal of Computational Intelligence Systems (2025).
% https://doi.org/10.1007/s44196-025-00969-3
% Base algorithm: M. Dehghani et al., Frilled Lizard Optimization (FLO).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = eflo(problem)

    dimension  = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE      = problem.maxFe;

    SearchAgents   = 50;
    Max_iterations = max(1, ceil(maxFE / (2 * SearchAgents)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % (1) Initialisation
    X = zeros(SearchAgents, dimension);
    for d = 1:dimension
        X(:, d) = lowerbound(d) + rand(SearchAgents, 1) .* (upperbound(d) - lowerbound(d));
    end

    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [fbest, bidx] = min(fit);
    xbest = X(bidx, :);
    bsf   = fbest;
    bsx   = xbest;

    for eval_count = 1:SearchAgents
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % (2) Main loop
    for t = 1:Max_iterations
        if FE >= maxFE, break; end

        [current_best, current_idx] = min(fit);
        if current_best < fbest
            fbest = current_best;
            xbest = X(current_idx, :);
        end

        % Adaptive step size ("frill factor")
        frillStd       = std(fit);
        frillThreshold = 1e-4;
        frillBase = 0.8 * sqrt(1 - t / Max_iterations) + ...
                    0.1 * sin(t / Max_iterations) - 0.1;
        if frillStd < frillThreshold
            frillFactor = 0.5 * frillBase;
        else
            frillFactor = frillBase;
        end
        frillFactor = max(0, frillFactor);

        for i = 1:SearchAgents
            if FE >= maxFE, break; end

            % PHASE 1: hunting with top-K prey selection
            prey_idx = find(fit < fit(i));
            if isempty(prey_idx)
                selected_prey = xbest;
            else
                [~, sorted_p] = sort(fit(prey_idx), 'ascend');
                sorted_better = prey_idx(sorted_p);
                frillCrest = min(5, numel(sorted_better));
                frillPreyGroup = sorted_better(1:frillCrest);
                frillPick = frillPreyGroup(randi(frillCrest));
                selected_prey = X(frillPick, :);
            end

            I = round(1 + rand);
            X_new_P1 = X(i, :) + frillFactor .* rand(1, dimension) .* (selected_prey - I .* X(i, :));

            X_new_P1 = max(X_new_P1, lowerbound);
            X_new_P1 = min(X_new_P1, upperbound);

            [f_new_P1, FE] = calculate_fitness(X_new_P1', problem, FE);
            if f_new_P1 < fit(i)
                X(i, :) = X_new_P1;
                fit(i)  = f_new_P1;
            end
            if f_new_P1 < bsf
                bsf = f_new_P1; bsx = X_new_P1;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end

            % PHASE 2: moving up the tree
            climbImpulse = (1 - 2 * rand(1, dimension)) .* ...
                           ((upperbound - lowerbound) / t) .* frillFactor;

            indices = 1:SearchAgents;
            indices(i) = [];
            chosen = randperm(numel(indices), 2);
            climbSpot1 = X(indices(chosen(1)), :);
            climbSpot2 = X(indices(chosen(2)), :);

            X_new_Climb = X(i, :) + climbImpulse + (climbSpot1 - climbSpot2);
            X_new_Climb = max(X_new_Climb, lowerbound);
            X_new_Climb = min(X_new_Climb, upperbound);

            [f_new_TreeClimb, FE] = calculate_fitness(X_new_Climb', problem, FE);
            if f_new_TreeClimb < fit(i)
                X(i, :) = X_new_Climb;
                fit(i)  = f_new_TreeClimb;
            end
            if f_new_TreeClimb < bsf
                bsf = f_new_TreeClimb; bsx = X_new_Climb;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end

            % Local search for the better half
            if fit(i) <= median(fit)
                frillShake = frillFactor * randn(1, dimension);
                X_local = X(i, :) + frillShake;
                X_local = max(X_local, lowerbound);
                X_local = min(X_local, upperbound);

                [f_local, FE] = calculate_fitness(X_local', problem, FE);
                if f_local < fit(i)
                    X(i, :) = X_local;
                    fit(i)  = f_local;
                end
                if f_local < bsf
                    bsf = f_local; bsx = X_local;
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

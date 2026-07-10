% ----------------------------------------------------------------------- %
% Dynamic Differential Annealed Optimization (DDAO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Npop     = 3       % Population size
%   MaxSubIt = 1000    % Number of sub-iterations (annealing samples)
%   T0       = 2000    % Initial temperature
%   alpha    = 0.995   % Temperature reduction rate
%
% Algorithm Concept:
%   - Hybrid of random search and classical simulated annealing
%     (inspired by the steel annealing process)
%   - Each iteration draws many candidate samples (sub-iterations)
%   - A forging operator combines population members with the best sample
%   - Metropolis acceptance governed by a cooling temperature schedule
%
% Reference:
% Hazim Nasir Ghafil, Karoly Jarmai,
% Dynamic differential annealed optimization: New metaheuristic optimization algorithm for engineering applications,
% Applied Soft Computing 93 (2020) 106392
% https://doi.org/10.1016/j.asoc.2020.106392
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ddao(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    % DDAO parameters
    Npop     = 3;
    MaxSubIt = 1000;
    T0       = 2000;
    alpha    = 0.995;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, Npop, dim);
    fitness_history = zeros(history_size, Npop);
    history_index = 1;

    % Initialize population
    popPhase = rand(Npop, dim) .* (ub - lb) + lb;
    [popCost, FE] = calculate_fitness(popPhase', problem, FE);
    popCost = popCost(:)';

    [BestCost, bidx] = min(popCost);
    BestPhase = popPhase(bidx, :);

    % Record initial evaluations
    for eval_count = 1:Npop
        curve(eval_count) = BestCost;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, popPhase, popCost', population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    T = T0;
    t = 1;   % mirrors the reference iteration counter (drives forging parity)

    while FE < maxFE
        FE_before = FE;

        % Create and evaluate MaxSubIt new random solutions
        newPhase = rand(MaxSubIt, dim) .* (ub - lb) + lb;
        newPhase = min(max(newPhase, lb), ub);
        [newCost, FE] = calculate_fitness(newPhase', problem, FE);
        newCost = newCost(:)';
        t = t + MaxSubIt;

        % Best of the annealing samples
        [~, so] = min(newCost);
        bnewPhase = newPhase(so, :);

        % Forging step
        kk = randi(Npop);
        bb = randi(Npop);
        if rem(t, 2) == 1
            MnewPhase = (popPhase(kk, :) - popPhase(bb, :)) + bnewPhase;
        else
            MnewPhase = (popPhase(kk, :) - popPhase(bb, :)) + bnewPhase * rand;
        end
        MnewPhase = min(max(MnewPhase, lb), ub);

        [MnewCost, FE] = calculate_fitness(MnewPhase', problem, FE);
        MnewCost = MnewCost(1);
        t = t + 1;

        % Acceptance (Metropolis criterion)
        for i = 1:Npop
            if MnewCost <= popCost(i)
                popPhase(i, :) = MnewPhase;
                popCost(i)     = MnewCost;
            else
                DELTA = MnewCost - popCost(i);
                P = exp(-DELTA / T);
                if rand <= P
                    popPhase(end, :) = MnewPhase;
                    popCost(end)     = MnewCost;
                end
            end
            if popCost(i) <= BestCost
                BestCost  = popCost(i);
                BestPhase = popPhase(i, :);
            end
        end

        % Update temperature
        T = alpha * T;

        % Record convergence curve and history for this decade
        for eval_count = (FE_before + 1):FE
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = BestCost;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, popPhase, popCost', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness  = BestCost;
    best_solution = BestPhase;
end

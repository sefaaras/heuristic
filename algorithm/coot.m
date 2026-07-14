% ----------------------------------------------------------------------- %
% Coot Optimization Algorithm (COOT) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                     % Population size (coots + leaders)
%   NLeader = ceil(0.1*N)      % Number of leaders
%
% Algorithm Concept:
%   - Inspired by the collective behaviour of coots on the water surface
%   - Four movements: random movement, chain movement, position adjustment
%     towards the group leaders, and leader movement towards the global best
%
% Reference:
% Iraj Naruei, Farshid Keynia,
% A new optimization method based on COOT bird natural life model,
% Expert Systems with Applications 183 (2021) 115352
% https://doi.org/10.1016/j.eswa.2021.115352
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = coot(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    Max_iter = (maxFE / N) + 1;

    if size(ub, 2) == 1
        ub = ones(1, dim) * ub;
        lb = ones(1, dim) * lb;
    end

    NLeader = ceil(0.1 * N);
    Ncoot = N - NLeader;
    gBest = zeros(1, dim);
    gBestScore = inf;

    % Initialize the positions of Coots
    CootPos = zeros(Ncoot, dim);
    LeaderPos = zeros(NLeader, dim);
    for i = 1:NLeader
        CootPos(i, :) = rand(1, dim) .* (ub - lb) + lb;
        LeaderPos(i, :) = rand(1, dim) .* (ub - lb) + lb;
    end
    CootFitness = zeros(1, Ncoot);
    LeaderFit = zeros(1, NLeader);

    % Initial evaluation of coots
    FE_before = FE;
    [CootFitness, FE] = calculate_fitness(CootPos', problem, FE);
    CootFitness = CootFitness(:)';
    for i = 1:size(CootPos, 1)
        if (gBestScore > CootFitness(1, i))
            gBestScore = CootFitness(1, i);
            gBest = CootPos(i, :);
        end
    end
    % Initial evaluation of leaders
    [LeaderFit, FE] = calculate_fitness(LeaderPos', problem, FE);
    LeaderFit = LeaderFit(:)';
    for i = 1:size(LeaderPos, 1)
        if (gBestScore > LeaderFit(1, i))
            gBestScore = LeaderFit(1, i);
            gBest = LeaderPos(i, :);
        end
    end
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = gBestScore;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                population_history, fitness_history, history_index, sampling_interval, history_size);
        end
    end

    l = 2;
    while FE < maxFE
        B = 2 - l * (1 / Max_iter);
        A = 1 - l * (1 / Max_iter);

        % Move coots
        for i = 1:size(CootPos, 1)
            if rand < 0.5
                R = -1 + 2 * rand;
                R1 = rand();
            else
                R = -1 + 2 * rand(1, dim);
                R1 = rand(1, dim);
            end
            k = 1 + mod(i, NLeader);
            if rand < 0.5
                CootPos(i, :) = 2 * R1 .* cos(2 * pi * R) .* (LeaderPos(k, :) - CootPos(i, :)) + LeaderPos(k, :);
                Tp = CootPos(i, :) > ub; Tm = CootPos(i, :) < lb; CootPos(i, :) = (CootPos(i, :) .* (~(Tp + Tm))) + ub .* Tp + lb .* Tm;
            else
                if rand < 0.5 && i ~= 1
                    CootPos(i, :) = (CootPos(i, :) + CootPos(i - 1, :)) / 2;
                else
                    Q = rand(1, dim) .* (ub - lb) + lb;
                    CootPos(i, :) = CootPos(i, :) + A * R1 .* (Q - CootPos(i, :));
                end
                Tp = CootPos(i, :) > ub; Tm = CootPos(i, :) < lb; CootPos(i, :) = (CootPos(i, :) .* (~(Tp + Tm))) + ub .* Tp + lb .* Tm;
            end
        end

        % Fitness of coots + update towards leaders
        FE_before = FE;
        [CootFitness, FE] = calculate_fitness(CootPos', problem, FE);
        CootFitness = CootFitness(:)';
        for i = 1:size(CootPos, 1)
            k = 1 + mod(i, NLeader);
            if CootFitness(1, i) < LeaderFit(1, k)
                Temp = LeaderPos(k, :);
                TemFit = LeaderFit(1, k);
                LeaderFit(1, k) = CootFitness(1, i);
                LeaderPos(k, :) = CootPos(i, :);
                CootFitness(1, i) = TemFit;
                CootPos(i, :) = Temp;
            end
        end
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = gBestScore;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                    population_history, fitness_history, history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE
            break;
        end

        % Fitness / movement of leaders (sequential, one evaluation each)
        for i = 1:size(LeaderPos, 1)
            if rand < 0.5
                R = -1 + 2 * rand;
                R3 = rand();
            else
                R = -1 + 2 * rand(1, dim);
                R3 = rand(1, dim);
            end
            if rand < 0.5
                Temp = B * R3 .* cos(2 * pi * R) .* (gBest - LeaderPos(i, :)) + gBest;
            else
                Temp = B * R3 .* cos(2 * pi * R) .* (gBest - LeaderPos(i, :)) - gBest;
            end
            Tp = Temp > ub; Tm = Temp < lb; Temp = (Temp .* (~(Tp + Tm))) + ub .* Tp + lb .* Tm;
            FE_before = FE;
            [TempFit, FE] = calculate_fitness(Temp', problem, FE);
            TempFit = TempFit(1);
            if (gBestScore > TempFit)
                LeaderFit(1, i) = gBestScore;
                LeaderPos(i, :) = gBest;
                gBestScore = TempFit;
                gBest = Temp;
            end
            for eval_count = (FE_before + 1):FE
                if eval_count <= maxFE
                    curve(eval_count) = gBestScore;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                        population_history, fitness_history, history_index, sampling_interval, history_size);
                end
            end
            if FE >= maxFE
                break;
            end
        end

        l = l + 1;
    end

    best_solution = gBest;
    best_fitness = gBestScore;

end

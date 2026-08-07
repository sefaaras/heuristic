% ----------------------------------------------------------------------- %
% Coot Optimization Algorithm (COOT)
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
%
% Implementation Note:
%   curve and best_fitness track the best point the run has evaluated. gBest is
%   updated in the leader phase alone, as the reference has it, so it stays the
%   leaders' attractor and the search trajectory is unchanged; reporting gBest
%   directly would lag the best point found by a full generation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
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

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
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

    % Best point evaluated so far, kept apart from the attractor gBest
    bsf_fit = inf;
    bsf_x = zeros(1, dim);

    % Initialize the positions of Coots
    CootPos = rand(Ncoot, dim) .* (ub - lb) + lb;
    LeaderPos = rand(NLeader, dim) .* (ub - lb) + lb;

    % Initial evaluation of coots
    FE_before = FE;
    [CootFitness, FE] = calculate_fitness(CootPos', problem, FE);
    CootFitness = CootFitness(:)';
    [bsf_fit, bsf_x] = track_best(CootFitness, CootPos, bsf_fit, bsf_x);
    for i = 1:size(CootPos, 1)
        if (gBestScore > CootFitness(1, i))
            gBestScore = CootFitness(1, i);
            gBest = CootPos(i, :);
        end
    end
    % Initial evaluation of leaders
    [LeaderFit, FE] = calculate_fitness(LeaderPos', problem, FE);
    LeaderFit = LeaderFit(:)';
    [bsf_fit, bsf_x] = track_best(LeaderFit, LeaderPos, bsf_fit, bsf_x);
    for i = 1:size(LeaderPos, 1)
        if (gBestScore > LeaderFit(1, i))
            gBestScore = LeaderFit(1, i);
            gBest = LeaderPos(i, :);
        end
    end
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bsf_fit;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                population_history, fitness_history, history_index, maxFE);
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
        [bsf_fit, bsf_x] = track_best(CootFitness, CootPos, bsf_fit, bsf_x);
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
                curve(eval_count) = bsf_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                    population_history, fitness_history, history_index, maxFE);
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
            [bsf_fit, bsf_x] = track_best(TempFit, Temp, bsf_fit, bsf_x);
            if (gBestScore > TempFit)
                LeaderFit(1, i) = gBestScore;
                LeaderPos(i, :) = gBest;
                gBestScore = TempFit;
                gBest = Temp;
            end
            for eval_count = (FE_before + 1):FE
                if eval_count <= maxFE
                    curve(eval_count) = bsf_fit;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, [CootPos; LeaderPos], [CootFitness, LeaderFit]', ...
                        population_history, fitness_history, history_index, maxFE);
                end
            end
            if FE >= maxFE
                break;
            end
        end

        l = l + 1;
    end

    best_solution = bsf_x;
    best_fitness = bsf_fit;

end

% Running best over every evaluated point
function [bf, bx] = track_best(fit, pos, bf, bx)
    [m, j] = min(fit(:));
    if ~isempty(m) && m < bf
        bf = m;
        bx = pos(j, :);
    end
end

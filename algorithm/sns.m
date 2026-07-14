% ----------------------------------------------------------------------- %
% Social Network Search (SNS) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nUser = 50            % Population size (number of network users)
%
% Algorithm Concept:
%   - Mimics social-network users' efforts to gain popularity
%   - Four decision moods act as search operators: imitation, conversation,
%     disputation and innovation (selected at random per user)
%
% Reference:
% Hadi Bayzidi, Siamak Talatahari, Meysam Saraee, Charles-Philippe Lamarche,
% Social Network Search for Solving Engineering Optimization Problems,
% Computational Intelligence and Neuroscience 2021 (2021) 8548639
% https://doi.org/10.1155/2021/8548639
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = sns(problem)

    % Extract problem parameters
    nDim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    nUser = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nUser, nDim);
    fitness_history = zeros(history_size, nUser);
    history_index = 1;

    % Level 1: Initializing
    x = zeros(nUser, nDim);
    for i = 1:nUser
        x(i, :) = LB + rand(1, nDim) .* (UB - LB);   % Eq. (8)
    end
    [f, FE] = calculate_fitness(x', problem, FE);
    f = f(:)';

    [fBest, bIdx] = min(f);
    xBest = x(bIdx, :);
    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = fBest;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, x, f, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    % Level 2: Increasing popularity (SNS Main Loop)
    while FE < maxFE
        FE_before = FE;

        for i = 1:nUser
            if FE >= maxFE, break; end

            % Select a random mood
            Mood = randi(4);

            % Select random user
            Id = [1:i - 1 i + 1:nUser];
            j = Id(randi(nUser - 1));
            Id(Id == j) = [];

            % Follow the procedure of the selected mood
            if Mood == 1
                r = x(j, :) - x(i, :);
                R = rand(1, nDim) .* r;
                nx = x(j, :) + (1 - 2 * rand(1, nDim)) .* R;
            elseif Mood == 2
                k = Id(randi(nUser - 2));
                D = sign(f(i) - f(j)) * (x(j, :) - x(i, :));
                nx = x(k, :) + rand(1, nDim) .* D;
            elseif Mood == 4
                Group = randperm(nUser, randi(nUser));
                M = mean(x(Group, :), 1);
                nx = x(i, :) + rand(1, nDim) .* (M - randi(2) * x(i, :));
            else
                d = randi(nDim);
                n = LB(d) + rand * (UB(d) - LB(d));
                nx = x(i, :);
                t = rand;
                nx(d) = t * n + (1 - t) * x(j, d);
            end

            % Clamp the new solution Eq. (6)
            nx = min(max(nx, LB), UB);

            % Evaluation of new generated solution (nx)
            [nf, FE] = calculate_fitness(nx', problem, FE);

            % Greedy selection
            if nf < f(i)
                f(i) = nf;
                x(i, :) = nx;
                if nf < fBest
                    fBest = nf;
                    xBest = nx;
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = fBest;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, x, f, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = xBest;
    best_fitness = fBest;

end

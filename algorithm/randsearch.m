% ----------------------------------------------------------------------- %
% Pure Random Search (RS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   batch = 100                 % Points drawn per evaluator call
%
% Algorithm Concept:
%   - Draws points uniformly at random from the box and keeps the best seen
%   - No state is carried between draws, so it is the baseline every search
%     heuristic has to beat: its expected progress comes only from the budget
%
% Reference:
% Samuel H. Brooks,
% A Discussion of Random Methods for Seeking Maxima,
% Operations Research 6 (1958) 244-251.
% https://doi.org/10.1287/opre.6.2.244
% ----------------------------------------------------------------------- %
% Implementation Note:
% Points are drawn 100 at a time only because one evaluator call per point
% would dominate the run time; the draws are independent and uniform either
% way, so the batch changes nothing but the speed. The last batch is trimmed to
% the evaluations the budget still allows.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = randsearch(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    batch = 100;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    best_fitness = inf;
    best_solution = zeros(1, dim);

    while FE < maxFE
        n = min(batch, maxFE - FE);
        X = repmat(lb, n, 1) + rand(n, dim) .* repmat(ub - lb, n, 1);
        [fitness, FE] = calculate_fitness(X', problem, FE);
        fitness = fitness(:);

        for i = 1:n
            if fitness(i) < best_fitness
                best_fitness = fitness(i);
                best_solution = X(i, :);
            end
            eval_count = FE - n + i;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

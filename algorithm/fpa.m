% ----------------------------------------------------------------------- %
% Flower Pollination Algorithm (FPA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 20                      % Population size (flowers)
%   p = 0.8                     % Switch probability; a draw above it pollinates globally
%   beta = 1.5                  % Levy exponent
%   scale = 0.01                % Levy step scale
%
% Algorithm Concept:
%   - Global pollination: a flower is carried towards the best by a Levy flight,
%     x + L*(x - best), whose heavy tail makes occasional long jumps
%   - Local pollination: a flower steps along the difference of two randomly
%     picked flowers, scaled by a uniform draw, which is a DE-like local move
%   - A draw above the switch probability selects the global move, so the search
%     is mostly local with rare long-range exploration
%   - Greedy selection per flower, with the global best updated from the trial
%
% Reference:
% Xin-She Yang,
% Flower Pollination Algorithm for Global Optimization,
% Unconventional Computation and Natural Computation, Lecture Notes in Computer
% Science 7445, Springer, 2012, pp. 240-249.
% https://doi.org/10.1007/978-3-642-32894-7_27
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Yang's released fpa_demo.m, including its two quirks: the local
% move starts from the trial vector S of the previous iteration rather than from
% the current flower, and greedy selection accepts ties (Fnew <= Fitness).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fpa(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    n = 20;
    p = 0.8;
    beta = 1.5;
    scale = 0.01;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Sol = repmat(lb, n, 1) + rand(n, dim) .* repmat(ub - lb, n, 1);
    [fitness, FE] = calculate_fitness(Sol', problem, FE);
    fitness = fitness(:);

    [best_fitness, ibest] = min(fitness);
    best_solution = Sol(ibest, :);
    S = Sol;

    for i = 1:min(n, maxFE)
        curve(i) = min(fitness(1:i));
        [population_history, fitness_history, history_index] = record_history(...
            i, Sol, fitness', population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        for i = 1:n
            if FE >= maxFE
                break;
            end

            if rand > p
                u = randn(1, dim) * sigma;
                v = randn(1, dim);
                L = scale * u ./ abs(v) .^ (1 / beta);   % Levy step
                S(i, :) = Sol(i, :) + L .* (Sol(i, :) - best_solution);
            else
                JK = randperm(n);
                S(i, :) = S(i, :) + rand * (Sol(JK(1), :) - Sol(JK(2), :));
            end
            S(i, :) = min(max(S(i, :), lb), ub);

            [fnew, FE] = calculate_fitness(S(i, :)', problem, FE);
            fnew = fnew(1);

            if fnew <= fitness(i)
                Sol(i, :) = S(i, :);
                fitness(i) = fnew;
            end
            if fnew <= best_fitness
                best_fitness = fnew;
                best_solution = S(i, :);
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Sol, fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

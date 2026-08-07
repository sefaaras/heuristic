% ----------------------------------------------------------------------- %
% Bernstein-Levy Differential Evolution (BDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30    % Number of pattern vectors
%
% Algorithm Concept:
%   - Two bijective direction vectors dv1, dv2 built from displaced
%     permutations of the pattern matrix
%   - An elitist direction vector dv3 built from the global minimiser pulled
%     towards the box corners with rand^3 weights
%   - Levy-distributed scale factor 1/gamrnd(0.5, randi([1 16])/2), applied
%     either scalar-wise (c = 1) or per pattern (c = N)
%   - Two mutation forms: bijective (dv1 + Scale*(dv3 - dv2)) or elitist
%     (a Gaussian-weighted combination of dv1, dv2 and dv3)
%   - Parameter-free crossover: the number of active dimensions is drawn from
%     a Bernstein polynomial basis (bernsteinMatrix) or from rand^randi([3 5])
%
% Reference:
% Pinar Civicioglu, Erkan Besdok,
% Bernstein-Levy differential evolution algorithm for numerical function
% optimization,
% Neural Computing and Applications 35 (2023) 6603-6621.
% https://doi.org/10.1007/s00521-022-08013-7
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bde(problem)

    D     = problem.dimension;
    low   = problem.lb;
    up    = problem.ub;
    maxFE = problem.maxFe;

    N     = 30;
    epoch = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation (fig. 1, lines 2-4)
    pattern_vectors = rand(N, D) .* (up - low) + low;
    [fitness_of_pattern_vectors, FE] = calculate_fitness(pattern_vectors', problem, FE);
    fitness_of_pattern_vectors = fitness_of_pattern_vectors(:);

    [globalminimum, ind] = min(fitness_of_pattern_vectors);
    globalminimizer = pattern_vectors(ind, :);
    bsf = globalminimum;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, pattern_vectors, fitness_of_pattern_vectors, ...
                population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Iterative search phase
    for epk = 1:epoch
        if FE >= maxFE, break; end

        % Bijective direction vectors (lines 6-7)
        while 1
            j1 = randperm(N);
            j2 = randperm(N);
            if sum(j1 == 1:N) == 0 && sum(j2 == 1:N) == 0 && sum(j1 == j2) == 0
                break;
            end
        end
        dv1 = pattern_vectors(j1, :) - pattern_vectors;
        dv2 = pattern_vectors(j2, :) - pattern_vectors;

        % Elitist direction vector (lines 8-13)
        mbest = pattern_vectors;
        for i = 1:N
            if rand < rand
                mbest(i, :) = globalminimizer + rand .^ 3 * (up  - globalminimizer);
                mbest(i, :) = mbest(i, :)     + rand .^ 3 * (low - globalminimizer);
            else
                mbest(i, :) = globalminimizer;
            end
        end
        dv3 = mbest - pattern_vectors;

        % Levy-distributed evolutionary step size (lines 15-18)
        if (rand < rand ^ 3), c = 1; else, c = N; end
        Scale = 1 ./ gamrnd(0.5, randi([1 16]) / 2, [c 1]);

        % Bi-form mutation operator (lines 21 / 25)
        if rand < rand
            dv = dv1 + Scale .* (dv3 - dv2);
        else
            w = 3 * randn(N, 3);
            dv = Scale .* (w(:, 1) .* dv1 + w(:, 2) .* dv2 + w(:, 3) .* dv3);
        end

        % Crossover control matrix (lines 27-34)
        map = zeros(N, D);
        for j = 1:N
            h = randperm(D);
            if rand < rand
                mm = double(bernsteinMatrix(randi(3), rand)) .^ 3;
                w2 = mm(randi(numel(mm)));
            else
                if rand < rand
                    w2 = rand .^ randi([3 5]);
                else
                    w2 = (1 - rand .^ randi([3 5]));
                end
            end
            map(j, h(1:ceil(w2 * D))) = 1;
        end

        % Mutation and crossover (line 36)
        mutation_patterns = pattern_vectors + map .* dv;

        % Border control (line 38)
        for i = 1:N
            for j = 1:D
                if mutation_patterns(i, j) < low(j) || mutation_patterns(i, j) > up(j)
                    mutation_patterns(i, j) = rand * (up(j) - low(j)) + low(j);
                end
            end
        end

        % Update (lines 40-43)
        [ft, FE] = calculate_fitness(mutation_patterns', problem, FE);
        ft = ft(:);
        ind = ft < fitness_of_pattern_vectors;
        pattern_vectors(ind, :)         = mutation_patterns(ind, :);
        fitness_of_pattern_vectors(ind) = ft(ind);

        [globalminimum, bestindex] = min(fitness_of_pattern_vectors);
        globalminimizer = pattern_vectors(bestindex, :);
        if globalminimum < bsf
            bsf = globalminimum;
        end

        for k = 1:N
            ec = FE - N + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pattern_vectors, fitness_of_pattern_vectors, ...
                    population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = globalminimum;
    best_solution = globalminimizer;
end

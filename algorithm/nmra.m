% ----------------------------------------------------------------------- %
% Naked Mole-Rat Algorithm (NMRA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n        = 50     % Colony size
%   breeders = n/5    % Rows 1..n/5 breed, rows n/5+1..n work
%   bp       = 0.5    % Breeding probability: a breeder moves when rand > bp
%   beta     = 1.5    % Levy exponent of the worker step, scaled by 0.01
%
% Algorithm Concept:
%   - Workers: x + L.*(x_a - x_b), L a Levy step per dimension and a, b two
%     distinct colony members drawn at random
%   - Breeders: with probability 1 - bp, (1 - lambda)*x + lambda*(d - x_r), d
%     the colony best, x_r a random breeder, lambda ~ U(0, 1)
%   - Greedy acceptance, ties accepted; d is refreshed once per iteration
%   - The breeder/worker split is by row index and is never re-sorted
%
% Reference:
% Rohit Salgotra, Urvinder Singh,
% The naked mole-rat algorithm,
% Neural Computing and Applications 31(12) (2019) 8837-8857.
% https://doi.org/10.1007/s00521-019-04464-7
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release (github.com/rohitsalgotra/
% Naked-Mole-Rat-Algorithm, NMRA.m v1.0 of 25 Aug 2019), which the paper cites.
% Kept as released, unlike the paper: the worker step is scaled by a Levy
% vector, not by lambda ~ U(0, 1), and every breeder reuses the lambda drawn
% for the last worker. The release hard-codes workers 11:n and breeder
% neighbours randperm(10), i.e. n/5 breeders at the paper's n = 50 (its demo
% runs n = 40 and leaves rows 9-10 idle); n = 50 makes the two agree. Its clamp
% acts after the evaluation and is then overwritten: 30-46 % of its points lie
% outside the box (cec14 D = 10), its best in 2 of 4 runs. Here candidates are
% clamped before evaluation; the reported best is updated at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nmra(problem)

    dim = problem.dimension;
    Lb = problem.lb;
    Ub = problem.ub;
    maxFE = problem.maxFe;

    n = 50;
    bp = 0.5;
    breeders = n / 5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    NMRsolution = zeros(n, dim);
    for i = 1:n
        NMRsolution(i, :) = Lb + (Ub - Lb) .* rand(1, dim);   % row by row, as in the release
    end
    [NMRfitness, FE] = calculate_fitness(NMRsolution', problem, FE);
    NMRfitness = NMRfitness(:)';

    [best_fitness, I] = min(NMRfitness);
    best_solution = NMRsolution(I, :);
    NMRbest = best_solution;

    for e = 1:min(n, maxFE)
        curve(e) = min(NMRfitness(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, NMRsolution, NMRfitness, population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE
        % Workers, updated in place so later workers see earlier moves
        for i = breeders + 1:n
            lambda = rand;   % only the last worker's draw reaches the breeders, as released
            ab = randperm(n);
            L = levy(dim);
            S = NMRsolution(i, :) + L .* (NMRsolution(ab(1), :) - NMRsolution(ab(2), :));
            S = min(max(S, Lb), Ub);   % harness: the release evaluates S unclamped
            [Fnew, FE] = calculate_fitness(S', problem, FE);
            if Fnew <= NMRfitness(i)
                NMRsolution(i, :) = S;
                NMRfitness(i) = Fnew;
                if Fnew < best_fitness
                    best_fitness = Fnew;
                    best_solution = S;
                end
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, NMRsolution, NMRfitness, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        % Breeders
        for z = 1:breeders
            if rand > bp
                NMRneighbours = randperm(breeders);   % randperm(10) in the release
                S = (1 - lambda) .* NMRsolution(z, :) + lambda * (NMRbest - NMRsolution(NMRneighbours(1), :));
                S = min(max(S, Lb), Ub);
                [Fnew, FE] = calculate_fitness(S', problem, FE);
                if Fnew <= NMRfitness(z)
                    NMRsolution(z, :) = S;
                    NMRfitness(z) = Fnew;
                    if Fnew < best_fitness
                        best_fitness = Fnew;
                        best_solution = S;
                    end
                end
                curve(FE) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, NMRsolution, NMRfitness, population_history, fitness_history, ...
                    history_index, maxFE);
                if FE >= maxFE, break; end
            end
        end

        [~, I] = min(NMRfitness);
        NMRbest = NMRsolution(I, :);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Mantegna Levy step of the release, scaled by 0.01
function L = levy(d)
    beta = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    L = 0.01 * step;
end

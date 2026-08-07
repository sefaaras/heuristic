% ----------------------------------------------------------------------- %
% Stochastic Social Learning Optimization (SSLO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP   = 30       % Population size
%   beta = exp(1)   % Shape of the inspiration-burst weight w
%   CR   = 0.9      % Initial crossover rate (self-adapted per individual)
%
% Algorithm Concept:
%   - Social learning: each individual takes an exponentially distributed
%     step 1.4588*log(1/rand) along the direction to a random peer, flipped
%     so that it always moves away from the worse of the two
%   - Stochastic inspiration burst: the worst individual occasionally jumps,
%     with a weight w decaying in its stagnation counter (bucket theory)
%   - Interpolation-based bound repair and a per-individual self-adaptive
%     crossover rate that is rolled back whenever a trial fails
%
% Reference:
% Jiaojiao Ye, Khamron Sunat, Sirapat Chiewchanwattana,
% Stochastic social learning optimization: Combining social learning and
% bucket theory for efficient optimization,
% Knowledge-Based Systems 341 (2026) 115767.
% https://doi.org/10.1016/j.knosys.2026.115767
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sslo(problem)

    D     = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    NP      = 30;
    MaxIter = max(1, ceil((maxFE - NP) / NP));
    Range   = ub - lb;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    X = repmat(lb, NP, 1) + repmat(ub - lb, NP, 1) .* rand(NP, D);

    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);

    [fmin, bestIdX] = min(fitness);
    bestX = X(bestIdX, :);
    bsf   = fmin;

    for eval_count = 1:NP
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    newX      = zeros(NP, D);
    beta      = exp(1);
    unsuccess = zeros(1, NP);
    CR        = 0.9 * ones(NP, 1);

    % Main loop
    for iter = 1:MaxIter
        if FE >= maxFE, break; end

        CROld = CR;
        [~, xId] = sort(fitness);

        for i = 1:NP
            if i == xId(end) && rand < 0.25
                w = exp(-(beta * max(0, iter - unsuccess(i) + randn(1, D) * 10) / MaxIter) .^ beta);
                if iter < 0.3 * MaxIter
                    newX(i, :) = X(i, :) + w .* sign(rand(1, D) - 0.5) .* abs(X(i, :));
                else
                    if unsuccess(i) <= 5
                        newX(i, :) = X(xId(1), :) + w .* rand .* sign(rand(1, D) - 0.5) .* Range;
                    else
                        unsuccess(i) = 0;
                        newX(i, :) = X(xId(1), :) + sign(rand(1, D) - 0.5) .* rand(1, D) .* Range;
                    end
                end
            else
                while true
                    jx = ceil(rand * NP);
                    if jx ~= i, break; end
                end
                Stp = X(i, :) - X(jx, :);
                if fitness(jx) < fitness(i)
                    Stp = -Stp;
                end
                newX(i, :) = X(i, :) + 1.4588 * log(1 ./ rand(1, D)) .* Stp;
            end

            % Interpolation-based bound repair
            vd = newX(i, :) < lb;
            newX(i, vd) = (X(i, vd) + lb(1, vd)) / 2;
            vd = newX(i, :) > ub;
            newX(i, vd) = (X(i, vd) + ub(1, vd)) / 2;

            if rand < 0.3, CR(i) = 0.0 + 1.0 * rand(1, 1); end
            rndqCR = CR(i) - 0.1 * rand(1, D);
            XCr = (rand(1, D) <= rndqCR);
            XCr(round(rand * D + 0.5)) = true;   % at least one changed dimension

            newX(i, :) = XCr .* newX(i, :) + (~XCr) .* X(i, :);
        end

        [newfitness, FE] = calculate_fitness(newX', problem, FE);
        newfitness = newfitness(:);

        for i = 1:NP
            if newfitness(i) <= fitness(i)
                fitness(i) = newfitness(i);
                X(i, :)    = newX(i, :);
                unsuccess(i) = 0;
            else
                CR(i) = CROld(i);
                unsuccess(i) = 1 + unsuccess(i);
            end
        end

        [fmin, bestIdX] = min(fitness);
        bestX = X(bestIdX, :);
        if fmin < bsf
            bsf = fmin;
        end

        for k = 1:NP
            ec = FE - NP + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = fmin;
    best_solution = bestX;
end

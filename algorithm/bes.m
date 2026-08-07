% ----------------------------------------------------------------------- %
% Bald Eagle Search (BES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 100            % Population size
%   a = 10, R = 1.5       % Spiral/search shape parameters
%
% Algorithm Concept:
%   - Inspired by the hunting strategy of bald eagles in three stages:
%     (1) select space, (2) search within the space (spiral flight),
%     (3) swoop towards the prey (the optimum)
%
% Reference:
% H.A. Alsattar, A.A. Zaidan, B.B. Zaidan,
% Novel meta-heuristic bald eagle search optimisation algorithm,
% Artificial Intelligence Review 53 (2020) 2237-2264
% https://doi.org/10.1007/s10462-019-09732-5
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bes(problem)

    % Extract problem parameters
    dim = problem.dimension;
    low = problem.lb;
    high = problem.ub;
    maxFE = problem.maxFe;

    nPop = 100;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    BestSol.cost = inf;
    BestSol.pos = zeros(1, dim);

    % Initialization
    for i = 1:nPop
        pop.pos(i, :) = low + (high - low) .* rand(1, dim);
    end
    [pop.cost, FE] = calculate_fitness(pop.pos', problem, FE);
    pop.cost = pop.cost(:)';
    for i = 1:nPop
        if pop.cost(i) < BestSol.cost
            BestSol.pos = pop.pos(i, :);
            BestSol.cost = pop.cost(i);
        end
    end
    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = BestSol.cost;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, pop.pos, pop.cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    while FE < maxFE
        % 1- select space
        FE_before = FE;
        [pop, BestSol, FE] = select_space(pop, nPop, BestSol, low, high, dim, problem, FE, maxFE);
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, BestSol.cost, pop.pos, pop.cost, curve, ...
            population_history, fitness_history, history_index);
        if FE >= maxFE, break; end

        % 2- search in space
        FE_before = FE;
        [pop, BestSol, FE] = search_space(pop, BestSol, nPop, low, high, problem, FE, maxFE);
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, BestSol.cost, pop.pos, pop.cost, curve, ...
            population_history, fitness_history, history_index);
        if FE >= maxFE, break; end

        % 3- swoop
        FE_before = FE;
        [pop, BestSol, FE] = swoop(pop, BestSol, nPop, low, high, problem, FE, maxFE);
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, BestSol.cost, pop.pos, pop.cost, curve, ...
            population_history, fitness_history, history_index);
    end

    best_solution = BestSol.pos;
    best_fitness = BestSol.cost;

end

% Record helper for a phase's FE range
function [pop_hist, fit_hist, hist_idx, curve] = record_block(FE_before, FE, maxFE, bestcost, pos, cost, curve, pop_hist, fit_hist, hist_idx)
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bestcost;
            [pop_hist, fit_hist, hist_idx] = record_history(...
                eval_count, pos, cost, pop_hist, fit_hist, hist_idx, maxFE);
        end
    end
end

% Stage 1: Select space
function [pop, BestSol, FE] = select_space(pop, npop, BestSol, low, high, dim, problem, FE, maxFE)
    Mean = mean(pop.pos);
    lm = 2;
    for i = 1:npop
        if FE >= maxFE, break; end
        newpos = BestSol.pos + lm * rand(1, dim) .* (Mean - pop.pos(i, :));
        newpos = max(newpos, low);
        newpos = min(newpos, high);
        [newcost, FE] = calculate_fitness(newpos', problem, FE);
        if newcost < pop.cost(i)
            pop.pos(i, :) = newpos;
            pop.cost(i) = newcost;
            if pop.cost(i) < BestSol.cost
                BestSol.pos = pop.pos(i, :);
                BestSol.cost = pop.cost(i);
            end
        end
    end
end

% Stage 2: Search in space
function [pop, best, FE] = search_space(pop, best, npop, low, high, problem, FE, maxFE)
    Mean = mean(pop.pos);
    a = 10;
    R = 1.5;
    for i = 1:npop - 1
        if FE >= maxFE, break; end
        A = randperm(npop);
        pop.pos = pop.pos(A, :);
        pop.cost = pop.cost(A);
        [x, y] = polr(a, R, npop);
        Step = pop.pos(i, :) - pop.pos(i + 1, :);
        Step1 = pop.pos(i, :) - Mean;
        newpos = pop.pos(i, :) + y(i) * Step + x(i) * Step1;
        newpos = max(newpos, low);
        newpos = min(newpos, high);
        [newcost, FE] = calculate_fitness(newpos', problem, FE);
        if newcost < pop.cost(i)
            pop.pos(i, :) = newpos;
            pop.cost(i) = newcost;
            if pop.cost(i) < best.cost
                best.pos = pop.pos(i, :);
                best.cost = pop.cost(i);
            end
        end
    end
end

% Stage 3: Swoop
function [pop, best, FE] = swoop(pop, best, npop, low, high, problem, FE, maxFE)
    Mean = mean(pop.pos);
    a = 10;
    R = 1.5;
    for i = 1:npop
        if FE >= maxFE, break; end
        A = randperm(npop);
        pop.pos = pop.pos(A, :);
        pop.cost = pop.cost(A);
        [x, y] = swoo_p(a, R, npop);
        Step = pop.pos(i, :) - 2 * Mean;
        Step1 = pop.pos(i, :) - 2 * best.pos;
        newpos = rand(1, length(Mean)) .* best.pos + x(i) * Step + y(i) * Step1;
        newpos = max(newpos, low);
        newpos = min(newpos, high);
        [newcost, FE] = calculate_fitness(newpos', problem, FE);
        if newcost < pop.cost(i)
            pop.pos(i, :) = newpos;
            pop.cost(i) = newcost;
            if pop.cost(i) < best.cost
                best.pos = pop.pos(i, :);
                best.cost = pop.cost(i);
            end
        end
    end
end

% Spiral parameters (swoop)
function [xR, yR] = swoo_p(a, ~, N)
    th = a * pi * exp(rand(N, 1));
    r = th;
    xR = r .* sinh(th);
    yR = r .* cosh(th);
    xR = xR / max(abs(xR));
    yR = yR / max(abs(yR));
end

% Spiral parameters (search)
function [xR, yR] = polr(a, R, N)
    th = a * pi * rand(N, 1);
    r = th + R * rand(N, 1);
    xR = r .* sin(th);
    yR = r .* cos(th);
    xR = xR / max(abs(xR));
    yR = yR / max(abs(yR));
end

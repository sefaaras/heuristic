% ----------------------------------------------------------------------- %
% Caterpillar Fungus Optimizer (CFO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Search_Num = 50    % Population size (Cordyceps individuals)
%
% Algorithm Concept:
%   - Phase 1 (exploration): the fungus searches for larvae with two
%     operators chosen at random per individual
%       * spiral rising : beta = 2*cos(pi*r1)*|(It/MaxIt)^(r1*randi([1 2]))|
%       * wave advance  : alpha = 2.5*r.*|cos(pi*r)|
%     both combining the population best with the neighbouring individual
%   - Phase 2 (parasitisation): re-parasitic behaviour (Gaussian scaled
%     difference to the best) or optimal parasitic behaviour (quadratic
%     iteration decay f around the best)
%   - Violating dimensions are re-drawn uniformly inside the box
%
% Reference:
% Yang, Liang, Zhou, Qian, Zheng, Shu, He, Wang, Jiang, Sang, Li,
% A novel bio-inspired caterpillar fungus (Ophiocordyceps sinensis)
% optimizer for SOFC parameter identification via GRNN,
% Renewable Energy 256 (2026) 123995.
% https://doi.org/10.1016/j.renene.2025.123995
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cfo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    Search_Num = 50;
    MaxIt      = max(1, ceil((maxFE - Search_Num) / (2 * Search_Num)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Cordycepes_P = repmat(lb, Search_Num, 1) + ...
                   repmat(ub - lb, Search_Num, 1) .* rand(Search_Num, dim);
    for n = 1:Search_Num
        Cordycepes_P(n, :) = BoundCheck(Cordycepes_P(n, :), ub, lb);
    end

    [Cordycepes_F, FE] = calculate_fitness(Cordycepes_P', problem, FE);
    Cordycepes_F = Cordycepes_F(:)';

    [bsf, bi] = min(Cordycepes_F);
    bsx = Cordycepes_P(bi, :);

    for eval_count = 1:Search_Num
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Cordycepes_P, Cordycepes_F, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    Population_inf = zeros(Search_Num, dim);

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        % Phase 1: exploration
        [Cordycepes_F, index] = sort(Cordycepes_F);
        Cordycepes_P = Cordycepes_P(index, :);
        [~, index] = min(Cordycepes_F);

        if rand < 0.5
            r1 = rand;
            beta = 2 * cos(pi * r1) * abs((It / MaxIt) ^ (r1 * randi([1 2])));
            Population_inf(1, :) = Cordycepes_P(index, :) - ...
                rand(1, dim) .* (Cordycepes_P(index, :) - Cordycepes_P(1, :)) + ...
                beta .* (Cordycepes_P(index, :) - Cordycepes_P(1, :));
        else
            r = rand(1, dim);
            alpha = 2.5 * r .* abs(cos(pi * r));
            Population_inf(1, :) = Cordycepes_P(1, :) - ...
                rand(1, dim) .* (Cordycepes_P(index, :) - Cordycepes_P(1, :)) + ...
                alpha .* (Cordycepes_P(index, :) - Cordycepes_P(1, :));
        end

        for n = 2:Search_Num
            if rand < 0.5
                r1 = rand;
                beta = 2 * cos(pi * r1) * abs((It / MaxIt) ^ (r1 * randi([1 2])));
                Population_inf(n, :) = Cordycepes_P(index, :) - ...
                    rand(1, dim) .* (Cordycepes_P(n-1, :) - Cordycepes_P(n, :)) + ...
                    beta .* (Cordycepes_P(index, :) - Cordycepes_P(n, :));
            else
                r = rand(1, dim);
                alpha = 2.5 * r .* abs(cos(pi * r));
                Population_inf(n, :) = Cordycepes_P(n, :) - ...
                    rand(1, dim) .* (Cordycepes_P(n-1, :) - Cordycepes_P(n, :)) + ...
                    alpha .* (Cordycepes_P(index, :) - Cordycepes_P(n, :));
            end
        end

        for n = 1:Search_Num
            Population_inf(n, :) = BoundCheck(Population_inf(n, :), ub, lb);
        end
        [Population_fitness, FE] = calculate_fitness(Population_inf', problem, FE);
        Population_fitness = Population_fitness(:)';

        for n = 1:Search_Num
            if Population_fitness(n) < Cordycepes_F(n)
                Cordycepes_F(n)    = Population_fitness(n);
                Cordycepes_P(n, :) = Population_inf(n, :);
            end
        end
        [mf, mi] = min(Population_fitness);
        if mf < bsf
            bsf = mf;
            bsx = Population_inf(mi, :);
        end
        for k = 1:Search_Num
            ec = FE - Search_Num + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Cordycepes_P, Cordycepes_F, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        % Phase 2: parasitisation
        [~, index] = min(Cordycepes_F);
        for n = 1:Search_Num
            if rand < 0.5
                % Re-parasitic behaviour
                e = 3 * randn(1, dim);
                Population_inf(n, :) = Cordycepes_P(n, :) + ...
                    e .* (rand * Cordycepes_P(index, :) - rand * Cordycepes_P(n, :));
            else
                % Optimal parasitic behaviour
                f = rand * ((1 / MaxIt ^ 2) * It ^ 2 - 2 / MaxIt * It + 1);
                Population_inf(n, :) = Cordycepes_P(index, :) + ...
                    f .* (rand * Cordycepes_P(index, :) - rand * Cordycepes_P(n, :));
            end
        end

        for n = 1:Search_Num
            Population_inf(n, :) = BoundCheck(Population_inf(n, :), ub, lb);
        end
        [Population_fitness, FE] = calculate_fitness(Population_inf', problem, FE);
        Population_fitness = Population_fitness(:)';

        for n = 1:Search_Num
            if Population_fitness(n) < Cordycepes_F(n)
                Cordycepes_F(n)    = Population_fitness(n);
                Cordycepes_P(n, :) = Population_inf(n, :);
            end
        end
        [mf, mi] = min(Population_fitness);
        if mf < bsf
            bsf = mf;
            bsx = Population_inf(mi, :);
        end
        for k = 1:Search_Num
            ec = FE - Search_Num + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Cordycepes_P, Cordycepes_F, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Bound handling: re-draw violating dimensions
function newX = BoundCheck(X, ub, lb)
    Flag4ub = X > ub;
    Flag4lb = X < lb;
    newX = (X .* (~(Flag4ub + Flag4lb))) + (Flag4ub + Flag4lb) .* ((ub - lb) .* rand(1, length(X)) + lb);
end

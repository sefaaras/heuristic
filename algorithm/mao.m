% ----------------------------------------------------------------------- %
% Mexican Axolotl Optimization (MAO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   np = 30               % Population size
%   dp = 0.5              % Damage (injury) probability
%   rp = 0.1              % Regeneration probability
%   k = 3                 % Tournament size
%   lambda = 0.5          % Transition step factor
%   cop = 0.5             % Crossover probability
%
% Algorithm Concept:
%   - Inspired by the life cycle of the axolotl (males/females)
%   - Transition from larvae to adult state driven by the best individuals
%   - Injury and restoration phase perturbs damaged body parts
%   - Reproduction and assortment via tournament selection and uniform
%     crossover of two eggs
%
% Reference:
% Yenny Villuendas-Rey, Jose Luis Velazquez-Rodriguez,
% Mariana Dayanara Alanis-Tamez, Marco-Antonio Moreno-Ibarra,
% Cornelio Yanez-Marquez,
% Mexican Axolotl Optimization: A Novel Bioinspired Heuristic,
% Mathematics 9(7) (2021) 781
% https://doi.org/10.3390/math9070781
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mao(problem)

    % Extract problem parameters
    dim = problem.dimension;
    Lb = problem.lb;
    Ub = problem.ub;
    maxFE = problem.maxFe;

    np = 30;         % Population size
    dp = 0.5;        % Damage probability
    rp = 0.1;        % Regeneration probability
    k = 3;           % Tournament size
    lambda = 0.5;    % lambda value
    cop = 0.5;       % Crossover probability

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Global best-so-far
    bf = inf;
    bs = zeros(1, dim);

    % Initialization: random population evaluated according to O
    axolotl = zeros(np, dim);
    for i = 1:np
        axolotl(i, :) = Lb + (Ub - Lb) .* rand(size(Lb));
    end
    O = Inf^10 * ones(np, 1);

    FE_before = FE;
    [axolotl, O, FE, bf, bs] = get_best(axolotl, axolotl, O, 1:np, problem, FE, maxFE, bf, bs);
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, axolotl, O, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Divide the population into males and females
    population = 1:np;
    male = population(1:2:end);
    female = population(2:2:end);
    M = size(male, 1);      % NOTE: faithful to the reference original, where
    F = size(female, 1);    % 1:size(male)/1:size(female) resolve to 1:1

    % Starting evaluations
    while FE < maxFE
        FE_before = FE;

        % Phase 1: Transition from larvae to adult state
        m = axolotl(male, :);
        [~, K] = min(O(male, :));
        mbest = m(K, :);

        f = axolotl(female, :);
        [~, K] = min(O(female, :));
        fbest = f(K, :);

        for j = 1:M
            pmj = O(male(j), :) / sum(O(male, :));
            if pmj < rand()
                m(j, :) = m(j, :) + (mbest - m(j, :)) * lambda;
            else
                Lb2 = Lb - m(j, :);
                Ub2 = Ub - m(j, :);
                ran_dom = Lb2 + (Ub2 - Lb2) .* rand(size(Lb2));
                m(j, :) = m(j, :) + ran_dom;
            end
        end
        [axolotl, O, FE, bf, bs] = get_best(axolotl, m, O, male, problem, FE, maxFE, bf, bs);

        for j = 1:F
            pfj = O(female(j), :) / sum(O(female, :));
            if pfj < rand()
                f(j, :) = f(j, :) + (fbest - f(j, :)) * lambda;
            else
                Lb2 = Lb - f(j, :);
                Ub2 = Ub - f(j, :);
                ran_dom = Lb2 + (Ub2 - Lb2) .* rand(size(Lb2));
                f(j, :) = f(j, :) + ran_dom;
            end
        end
        [axolotl, O, FE, bf, bs] = get_best(axolotl, f, O, female, problem, FE, maxFE, bf, bs);

        % Phase 2: Injury and restoration
        list_damage = [];
        axolotl_damage = axolotl(:, :);
        for i = 1:np
            if rand() < dp
                for j = 1:dim
                    if rand() < rp
                        Lb2 = Lb - axolotl_damage(i, :);
                        Ub2 = Ub - axolotl_damage(i, :);
                        ran_dom = Lb2 + (Ub2 - Lb2) .* rand(size(Lb2));
                        axolotl_damage(i, :) = axolotl_damage(i, :) + ran_dom;
                        list_damage = [list_damage i];
                    end
                end
            end
        end
        list_damage = unique(list_damage);
        axolotl_p2 = axolotl_damage(list_damage, :);
        [axolotl, O, FE, bf, bs] = get_best(axolotl, axolotl_p2, O, list_damage, problem, FE, maxFE, bf, bs);

        % Phase 3: Reproduction and assortment
        f = axolotl(female, :);
        fitness_female = O(female, :);
        m = axolotl(male, :);
        fitness_male = O(male, :);
        for j = 1:F
            fj = f(j, :);
            % Tournament selection of a mate
            [mj, id_win] = Tournament_Selection(k, m, fitness_male);
            % Uniform crossover producing two eggs
            [egg1, egg2] = UniformCrossover(fj, mj, cop);
            % Evaluate the eggs
            [oegg1, FE, bf, bs] = fobj(egg1, problem, FE, bf, bs);
            [oegg2, FE, bf, bs] = fobj(egg2, problem, FE, bf, bs);

            omj = fitness_male(id_win, :);
            ofj = fitness_female(j, :);
            ranking = [egg1; egg2; axolotl(male(id_win), :); axolotl(female(j), :)];
            [bestsrt, id_best] = sort([oegg1, oegg2, omj, ofj]);

            % Assign the two best to the female and male slots
            axolotl(female(j), :) = ranking(id_best(1), :);
            O(female(j)) = bestsrt(1);
            axolotl(male(id_win), :) = ranking(id_best(2), :);
            O(male(id_win)) = bestsrt(2);
        end

        % Record convergence curve and history over this generation's FEs
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = bf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, axolotl, O, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    best_fitness = bf;
    best_solution = bs;

end

% Evaluate new solutions one at a time and greedily accept them
function [axolotl, O, FE, bf, bs] = get_best(axolotl, newaxolotl, O, ids, problem, FE, maxFE, bf, bs)
    for gb_j = 1:size(newaxolotl, 1)
        if FE >= maxFE
            break;
        end
        [fnew, FE, bf, bs] = fobj(newaxolotl(gb_j, :), problem, FE, bf, bs);
        if fnew <= O(ids(gb_j))
            O(ids(gb_j)) = fnew;
            axolotl(ids(gb_j), :) = newaxolotl(gb_j, :);
        end
    end
end

% Objective function: evaluate, update FE and best-so-far
function [z, FE, bf, bs] = fobj(u, problem, FE, bf, bs)
    [z, FE] = calculate_fitness(u', problem, FE);
    if z < bf
        bf = z;
        bs = u;
    end
end

% Tournament selection
function [winner, id_win] = Tournament_Selection(k, axolotl, fitness)
    n = size(axolotl, 1);
    r = randi(n, k, 1);
    select_fit = fitness(r);
    [~, id_min] = min(select_fit);
    id_win = r(id_min);
    winner = axolotl(id_win, :);
end

% Uniform crossover
function [egg1, egg2] = UniformCrossover(f, m, cop)
    n = length(f);
    egg1 = [];
    egg2 = [];
    for i = 1:n
        if rand() < cop
            egg1 = [egg1 f(i)];
            egg2 = [egg2 m(i)];
        else
            egg1 = [egg1 m(i)];
            egg2 = [egg2 f(i)];
        end
    end
end

% ----------------------------------------------------------------------- %
% Black Widow Optimization Algorithm (BWOA) for unconstrained benchmark
% problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                     % Population size
%   pc = 0.8                   % Crossover (procreating) percentage
%   pMutation = 0.4            % Mutation percentage
%   pCannibalism = 0.5         % Cannibalism rate
%
% Algorithm Concept:
%   - Inspired by the mating behaviour of black widow spiders
%   - Procreation (crossover), cannibalism (removal of weak individuals)
%     and mutation drive the search
%
% Reference:
% Vahideh Hayyolalam, Ali Asghar Pourhaji Kazem,
% Black Widow Optimization Algorithm: A novel meta-heuristic approach for
% solving engineering optimization problems,
% Engineering Applications of Artificial Intelligence 87 (2020) 103249
% https://doi.org/10.1016/j.engappai.2019.103249
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = bwoa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 40;
    nPop = N;
    nvar = dim;

    pc = 0.8;
    nCross = round(pc * nPop / 2) * 2;   % Number of selected parents
    pMutation = 0.4;
    nMutation = round(pMutation * nPop);
    pCannibalism = 0.5;
    nCannibalism = round(pCannibalism * nvar);

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nPop, dim);
    fitness_history = zeros(history_size, nPop);
    history_index = 1;

    individual.Position = [];
    individual.Cost = [];
    pop = repmat(individual, nPop, 1);

    bf = inf;              % best-so-far cost (framework convention)
    bs = zeros(1, dim);    % best-so-far position

    % Generating the initial population
    for i = 1:nPop
        pop(i).Position = initializationBW(dim, ub, lb);
        [pop(i).Cost, FE, bf, bs] = evalp(pop(i).Position, problem, FE, bf, bs);
    end

    % Sorting the population
    Costs = [pop.Cost];
    [Costs, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
    WorstCost = Costs(end);

    [population_history, fitness_history, history_index, curve] = record_block(...
        0, FE, maxFE, bf, pop, dim, curve, ...
        population_history, fitness_history, history_index, sampling_interval, history_size);

    while FE < maxFE
        FE_before = FE;

        % Crossover - generating the offspring population
        crosspop = repmat(individual, nCross, 1);
        [crosspop, FE, bf, bs] = BwCrossover(crosspop, pop, nvar, nCross, nPop, nCannibalism, problem, FE, maxFE, bf, bs);

        % Mutation
        pop3 = repmat(individual, nMutation, 1);
        randnum = randperm(nCross);
        for k = 1:nMutation
            i = randnum(k);
            q = Mutate(pop(i));
            [q.Cost, FE, bf, bs] = evalp(q.Position, problem, FE, bf, bs);
            pop3(k) = q;
        end

        pop = [crosspop
            pop3];

        Costs = [pop.Cost];
        [Costs, SortOrder] = sort(Costs);
        pop = pop(SortOrder);
        WorstCost = max(WorstCost, Costs(end)); %#ok<NASGU>

        pop = pop(1:nPop);
        Costs = Costs(1:nPop);

        % Record convergence curve and history
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, bf, pop, dim, curve, ...
            population_history, fitness_history, history_index, sampling_interval, history_size);
    end

    best_solution = bs;
    best_fitness = bf;

end

%% --- Objective evaluation: threads FE and best-so-far ---
function [z, FE, bf, bs] = evalp(pos, problem, FE, bf, bs)
    [z, FE] = calculate_fitness(pos', problem, FE);
    if z < bf
        bf = z;
        bs = pos;
    end
end

%% --- Record helper for a block's FE range ---
function [pop_hist, fit_hist, hist_idx, curve] = record_block(FE_before, FE, maxFE, bestcost, pop, dim, curve, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    n = numel(pop);
    pos = zeros(n, dim);
    cost = zeros(1, n);
    for i = 1:n
        pos(i, :) = pop(i).Position;
        cost(i) = pop(i).Cost;
    end
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bestcost;
            [pop_hist, fit_hist, hist_idx] = record_history(...
                eval_count, pos, cost, pop_hist, fit_hist, hist_idx, sampling_interval, history_size);
        end
    end
end

%% --- Initialization ---
function X = initializationBW(dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(1, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Mutation (swap two genes) ---
function q = Mutate(p)
    x = p.Position;
    nvar = numel(x);
    randrand = randperm(nvar);
    j1 = randrand(1);
    j2 = randrand(2);
    n1j = x(j1);
    n2j = x(j2);
    x(j1) = n2j;
    x(j2) = n1j;
    q.Position = x;
end

%% --- Crossover with sexual/sibling cannibalism ---
function [crosspop, FE, bf, bs] = BwCrossover(crosspop, pop, nvar, nCross, ~, nCannibalism, problem, FE, maxFE, bf, bs)
    individual.Position = [];
    individual.Cost = [];
    a = repmat(individual, nvar, 1);

    indexno = randperm(nCross);

    for k = 1:2:nCross
        % Parents' choosing
        r1 = indexno(k);
        r2 = indexno(k + 1);
        p1 = pop(r1);
        p2 = pop(r2);
        % Sexual cannibalism
        if pop(r1).Cost < pop(r2).Cost
            a(1) = pop(r1);
        else
            a(1) = pop(r2);
        end
        % Reproduction
        for i = 1:2:nvar
            x1 = p1.Position;
            x2 = p2.Position;
            alpha = rand(size(x1));
            y1 = alpha .* x1 + (1 - alpha) .* x2;
            y2 = alpha .* x2 + (1 - alpha) .* x1;
            a(i + 1).Position = y1;
            a(i + 2).Position = y2;
            [a(i + 1).Cost, FE, bf, bs] = evalp(a(i + 1).Position, problem, FE, bf, bs);
            [a(i + 2).Cost, FE, bf, bs] = evalp(a(i + 2).Position, problem, FE, bf, bs);
        end

        Costs = [a.Cost];
        [~, order] = sort(Costs);
        a = a(order);
        % Sibling cannibalism
        if nvar > 2
            for l = 0:nCannibalism
                crosspop(k + l) = a(l + 1);
            end
        elseif nvar == 2
            for l = 0:nCannibalism + 1
                crosspop(k + l) = a(l + 1);
            end
        elseif nvar == 1
            for l = 0:nCannibalism + 2
                crosspop(k + l) = a(l + 1);
            end
        end
    end
end

% ----------------------------------------------------------------------- %
% Imperialist Competitive Algorithm (ICA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50, nEmp = 10        % Countries, of which the best become imperialists
%   alpha = 1                   % Selection pressure of the roulette over empires
%   beta = 1.5                  % Assimilation coefficient
%   pRevolution = 0.05, mu = 0.1 % Revolution probability and share of coordinates
%   zeta = 0.2                  % Weight of the colonies' mean cost in an empire's total
%
% Algorithm Concept:
%   - The best nEmp countries become imperialists, the rest are handed out as
%     colonies by a roulette over empire cost, so a stronger empire starts larger
%   - Assimilation moves every colony a beta-scaled random fraction of the way
%     towards its imperialist, which is the algorithm's exploitation step
%   - Revolution perturbs mu of the coordinates with a Gaussian of 10 % of the
%     box: the imperialist keeps its trial only if better, a colony always does
%   - Intra-empire competition swaps a colony with its imperialist when it is better
%   - Inter-empire competition hands the weakest empire's worst colony to a
%     roulette winner; an empire that loses its last colony is absorbed, so the
%     empire count falls over the run
%
% Reference:
% Esmaeil Atashpaz-Gargari, Caro Lucas,
% Imperialist competitive algorithm: An algorithm for optimization inspired by
% imperialistic competition,
% 2007 IEEE Congress on Evolutionary Computation (CEC), 2007, pp. 4661-4667.
% https://doi.org/10.1109/CEC.2007.4425083
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Yarpiz's YPEA118 implementation (yarpiz.com/247), which is the
% reference this repository uses for the classics; its struct-of-empires is
% flattened into position and cost arrays with an empire index per country. Its
% asymmetry is kept: a revolting imperialist keeps its trial only when better,
% while a revolting colony takes it unconditionally. When empires have merged
% into one, inter-empire competition returns untouched, as it does there, and
% the run continues as a single empire.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ica(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;
    nEmp = 10;
    alpha = 1;
    beta = 1.5;
    pRevolution = 0.05;
    mu = 0.1;
    zeta = 0.2;

    nmu = ceil(mu * dim);
    sigma = 0.1 * (ub - lb);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    pos = repmat(lb, nPop, 1) + rand(nPop, dim) .* repmat(ub - lb, nPop, 1);
    [cost, FE] = calculate_fitness(pos', problem, FE);
    cost = cost(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:min(nPop, maxFE)
        if cost(i) < best_fitness
            best_fitness = cost(i);
            best_solution = pos(i, :);
        end
        curve(i) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            i, pos, cost', population_history, fitness_history, history_index, maxFE);
    end

    [cost, order] = sort(cost);
    pos = pos(order, :);

    imp_pos = pos(1:nEmp, :);
    imp_cost = cost(1:nEmp);
    col_pos = pos(nEmp+1:end, :);
    col_cost = cost(nEmp+1:end);

    % Colonies are dealt out by a roulette over imperialist cost
    P = exp(-alpha * imp_cost / max(imp_cost));
    P = P / sum(P);
    col_emp = zeros(size(col_cost));
    for j = 1:numel(col_cost)
        col_emp(j) = roulette(P);
    end

    while FE < maxFE
        n_emp = numel(imp_cost);

        % Assimilation: every colony moves towards its imperialist
        if ~isempty(col_cost)
            step = beta * rand(size(col_pos)) .* (imp_pos(col_emp, :) - col_pos);
            col_pos = min(max(col_pos + step, repmat(lb, size(col_pos, 1), 1)), ...
                          repmat(ub, size(col_pos, 1), 1));
            [col_cost, FE] = calculate_fitness(col_pos', problem, FE);
            col_cost = col_cost(:);
            [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                stage(col_pos, col_cost, imp_pos, imp_cost, col_pos, col_cost, FE, ...
                      numel(col_cost), best_fitness, best_solution, curve, ...
                      population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE, break; end
        end

        % Revolution of the imperialists, accepted only if better
        new_imp = imp_pos;
        for k = 1:n_emp
            jj = randperm(dim, nmu);
            new_imp(k, jj) = imp_pos(k, jj) + sigma(jj) .* randn(1, nmu);
        end
        new_imp = min(max(new_imp, repmat(lb, n_emp, 1)), repmat(ub, n_emp, 1));
        [new_cost, FE] = calculate_fitness(new_imp', problem, FE);
        new_cost = new_cost(:);
        better = new_cost < imp_cost;
        imp_pos(better, :) = new_imp(better, :);
        imp_cost(better) = new_cost(better);
        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            stage(new_imp, new_cost, imp_pos, imp_cost, col_pos, col_cost, FE, n_emp, ...
                  best_fitness, best_solution, curve, population_history, fitness_history, ...
                  history_index, maxFE);
        if FE >= maxFE, break; end

        % Revolution of the colonies, taken unconditionally
        revolt = find(rand(numel(col_cost), 1) <= pRevolution);
        if ~isempty(revolt)
            for idx = revolt'
                jj = randperm(dim, nmu);
                col_pos(idx, jj) = col_pos(idx, jj) + sigma(jj) .* randn(1, nmu);
            end
            col_pos(revolt, :) = min(max(col_pos(revolt, :), repmat(lb, numel(revolt), 1)), ...
                                     repmat(ub, numel(revolt), 1));
            [rev_cost, FE] = calculate_fitness(col_pos(revolt, :)', problem, FE);
            col_cost(revolt) = rev_cost(:);
            [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                stage(col_pos(revolt, :), col_cost(revolt), imp_pos, imp_cost, col_pos, col_cost, ...
                      FE, numel(revolt), best_fitness, best_solution, curve, ...
                      population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE, break; end
        end

        % Intra-empire competition: a colony better than its imperialist trades places
        for j = 1:numel(col_cost)
            k = col_emp(j);
            if col_cost(j) < imp_cost(k)
                tmp_pos = imp_pos(k, :); tmp_cost = imp_cost(k);
                imp_pos(k, :) = col_pos(j, :); imp_cost(k) = col_cost(j);
                col_pos(j, :) = tmp_pos; col_cost(j) = tmp_cost;
            end
        end

        % Inter-empire competition
        if n_emp > 1
            total = imp_cost;
            for k = 1:n_emp
                mine = (col_emp == k);
                if any(mine)
                    total(k) = imp_cost(k) + zeta * mean(col_cost(mine));
                end
            end
            [~, weakest] = max(total);
            P = exp(-alpha * total / max(total));
            P(weakest) = 0;
            P = P / sum(P);
            if any(isnan(P))
                P(isnan(P)) = 0;
                if all(P == 0), P(:) = 1; end
                P = P / sum(P);
            end

            mine = find(col_emp == weakest);
            if ~isempty(mine)
                [~, worst_local] = max(col_cost(mine));
                col_emp(mine(worst_local)) = roulette(P);
            else
                % The weakest empire has no colony left, so it becomes one itself
                winner = roulette(P);
                col_pos(end+1, :) = imp_pos(weakest, :); %#ok<AGROW>
                col_cost(end+1, 1) = imp_cost(weakest);  %#ok<AGROW>
                col_emp(end+1, 1) = winner;              %#ok<AGROW>
                imp_pos(weakest, :) = [];
                imp_cost(weakest) = [];
                col_emp(col_emp > weakest) = col_emp(col_emp > weakest) - 1;
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

function k = roulette(P)
    c = cumsum(P);
    k = find(rand * c(end) <= c, 1, 'first');
    if isempty(k)
        k = numel(P);   % floating-point rounding can leave the draw past the last edge
    end
end

% Curve and history for one stage's evaluations; the recorder sees every country
function [bf, bx, curve, ph, fh, hi] = stage(new_pos, new_cost, imp_pos, imp_cost, ...
                                             col_pos, col_cost, FE, n, bf, bx, curve, ph, fh, hi, maxFE)
    pop = [imp_pos; col_pos];
    fit = [imp_cost; col_cost];
    for i = 1:n
        if new_cost(i) < bf
            bf = new_cost(i);
            bx = new_pos(i, :);
        end
        eval_count = FE - n + i;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bf;
            [ph, fh, hi] = record_history(eval_count, pop, fit', ph, fh, hi, maxFE);
        end
    end
end

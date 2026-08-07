% ----------------------------------------------------------------------- %
% Wild Horse Optimizer (WHO)
% ----------------------------------------------------------------------- %
% NOTE: stored as `whoa` because `who` is a built-in MATLAB command that a
% path function cannot override; call this algorithm as 'whoa'.
%
% Algorithm Parameters:
%   N  = 30      % Total population
%   PS = 0.2     % Stallions percentage
%   PC = 0.13    % Crossover percentage
%
% Algorithm Concept:
%   - Herds led by stallions; foals graze around their stallion, groups
%     exchange members via crossover, and stallions compete toward the best
%     stallion (WH), the global leader.
%
% Reference:
% Iraj Naruei, Farshid Keynia,
% Wild horse optimizer: a new meta-heuristic algorithm for solving
% engineering optimization problems,
% Engineering with Computers 38 (Suppl. 4) (2022) 3025-3056.
% https://doi.org/10.1007/s00366-021-01438-z
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = whoa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    Max_iter = ceil(maxFE / N);
    PS = 0.2;     % Stallions percentage
    PC = 0.13;    % Crossover percentage
    NStallion = ceil(PS * N);
    Nfoal = N - NStallion;

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    empty.pos = [];
    empty.cost = [];

    group = repmat(empty, Nfoal, 1);
    for i = 1:Nfoal
        group(i).pos = lb + rand(1, dim) .* (ub - lb);
        [group(i).cost, FE] = calculate_fitness(group(i).pos', problem, FE);
    end

    Stallion = repmat(empty, NStallion, 1);
    for i = 1:NStallion
        Stallion(i).pos = lb + rand(1, dim) .* (ub - lb);
        [Stallion(i).cost, FE] = calculate_fitness(Stallion(i).pos', problem, FE);
    end

    ngroup = length(group);
    a = randperm(ngroup);
    group = group(a);

    i = 0;
    k = 1;
    for j = 1:ngroup
        i = i + 1;
        Stallion(i).group(k) = group(j);
        if i == NStallion
            i = 0;
            k = k + 1;
        end
    end
    Stallion = exchange(Stallion);
    [~, index] = min([Stallion.cost]);

    WH = Stallion(index);   % global leader
    gBest = WH.pos;
    gBestScore = WH.cost;

    bsf = gBestScore;
    % Record initial evaluations
    [population_history, fitness_history, history_index] = record_pop(...
        Stallion, N, dim, 1, min(FE, maxFE), bsf, curve, ...
        population_history, fitness_history, history_index, maxFE);
    for eval_count = 1:min(FE, maxFE)
        curve(eval_count) = bsf;
    end

    l = 2;
    while l < Max_iter + 1 && FE < maxFE
        FE_before = FE;
        TDR = 1 - l * ((1) / Max_iter);

        for i = 1:NStallion
            ngroup = length(Stallion(i).group);
            [~, index] = sort([Stallion(i).group.cost]);
            Stallion(i).group = Stallion(i).group(index);

            % r3/rr pre-initialised so the stallion update is defined even if every foal crossed over
            z = rand(1, dim) < TDR;
            r1 = rand; r2 = rand(1, dim);
            idx = (z == 0);
            r3 = r1 .* idx + r2 .* ~idx;
            rr = -2 + 4 * r3;

            for j = 1:ngroup
                if rand > PC
                    z = rand(1, dim) < TDR;
                    r1 = rand;
                    r2 = rand(1, dim);
                    idx = (z == 0);
                    r3 = r1 .* idx + r2 .* ~idx;
                    rr = -2 + 4 * r3;
                    Stallion(i).group(j).pos = 2 * r3 .* cos(2 * pi * rr) .* (Stallion(i).pos - Stallion(i).group(j).pos) + (Stallion(i).pos);
                else
                    A = randperm(NStallion);
                    A(A == i) = [];
                    a = A(1);
                    c = A(2);
                    x1 = Stallion(c).group(end).pos;
                    x2 = Stallion(a).group(end).pos;
                    y1 = (x1 + x2) / 2;   % Crossover
                    Stallion(i).group(j).pos = y1;
                end

                Stallion(i).group(j).pos = min(Stallion(i).group(j).pos, ub);
                Stallion(i).group(j).pos = max(Stallion(i).group(j).pos, lb);

                [Stallion(i).group(j).cost, FE] = calculate_fitness(Stallion(i).group(j).pos', problem, FE);
                if Stallion(i).group(j).cost < bsf, bsf = Stallion(i).group(j).cost; end
                if FE <= maxFE, curve(FE) = bsf; end
                if FE >= maxFE, break; end
            end
            if FE >= maxFE, break; end

            R = rand;
            if R < 0.5
                kk = 2 * r3 .* cos(2 * pi * rr) .* (WH.pos - (Stallion(i).pos)) + WH.pos;
            else
                kk = 2 * r3 .* cos(2 * pi * rr) .* (WH.pos - (Stallion(i).pos)) - WH.pos;
            end
            kk = min(kk, ub);
            kk = max(kk, lb);
            [fk, FE] = calculate_fitness(kk', problem, FE);
            if fk < Stallion(i).cost
                Stallion(i).pos = kk;
                Stallion(i).cost = fk;
            end
            if fk < bsf, bsf = fk; end
            if FE <= maxFE, curve(FE) = bsf; end
            if FE >= maxFE, break; end
        end

        Stallion = exchange(Stallion);
        [value, index] = min([Stallion.cost]);
        if value < WH.cost
            WH = Stallion(index);
        end
        gBest = WH.pos;
        gBestScore = WH.cost;

        % Record population history for this iteration's FE block
        [population_history, fitness_history, history_index] = record_pop(...
            Stallion, N, dim, FE_before + 1, min(FE, maxFE), bsf, curve, ...
            population_history, fitness_history, history_index, maxFE);

        l = l + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness = gBestScore;
    best_solution = gBest;
end

% Assemble population and record its history over an FE block
function [pop_hist, fit_hist, hist_idx] = record_pop(Stallion, N, dim, fe_from, fe_to, ~, ~, pop_hist, fit_hist, hist_idx, maxFE)
    if fe_to < fe_from, return; end
    allPos = zeros(N, dim);
    allCost = zeros(1, N);
    idx = 1;
    for i = 1:numel(Stallion)
        allPos(idx, :) = Stallion(i).pos;
        allCost(idx) = Stallion(i).cost;
        idx = idx + 1;
        for j = 1:numel(Stallion(i).group)
            if idx > N, break; end
            allPos(idx, :) = Stallion(i).group(j).pos;
            allCost(idx) = Stallion(i).group(j).cost;
            idx = idx + 1;
        end
    end
    for eval_count = fe_from:fe_to
        [pop_hist, fit_hist, hist_idx] = record_history(...
            eval_count, allPos, allCost, pop_hist, fit_hist, hist_idx, ...
            maxFE);
    end
end

% Exchange best group member with the stallion
function Stallion = exchange(Stallion)
    nStallion = length(Stallion);
    for i = 1:nStallion
        [value, index] = min([Stallion(i).group.cost]);
        if value < Stallion(i).cost
            bestgroup = Stallion(i).group(index);
            Stallion(i).group(index).pos = Stallion(i).pos;
            Stallion(i).group(index).cost = Stallion(i).cost;
            Stallion(i).pos = bestgroup.pos;
            Stallion(i).cost = bestgroup.cost;
        end
    end
end

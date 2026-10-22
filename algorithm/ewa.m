% ----------------------------------------------------------------------- %
% Earthworm Optimisation Algorithm (EWA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50                    % Earthworms
%   Keep    = 2                     % Elites, carried over and never mutated
%   alpha   = 0.98                  % Similarity factor of Reproduction 1
%   beta    = 1 -> 0 (x0.9 per gen) % Weight of Reproduction 1 against 2
%   pcross  = 1                     % Uniform-crossover probability
%   pmutate = 0.01                  % Per-gene rate of the mutation step
%
% Algorithm Concept:
%   - Reproduction 1: x1 = ub + lb - alpha*x, a contracted reflection of the
%     worm through the centre of the box
%   - Reproduction 2: uniform crossover of a random worm with one drawn by
%     roulette on 1/f that never picks the best; the elites copy a random worm
%   - Offspring beta*x1 + (1 - beta)*x2, written back in place, so later worms
%     can mate with offspring of the same generation
%   - Mutation as released: each non-elite becomes (w + xb0)/2, w a copy of xb0
%     whose genes turn into the population mean at rate pmutate, cumulatively
%   - The two previous elites replace the two worst; a duplicate worm gets one
%     gene redrawn at random
%
% Reference:
% Gai-Ge Wang, Suash Deb, Leandro dos Santos Coelho,
% Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm for
% global optimisation problems,
% International Journal of Bio-Inspired Computation 12(1) (2018) 1-22.
% https://doi.org/10.1504/IJBIC.2018.093328
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release, File Exchange 53479 v2.0.0.0 (EWA_FEs_V2.m,
% Oct 2015), which is not the paper: its "Cauchy mutation" draws no Cauchy
% number and pulls toward xb0, the best of the INITIAL population, never
% updated; both kept, as are ClearDups' redraws without re-evaluation (stale
% costs in the population; the reported best sees true evaluations only) and
% 1/f roulette weights, which assume f > 0. Harness: the integer redraw becomes
% lb + (ub-lb)*rand; offspring are clamped before their first evaluation (no-op
% on a box symmetric about 0); both batches spend popsize FEs as released, though
% 62-66 % of them repeat an already evaluated point on cec14 D = 10; the
% roulette's rejection loop (endless at f = 0) falls back to its direct draw.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ewa(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    popsize = 50;
    Keep    = 2;
    alpha   = 0.98;
    beta    = 1;
    gamma   = 0.9;
    pcross  = 1;
    pmutate = 0.01;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Init.m: one row per worm, duplicates cleared before the first evaluation
    chrom = zeros(popsize, dim);
    for i = 1:popsize
        chrom(i, :) = lb + span .* rand(1, dim);
    end
    chrom = clear_dups(chrom, lb, span);

    m = min(popsize, maxFE);
    chrom = chrom(1:m, :);
    [cost, FE] = calculate_fitness(chrom', problem, FE);
    cost = cost(:)';

    bsf  = inf;
    bsfx = chrom(1, :);
    for e = 1:m
        if cost(e) < bsf
            bsf  = cost(e);
            bsfx = chrom(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, chrom, cost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    [cost, ord] = sort(cost, 'ascend');
    chrom = chrom(ord, :);
    bestChome = chrom(1, :);   % set once from the initial population, as released

    while FE < maxFE
        chromKeep = chrom(1:Keep, :);
        costKeep  = cost(1:Keep);
        X = chrom;   % generation-start population, the source of the mutation mean
        InverseCost = 1 ./ cost;
        sumInv = sum(InverseCost);
        beta = gamma * beta;

        for i = 1:popsize
            % Reproduction 1
            OP = ub + lb - alpha * chrom(i, :);

            % Reproduction 2; r1 is drawn for every worm, as released
            r1 = round(popsize * rand + 0.5);
            if i >= Keep + 1
                r2 = round(popsize * rand + 0.5);
                sel = roulette_mate(InverseCost, sumInv, popsize);
                Parent1 = chrom(r2, :);
                Parent2 = chrom(sel, :);
                child = zeros(1, dim);
                for j = 1:dim
                    if pcross > rand
                        if rand > 0.5
                            child(j) = Parent1(j);
                        else
                            child(j) = Parent2(j);
                        end
                    else
                        if rand > 0.5
                            child(j) = Parent2(j);
                        else
                            child(j) = Parent1(j);
                        end
                    end
                end
            else
                child = chrom(r1, :);
            end

            chrom(i, :) = beta * OP + (1 - beta) * child;
        end

        % First evaluation: the offspring
        chrom = min(max(chrom, lb), ub);
        m = min(popsize, maxFE - FE);
        [fu, FE] = calculate_fitness(chrom(1:m, :)', problem, FE);
        fu = fu(:)';
        cost = inf(1, popsize);
        cost(1:m) = fu;
        for g = 1:m
            if fu(g) < bsf
                bsf  = fu(g);
                bsfx = chrom(g, :);
            end
            ec = FE - m + g;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, chrom(1:m, :), cost(1:m), population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);

        % "Cauchy mutation" as released: w accumulates across the worms of one generation
        cauchy_w = bestChome;
        for ind = Keep + 1:popsize
            for parnum = 1:dim
                if pmutate > rand
                    cauchy_w(parnum) = sum(X(:, parnum)) / popsize;
                end
            end
            chrom(ind, :) = (cauchy_w + bestChome) / 2;
        end

        % Second evaluation: the whole population again, the unchanged elites included, as released
        chrom = min(max(chrom, lb), ub);
        m = min(popsize, maxFE - FE);
        [fu, FE] = calculate_fitness(chrom(1:m, :)', problem, FE);
        fu = fu(:)';
        cost(1:m) = fu;
        for g = 1:m
            if fu(g) < bsf
                bsf  = fu(g);
                bsfx = chrom(g, :);
            end
            ec = FE - m + g;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, chrom(1:m, :), cost(1:m), population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);

        % Elitism: the previous generation's elites replace the worst
        for k = 1:Keep
            chrom(popsize - k + 1, :) = chromKeep(k, :);
            cost(popsize - k + 1)     = costKeep(k);
        end

        % ClearDups: the redrawn gene's cost is deliberately not recomputed, as released
        chrom = clear_dups(chrom, lb, span);
        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsfx;
end

% Roulette for the second parent, as released: Random_Cost is drawn above the best's weight
function sel = roulette_mate(InverseCost, sumInv, popsize)
    if InverseCost(1) >= 0 && ~isnan(sumInv) && ~(sumInv > InverseCost(1))
        sel = direct_draw(InverseCost);   % the release would loop forever here
        return;
    end
    Random_Cost = 0;
    tries = 0;
    while InverseCost(1) >= Random_Cost
        tries = tries + 1;
        if tries > 1e4
            sel = direct_draw(InverseCost);
            return;
        end
        Random_Cost = rand * sumInv;
    end
    Select_Cost = InverseCost(1);
    sel = 1;
    while Select_Cost < Random_Cost
        sel = sel + 1;
        if sel >= popsize
            break;
        end
        Select_Cost = Select_Cost + InverseCost(sel);
    end
end

% The distribution the rejection loop samples: worms 2..N in proportion to 1/f
function sel = direct_draw(InverseCost)
    c = cumsum(InverseCost(2:end));
    k = find(c >= rand * c(end), 1);
    if isempty(k)
        k = numel(c);
    end
    sel = 1 + k;
end

% Simon's ClearDups: permutation-invariant duplicate test, one gene redrawn per duplicate
function chrom = clear_dups(chrom, lb, span)
    [n, dim] = size(chrom);
    srt = sort(chrom, 2);
    for i = 1:n - 1
        js = i + find(all(srt(i + 1:n, :) == srt(i, :), 2))';
        for j = js
            p = ceil(dim * rand);
            chrom(j, p) = lb(p) + span(p) * rand;
            srt(j, :) = sort(chrom(j, :));
        end
    end
end

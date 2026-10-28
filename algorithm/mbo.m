% ----------------------------------------------------------------------- %
% Monarch Butterfly Optimization (MBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize     = 50        % Butterflies; Land 1 holds NP1 = ceil(p*50) = 21
%   partition   = 5/12      % p: Land 1 share and migration ratio
%   period      = 1.2       % peri: the migration draw is r = rand*peri
%   BAR         = 5/12      % Butterfly adjusting rate
%   maxStepSize = 1         % S_max; walk weight alpha = S_max/t^2, t the generation
%   Maxgen      = 50        % Levy step count StepSize = ceil(exprnd(2*Maxgen))
%   Keep        = 2         % Elites carried over, replacing the worst two
%
% Algorithm Concept:
%   - The sorted population is split: Land 1 is the best NP1 butterflies, Land 2
%     the remaining NP2
%   - Migration builds NP1 offspring gene by gene: from a random Land 1 butterfly
%     when rand*peri <= p, else from a random Land 2 butterfly
%   - Butterfly adjusting builds NP2 offspring: gene k is the best's with
%     probability p, else a Land 2 value plus alpha*(dx_k - 0.5) when rand > BAR
%   - dx_k is a Levy-type walk, the sum of StepSize Cauchy draws tan(pi*u)
%   - Offspring replace the whole population; the two previous elites then
%     replace its worst two
%
% Reference:
% Gai-Ge Wang, Suash Deb, Zhihua Cui,
% Monarch butterfly optimization,
% Neural Computing and Applications 31(7) (2019) 1995-2014.
% https://doi.org/10.1007/s00521-015-1923-y
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release, github.com/ggw0122/MBO commit 2b21337
% (MBO_FEs_V2.m, 2 Nov 2021; the same code as File Exchange 101400 v1.0.1).
% Kept as released: the adjusting operator's Land 2 branch copies gene 1 of the
% random butterfly into every gene k (chrom(1), not chrom(k)); the Levy step
% count keeps Maxgen = 50 from the release's Init although its FE loop runs
% maxFe/50 generations (tied to the budget it would cost O(maxFe) draws per gene);
% exprnd is drawn as -mu*log(rand), the toolbox's own formula. Harness: the
% release's problem file initialises in [lb, ub + 1) and evaluates it unclamped,
% here lb + (ub-lb)*rand; ClearDups' integer redraw becomes continuous; the
% reported best is updated at every evaluation, not once per generation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mbo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    popsize     = 50;
    Keep        = 2;
    maxStepSize = 1.0;
    partition   = 5/12;
    period      = 1.2;
    BAR         = partition;
    Maxgen      = 50;
    NP1 = ceil(partition * popsize);
    NP2 = popsize - NP1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % One row per butterfly, duplicates cleared before the first evaluation, as in Init.m
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

    GenIndex = 1;
    while FE < maxFE
        chromKeep = chrom(1:Keep, :);
        costKeep  = cost(1:Keep);
        Pop1 = chrom(1:NP1, :);
        Pop2 = chrom(NP1 + 1:popsize, :);
        % Recorded population: offspring overwrite the generation-start slots as they are evaluated
        Xrec = chrom;
        frec = cost;

        % Migration operator: two draws per gene, (r1, donor index), butterfly by butterfly
        U  = rand(2, dim, NP1);
        r1 = reshape(U(1, :, :), dim, NP1)' * period;
        u2 = reshape(U(2, :, :), dim, NP1)';
        fromLand1 = r1 <= partition;
        col = repmat(1:dim, NP1, 1);
        Land1 = zeros(NP1, dim);
        r2 = round(NP1 * u2(fromLand1) + 0.5);
        Land1(fromLand1) = Pop1(sub2ind([NP1, dim], r2, col(fromLand1)));
        r3 = round(NP2 * u2(~fromLand1) + 0.5);
        Land1(~fromLand1) = Pop2(sub2ind([NP2, dim], r3, col(~fromLand1)));

        Land1 = min(max(Land1, lb), ub);
        m = min(NP1, maxFE - FE);
        [fu, FE] = calculate_fitness(Land1(1:m, :)', problem, FE);
        fu = fu(:)';
        cost1 = inf(1, NP1);
        cost1(1:m) = fu;
        for g = 1:m
            if fu(g) < bsf
                bsf  = fu(g);
                bsfx = Land1(g, :);
            end
            Xrec(g, :) = Land1(g, :);
            frec(g)    = fu(g);
            ec = FE - m + g;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        % Butterfly adjusting operator
        Land2 = zeros(NP2, dim);
        scale = maxStepSize / (GenIndex^2);
        for k2 = 1:NP2
            StepSize = ceil(-(2 * Maxgen) * log(rand));   % exprnd(2*Maxgen)
            delataX = sum(tan(pi * rand(StepSize, dim)), 1);   % LevyFlight: same draws as rand(1,StepSize) per gene
            for j = 1:dim
                if rand <= partition
                    Land2(k2, j) = chrom(1, j);
                else
                    r4 = round(NP2 * rand + 0.5);
                    Land2(k2, j) = Pop2(r4, 1);   % gene 1 for every j, as released
                    if rand > BAR
                        Land2(k2, j) = Land2(k2, j) + scale * (delataX(j) - 0.5);
                    end
                end
            end
        end

        Land2 = min(max(Land2, lb), ub);
        m = min(NP2, maxFE - FE);
        [fu, FE] = calculate_fitness(Land2(1:m, :)', problem, FE);
        fu = fu(:)';
        cost2 = inf(1, NP2);
        cost2(1:m) = fu;
        for g = 1:m
            if fu(g) < bsf
                bsf  = fu(g);
                bsfx = Land2(g, :);
            end
            Xrec(NP1 + g, :) = Land2(g, :);
            frec(NP1 + g)    = fu(g);
            ec = FE - m + g;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        % CombinePopulation, then the previous elites replace the worst
        chrom = [Land1; Land2];
        cost  = [cost1, cost2];
        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);
        for k3 = 1:Keep
            chrom(popsize - k3 + 1, :) = chromKeep(k3, :);
            cost(popsize - k3 + 1)     = costKeep(k3);
        end
        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);

        GenIndex = GenIndex + 1;
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsfx;
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

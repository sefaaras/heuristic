% ----------------------------------------------------------------------- %
% Elephant Herding Optimization (EHO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50              % Elephants
%   numClan = 5               % Clans, 10 elephants each
%   alpha   = 0.5             % Scale of the pull toward the matriarch, Eq. (1)
%   beta    = 0.1             % Matriarch factor on the clan centre, Eq. (2)
%   Keep    = 2               % Elites carried over, replacing the worst two
%
% Algorithm Concept:
%   - The sorted elephants are dealt into the clans in turn, so clan c's
%     matriarch is the c-th best elephant overall
%   - Clan updating, Eq. (1): x + alpha*(x_matriarch - x).*r, r ~ U(0,1) per gene
%   - Matriarch, Eq. (2): x_new = beta*x_center, x_center the clan mean, Eq. (3)
%   - Separating, Eq. (4): each clan's worst is replaced by
%     x_min + (x_max - x_min + 1)*rand
%   - The new clans replace the population; the two previous elites then
%     replace its worst two
%
% Reference:
% Gai-Ge Wang, Suash Deb, Leandro dos Santos Coelho,
% Elephant Herding Optimization,
% 2015 3rd International Symposium on Computational and Business Intelligence
% (ISCBI), Bali, Indonesia, IEEE (2015) 1-5.
% https://doi.org/10.1109/ISCBI.2015.8
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release, File Exchange 53486 v1.0.0.0 (EHO_FEs_V2.m,
% 19 Feb 2016), which adds the two carried-over elites and ClearDups to the paper.
% Kept from paper and release, deliberately: Eq. (2) puts the matriarch at
% beta*x_center, a contraction toward the ORIGIN, not the clan, so the search has
% a structural centre bias (clamped back when the box excludes 0 and it falls
% out); the release fires it when sum(x_new - x) == 0. Eq. (4) draws from
% [lb, ub + 1), the +1 in problem units, then clamps: a gene lands on ub with
% probability 1/(w + 1), w its width; 1/201 on the CEC box, >= 1/2 on the 128
% CEC2020RW dims with w <= 1 (15 of 57 problems). Harness: init lb + (ub-lb)*rand,
% not the problem file's integers; ClearDups' redraw is continuous; best per FE.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = eho(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    popsize = 50;
    numClan = 5;
    nEach   = popsize / numClan;
    Keep    = 2;
    alpha   = 0.5;
    beta    = 0.1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % One row per elephant, duplicates cleared before the first evaluation, as in Init.m
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

    while FE < maxFE
        chromKeep = chrom(1:Keep, :);
        costKeep  = cost(1:Keep);
        % Recorded population: each new elephant overwrites the slot of the one it was made from
        Xrec = chrom;
        frec = cost;

        % Member j of clan c is row (j-1)*numClan + c, so row c is clan c's matriarch
        center = zeros(numClan, dim);
        for c = 1:numClan
            s = zeros(1, dim);
            for e = 1:nEach
                s = s + chrom((e - 1) * numClan + c, :);
            end
            center(c, :) = (1 / nEach) * s;   % Eq. (3), summed in the release's order
        end

        % Clan updating operator, Eq. (1); Eq. (2) for a null move, which is always the matriarch's
        newChrom = zeros(popsize, dim);
        for j = 1:nEach
            for c = 1:numClan
                row = (j - 1) * numClan + c;
                x  = chrom(row, :);
                xn = x + alpha * (chrom(c, :) - x) .* rand(1, dim);
                if sum(xn - x) == 0
                    xn = beta * center(c, :);   % scaled toward the origin, kept as released
                end
                newChrom(row, :) = xn;
            end
        end

        % Separating operator, Eq. (4): the clan's last (worst) member
        for c = 1:numClan
            newChrom((nEach - 1) * numClan + c, :) = lb + (span + 1) .* rand(1, dim);
        end

        % Clan by clan: ClearDups, clamp, evaluate, then PopSort within the clan
        newCost = inf(1, popsize);
        for c = 1:numClan
            rows = c:numClan:popsize;
            C = clear_dups(newChrom(rows, :), lb, span);
            C = min(max(C, lb), ub);
            m = min(nEach, maxFE - FE);
            [fu, FE] = calculate_fitness(C(1:m, :)', problem, FE);
            fu = fu(:)';
            for g = 1:m
                if fu(g) < bsf
                    bsf  = fu(g);
                    bsfx = C(g, :);
                end
                Xrec(rows(g), :) = C(g, :);
                frec(rows(g))    = fu(g);
                ec = FE - m + g;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Xrec, frec, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
            [fc, ord] = sort(fu, 'ascend');
            newChrom(rows, :) = C(ord, :);
            newCost(rows)     = fc;
        end
        if FE >= maxFE, break; end

        % CombineClan interleaves the sorted clans back into rows (j-1)*numClan + c
        [cost, ord] = sort(newCost, 'ascend');
        chrom = newChrom(ord, :);
        for k3 = 1:Keep
            chrom(popsize - k3 + 1, :) = chromKeep(k3, :);
            cost(popsize - k3 + 1)     = costKeep(k3);
        end
        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);
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

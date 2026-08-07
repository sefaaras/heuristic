% ----------------------------------------------------------------------- %
% Adaptive Gaining-Sharing Knowledge (AGSK)
% CEC 2020 competition runner-up
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size     = 40*D (D > 5) or 100 (D <= 5)   % Non-linear reduction to 12
%   min_pop_size = 12
%   KF_pool      = [0.1 1.0 0.5 1.0]              % Knowledge factor settings
%   KR_pool      = [0.2 0.1 0.9 0.9]              % Knowledge ratio settings
%   KW_ind       = [0.85 0.05 0.05 0.05]          % Initial setting probabilities
%   K            = U(0,1) w.p. 0.5, else randi(20)  % Per-individual knowledge rate
%
% Algorithm Concept:
%   - Two phases whose share of the dimensions shifts over the budget,
%     D_junior = ceil(D * (1 - FE/maxFE)^K_i), so early generations are junior
%     (local, nearest-neighbour) and late ones senior (elite/middle/worst)
%   - Junior phase: the two fitness-rank neighbours of the individual plus a
%     random third; senior phase: a random member of the best 5 %, of the
%     middle 90 % and of the worst 5 %
%   - Both phases flip the direction of the last difference term depending on
%     whether the individual is worse than its reference partner
%   - The four (KF, KR) settings are selected by a probability vector adapted
%     from the fitness improvement each setting produced, floored at 0.05
%   - Non-linear population size reduction, as in NL-SHADE
%
% Reference:
% Ali W. Mohamed, Anas A. Hadi, Ali K. Mohamed, Noor H. Awad,
% Evaluating the Performance of Adaptive Gaining-Sharing Knowledge Based
% Algorithm on CEC 2020 Benchmark Problems,
% 2020 IEEE Congress on Evolutionary Computation (CEC), 2020, pp. 1-8.
% https://doi.org/10.1109/CEC48606.2020.9185901
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (Adaptive_GSK_ALI.m, submission
% E-24505 in the CEC2020 archive) with its three helpers. Three reference
% defects had to be handled: bsf_solution is assigned only inside the offspring
% loop, so a run that never improves on its initial best returns an undefined
% solution (it is tracked from initialisation instead); KW_ind starts as [] and
% is filled only inside the FE < 0.1*maxFE branch, crashing on budgets that skip
% it (initialised to its documented [0.85 0.05 0.05 0.05], the same fix as
% apgsk_imode); and the junior rank lookup is O(N^2) via find() in a loop,
% untenable at 40*D individuals (the permutation is inverted once instead).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = agsk(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';      % 1 x dim
    ub    = problem.ub(:)';      % 1 x dim
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % AGSK control parameters
    KF_pool = [0.1 1.0 0.5 1.0];
    KR_pool = [0.2 0.1 0.9 0.9];

    if dim > 5
        pop_size = 40 * dim;
    else
        pop_size = 100;
    end
    max_pop_size = pop_size;
    min_pop_size = 12;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial population
    popold = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);

    [fitness, FE] = calculate_fitness(popold', problem, FE);
    fitness = fitness(:);

    bsf          = inf;
    bsf_solution = popold(1, :);
    for i = 1:pop_size
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = popold(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, popold, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Per-individual knowledge rate K: U(0,1) w.p. 0.5, else a uniform integer in [1,20]
    Kind = rand(pop_size, 1);
    K = zeros(pop_size, 1);
    K(Kind <  0.5) = rand(sum(Kind <  0.5), 1);
    K(Kind >= 0.5) = ceil(20 * rand(sum(Kind >= 0.5), 1));

    KW_ind  = [0.85 0.05 0.05 0.05];
    All_Imp = zeros(1, 4);

    % Main loop
    while FE < maxFE
        % Setting-selection probabilities
        if FE >= 0.1 * maxFE
            KW_ind = 0.95 * KW_ind + 0.05 * All_Imp;
            KW_ind = KW_ind ./ sum(KW_ind);
        end

        u = rand(pop_size, 1);
        edges = [0, cumsum(KW_ind)];
        K_rand_ind = ones(pop_size, 1);
        for s = 2:4
            K_rand_ind(u > edges(s) & u <= edges(s + 1)) = s;
        end
        KF = KF_pool(K_rand_ind)';
        KR = KR_pool(K_rand_ind)';

        % Junior / senior dimension split
        D_junior = ceil(dim * (1 - FE / maxFE) .^ K);

        pop = popold;
        [~, indBest] = sort(fitness, 'ascend');

        [Rg1, Rg2, Rg3] = juniorR1R2R3(indBest);
        [R1, R2, R3]    = seniorR1R2R3(indBest);

        self = (1:pop_size)';

        % Junior gaining-sharing phase
        Junior = zeros(pop_size, dim);
        worse = fitness(self) > fitness(Rg3);
        if any(worse)
            Junior(worse, :) = pop(worse, :) + KF(worse, ones(1, dim)) .* ...
                (pop(Rg1(worse), :) - pop(Rg2(worse), :) + pop(Rg3(worse), :) - pop(worse, :));
        end
        better = ~worse;
        if any(better)
            Junior(better, :) = pop(better, :) + KF(better, ones(1, dim)) .* ...
                (pop(Rg1(better), :) - pop(Rg2(better), :) + pop(better, :) - pop(Rg3(better), :));
        end

        % Senior gaining-sharing phase
        Senior = zeros(pop_size, dim);
        worse = fitness(self) > fitness(R2);
        if any(worse)
            Senior(worse, :) = pop(worse, :) + KF(worse, ones(1, dim)) .* ...
                (pop(R1(worse), :) - pop(worse, :) + pop(R2(worse), :) - pop(R3(worse), :));
        end
        better = ~worse;
        if any(better)
            Senior(better, :) = pop(better, :) + KF(better, ones(1, dim)) .* ...
                (pop(R1(better), :) - pop(R2(better), :) + pop(better, :) - pop(R3(better), :));
        end

        Junior = boundConstraint(Junior, pop, lu);
        Senior = boundConstraint(Senior, pop, lu);

        % Dimension masks
        junior_mask = rand(pop_size, dim) <= (D_junior(:, ones(1, dim)) ./ dim);
        senior_mask = ~junior_mask;

        junior_mask = junior_mask & (rand(pop_size, dim) <= KR(:, ones(1, dim)));
        senior_mask = senior_mask & (rand(pop_size, dim) <= KR(:, ones(1, dim)));

        ui = pop;
        ui(junior_mask) = Junior(junior_mask);
        ui(senior_mask) = Senior(senior_mask);

        [children_fitness, FE] = calculate_fitness(ui', problem, FE);
        children_fitness = children_fitness(:);

        for i = 1:pop_size
            if children_fitness(i) < bsf
                bsf          = children_fitness(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - pop_size + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Improvement of each (KF, KR) setting
        dif = abs(fitness - children_fitness);
        child_better = (fitness > children_fitness);

        All_Imp = zeros(1, 4);
        for s = 1:4
            sel = child_better & (K_rand_ind == s);
            if any(sel)
                All_Imp(s) = sum(dif(sel));
            end
        end

        if sum(All_Imp) ~= 0
            All_Imp = All_Imp ./ sum(All_Imp);
            [~, Imp_Ind] = sort(All_Imp);
            for imp_i = 1:length(All_Imp) - 1
                All_Imp(Imp_Ind(imp_i)) = max(All_Imp(Imp_Ind(imp_i)), 0.05);
            end
            All_Imp(Imp_Ind(end)) = 1 - sum(All_Imp(Imp_Ind(1:end-1)));
        else
            All_Imp(:) = 1 / length(All_Imp);
        end

        % Selection
        [fitness, sel_idx] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(sel_idx == 2, :) = ui(sel_idx == 2, :);

        % Non-linear population size reduction
        r = FE / maxFE;
        plan_pop_size = round((min_pop_size - max_pop_size) * (r .^ (1 - r)) + max_pop_size);

        if pop_size > plan_pop_size
            reduction_ind_num = pop_size - plan_pop_size;
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = pop_size - min_pop_size;
            end

            pop_size = pop_size - reduction_ind_num;
            for rr = 1:reduction_ind_num
                [~, ord] = sort(fitness, 'ascend');
                worst_ind = ord(end);
                popold(worst_ind, :) = [];
                fitness(worst_ind)   = [];
                K(worst_ind)         = [];
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function [R1, R2, R3] = juniorR1R2R3(indBest)
% Junior partners: each individual's two rank neighbours plus a distinct random third
    pop_size = length(indBest);

    rnk = zeros(pop_size, 1);
    rnk(indBest) = (1:pop_size)';    % individual -> its rank

    prev = max(rnk - 1, 1);
    next = min(rnk + 1, pop_size);
    prev(rnk == 1)        = 2;
    next(rnk == 1)        = 3;
    prev(rnk == pop_size) = pop_size - 2;
    next(rnk == pop_size) = pop_size - 1;

    R1 = indBest(prev);
    R2 = indBest(next);

    R0 = (1:pop_size)';
    R3 = floor(rand(pop_size, 1) * pop_size) + 1;
    for i = 1:1000
        pos = (R3 == R2) | (R3 == R1) | (R3 == R0);
        if ~any(pos)
            break;
        end
        R3(pos) = floor(rand(sum(pos), 1) * pop_size) + 1;
        if i == 1000
            error('agsk:juniorR3', 'Cannot generate R3 in 1000 iterations');
        end
    end
end

function [R1, R2, R3] = seniorR1R2R3(indBest)
% Senior partners: one random member each from the best 5 %, middle 90 % and worst 5 %
    pop_size = length(indBest);
    lo = round(pop_size * 0.05);
    hi = round(pop_size * 0.95);

    best   = indBest(1:max(lo, 1));
    middle = indBest(max(lo, 1) + 1:max(hi, max(lo, 1) + 1));
    worst  = indBest(min(hi, pop_size - 1) + 1:end);

    % max(...,1) guards a rand() that lands exactly on 0, which would index 0
    R1 = best(max(ceil(length(best)     * rand(pop_size, 1)), 1));
    R2 = middle(max(ceil(length(middle) * rand(pop_size, 1)), 1));
    R3 = worst(max(ceil(length(worst)   * rand(pop_size, 1)), 1));
end

function vi = boundConstraint(vi, pop, lu)
% L-SHADE bound handling: violating component moved to the parent/bound midpoint
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

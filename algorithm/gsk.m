% ----------------------------------------------------------------------- %
% Gaining-Sharing Knowledge Based Algorithm (GSK)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 100               % Population size
%   K  = 10                      % Knowledge rate (exponent of the phase schedule)
%   KF = 0.5                     % Knowledge factor (step size)
%   KR = 0.9                     % Knowledge ratio (per-dimension update probability)
%   G_Max = fix(maxFE/pop_size)  % Planned number of generations
%
% Algorithm Concept:
%   - The DIMENSIONS of each individual, not the population, are split between
%     two phases
%   - JUNIOR: learn from immediate fitness-rank NEIGHBOURS -- the better and
%     worse neighbour plus one random individual, moving towards that random one
%     if it is fitter and away from it otherwise
%   - SENIOR: learn from the whole ranked society split into top 5%, middle 90%
%     and bottom 5%, moving towards the elite and away from the bottom
%   - The SPLIT MOVES OVER TIME: junior dimensions number ceil(D*(1-g/G_Max)^K),
%     so a run starts local and neighbour-driven and ends global and
%     elite-driven, with K setting how fast the transition happens
%   - KR gates each phase's dimensions, so only a fraction update per generation
%   - Violating components take the midpoint between parent and violated bound
%
% Reference:
% Ali Wagdy Mohamed, Anas A. Hadi, Ali Khater Mohamed,
% Gaining-sharing knowledge based algorithm for solving optimization problems:
% a novel nature-inspired algorithm,
% International Journal of Machine Learning and Cybernetics, vol. 11,
% pp. 1501-1529, 2020.
% https://doi.org/10.1007/s13042-019-01053-x
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (GSK.m with Gained_Shared_Junior and
% Gained_Shared_Senior_R1R2R3.m). GSK is the fixed-parameter ancestor of agsk.m
% in this folder, so the same two reference defects are handled the same way:
% bsf_solution is assigned only inside the offspring loop and is undefined if
% nothing ever improves (the best initial individual seeds it), and the junior
% partner selection is O(N^2) via find() in a loop (the rank permutation is
% inverted once instead, identically).
% Because the framework counts the initial population in the budget, g can pass
% G_Max and make (1-g/G_Max)^K complex; g/G_Max is clamped to 1, the limit the
% schedule heads to anyway.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gsk(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters
    pop_size = 100;
    KF = 0.5;
    KR = 0.9;
    K  = 10 * ones(pop_size, 1);
    G_Max = max(1, fix(maxFE / pop_size));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    popold = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
    pop    = popold;

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    % bsf_solution is seeded with the best initial individual; the reference leaves it undefined
    bsf          = inf;
    bsf_solution = pop(1, :);
    for i = 1:pop_size
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    g = 0;

    % Main loop
    while FE < maxFE
        g = g + 1;

        % Dimension split; the ratio is clamped so the power stays real
        ratio = min(g / G_Max, 1);
        D_junior = ceil(dim * (1 - ratio) .^ K);
        pop = popold;

        [~, indBest] = sort(fitness, 'ascend');
        [Rg1, Rg2, Rg3] = juniorR1R2R3(indBest);
        [R1, R2, R3]    = seniorR1R2R3(indBest);

        R0 = (1:pop_size)';

        % Junior phase: learn from the two rank neighbours
        junior = zeros(pop_size, dim);
        ind = fitness(R0) > fitness(Rg3);
        if any(ind)
            junior(ind, :) = pop(ind, :) + KF * ( ...
                pop(Rg1(ind), :) - pop(Rg2(ind), :) + pop(Rg3(ind), :) - pop(ind, :));
        end
        ind = ~ind;
        if any(ind)
            junior(ind, :) = pop(ind, :) + KF * ( ...
                pop(Rg1(ind), :) - pop(Rg2(ind), :) + pop(ind, :) - pop(Rg3(ind), :));
        end

        % Senior phase: learn from the ranked society
        senior = zeros(pop_size, dim);
        ind = fitness(R0) > fitness(R2);
        if any(ind)
            senior(ind, :) = pop(ind, :) + KF * ( ...
                pop(R1(ind), :) - pop(ind, :) + pop(R2(ind), :) - pop(R3(ind), :));
        end
        ind = ~ind;
        if any(ind)
            senior(ind, :) = pop(ind, :) + KF * ( ...
                pop(R1(ind), :) - pop(R2(ind), :) + pop(ind, :) - pop(R3(ind), :));
        end

        junior = boundConstraint(junior, pop, lu);
        senior = boundConstraint(senior, pop, lu);

        % Dimension masks: phase split first, then the KR gate
        junior_mask = rand(pop_size, dim) <= (D_junior(:, ones(1, dim)) ./ dim);
        senior_mask = ~junior_mask;

        junior_mask = junior_mask & (rand(pop_size, dim) <= KR);
        senior_mask = senior_mask & (rand(pop_size, dim) <= KR);

        ui = pop;
        ui(junior_mask) = junior(junior_mask);
        ui(senior_mask) = senior(senior_mask);

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

        % Selection
        [fitness, better] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(better == 2, :) = ui(better == 2, :);
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
            error('gsk:juniorR3', 'Cannot generate R3 in 1000 iterations');
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

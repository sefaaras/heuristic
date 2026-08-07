% ----------------------------------------------------------------------- %
% Harmony Search (HS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   HMS  = 10                    % Harmony memory size
%   HMCR = 0.9                   % Harmony memory considering rate
%   PAR  = 0.1                   % Pitch adjusting rate
%   bw   = 0.02 * (ub - lb)      % Bandwidth (fret width)
%
% Algorithm Concept:
%   - A new candidate is assembled VARIABLE BY VARIABLE from different members
%     of the population, so there is no notion of a parent at all
%   - For each variable of a new harmony: with probability HMCR copy it from a
%     RANDOMLY CHOSEN memory row (not the best one), then with probability PAR
%     nudge it by +/- bw; otherwise draw it uniformly from the whole range
%   - HMCR is therefore how much the memory is trusted, and 1-HMCR a permanent
%     per-variable random-restart rate
%   - Since variables are drawn independently, one harmony can recombine k
%     memory rows at once, which is why HS behaves like an estimation-of-
%     distribution method on separable problems
%   - Exactly ONE harmony is produced per iteration and replaces the WORST row
%     if better, so the memory improves monotonically and very gradually
%
% Reference:
% Zong Woo Geem, Joong Hoon Kim, G. V. Loganathan,
% A New Heuristic Optimization Algorithm: Harmony Search,
% Simulation, vol. 76, no. 2, pp. 60-68, 2001.
% https://doi.org/10.1177/003754970107600201
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author MATLAB release could be located, so this is implemented from the
% paper, with Yarpiz's ypea_hs.m as a structural cross-check for the parameter
% values. Two of that template's choices are deliberately not followed: it
% generates 20 harmonies per iteration and merges them, where the published
% algorithm improvises ONE and compares it with the worst memory row; and it
% multiplies the bandwidth by 0.99 every iteration, which is tuned to its own
% 100-iteration default and would freeze pitch adjustment within the first few
% percent of the budgets used here. Geem's published HS holds bw fixed -- the
% varying-bw variants are Mahdavi et al.'s later work -- so it is held fixed.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hs(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    HMS  = 10;
    HMCR = 0.9;
    PAR  = 0.1;
    bw   = 0.02 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial harmony memory
    HM = repmat(lb, HMS, 1) + rand(HMS, D) .* repmat(span, HMS, 1);

    [f, FE] = calculate_fitness(HM', problem, FE);
    f = f(:);

    bsf  = inf;
    bsfx = HM(1, :);
    for i = 1:HMS
        if f(i) < bsf
            bsf  = f(i);
            bsfx = HM(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, HM, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop: one improvised harmony per iteration
    while FE < maxFE
        % Start from a fresh random harmony, then overwrite what the memory-considering rate claims
        x = lb + rand(1, D) .* span;

        useHM = rand(1, D) <= HMCR;
        if any(useHM)
            src = randi(HMS, 1, D);                     % an independent row per variable
            idx = find(useHM);
            x(idx) = HM(sub2ind([HMS D], src(idx), idx));

            adj = idx(rand(1, numel(idx)) <= PAR);      % pitch adjustment
            if ~isempty(adj)
                x(adj) = x(adj) + bw(adj) .* (2 * rand(1, numel(adj)) - 1);
            end
        end

        x = min(max(x, lb), ub);

        [fx, FE] = calculate_fitness(x', problem, FE);
        fx = fx(1);

        if fx < bsf
            bsf  = fx;
            bsfx = x;
        end

        % Replace the worst memory row if the new harmony beats it
        [fworst, iworst] = max(f);
        if fx < fworst
            HM(iworst, :) = x;
            f(iworst)     = fx;
        end

        if FE >= 1 && FE <= maxFE
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, HM, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

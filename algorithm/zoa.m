% ----------------------------------------------------------------------- %
% Zebra Optimization Algorithm (ZOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30    % Population size (zebras)
%   R            = 0.1   % Escape-step scale of strategy S1
%
% Algorithm Concept:
%   - Phase 1 (foraging, exploration): move toward the pioneer zebra PZ, the
%     best one, x + r.*(PZ - I*x) with r per dimension and I in {1, 2}
%   - Phase 2 (defence): one Ps and one attacked zebra AZ per iteration decide
%     for the whole herd between the two strategies below
%   - S1, Ps < 0.5 (escaping a lion): x + R*(2r - 1)*(1 - t/T).*x, a step
%     proportional to x that vanishes at the end of the run
%   - S2 (attacking other predators): x + r.*(AZ - I*x)
%   - Greedy acceptance after each phase, ties accepted
%
% Reference:
% Eva Trojovska, Mohammad Dehghani, Pavel Trojovsky,
% Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm for
% Solving Optimization Algorithm,
% IEEE Access 10 (2022) 49445-49473.
% https://doi.org/10.1109/ACCESS.2022.3172789
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 122942 v1.0.0 (ZOA.m,
% 7 Jan 2023); the paper prints R = 0.01 and a strict < in the greedy
% acceptance where the release uses R = 0.1 and <=.
% A candidate equal to its zebra or to an earlier one of its batch is not
% re-evaluated, though the release spends an FE on it (1.2-6.7 % of its FEs on
% cec14 D = 10): mostly the pioneer or the attacked zebra moving with I = 1,
% and every S1 move at t = T. T counts the release's 2N evaluation slots per
% iteration, the skipped ones included, recounted from the budget every
% iteration, so t/T still reaches 1 on the last one. PZ is refreshed when an
% iteration starts, as released; the reported best at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = zoa(problem)

    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents = 30;
    R = 0.1;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = initialization(SearchAgents, dim, upperbound, lowerbound);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);

    for e = 1:min(SearchAgents, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    skipped = 0;   % candidates not evaluated so far: null moves and in-batch repeats
    t = 0;
    while FE < maxFE
        t = t + 1;
        % Two evaluation slots per zebra per iteration, as released; T >= t holds while FE < maxFE
        T = ceil((maxFE - SearchAgents + skipped) / (2 * SearchAgents));

        % The pioneer zebra PZ is refreshed once per iteration, as in the release
        [best, location] = min(fit);
        if t == 1 || best < fbest
            fbest = best;
            PZ = X(location, :);
        end

        % Phase 1: foraging behaviour
        Xnew = X;
        for i = 1:SearchAgents
            I = round(1 + rand);
            Xnew(i, :) = X(i, :) + rand(1, dim) .* (PZ - I .* X(i, :));   % Eq. (3)
            Xnew(i, :) = max(Xnew(i, :), lowerbound);
            Xnew(i, :) = min(Xnew(i, :), upperbound);
        end
        rows = find(any(Xnew ~= X, 2))';   % a candidate identical to its zebra is not re-evaluated
        [U, ufirst, grp] = unique(Xnew(rows, :), 'rows', 'stable');   % nor one repeated earlier in this batch
        skipped = skipped + SearchAgents - size(U, 1);
        if ~isempty(rows)
            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(:)';
            for j = 1:numel(rows)
                i = rows(j);
                fnew = fu(grp(j));
                if fnew <= fit(i)   % Eq. (4)
                    X(i, :) = Xnew(i, :);
                    fit(i) = fnew;
                    if fnew < best_fitness
                        best_fitness = fnew;
                        best_solution = Xnew(i, :);
                    end
                end
                ec = FE - numel(fu) + grp(j);
                if ufirst(grp(j)) == j && ec <= maxFE   % one sample per evaluation
                    curve(ec) = best_fitness;
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
        if FE >= maxFE, break; end

        % Phase 2: defence strategies against predators
        Ps = rand;
        k = randperm(SearchAgents, 1);
        AZ = X(k, :);   % a copy: the attacked zebra's own move this phase does not shift AZ
        Xnew = X;
        for i = 1:SearchAgents
            if Ps < 0.5
                Xnew(i, :) = X(i, :) + R * (2 * rand(1, dim) - 1) * (1 - t / T) .* X(i, :);   % Eq. (5), S1
            else
                I = round(1 + rand(1, 1));
                Xnew(i, :) = X(i, :) + rand(1, dim) .* (AZ - I .* X(i, :));   % Eq. (5), S2
            end
            Xnew(i, :) = max(Xnew(i, :), lowerbound);
            Xnew(i, :) = min(Xnew(i, :), upperbound);
        end
        rows = find(any(Xnew ~= X, 2))';
        [U, ufirst, grp] = unique(Xnew(rows, :), 'rows', 'stable');
        skipped = skipped + SearchAgents - size(U, 1);
        if ~isempty(rows)
            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(:)';
            for j = 1:numel(rows)
                i = rows(j);
                fnew = fu(grp(j));
                if fnew <= fit(i)   % Eq. (6)
                    X(i, :) = Xnew(i, :);
                    fit(i) = fnew;
                    if fnew < best_fitness
                        best_fitness = fnew;
                        best_solution = Xnew(i, :);
                    end
                end
                ec = FE - numel(fu) + grp(j);
                if ufirst(grp(j)) == j && ec <= maxFE   % one sample per evaluation
                    curve(ec) = best_fitness;
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Initialization
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

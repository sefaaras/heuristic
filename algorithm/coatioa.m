% ----------------------------------------------------------------------- %
% Coati Optimization Algorithm (COA)
% Stored as coatioa; the acronym COA collides with coa (Coyote)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30   % Population size (coatis); the first half is on the tree
%
% Algorithm Concept:
%   - Phase 1, first half (on the tree): attack the iguana, i.e. the best coati,
%     x + r*(iguana - I*x) with I in {1, 2}
%   - Phase 1, second half (on the ground): the iguana falls to a random point;
%     move toward it when it is better, away from it otherwise
%   - Phase 2 (escaping predators): x + (1 - 2r)*(lb/t + r*(ub - lb)/t), a
%     random step that shrinks with the iteration count t
%   - Greedy acceptance after each move
%
% Reference:
% Mohammad Dehghani, Zeinab Montazeri, Eva Trojovska, Pavel Trojovsky,
% Coati Optimization Algorithm: A new bio-inspired metaheuristic algorithm
% for solving optimization problems,
% Knowledge-Based Systems 259 (2023) 110011.
% https://doi.org/10.1016/j.knosys.2022.110011
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 116965 v1.0.0 (COA.m,
% 30 Aug 2022). Reference defects kept: the ground iguana draws one scalar r,
% so it lies on the box diagonal, and the Phase 2 candidate is clamped to the
% local box [lb/t, ub/t] itself, which for lb < 0 < ub shrinks onto the origin
% instead of around the coati; the global clamp after it is a harness addition.
% That clamp sends most late candidates to one corner, so 40-84 % of the
% release's Phase 2 FEs repeat a point (cec14 D = 10): a candidate equal to its
% coati or to an earlier one of its batch is not re-evaluated, and a budget
% buys more iterations. The iguanas count toward the reported best, updated at
% every evaluation; the guiding best is refreshed once per iteration.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = coatioa(problem)

    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents = 30;
    half = SearchAgents / 2;

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

    t = 0;
    while FE < maxFE
        t = t + 1;

        % The iguana of the tree phase is the best coati, refreshed once per iteration as in the release
        [best, location] = min(fit);
        if t == 1 || best < fbest
            fbest = best;
            Xbest = X(location, :);
        end

        % Phase 1, first half: hunting the iguana on the tree
        Xnew = X;
        for i = 1:half
            I = round(1 + rand(1, 1));
            Xnew(i, :) = X(i, :) + rand(1, 1) .* (Xbest - I .* X(i, :));   % Eq. (4)
            Xnew(i, :) = max(Xnew(i, :), lowerbound);
            Xnew(i, :) = min(Xnew(i, :), upperbound);
        end
        rows = find(any(Xnew ~= X, 2))';   % a candidate identical to its coati is not re-evaluated
        [U, ufirst, grp] = unique(Xnew(rows, :), 'rows', 'stable');   % nor one repeated earlier in this batch
        if ~isempty(rows)
            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(:)';
            for j = 1:numel(rows)
                i = rows(j);
                fnew = fu(grp(j));
                if fnew < fit(i)   % Eq. (7)
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

        % Phase 1, second half: the release draws I and r after evaluating the iguana; evaluations use no RNG
        iguana = zeros(half, dim);
        I_g = zeros(1, half);
        r_g = zeros(1, half);
        for j = 1:half
            iguana(j, :) = lowerbound + rand .* (upperbound - lowerbound);   % Eq. (5), one scalar rand
            I_g(j) = round(1 + rand(1, 1));
            r_g(j) = rand(1, 1);
        end
        [f_iguana, FE] = calculate_fitness(iguana', problem, FE);
        f_iguana = f_iguana(:)';
        for j = 1:half
            % The iguana never joins the band, but a better one is reported
            if f_iguana(j) < best_fitness
                best_fitness = f_iguana(j);
                best_solution = iguana(j, :);
            end
            ec = FE - half + j;
            if ec <= maxFE
                curve(ec) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        Xnew = X;
        for j = 1:half
            i = half + j;
            if fit(i) > f_iguana(j)   % Eq. (6)
                Xnew(i, :) = X(i, :) + r_g(j) .* (iguana(j, :) - I_g(j) .* X(i, :));
            else
                Xnew(i, :) = X(i, :) + r_g(j) .* (X(i, :) - iguana(j, :));
            end
            Xnew(i, :) = max(Xnew(i, :), lowerbound);
            Xnew(i, :) = min(Xnew(i, :), upperbound);
        end
        rows = find(any(Xnew ~= X, 2))';
        [U, ufirst, grp] = unique(Xnew(rows, :), 'rows', 'stable');
        if ~isempty(rows)
            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(:)';
            for j = 1:numel(rows)
                i = rows(j);
                fnew = fu(grp(j));
                if fnew < fit(i)   % Eq. (7)
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

        % Phase 2: escaping from predators inside the local bounds of Eqs. (9)-(10)
        LO_LOCAL = lowerbound / t;
        HI_LOCAL = upperbound / t;
        Xnew = X;
        for i = 1:SearchAgents
            Xnew(i, :) = X(i, :) + (1 - 2 * rand) .* (LO_LOCAL + rand(1, 1) .* (HI_LOCAL - LO_LOCAL));   % Eq. (8)
            Xnew(i, :) = max(Xnew(i, :), LO_LOCAL);
            Xnew(i, :) = min(Xnew(i, :), HI_LOCAL);
            Xnew(i, :) = max(Xnew(i, :), lowerbound);   % harness: the local box leaves [lb, ub] when lb > 0 or ub < 0
            Xnew(i, :) = min(Xnew(i, :), upperbound);
        end
        rows = find(any(Xnew ~= X, 2))';
        [U, ufirst, grp] = unique(Xnew(rows, :), 'rows', 'stable');
        if ~isempty(rows)
            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(:)';
            for j = 1:numel(rows)
                i = rows(j);
                fnew = fu(grp(j));
                if fnew < fit(i)   % Eq. (11)
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

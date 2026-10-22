% ----------------------------------------------------------------------- %
% Spotted Hyena Optimizer (SHO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 30          % Population size (hyenas)
%   h            = 5 -> 0      % Control parameter, linear over the budget, Eq. (5)
%   M            = U(0.5, 1)   % Fitness margin above the best that sizes the cluster
%
% Algorithm Concept:
%   - Elite ranking: the N best of the previous population and the previous
%     elite, so the ranking runs one iteration behind the evaluations
%   - Cluster size NOH (Eq. 9): elite hyenas whose fitness is within best + M
%   - Encircling (Eqs. 6-7): P_k - E*|B*P_k - x| per coordinate, E = 2h*r1 - h,
%     B = 2*r2, summed over the NOH best and divided by NOH + 1 (Eq. 10)
%   - P_k is the FIRST coordinate of the k-th elite hyena for every coordinate;
%     an empty cluster (NOH = 0) sends every hyena to the origin
%   - Hyenas are clamped to the box just before they are evaluated
%
% Reference:
% Gaurav Dhiman, Vijay Kumar,
% Spotted hyena optimizer: A novel bio-inspired based metaheuristic technique
% for engineering applications,
% Advances in Engineering Software 114 (2017) 48-70.
% https://doi.org/10.1016/j.advengsoft.2017.05.014
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 73887 v2.0.0 (sho.m,
% noh.m, init.m, 6 Jun 2020); seeded runs are bit-identical to it. Kept as
% released, all departing from the paper: sorted_population(k) indexes the
% matrix linearly, so every coordinate is pulled toward the first coordinate of
% the k-th best hyena; Eq. (10) divides by NOH + 1, not by N, and NOH = 0 (no
% other elite hyena within M of the best: iteration 1, and recurring) sends every
% hyena to the origin; the elite merges the previous iteration's population, so
% the last round's evaluations never rank. T = ceil(maxFe/N) evaluation rounds
% and Max_iterations = T + 1 as in the release's loop, so the last evaluated move
% has h = 10/(T + 1); the best is tracked at every evaluation, not the elite's.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sho(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    T = max(1, ceil(maxFE / N));   % evaluation rounds of N hyenas
    Max_iterations = T + 1;        % the release's loop evaluates Max_iterations - 1 rounds

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % The release evaluates the initial hyenas inside its loop, after a clamp that is a no-op here
    hyena_pos = initialization(N, dim, ub, lb);
    nev = min(N, maxFE);
    [hyena_fitness, FE] = calculate_fitness(hyena_pos(1:nev, :)', problem, FE);
    hyena_fitness = hyena_fitness(:)';

    fbest = Inf;
    Xbest = hyena_pos(1, :);
    for e = 1:nev
        if hyena_fitness(e) < fbest
            fbest = hyena_fitness(e);
            Xbest = hyena_pos(e, :);
        end
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, hyena_pos(1:nev, :), hyena_fitness, population_history, fitness_history, ...
            history_index, maxFE);
    end

    Xrec = hyena_pos;   % position at which each entry of hyena_fitness was measured
    Iteration = 1;
    while FE < maxFE
        if Iteration == 1
            [fitness_sorted, FS] = sort(hyena_fitness);
            sorted_population = hyena_pos(FS, :);
        else
            % Elite merge with the PREVIOUS round's population, as released
            double_population = [pre_population; best_hyenas];
            double_fitness = [pre_fitness best_hyena_fitness];
            [double_fitness_sorted, FS] = sort(double_fitness);
            double_sorted_population = double_population(FS, :);
            fitness_sorted = double_fitness_sorted(1:N);
            sorted_population = double_sorted_population(1:N, :);
        end
        best_hyenas = sorted_population;
        best_hyena_fitness = fitness_sorted;

        NOH = noh(best_hyena_fitness);   % Eq. (9)

        pre_population = hyena_pos;
        pre_fitness = hyena_fitness;

        a = 5 - Iteration * ((5) / Max_iterations);   % Eq. (5)
        % r1, r2 per (hyena i, coordinate j, cluster member k) in the release's i, j, k loop order
        rr = rand(2, NOH, dim, N);
        CV = zeros(N, dim);
        for k = 1:NOH
            r1 = reshape(rr(1, k, :, :), dim, N)';
            r2 = reshape(rr(2, k, :, :), dim, N)';
            Var1 = 2 * a * r1 - a;
            Var2 = 2 * r2;
            % sorted_population(k) is the release's linear index: row k, column 1
            distance_to_hyena = abs(Var2 * sorted_population(k) - hyena_pos);   % Eq. (6)
            HYE = sorted_population(k) - Var1 .* distance_to_hyena;            % Eq. (7)
            CV = CV + HYE;                                                       % Eq. (8)
        end
        hyena_pos = CV / (NOH + 1);   % Eq. (10) as released
        Iteration = Iteration + 1;

        % The release clamps each hyena just before evaluating it at the top of the next iteration
        hyena_pos = min(max(hyena_pos, lb), ub);
        nev = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(hyena_pos(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for i = 1:nev
            Xrec(i, :) = hyena_pos(i, :);
            hyena_fitness(i) = fnew(i);
            if fnew(i) < fbest
                fbest = fnew(i);
                Xbest = hyena_pos(i, :);
            end
            ec = FE - nev + i;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, hyena_fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Cluster size: hyenas 2..N whose fitness is within a U(0.5, 1) margin of the best
function X = noh(best_hyena_fitness)
    M = (1 - 0.5) .* rand(1, 1) + 0.5;
    M = M + best_hyena_fitness(1);
    X = 0;
    for i = 2:numel(best_hyena_fitness)
        if M >= best_hyena_fitness(i)
            X = X + 1;
        end
    end
end

% Initialization, one column per dimension as in the release
function Pos = initialization(N, dim, ub, lb)
    Pos = zeros(N, dim);
    for i = 1:dim
        Pos(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

% ----------------------------------------------------------------------- %
% Mayfly Optimization Algorithm (MOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 20, nPopf = 20      % Male / female population sizes
%   g = 0.8 (gdamp = 1)        % Inertia weight (and damping)
%   a1 = 1.0, a2 = 1.5, a3 = 1.5   % Learning coefficients
%   beta = 2                   % Distance-sight coefficient
%   dance = 5, fl = 1          % Nuptial dance / random flight
%   nc = 20, nm, mu = 0.01     % Mating / mutation parameters
%
% Algorithm Concept:
%   - Inspired by the flight and mating behaviour of mayflies
%   - Males perform a nuptial dance; females fly towards males; offspring
%     are produced through crossover and mutation
%
% Reference:
% Konstantinos Zervoudakis, Stelios Tsafarakis,
% A mayfly optimization algorithm,
% Computers & Industrial Engineering 145 (2020) 106559
% https://doi.org/10.1016/j.cie.2020.106559
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = moa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    LowerBound = problem.lb;
    UpperBound = problem.ub;
    maxFE = problem.maxFe;

    ProblemSize = [1 dim];

    nPop = 20;
    nPopf = 20;
    g = 0.8;
    gdamp = 1;
    a1 = 1.0;
    a2 = 1.5; a3 = 1.5;
    beta = 2;
    dance = 5;
    fl = 1;
    dance_damp = 0.8;
    fl_damp = 0.99;
    nc = 20;
    nm = round(0.05 * nPop);
    mu = 0.01;
    VelMax = 0.1 * (UpperBound - LowerBound);
    VelMin = -VelMax;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nPop, dim);
    fitness_history = zeros(history_size, nPop);
    history_index = 1;

    % Initialization
    empty_mayfly.Position = [];
    empty_mayfly.Cost = [];
    empty_mayfly.Velocity = [];
    empty_mayfly.Best.Position = [];
    empty_mayfly.Best.Cost = [];
    Mayfly = repmat(empty_mayfly, nPop, 1);    % Males
    Mayflyf = repmat(empty_mayfly, nPopf, 1);  % Females
    GlobalBest.Cost = inf;
    GlobalBest.Position = zeros(1, dim);

    for i = 1:nPop
        Mayfly(i).Position = unifrnd(LowerBound, UpperBound, ProblemSize);
        Mayfly(i).Velocity = zeros(ProblemSize);
        [Mayfly(i).Cost, FE] = calculate_fitness(Mayfly(i).Position', problem, FE);
        Mayfly(i).Best.Position = Mayfly(i).Position;
        Mayfly(i).Best.Cost = Mayfly(i).Cost;
        if Mayfly(i).Best.Cost < GlobalBest.Cost
            GlobalBest = Mayfly(i).Best;
        end
    end
    for i = 1:nPopf
        Mayflyf(i).Position = unifrnd(LowerBound, UpperBound, ProblemSize);
        Mayflyf(i).Velocity = zeros(ProblemSize);
        [Mayflyf(i).Cost, FE] = calculate_fitness(Mayflyf(i).Position', problem, FE);
    end

    [population_history, fitness_history, history_index, curve] = record_block(...
        0, FE, maxFE, GlobalBest.Cost, Mayfly, curve, ...
        population_history, fitness_history, history_index, sampling_interval, history_size);

    % Mayfly Main Loop
    while FE < maxFE
        FE_before = FE;

        for i = 1:nPopf
            % Update Females
            e = unifrnd(-1, +1, ProblemSize);
            rmf = (Mayfly(i).Position - Mayflyf(i).Position);
            if Mayflyf(i).Cost > Mayfly(i).Cost
                Mayflyf(i).Velocity = g * Mayflyf(i).Velocity ...
                    + a3 * exp(-beta .* rmf.^2) .* (Mayfly(i).Position - Mayflyf(i).Position);
            else
                Mayflyf(i).Velocity = g * Mayflyf(i).Velocity + fl * (e);
            end
            Mayflyf(i).Velocity = max(Mayflyf(i).Velocity, VelMin);
            Mayflyf(i).Velocity = min(Mayflyf(i).Velocity, VelMax);
            Mayflyf(i).Position = Mayflyf(i).Position + Mayflyf(i).Velocity;
            Mayflyf(i).Position = max(Mayflyf(i).Position, LowerBound);
            Mayflyf(i).Position = min(Mayflyf(i).Position, UpperBound);
            [Mayflyf(i).Cost, FE] = calculate_fitness(Mayflyf(i).Position', problem, FE);
        end

        for i = 1:nPop
            % Update Males
            rpbest = (Mayfly(i).Best.Position - Mayfly(i).Position);
            rgbest = (GlobalBest.Position - Mayfly(i).Position);
            e = unifrnd(-1, +1, ProblemSize);
            if Mayfly(i).Cost > GlobalBest.Cost
                Mayfly(i).Velocity = g * Mayfly(i).Velocity ...
                    + a1 * exp(-beta .* rpbest.^2) .* (Mayfly(i).Best.Position - Mayfly(i).Position) ...
                    + a2 * exp(-beta .* rgbest.^2) .* (GlobalBest.Position - Mayfly(i).Position);
            else
                Mayfly(i).Velocity = g * Mayfly(i).Velocity + dance * (e);
            end
            Mayfly(i).Velocity = max(Mayfly(i).Velocity, VelMin);
            Mayfly(i).Velocity = min(Mayfly(i).Velocity, VelMax);
            Mayfly(i).Position = Mayfly(i).Position + Mayfly(i).Velocity;
            Mayfly(i).Position = max(Mayfly(i).Position, LowerBound);
            Mayfly(i).Position = min(Mayfly(i).Position, UpperBound);
            [Mayfly(i).Cost, FE] = calculate_fitness(Mayfly(i).Position', problem, FE);
            if Mayfly(i).Cost < Mayfly(i).Best.Cost
                Mayfly(i).Best.Position = Mayfly(i).Position;
                Mayfly(i).Best.Cost = Mayfly(i).Cost;
                if Mayfly(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = Mayfly(i).Best;
                end
            end
        end

        [~, SortMayflies] = sort([Mayfly.Cost]);
        Mayfly = Mayfly(SortMayflies);
        [~, SortMayflies] = sort([Mayflyf.Cost]);
        Mayflyf = Mayflyf(SortMayflies);

        % MATE
        MayflyOffspring = repmat(empty_mayfly, nc / 2, 2);
        for k = 1:nc / 2
            i1 = k;
            i2 = k;
            p1 = Mayfly(i1);
            p2 = Mayflyf(i2);
            [MayflyOffspring(k, 1).Position, MayflyOffspring(k, 2).Position] = Crossover(p1.Position, p2.Position, LowerBound, UpperBound);
            [MayflyOffspring(k, 1).Cost, FE] = calculate_fitness(MayflyOffspring(k, 1).Position', problem, FE);
            if MayflyOffspring(k, 1).Cost < GlobalBest.Cost
                GlobalBest = MayflyOffspring(k, 1);
            end
            [MayflyOffspring(k, 2).Cost, FE] = calculate_fitness(MayflyOffspring(k, 2).Position', problem, FE);
            if MayflyOffspring(k, 2).Cost < GlobalBest.Cost
                GlobalBest = MayflyOffspring(k, 2);
            end
            MayflyOffspring(k, 1).Best.Position = MayflyOffspring(k, 1).Position;
            MayflyOffspring(k, 1).Best.Cost = MayflyOffspring(k, 1).Cost;
            MayflyOffspring(k, 1).Velocity = zeros(ProblemSize);
            MayflyOffspring(k, 2).Best.Position = MayflyOffspring(k, 2).Position;
            MayflyOffspring(k, 2).Best.Cost = MayflyOffspring(k, 2).Cost;
            MayflyOffspring(k, 2).Velocity = zeros(ProblemSize);
        end
        MayflyOffspring = MayflyOffspring(:);

        % Mutation
        MutMayflies = repmat(empty_mayfly, nm, 1);
        for k = 1:nm
            i = randi([1 nPop]);
            p = MayflyOffspring(i);
            MutMayflies(k).Position = Mutate(p.Position, mu, LowerBound, UpperBound);
            [MutMayflies(k).Cost, FE] = calculate_fitness(MutMayflies(k).Position', problem, FE);
            if MutMayflies(k).Cost < GlobalBest.Cost
                GlobalBest = MutMayflies(k);
            end
            MutMayflies(k).Best.Position = MutMayflies(k).Position;
            MutMayflies(k).Best.Cost = MutMayflies(k).Cost;
            MutMayflies(k).Velocity = zeros(ProblemSize);
        end

        % Create merged population
        MayflyOffspring = [MayflyOffspring
            MutMayflies]; %#ok
        split = round((size(MayflyOffspring, 1)) / 2);
        newmayflies = MayflyOffspring(1:split);
        Mayfly = [Mayfly
            newmayflies]; %#ok
        newmayflies = MayflyOffspring(split + 1:size(MayflyOffspring, 1));
        Mayflyf = [Mayflyf
            newmayflies]; %#ok
        [~, SortMayflies] = sort([Mayfly.Cost]);
        Mayfly = Mayfly(SortMayflies);
        Mayfly = Mayfly(1:nPop);         % Keep best males
        [~, SortMayflies] = sort([Mayflyf.Cost]);
        Mayflyf = Mayflyf(SortMayflies);
        Mayflyf = Mayflyf(1:nPopf);       % Keep best females

        g = g * gdamp;
        dance = dance * dance_damp;
        fl = fl * fl_damp;

        % Record convergence curve and history
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, GlobalBest.Cost, Mayfly, curve, ...
            population_history, fitness_history, history_index, sampling_interval, history_size);
    end

    best_solution = GlobalBest.Position;
    best_fitness = GlobalBest.Cost;

end

%% --- Record helper for a block's FE range ---
function [pop_hist, fit_hist, hist_idx, curve] = record_block(FE_before, FE, maxFE, bestcost, Mayfly, curve, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    nPop = numel(Mayfly);
    pos = reshape([Mayfly.Position], [], nPop)';   % nPop x dim
    cost = [Mayfly.Cost];                          % 1 x nPop
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bestcost;
            [pop_hist, fit_hist, hist_idx] = record_history(...
                eval_count, pos, cost, pop_hist, fit_hist, hist_idx, sampling_interval, history_size);
        end
    end
end

%% --- Crossover ---
function [off1, off2] = Crossover(x1, x2, LowerBound, UpperBound)
    L = unifrnd(0, 1, size(x1));
    off1 = L .* x1 + (1 - L) .* x2;
    off2 = L .* x2 + (1 - L) .* x1;
    off1 = max(off1, LowerBound); off1 = min(off1, UpperBound);
    off2 = max(off2, LowerBound); off2 = min(off2, UpperBound);
end

%% --- Mutation ---
function y = Mutate(x, mu, LowerBound, UpperBound)
    nVar = numel(x);
    nmu = ceil(mu * nVar);
    j = randsample(nVar, nmu);
    sigma(1:nVar) = 0.1 * (UpperBound - LowerBound);
    y = x;
    y(j) = x(j) + sigma(j) .* randn(size(x(j)));
    y = max(y, LowerBound); y = min(y, UpperBound);
end

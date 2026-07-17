% ----------------------------------------------------------------------- %
% Child Drawing Development Optimization (CDDO) for unconstrained problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 50    % Population size (children)
%   LR = 0.01    % Child level (learning) rate
%   SR = 0.9     % Child skill rate
%   PS = 10      % Pattern matrix size
%   CR = 0.1     % Creativity rate
%
% Algorithm Concept:
%   - Simulates a child's cognitive/drawing development; drawings improve via
%     learning/skill rates and the Golden Ratio (GR), steered by personal
%     best and the global best, with a learnt-pattern matrix.
%
% Reference:
% Sabat Abdulhameed, Tarik A. Rashid,
% Child Drawing Development Optimization Algorithm Based on Child's Cognitive
% Development,
% Arabian Journal for Science and Engineering 47 (2022) 1337-1351.
% https://doi.org/10.1007/s13369-021-05928-6
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = cddo(problem)

    % Extract problem parameters
    lb = problem.lb;
    ub = problem.ub;
    nVar = problem.dimension;
    maxFE = problem.maxFe;

    VarSize = [1 nVar];
    VarMin = round(lb);
    VarMax = round(ub);
    nPop = 50;
    MaxIt = ceil(maxFE / nPop);
    LR = 0.01;
    SR = 0.9;
    PS = 10;
    CR = 0.1;
    ncount = 0; %#ok<NASGU>

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nPop, nVar);
    fitness_history = zeros(history_size, nPop);
    history_index = 1;

    %% Initialization
    empty_child.Drawing = [];
    empty_child.GR = [];
    empty_child.Cost = [];
    empty_child.Best.Drawing = [];
    empty_child.Best.Cost = [];

    GlobalBest.Cost = inf;

    child = repmat(empty_child, nPop, 1);
    CP = repmat(empty_child, PS, 1);

    for i = 1:nPop
        p1 = randi([1 nVar]); %#ok<NASGU>
        p2 = randi([1 nVar]);
        p3 = randi([1 nVar]);

        child(i).Drawing = unifrnd(VarMin, VarMax, VarSize);
        child(i).GR = child(i).Drawing(1, p2) + child(i).Drawing(1, p3) / child(i).Drawing(1, p2);
        [child(i).Cost, FE] = calculate_fitness(child(i).Drawing', problem, FE);
        child(i).Best.Drawing = child(i).Drawing;
        child(i).Best.Cost = child(i).Cost;
        if child(i).Best.Cost < GlobalBest.Cost
            GlobalBest = child(i).Best;
        end
        if FE <= maxFE
            curve(FE) = GlobalBest.Cost;
        end
    end

    [population_history, fitness_history, history_index] = record_children(...
        child, nPop, nVar, 1, min(FE, maxFE), population_history, fitness_history, ...
        history_index, sampling_interval, history_size);

    [~, SortOrder] = sort([child.Cost]);
    for o = 1:PS
        CP(o) = child(SortOrder(o));
    end
    k = randi([1 PS]);

    for it = 1:MaxIt
        if FE >= maxFE, break; end
        FE_before = FE;

        for i = 1:nPop
            T = randi([VarMin(1) VarMax(1)], 1);
            p1 = randi([1 nVar]);
            p2 = randi([1 nVar]); %#ok<NASGU>
            p3 = randi([1 nVar]); %#ok<NASGU>

            if (child(i).Drawing(1, p1) <= T)
                child(i).Drawing = child(i).GR ...
                    + SR * rand(VarSize) .* (child(i).Best.Drawing - child(i).Drawing) ...
                    + LR * rand(VarSize) .* (GlobalBest.Drawing - child(i).Drawing);
                LR = randi([6, 10]) / 10;
                SR = randi([6, 10]) / 10;
            elseif (1.5 < child(i).GR < 2)
                child(i).Drawing = CP(k).Drawing - CR * child(i).Best.Drawing;
                LR = randi([0, 5]) / 10;
                SR = randi([0, 5]) / 10;
                ncount = ncount + 1; %#ok<NASGU>
            end

            [child(i).Cost, FE] = calculate_fitness(child(i).Drawing', problem, FE);

            if child(i).Cost < child(i).Best.Cost
                child(i).Best.Drawing = child(i).Drawing;
                child(i).Best.Cost = child(i).Cost;
                if child(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = child(i).Best;
                    CP(k) = child(i);
                end
            end

            if FE <= maxFE
                curve(FE) = GlobalBest.Cost;
            end
            if FE >= maxFE, break; end
        end

        [population_history, fitness_history, history_index] = record_children(...
            child, nPop, nVar, FE_before + 1, min(FE, maxFE), population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    curve(min(FE, maxFE):end) = GlobalBest.Cost;

    best_solution = GlobalBest.Drawing;
    best_fitness = GlobalBest.Cost;
end

%% --- Assemble children into a matrix and record over an FE block ---
function [pop_hist, fit_hist, hist_idx] = record_children(child, nPop, dim, fe_from, fe_to, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    if fe_to < fe_from, return; end
    pop = reshape([child.Drawing], dim, nPop)';
    costs = [child.Cost];
    for eval_count = fe_from:fe_to
        [pop_hist, fit_hist, hist_idx] = record_history(...
            eval_count, pop, costs, pop_hist, fit_hist, hist_idx, ...
            sampling_interval, history_size);
    end
end

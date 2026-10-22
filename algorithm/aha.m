% ----------------------------------------------------------------------- %
% Artificial Hummingbird Algorithm (AHA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop      = 50                    % Population size (hummingbirds), one food source each
%   migration = every 2*nPop iterations  % Worst source re-drawn uniformly in the box
%
% Algorithm Concept:
%   - Flight skill sets the moved dimensions: axial (one), diagonal (a random
%     2..D-1 of them) or omnidirectional (all), each with probability 1/3
%   - Guided foraging (p = 0.5): v = x_tar + a*Dir.*(x - x_tar), a ~ N(0,1), with
%     x_tar the source this bird has left unvisited longest (ties: the best)
%   - Territorial foraging: v = x + b*Dir.*x, b ~ N(0,1)
%   - Visit table: +1 per visit; a bird that improves its source becomes the
%     longest-unvisited source for every other bird
%   - Migration: every 2*nPop iterations the worst source is re-drawn at random
%   - Out-of-box coordinates are re-drawn uniformly; greedy acceptance
%
% Reference:
% Weiguo Zhao, Liying Wang, Seyedali Mirjalili,
% Artificial hummingbird algorithm: A new bio-inspired optimizer with its
% engineering applications,
% Computer Methods in Applied Mechanics and Engineering 388 (2022) 114194.
% https://doi.org/10.1016/j.cma.2021.114194
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 101133 v1.0.1 (AHA.m of
% "AHA for 50 benchmark functions", 28 Oct 2021; the 23-function copy
% differs only in whitespace); nPop = 50 as in its main.m. The release has no
% iteration-dependent schedule, so the run simply continues until maxFe and
% may stop inside an iteration. It updates its best once per iteration; here
% the reported best is updated at every evaluation (the best never guides).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = aha(problem)

    dim = problem.dimension;
    Low = problem.lb;
    Up = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Row by row, as released, so the random stream matches
    PopPos = zeros(nPop, dim);
    for i = 1:nPop
        PopPos(i, :) = rand(1, dim) .* (Up - Low) + Low;
    end
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    PopFit = PopFit(:)';

    [best_fitness, bidx] = min(PopFit);
    best_solution = PopPos(bidx, :);
    for e = 1:min(nPop, maxFE)
        curve(e) = min(PopFit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, PopPos, PopFit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    VisitTable = zeros(nPop);
    VisitTable(logical(eye(nPop))) = NaN;

    It = 0;
    while FE < maxFE
        It = It + 1;
        DirectVector = zeros(nPop, dim);

        for i = 1:nPop
            r = rand;
            if r < 1/3   % diagonal flight
                RandDim = randperm(dim);
                if dim >= 3
                    RandNum = ceil(rand * (dim - 2) + 1);
                else
                    RandNum = ceil(rand * (dim - 1) + 1);
                end
                DirectVector(i, RandDim(1:RandNum)) = 1;
            elseif r > 2/3   % omnidirectional flight
                DirectVector(i, :) = 1;
            else   % axial flight
                RandNum = ceil(rand * dim);
                DirectVector(i, RandNum) = 1;
            end

            if rand < 0.5   % guided foraging
                [MaxUnvisitedTime, TargetFoodIndex] = max(VisitTable(i, :));
                MUT_Index = find(VisitTable(i, :) == MaxUnvisitedTime);
                if length(MUT_Index) > 1
                    [~, Ind] = min(PopFit(MUT_Index));
                    TargetFoodIndex = MUT_Index(Ind);
                end

                newPopPos = PopPos(TargetFoodIndex, :) + randn * DirectVector(i, :) .* ...
                    (PopPos(i, :) - PopPos(TargetFoodIndex, :));
                newPopPos = SpaceBound(newPopPos, Up, Low);
                [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);

                if newPopFit < PopFit(i)
                    PopFit(i) = newPopFit;
                    PopPos(i, :) = newPopPos;
                    VisitTable(i, :) = VisitTable(i, :) + 1;
                    VisitTable(i, TargetFoodIndex) = 0;
                    VisitTable(:, i) = max(VisitTable, [], 2) + 1;
                    VisitTable(i, i) = NaN;
                else
                    VisitTable(i, :) = VisitTable(i, :) + 1;
                    VisitTable(i, TargetFoodIndex) = 0;
                end
            else   % territorial foraging
                newPopPos = PopPos(i, :) + randn * DirectVector(i, :) .* PopPos(i, :);
                newPopPos = SpaceBound(newPopPos, Up, Low);
                [newPopFit, FE] = calculate_fitness(newPopPos', problem, FE);

                if newPopFit < PopFit(i)
                    PopFit(i) = newPopFit;
                    PopPos(i, :) = newPopPos;
                    VisitTable(i, :) = VisitTable(i, :) + 1;
                    VisitTable(:, i) = max(VisitTable, [], 2) + 1;
                    VisitTable(i, i) = NaN;
                else
                    VisitTable(i, :) = VisitTable(i, :) + 1;
                end
            end

            if newPopFit < best_fitness
                best_fitness = newPopFit;
                best_solution = newPopPos;
            end
            curve(FE) = best_fitness;   % one evaluation per step, so FE <= maxFE here
            [population_history, fitness_history, history_index] = record_history(...
                FE, PopPos, PopFit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        if mod(It, 2 * nPop) == 0   % migration foraging
            [~, MigrationIndex] = max(PopFit);
            PopPos(MigrationIndex, :) = rand(1, dim) .* (Up - Low) + Low;
            [fm, FE] = calculate_fitness(PopPos(MigrationIndex, :)', problem, FE);
            PopFit(MigrationIndex) = fm;
            VisitTable(MigrationIndex, :) = VisitTable(MigrationIndex, :) + 1;
            VisitTable(:, MigrationIndex) = max(VisitTable, [], 2) + 1;
            VisitTable(MigrationIndex, MigrationIndex) = NaN;

            if fm < best_fitness
                best_fitness = fm;
                best_solution = PopPos(MigrationIndex, :);
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, PopPos, PopFit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Out-of-box coordinates re-drawn uniformly; D draws per call whether used or not
function X = SpaceBound(X, Up, Low)
    Dim = length(X);
    S = (X > Up) + (X < Low);
    X = (rand(1, Dim) .* (Up - Low) + Low) .* S + X .* (~S);
end

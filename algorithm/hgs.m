% ----------------------------------------------------------------------- %
% Hunger Games Search (HGS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                % Population size
%   VC2 = 0.03            % Variation control constant
%
% Algorithm Concept:
%   - Inspired by the hunger-driven activities and behaviours of animals
%   - A hunger weight for each individual biases the cooperative search
%     between exploration and exploitation
%
% Reference:
% Yutao Yang, Huiling Chen, Ali Asghar Heidari, Amir H. Gandomi,
% Hunger games search: Visions, conception, implementation, deep analysis,
% perspectives, and towards performance shifts,
% Expert Systems with Applications 177 (2021) 114864
% https://doi.org/10.1016/j.eswa.2021.114864
% ----------------------------------------------------------------------- %
% Implementation Note:
% Two divisions the reference leaves unguarded, both of which produced NaN
% positions. The hunger ratio divides by (Worstest - Destination): an Inf
% objective, which CEC2020RW RC25 really returns near its interior pole, makes
% that Inf/Inf, so the finite range is kept as the reference and a non-finite
% position is scored as maximally bad instead. The hunger weight divides by
% sumHungry, which is exactly zero once every position sits at the best -- seen
% on CEC2014 F30 at D = 100 -- and the run then filled with NaN; that case now
% takes the reference's own no-modulation branch, weight4 = 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hgs(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Max_iter = (maxFE / N) + 1;

    % initialize position
    bestPositions = zeros(1, dim);
    tempPosition = zeros(N, dim);

    Destination_fitness = inf;   % change this to -inf for maximization problems
    Worstest_fitness = -inf;
    AllFitness = inf * ones(N, 1);   % record the fitness of all positions
    VC1 = ones(N, 1);                % record the variation control of all positions

    weight3 = ones(N, dim);   % hungry weight of each position
    weight4 = ones(N, dim);   % hungry weight of each position

    X = initialization(N, dim, ub, lb);
    it = 1;

    hungry = zeros(1, size(X, 1));   % record the hungry of all positions
    count = 0;

    while FE < maxFE
        FE_before = FE;

        VC2 = 0.03;   % The variable of variation control
        sumHungry = 0;

        % Boundary control then evaluate
        for i = 1:size(X, 1)
            Flag4ub = X(i, :) > ub;
            Flag4lb = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        [AllFitness, FE] = calculate_fitness(X', problem, FE);
        AllFitness = AllFitness(:);

        [AllFitnessSorted, IndexSorted] = sort(AllFitness);
        bestFitness = AllFitnessSorted(1);
        worstFitness = AllFitnessSorted(size(X, 1));

        % update the best fitness value and best position
        if bestFitness < Destination_fitness
            bestPositions = X(IndexSorted(1), :);
            Destination_fitness = bestFitness;
            count = 0;
        end

        % isfinite: an Inf objective must not become the normalisation reference
        if isfinite(worstFitness) && worstFitness > Worstest_fitness
            Worstest_fitness = worstFitness;
        end

        for i = 1:size(X, 1)
            % calculate the variation control of all positions
            VC1(i) = sech(abs(AllFitness(i) - Destination_fitness));
            % calculate the hungry of each position
            if Destination_fitness == AllFitness(i)
                hungry(1, i) = 0;
                count = count + 1;
                tempPosition(count, :) = X(i, :);
            else
                temprand = rand();
                if ~isfinite(AllFitness(i))
                    ratio = 1;   % a non-finite objective is maximally bad, not a NaN ratio
                elseif Worstest_fitness > Destination_fitness
                    ratio = (AllFitness(i) - Destination_fitness) / (Worstest_fitness - Destination_fitness);
                else
                    ratio = 0;   % no spread yet, so nothing separates this position from the best
                end
                c = ratio * temprand * 2 * (ub - lb);
                if c < 100
                    b = 100 * (1 + temprand);
                else
                    b = c;
                end
                hungry(1, i) = hungry(1, i) + max(b);
                sumHungry = sumHungry + hungry(1, i);
            end
        end

        % calculate the hungry weight of each position
        for i = 1:size(X, 1)
            for j = 2:size(X, 2)
                weight3(i, j) = (1 - exp(-abs(hungry(1, i) - sumHungry))) * rand() * 2;
                if rand() < VC2 && sumHungry > 0
                    weight4(i, j) = hungry(1, i) * size(X, 1) / sumHungry * rand();
                else
                    weight4(i, j) = 1;
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Destination_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, AllFitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if FE >= maxFE
            break;
        end

        % Update the Position of search agents
        shrink = 2 * (1 - it / Max_iter);   % a decreases linearly from 2 to 0
        for i = 1:size(X, 1)
            if rand < VC2
                X(i, :) = X(i, j) * (1 + randn(1));
            else
                A = randi([1, count]);
                for j = 1:size(X, 2)
                    r = rand();
                    vb = 2 * shrink * r - shrink;   % [-a,a]
                    if r > VC1(i)
                        X(i, j) = weight4(i, j) * tempPosition(A, j) + vb * weight3(i, j) * abs(tempPosition(A, j) - X(i, j));
                    else
                        X(i, j) = weight4(i, j) * tempPosition(A, j) - vb * weight3(i, j) * abs(tempPosition(A, j) - X(i, j));
                    end
                end
            end
        end

        it = it + 1;
    end

    best_solution = bestPositions;
    best_fitness = Destination_fitness;

end

% Initialization Function
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

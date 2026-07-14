% ----------------------------------------------------------------------- %
% Slime Mould Algorithm (SMA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                % Population size (number of slime moulds)
%   z = 0.03              % Random-relocation probability
%
% Algorithm Concept:
%   - Inspired by the oscillation/diffusion behaviour of Physarum
%     polycephalum during foraging
%   - Adaptive fitness weights model the positive/negative feedback of
%     the propagation wave, guiding the search towards food (the optimum)
%
% Reference:
% Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili,
% Slime mould algorithm: A new method for stochastic optimization,
% Future Generation Computer Systems 111 (2020) 300-323
% https://doi.org/10.1016/j.future.2020.03.055
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = sma(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    Max_iter = ceil(maxFE / N);

    % initialize position
    bestPositions = zeros(1, dim);
    Destination_fitness = inf;      % change this to -inf for maximization problems
    AllFitness = inf * ones(N, 1);  % record the fitness of all slime mould
    weight = ones(N, dim);          % fitness weight of each slime mould

    X = initialization(N, dim, ub, lb);
    it = 1;
    lb = ones(1, dim) .* lb;   % lower boundary
    ub = ones(1, dim) .* ub;   % upper boundary
    z = 0.03;                  % parameter

    while FE < maxFE
        FE_before = FE;

        % Boundary control then evaluate the whole population
        for i = 1:N
            Flag4ub = X(i, :) > ub;
            Flag4lb = X(i, :) < lb;
            X(i, :) = (X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        [AllFitness, FE] = calculate_fitness(X', problem, FE);
        AllFitness = AllFitness(:);

        [SmellOrder, SmellIndex] = sort(AllFitness);  % Eq.(2.6)
        worstFitness = SmellOrder(N);
        bestFitness = SmellOrder(1);

        S = bestFitness - worstFitness + eps;  % plus eps to avoid denominator zero

        % calculate the fitness weight of each slime mould
        for i = 1:N
            for j = 1:dim
                if i <= (N / 2)  % Eq.(2.5)
                    weight(SmellIndex(i), j) = 1 + rand() * log10((bestFitness - SmellOrder(i)) / (S) + 1);
                else
                    weight(SmellIndex(i), j) = 1 - rand() * log10((bestFitness - SmellOrder(i)) / (S) + 1);
                end
            end
        end

        % update the best fitness value and best position
        if bestFitness < Destination_fitness
            bestPositions = X(SmellIndex(1), :);
            Destination_fitness = bestFitness;
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Destination_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, AllFitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE
            break;
        end

        a = atanh(-(it / Max_iter) + 1);  % Eq.(2.4)
        b = 1 - it / Max_iter;
        % Update the Position of search agents
        for i = 1:N
            if rand < z  % Eq.(2.7)
                X(i, :) = (ub - lb) * rand + lb;
            else
                p = tanh(abs(AllFitness(i) - Destination_fitness));  % Eq.(2.2)
                vb = unifrnd(-a, a, 1, dim);  % Eq.(2.3)
                vc = unifrnd(-b, b, 1, dim);
                for j = 1:dim
                    r = rand();
                    A = randi([1, N]);  % two positions randomly selected from population
                    B = randi([1, N]);
                    if r < p  % Eq.(2.1)
                        X(i, j) = bestPositions(j) + vb(j) * (weight(i, j) * X(A, j) - X(B, j));
                    else
                        X(i, j) = vc(j) * X(i, j);
                    end
                end
            end
        end
        it = it + 1;
    end

    best_solution = bestPositions;
    best_fitness = Destination_fitness;

end

%% --- Initialization Function ---
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

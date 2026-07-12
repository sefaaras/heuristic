% ----------------------------------------------------------------------- %
% Multi-Verse Optimizer (MVO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 60                % Population size (number of universes)
%   WEP_Max = 1           % Maximum Wormhole Existence Probability
%   WEP_Min = 0.2         % Minimum Wormhole Existence Probability
%
% Algorithm Concept:
%   - Inspired by the cosmological concepts of white holes, black holes and
%     wormholes
%   - White/black hole tunnels exchange objects between universes via a
%     roulette-wheel selection based on inflation rates (fitness)
%   - Wormholes transport objects towards the best universe using WEP and
%     Travelling Distance Rate (TDR)
%
% Reference:
% Seyedali Mirjalili, Seyed Mohammad Mirjalili, Abdolreza Hatamlou,
% Multi-Verse Optimizer: a nature-inspired algorithm for global optimization,
% Neural Computing and Applications 27 (2016) 495-513
% https://doi.org/10.1007/s00521-015-1870-7
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = mvo(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 60;                       % Population size

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    % Best universe (elite) tracking
    Best_universe = zeros(1, dim);
    Best_universe_Inflation_rate = inf;

    % Initialize the positions of universes
    Universes = initialization(N, dim, ub, lb);

    % Wormhole Existence Probability bounds
    WEP_Max = 1;
    WEP_Min = 0.2;

    % Analogue of the original time budget
    Max_time = maxFE / N;
    Time = 1;

    while FE < maxFE
        FE_before = FE;

        % WEP - Eq. (3.3)
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time);

        % Travelling Distance Rate - Eq. (3.4)
        TDR = 1 - ((Time)^(1/6) / (Max_time)^(1/6));

        % Boundary checking for all universes
        for i = 1:N
            Flag4ub = Universes(i, :) > ub;
            Flag4lb = Universes(i, :) < lb;
            Universes(i, :) = (Universes(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end

        % Evaluate inflation rates (fitness) of the universes
        [Inflation_rates, FE] = calculate_fitness(Universes', problem, FE);
        Inflation_rates = Inflation_rates(:)';   % 1 x N row

        % Elitism
        for i = 1:N
            if Inflation_rates(1, i) < Best_universe_Inflation_rate
                Best_universe_Inflation_rate = Inflation_rates(1, i);
                Best_universe = Universes(i, :);
            end
        end

        % Sort inflation rates
        [sorted_Inflation_rates, sorted_indexes] = sort(Inflation_rates);
        Sorted_universes = zeros(N, dim);
        for newindex = 1:N
            Sorted_universes(newindex, :) = Universes(sorted_indexes(newindex), :);
        end

        % Normalized inflation rates (NI in Eq. (3.1))
        normalized_sorted_Inflation_rates = normr(sorted_Inflation_rates);

        % Keep the elite universe
        Universes(1, :) = Sorted_universes(1, :);

        % Update the position of universes
        for i = 2:N   % start from 2 since the first one is the elite
            Back_hole_index = i;
            for j = 1:dim
                r1 = rand();
                if r1 < normalized_sorted_Inflation_rates(i)
                    % White hole - Eq. (3.1)
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates);
                    if White_hole_index == -1
                        White_hole_index = 1;
                    end
                    Universes(Back_hole_index, j) = Sorted_universes(White_hole_index, j);
                end

                if (size(lb, 2) == 1)
                    % Eq. (3.2) if the boundaries are all the same
                    r2 = rand();
                    if r2 < WEP
                        r3 = rand();
                        if r3 < 0.5
                            Universes(i, j) = Best_universe(1, j) + TDR * ((ub - lb) * rand + lb);
                        end
                        if r3 > 0.5
                            Universes(i, j) = Best_universe(1, j) - TDR * ((ub - lb) * rand + lb);
                        end
                    end
                end

                if (size(lb, 2) ~= 1)
                    % Eq. (3.2) if the bounds differ per variable
                    r2 = rand();
                    if r2 < WEP
                        r3 = rand();
                        if r3 < 0.5
                            Universes(i, j) = Best_universe(1, j) + TDR * ((ub(j) - lb(j)) * rand + lb(j));
                        end
                        if r3 > 0.5
                            Universes(i, j) = Best_universe(1, j) - TDR * ((ub(j) - lb(j)) * rand + lb(j));
                        end
                    end
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = Best_universe_Inflation_rate;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Universes, Inflation_rates', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        Time = Time + 1;
    end

    best_fitness = Best_universe_Inflation_rate;
    best_solution = Best_universe;

end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Roulette Wheel Selection ---
function choice = RouletteWheelSelection(weights)
    accumulation = cumsum(weights);
    p = rand() * accumulation(end);
    chosen_index = -1;
    for index = 1:length(accumulation)
        if (accumulation(index) > p)
            chosen_index = index;
            break;
        end
    end
    choice = chosen_index;
end

% ----------------------------------------------------------------------- %
% Golden Eagle Optimizer (GEO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopulationSize = 50   % Population size (eagles)
%   AttackPropensity: linspace 0.5 -> 2   (exploitation grows over time)
%   CruisePropensity: linspace 1   -> 0.5 (exploration shrinks over time)
%
% Algorithm Concept:
%   - Each eagle spirals toward a prey chosen from the flock memory
%   - Movement = attack vector (toward prey) + cruise vector (orthogonal)
%   - Flock memory keeps the best position visited by each eagle
%
% Reference:
% Abdolkarim Mohammadi-Balani, Mahmoud Dehghan Nayeri, Adel Azar,
% Mohammadreza Taghizadeh-Yazdi,
% Golden eagle optimizer: A nature-inspired metaheuristic algorithm,
% Computers & Industrial Engineering 152 (2021) 107050.
% https://doi.org/10.1016/j.cie.2020.107050
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = geo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    PopulationSize = 50;
    nvars = dim;
    Max_iter = ceil(maxFE / PopulationSize);

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, PopulationSize, dim);
    fitness_history = zeros(history_size, PopulationSize);
    history_index = 1;

    % Attack / cruise propensity sweep over the actual iteration budget
    AttackPropensity = linspace(0.5, 2,   Max_iter);
    CruisePropensity = linspace(1,   0.5, Max_iter);

    x = initialization(PopulationSize, dim, ub, lb);
    [FitnessScores, FE] = calculate_fitness(x', problem, FE);
    FitnessScores = FitnessScores(:)';

    FlockMemoryF = FitnessScores;
    FlockMemoryX = x;

    [bsf, bidx] = min(FlockMemoryF);
    best_pos = FlockMemoryX(bidx, :);

    for e = 1:PopulationSize
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, x, FitnessScores, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    t = 0;
    while FE < maxFE
        t = t + 1;
        idxProp = min(t, Max_iter);

        % Prey selection (one-to-one mapping)
        DestinationEagle = randperm(PopulationSize)';

        % Attack vector (toward memory of the destination eagle), Eq.(1)
        AttackVectorInitial = FlockMemoryX(DestinationEagle, :) - x;
        Radius = rownorm(AttackVectorInitial);          % PopulationSize x 1
        ConvergedEagles = Radius == 0;
        UnconvergedEagles = ~ConvergedEagles;

        CruiseVectorInitial = 2 .* rand(PopulationSize, nvars) - 1; % [-1,1]

        AttackVectorInitial(ConvergedEagles, :) = 0;
        CruiseVectorInitial(ConvergedEagles, :) = 0;

        % Determine constrained / free variables (Eq.4)
        for i1 = 1:PopulationSize
            if UnconvergedEagles(i1)
                vConstrained = false(1, nvars);
                nz = find(AttackVectorInitial(i1, :));
                idx = nz(randi(numel(nz)));
                vConstrained(idx) = true;
                vFree = ~vConstrained;
                CruiseVectorInitial(i1, idx) = -sum(AttackVectorInitial(i1, vFree) .* CruiseVectorInitial(i1, vFree), 2) ./ AttackVectorInitial(i1, vConstrained);
            end
        end

        % Unit vectors
        AttackVectorUnit = AttackVectorInitial ./ (rownorm(AttackVectorInitial) + eps);
        CruiseVectorUnit = CruiseVectorInitial ./ (rownorm(CruiseVectorInitial) + eps);
        AttackVectorUnit(ConvergedEagles, :) = 0;
        CruiseVectorUnit(ConvergedEagles, :) = 0;

        % Movement vectors (Eq.6)
        AttackVector = rand(PopulationSize, 1) .* AttackPropensity(idxProp) .* Radius .* AttackVectorUnit;
        CruiseVector = rand(PopulationSize, 1) .* CruisePropensity(idxProp) .* Radius .* CruiseVectorUnit;
        StepVector = AttackVector + CruiseVector;

        x = x + StepVector;

        % Enforce bounds
        lbExtended = repmat(lb, PopulationSize, 1);
        ubExtended = repmat(ub, PopulationSize, 1);
        x(x < lbExtended) = lbExtended(x < lbExtended);
        x(x > ubExtended) = ubExtended(x > ubExtended);

        [FitnessScores, FE] = calculate_fitness(x', problem, FE);
        FitnessScores = FitnessScores(:)';

        % Update flock memory (per-eagle best) and running best-so-far
        for i = 1:PopulationSize
            if FitnessScores(i) < FlockMemoryF(i)
                FlockMemoryF(i) = FitnessScores(i);
                FlockMemoryX(i, :) = x(i, :);
                if FlockMemoryF(i) < bsf
                    bsf = FlockMemoryF(i);
                    best_pos = FlockMemoryX(i, :);
                end
            end
            ec = FE - PopulationSize + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, x, FitnessScores, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

%% --- Row-wise L2 norm (column vector of length = #rows) ---
function n = rownorm(A)
    n = sqrt(sum(A.^2, 2));
end

%% --- Initialization ---
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

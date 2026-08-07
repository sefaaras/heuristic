% ----------------------------------------------------------------------- %
% Horse herd Optimization Algorithm (HOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nHorse = 50            % Population size (horses)
%   w = 1                  % Inertia weight
%   phiD = phiI = 0.02     % Fraction of the herd counted as bad / good
%   VelMax = 0.1*(ub-lb)   % Velocity clamp
%   g = 1.5, h = 0.5..1.5, s/i/d/r = 0.05..0.5   % Behaviour coefficients per age group
%
% Algorithm Concept:
%   - Horses are ranked by personal-best cost into four age groups:
%     Alpha (top 10%), Beta (30%), Gamma (60%), Delta (the rest)
%   - Each group mixes a different subset of six behaviours into its velocity:
%     grazing (own pbest), hierarchy (global best), sociability (herd mean),
%     imitation (good group), defense (away from bad group), roaming (random)
%   - Younger groups get more random and social terms, older ones converge on
%     the global best, so exploration decays with rank rather than with time
%   - Velocity is clamped and the position updated PSO-style
%
% Reference:
% Farhad MiarNaeimi, Gholamreza Azizyan, Mohsen Rashki,
% Horse herd optimization algorithm: A nature-inspired algorithm for
% high-dimensional optimization problems,
% Knowledge-Based Systems 213 (2021) 106711.
% https://doi.org/10.1016/j.knosys.2020.106711
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hoa(problem)

    dim = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    nHorse = 50;
    Max_iter = ceil(maxFE / nHorse); %#ok<NASGU>

    VelMax = 0.1 * (VarMax - VarMin);
    VelMin = -VelMax;

    w = 1;
    phiD = 0.02;
    phiI = 0.02;

    % Behavior coefficients (age groups Alpha/Beta/Gamma/Delta)
    g_Alpha = 1.50; d_Alpha = 0.5;  h_Alpha = 1.5;
    g_Beta = 1.50;  h_Beta = 0.9;   s_Beta = 0.20;  d_Beta = 0.20;
    g_Gamma = 1.50; h_Gamma = 0.50; s_Gamma = 0.10; i_Gamma = 0.30; d_Gamma = 0.10; r_Gamma = 0.05;
    g_Delta = 1.50; r_Delta = 0.10;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialization
    Position = zeros(nHorse, dim);
    for i = 1:nHorse
        Position(i, :) = unifrnd(VarMin, VarMax, [1 dim]);
    end
    Velocity = zeros(nHorse, dim);
    [Cost, FE] = calculate_fitness(Position', problem, FE);
    Cost = Cost(:);
    PBestPos = Position;
    PBestCost = Cost;
    [GlobalBestCost, gidx] = min(PBestCost);
    GlobalBestPos = PBestPos(gidx, :);

    for e = 1:nHorse
        if e <= maxFE
            curve(e) = GlobalBestCost;
            [population_history, fitness_history, history_index] = record_history(...
                e, Position, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    while FE < maxFE
        % Rank horses by personal-best cost
        [~, order] = sort(PBestCost);
        rank = zeros(nHorse, 1);
        rank(order) = 1:nHorse;

        MeanPosition = mean(PBestCost);                                   % scalar (per reference code)
        badIdx = order(max(1, round((1 - phiD) * nHorse)):nHorse);
        BadPosition = mean(PBestPos(badIdx, 1));                          % scalar
        goodIdx = order(1:max(1, round(phiI * nHorse)));
        GoodPosition = mean(PBestPos(goodIdx, 1));                        % scalar

        for i = 1:nHorse
            CC = rank(i);
            if CC <= 0.1 * nHorse
                Velocity(i, :) = + h_Alpha * rand(1, dim) .* (GlobalBestPos - Position(i, :)) ...
                                 - d_Alpha * rand(1, dim) .* (Position(i, :)) ...
                                 + g_Alpha * (0.95 + 0.1 * rand) * (PBestPos(i, :) - Position(i, :));
            elseif CC <= 0.3 * nHorse
                Velocity(i, :) = s_Beta * rand(1, dim) .* (MeanPosition - Position(i, :)) ...
                                 - d_Beta * rand(1, dim) .* (BadPosition - Position(i, :)) ...
                                 + h_Beta * rand(1, dim) .* (GlobalBestPos - Position(i, :)) ...
                                 + g_Beta * (0.95 + 0.1 * rand) * (PBestPos(i, :) - Position(i, :));
            elseif CC <= 0.6 * nHorse
                Velocity(i, :) = s_Gamma * rand(1, dim) .* (MeanPosition - Position(i, :)) ...
                                 + r_Gamma * rand(1, dim) .* (Position(i, :)) ...
                                 - d_Gamma * rand(1, dim) .* (BadPosition - Position(i, :)) ...
                                 + h_Gamma * rand(1, dim) .* (GlobalBestPos - Position(i, :)) ...
                                 + i_Gamma * rand(1, dim) .* (GoodPosition - Position(i, :)) ...
                                 + g_Gamma * (0.95 + 0.1 * rand) * (PBestPos(i, :) - Position(i, :));
            else
                Velocity(i, :) = + r_Delta * rand(1, dim) .* (Position(i, :)) ...
                                 + g_Delta * (0.95 + 0.1 * rand) * (PBestPos(i, :) - Position(i, :));
            end

            % Apply velocity limits
            Velocity(i, :) = max(Velocity(i, :), VelMin);
            Velocity(i, :) = min(Velocity(i, :), VelMax);

            % Update position
            Position(i, :) = Position(i, :) + Velocity(i, :);

            % Velocity mirror effect + position limits
            IsOutside = (Position(i, :) < VarMin | Position(i, :) > VarMax);
            Velocity(i, IsOutside) = -Velocity(i, IsOutside);
            Position(i, :) = max(Position(i, :), VarMin);
            Position(i, :) = min(Position(i, :), VarMax);
        end

        [Cost, FE] = calculate_fitness(Position', problem, FE);
        Cost = Cost(:);

        for i = 1:nHorse
            if Cost(i) < PBestCost(i)
                PBestPos(i, :) = Position(i, :);
                PBestCost(i) = Cost(i);
                if PBestCost(i) < GlobalBestCost
                    GlobalBestCost = PBestCost(i);
                    GlobalBestPos = PBestPos(i, :);
                end
            end
            ec = FE - nHorse + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = GlobalBestCost;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Position, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Coefficient decay (w = 1 -> unchanged, per reference)
        d_Alpha = d_Alpha * w; g_Alpha = g_Alpha * w;
        d_Beta = d_Beta * w; s_Beta = s_Beta * w; g_Beta = g_Beta * w;
        d_Gamma = d_Gamma * w; s_Gamma = s_Gamma * w; r_Gamma = r_Gamma * w; i_Gamma = i_Gamma * w; g_Gamma = g_Gamma * w;
        r_Delta = r_Delta * w; g_Delta = g_Delta * w;

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = GlobalBestCost;
    best_fitness = GlobalBestCost;
    best_solution = GlobalBestPos;
end

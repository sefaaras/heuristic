% ----------------------------------------------------------------------- %
% Bezier Curve-based Optimization (BCO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize = 50                                   % Population size
%   Alpha0  = 1+80*It/MaxIt+0.02*(10*It/MaxIt)^3   % Weight schedule
%   A0      = sin(pi/2*(1-It/MaxIt))               % Exploration operator
%
% Algorithm Concept:
%   - Candidate solutions are generated as points on Bezier curves whose
%     control points are built from the current individual, random peers,
%     the population mean and the best solution
%   - A > 1 (exploration): cubic (Eq. 18) or quadratic (Eq. 16) Bezier curves
%     with far-reaching control points
%   - A <= 1 (exploitation): linear Bezier interpolation towards the best
%     (Eq. 9), towards the population mean (Eq. 12) or dimension-wise
%     towards a random peer (Eq. 14)
%
% Reference:
% W. Zhao, Y. Xie, L. Wang, Z. Zhang, N. Khodadadi, S. Mirjalili,
% An effective Bezier curve-based optimization (BCO) for large-scale
% numerical problems and 3D unmanned aerial vehicle path planning with
% efficient multiple threats evasion,
% Advanced Engineering Informatics 73 (2026) 104524.
% https://doi.org/10.1016/j.aei.2026.104524
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bco(problem)

    Dim   = problem.dimension;
    Low   = problem.lb;
    Up    = problem.ub;
    maxFE = problem.maxFe;

    PopSize = 50;
    MaxIt   = max(1, ceil((maxFE - PopSize) / PopSize));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    PopPos = zeros(PopSize, Dim);
    for i = 1:PopSize
        PopPos(i, :) = rand(1, Dim) .* (Up - Low) + Low;
    end
    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    PopFit = PopFit(:);

    [BestFit, idx] = min(PopFit);
    BestPos = PopPos(idx, :);
    bsf     = BestFit;

    for eval_count = 1:PopSize
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, PopPos, PopFit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    P1 = zeros(1, Dim);

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        Alpha0 = 1 + 80 * It / MaxIt + 0.02 * (10 * It / MaxIt) ^ 3;
        A0     = sin(pi / 2 * (1 - It / MaxIt));

        for i = 1:PopSize
            if FE >= maxFE, break; end

            Alpha = 1 + (2 * rand - 1) / Alpha0;          % Eq. (10)
            A     = (0.4 + 2 * log(1 / rand)) * A0;       % Eq. (19)

            if A > 1
                if rand > 0.5
                    j = i;
                    k = i;
                    while j == i
                        j = randi(PopSize);
                    end
                    r1 = rand;
                    if r1 < 0.8                            % Eq. (17)
                        while k == i || k == j
                            k = randi(PopSize);
                        end
                        P3 = mean(PopPos) + 2 * randn(1, Dim) .* (PopPos(k, :) - PopPos(j, :));
                    else
                        if r1 > 0.9
                            while k == i || k == j
                                k = randi(PopSize);
                            end
                            P3 = PopPos(j, :) + 2 * randn * (PopPos(k, randi(Dim)) - PopPos(j, :));
                        else
                            P3 = PopPos(j, :) + 2 * randn * (Low + rand * (Up - Low));
                        end
                    end
                    P0 = PopPos(i, :);
                    P1 = PopPos(j, :);
                    P2 = PopPos(k, :);
                    NewPos = (1 - Alpha) ^ 3 * P0 + 3 * (1 - Alpha) ^ 2 * Alpha * P1 + ...
                             3 * (1 - Alpha) * Alpha ^ 2 * P2 + Alpha ^ 3 * P3;    % Eq. (18)
                else
                    j = i;
                    while i == j
                        j = randi(PopSize);
                    end
                    P0 = PopPos(i, :);
                    P1 = PopPos(j, :);
                    if rand > 0.5                          % Eq. (15)
                        P2 = PopPos(i, :) + randn * (BestPos - PopPos(j, :));
                    else
                        P2 = PopPos(i, :) + randn * (mean(PopPos) - PopPos(j, :));
                    end
                    NewPos = (1 - Alpha) ^ 2 * P0 + 2 * (1 - Alpha) * Alpha * P1 + Alpha ^ 2 * P2;  % Eq. (16)
                end
            else
                if rand > 0.5
                    j = i;
                    while i == j
                        j = randi(PopSize);
                    end
                    P0 = PopPos(i, :);
                    if rand < 0.5                          % Eq. (8)
                        P1 = BestPos + (PopPos(j, :) - PopPos(i, :)) / 2;
                    else
                        P1 = BestPos - (PopPos(j, :) + PopPos(i, :)) / 2;
                    end
                    NewPos = (1 - Alpha) * P0 + Alpha * P1;                        % Eq. (9)
                else
                    if rand > 0.5
                        P0 = PopPos(i, :);
                        if rand > 0.5                      % Eq. (11)
                            P1 = mean(PopPos) + randn * (PopPos(i, :) - mean(PopPos));
                        else
                            P1 = mean(PopPos) + randn * mean(PopPos);
                        end
                        NewPos = (1 - Alpha) .* P0 + Alpha .* P1;                  % Eq. (12)
                    else
                        P0 = PopPos(i, :);
                        for j = 1:Dim
                            if rand > 0.5                  % Eq. (13)
                                k = i;
                                while i == k
                                    k = randi(PopSize);
                                end
                                P1(j) = (1 - Alpha) * PopPos(i, j) + Alpha * PopPos(k, j);
                            else
                                P1(j) = PopPos(i, j);
                            end
                        end
                        NewPos = (1 - Alpha) * P0 + Alpha * P1;                    % Eq. (14)
                    end
                end
            end

            NewPos = SpaceBound(NewPos, Up, Low);
            [NewFit, FE] = calculate_fitness(NewPos', problem, FE);

            if NewFit < PopFit(i)
                PopFit(i)     = NewFit;
                PopPos(i, :)  = NewPos;
                if PopFit(i) < BestFit
                    BestPos = PopPos(i, :);
                    BestFit = PopFit(i);
                end
            end

            if NewFit < bsf
                bsf = NewFit;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = BestFit;
    best_solution = BestPos;
end

% Bound handling: random re-draw for violating dimensions
function X = SpaceBound(X, Up, Low)
    Dim = length(X);
    S = (X > Up) + (X < Low);
    X = (rand(1, Dim) .* (Up - Low) + Low) .* S + X .* (~S);
end

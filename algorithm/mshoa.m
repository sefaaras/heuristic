% ----------------------------------------------------------------------- %
% Mantis Shrimp Optimization Algorithm (MShOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 50   % Population size (mantis shrimps)
%   b               = 1    % Shape parameter
%
% Algorithm Concept:
%   - Each shrimp carries a polarization state in {1,2,3} that selects its
%     move for the current turn:
%       1) foraging  (Eq. 12): random navigation between the best and a peer
%       2) attack    (Eq. 14): cosine-scaled strike around the best
%       3) burrow / defense (Eq. 15): small signed contraction of the best
%   - The polarization of the next turn is decided by getPolarization, which
%     measures the angle between the old and the new position vectors and
%     compares it with the linear-horizontal, linear-vertical and circular
%     polarization references (45/90/135 degree bands with fi = 10)
%
% Reference:
% J. A. Sanchez Cortez, H. Peraza Vazquez, A. F. Pena Delgado,
% A Novel Bio-Inspired Optimization Algorithm Based on Mantis Shrimp
% Survival Tactics,
% Mathematics 2025, 13(9), 1500.
% https://doi.org/10.3390/math13091500
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mshoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 50;
    MaxIt = max(1, ceil((maxFE - SearchAgents_no) / SearchAgents_no));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Positions = repmat(lb, SearchAgents_no, 1) + ...
                repmat(ub - lb, SearchAgents_no, 1) .* rand(SearchAgents_no, dim);

    [Fitness, FE] = calculate_fitness(Positions', problem, FE);
    Fitness = Fitness(:);

    [vMin, minIdx] = min(Fitness);
    theBestVct = Positions(minIdx, :);
    bsf = vMin;

    for eval_count = 1:SearchAgents_no
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Positions, Fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    Polarization = randi([1, 3], 1, SearchAgents_no);
    x = zeros(SearchAgents_no, dim);

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        for ii = 1:size(Fitness, 1)
            if Polarization(ii) == 1
                % Strategy 1: foraging -- Eq. (12)
                r = randi(size(Fitness, 1), 1);
                while ii == r
                    r = randi(size(Fitness, 1), 1);
                end
                D = -1 + 2 .* rand;
                x(ii, :) = (theBestVct - Positions(ii, :)) + D .* (Positions(r, :) - theBestVct);

            elseif Polarization(ii) == 2
                % Strategy 2: attack -- Eq. (14)
                t = 180 + (360 - 180) .* rand;
                x(ii, :) = theBestVct .* cos(t);

            elseif Polarization(ii) == 3
                % Strategy 3: burrow / defense / shelter -- Eq. (15)
                k = 0 + (0.3 - 0) .* rand;
                bin = ((-1) .^ randi(2));
                x(ii, :) = theBestVct + rand .* bin .* k .* theBestVct;
                if bin == 1
                    t = 180 + (360 - 180) .* rand;
                    x(ii, :) = x(ii, :) .* cos(t);
                end
            end

            % Bound handling: re-draw violating dimensions
            Flag4ub = x(ii, :) > ub;
            Flag4lb = x(ii, :) < lb;
            x(ii, :) = (x(ii, :) .* (~(Flag4ub + Flag4lb))) + ...
                       (Flag4ub + Flag4lb) .* (lb + rand(1, dim) .* (ub - lb));
        end

        Polarization = getPolarization(Positions, Fitness, x, SearchAgents_no);
        Positions = x;

        [Fit, FE] = calculate_fitness(Positions', problem, FE);
        Fit = Fit(:);

        [VMin, MinIdx] = min(Fit);
        if VMin < vMin
            theBestVct = Positions(MinIdx, :);
            vMin = VMin;
        end
        Fitness = Fit;

        if vMin < bsf
            bsf = vMin;
        end
        for k = 1:SearchAgents_no
            ec = FE - SearchAgents_no + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Positions, Fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = vMin;
    best_solution = theBestVct;
end

% Polarization state of the next turn
function Polarization = getPolarization(Positions, Fitness, x, SearchAgents_no)
    fi = 10; a = 45 - fi; b = 45 + fi; c = 135 - fi; d = 135 + fi;
    k1 = randi(SearchAgents_no);
    k2 = k1 + randi(SearchAgents_no - 1);
    Positions = circshift(Positions, k1, 1);
    x = circshift(x, k2, 1);

    n = size(Fitness, 1);
    dif_1 = zeros(n, 1); Idx_Ang1 = zeros(n, 1);
    dif_2 = zeros(n, 1); Idx_Ang2 = zeros(n, 1);

    for i = 1:n
        v1 = Positions(i, :) / norm(Positions(i, :));
        v2 = x(i, :) / norm(x(i, :));
        angulo = rad2deg(acos(dot(v1, v2)));
        [LinH, LinV, PolC] = refs(angulo, a, b, c, d);
        ref = [LinH LinV PolC];
        [dif_1(i, 1), Idx_Ang1(i, 1)] = min(ref);
    end

    for i = 1:n
        angulo = randi([1, 90]);
        [LinH, LinV, PolC] = refs(angulo, a, b, c, d);
        ref = [LinH LinV PolC];
        [dif_2(i, 1), Idx_Ang2(i, 1)] = min(ref);
    end

    Polarization = zeros(1, n);
    for i = 1:n
        eyes = min(dif_1(i), dif_2(i));
        if eyes == dif_1(i)
            lightPol = Idx_Ang1(i);
        else
            lightPol = Idx_Ang2(i);
        end
        Polarization(1, i) = lightPol;
    end
end

% Linear/circular polarization references for a given angle
function [LinH, LinV, PolC] = refs(angulo, a, b, c, d)
    if angulo > 90
        LinH = 180 - angulo;
        LinV = angulo - 90;
        if angulo <= 135
            PolC = 135 - angulo;
        else
            PolC = angulo - 135;
        end
        if (angulo <= d && angulo >= c)
            PolC = 0;
        end
    else
        LinH = angulo - 0;
        LinV = 90 - angulo;
        if angulo <= 45
            PolC = 45 - angulo;
        else
            PolC = angulo - 45;
        end
        if (angulo <= b && angulo >= a)
            PolC = 0;
        end
    end
end

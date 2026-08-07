% ----------------------------------------------------------------------- %
% Kangaroo Escape Optimizer (KEO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   numKangaroos    = 50                 % Population size
%   Group_Size      = round(0.05*N)      % Random-group size for safe-area pick
%   EnergyThreshold = 0.5                % Long-jump / zig-zag switch level
%
% Algorithm Concept:
%   - Stage 1 (escape): energy level driven by a logistic chaotic map decides
%     between a long-jump escape (with the decoy-drop mask) and a zig-zag
%     escape (angular perturbation of the direction towards the best kangaroo)
%   - Stage 2 (safer areas): move towards a randomly picked kangaroo, the best
%     of a small random group, or the overall best, modulated by a decoy mask
%
% Reference:
% Sulaiman Z. Almutairi, Abdullah M. Shaheen,
% A novel kangaroo escape optimizer for parameter estimation of solar
% photovoltaic cells/modules via one, two and three-diode equivalent
% circuit modeling,
% Scientific Reports 15 (2025) 32669.
% https://doi.org/10.1038/s41598-025-19917-4
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = keo(problem)

    dim   = problem.dimension;
    Low   = problem.lb;
    Up    = problem.ub;
    maxFE = problem.maxFe;

    numKangaroos = 50;
    Group_Size      = max(1, round(0.05 * numKangaroos));
    EnergyThreshold = 0.5;

    MaxIt = max(1, ceil((maxFE - numKangaroos) / numKangaroos));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Kangeroo = repmat(Low, numKangaroos, 1) + ...
               rand(numKangaroos, dim) .* repmat(Up - Low, numKangaroos, 1);

    [KangerooFit, FE] = calculate_fitness(Kangeroo', problem, FE);
    KangerooFit = KangerooFit(:)';

    BestF = inf;
    BestX = Kangeroo(1, :);
    for i = 1:numKangaroos
        if KangerooFit(i) <= BestF
            BestF = KangerooFit(i);
            BestX = Kangeroo(i, :);
        end
    end
    bsf = BestF;

    for eval_count = 1:numKangaroos
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Kangeroo, KangerooFit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    chaotic_val = 0.7;   % logistic-map state (persists across iterations)

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        Decoy_Drop = zeros(numKangaroos, dim);   % reset every iteration

        for ii = 1:numKangaroos
            r = rand;
            if rand > 0.5
                % Stage 1: escape (long jump or zig-zag) driven by energy
                if It == 1
                    chaotic_val = 0.7;
                else
                    chaotic_val = 4 * chaotic_val * (1 - chaotic_val);
                end
                Energy_Level = (1 - rand * (It / MaxIt)) * (0.95 + 0.05 * chaotic_val);

                if Energy_Level > EnergyThreshold && rand > rand
                    % Stage 1.1: long-jump escape with the decoy drop
                    Jump = 2 * randn * Decoy_Drop(ii, :) .* Kangeroo(ii, :);
                    newKangeroo = Kangeroo(ii, :) + Jump;
                else
                    % Stage 1.2: zig-zag escape (angular perturbation)
                    beta1 = 1;              % zigzag step scale
                    theta_max_deg = 30;     % max angle in degrees
                    newKangeroo = zigzag_escape(Kangeroo(ii, :), BestX, beta1, theta_max_deg);
                end
            else
                % Stage 2: move towards a safer area
                if r < 1/3
                    Decoy_Drop(ii, :) = 1;
                elseif r < 2/3
                    Decoy_Drop(ii, :) = round(rand(1, dim));
                else
                    Decoy_Drop(ii, :) = round(rand(1, dim) .* rand(1, dim));
                end

                if It < 2 * MaxIt / 4 || rand > rand
                    Safer_Area = randi(numKangaroos);
                else
                    if rand < 0.75
                        Safe_group = randi(numKangaroos, 1, Group_Size);
                        [~, Selected_one] = min(KangerooFit(Safe_group));
                        Safer_Area = Safe_group(Selected_one);
                    else
                        [~, Safer_Area] = min(KangerooFit);
                    end
                end
                newKangeroo = Kangeroo(Safer_Area, :) + ...
                              randn * Decoy_Drop(ii, :) .* (Kangeroo(ii, :) - Kangeroo(Safer_Area, :));
            end

            % Bound check
            F_ub = newKangeroo > Up;
            F_lb = newKangeroo < Low;
            newKangeroo = (newKangeroo .* (~(F_ub + F_lb))) + Up .* F_ub + Low .* F_lb;

            [newKangerooFit, FE] = calculate_fitness(newKangeroo', problem, FE);

            if newKangerooFit < KangerooFit(ii)
                KangerooFit(ii)  = newKangerooFit;
                Kangeroo(ii, :)  = newKangeroo;
            end

            if newKangerooFit < bsf
                bsf = newKangerooFit;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Kangeroo, KangerooFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        for ii = 1:numKangaroos
            if KangerooFit(ii) < BestF
                BestF = KangerooFit(ii);
                BestX = Kangeroo(ii, :);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = BestF;
    best_solution = BestX;
end

% Zig-Zag Escape Mechanism (angular perturbation)
function Xi_new = zigzag_escape(Xi, BestX, beta1, theta_max_deg)
    D = length(Xi);
    theta_max = deg2rad(theta_max_deg);
    r = rand;
    theta = theta_max * (2 * r - 1);

    V = BestX - Xi;                 % current direction vector
    V_unit = V / norm(V + eps);     % unit direction

    if D >= 2
        rand_vec = rand(1, D);
        rand_vec = rand_vec - dot(rand_vec, V_unit) * V_unit;   % Gram-Schmidt
        U = rand_vec / norm(rand_vec + eps);
    else
        U = 1;
    end

    V_rot = cos(theta) * V + sin(theta) * U * norm(V);

    Xi_new = Xi + beta1 * sign(theta) * randn(1, D) .* V_rot;
end

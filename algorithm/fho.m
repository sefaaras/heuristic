% ----------------------------------------------------------------------- %
% Fire Hawk Optimizer (FHO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop = 25                       % Population size (fire hawks + prey)
%   HN   = randi([1 ceil(nPop/5)])  % Number of fire hawks (re-drawn each iter)
%
% Algorithm Concept:
%   - Fire hawks carry burning sticks to spread fire; each controls a
%     territory of prey chosen by proximity
%   - Fire hawk move: FHl + (r1*GB - r2*FHnear)   (GB = global best position)
%   - Prey move: toward its hawk & local safe point, or another hawk &
%     global safe point
%
% Reference:
% Mahdi Azizi, Siamak Talatahari, Amir H. Gandomi,
% Fire Hawk Optimizer: a novel metaheuristic algorithm,
% Artificial Intelligence Review 56 (2023) 287-363.
% https://doi.org/10.1007/s10462-022-10173-w
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fho(problem)

    dim = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    nPop = 25;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialization
    Pop = zeros(nPop, dim);
    for i = 1:nPop
        Pop(i, :) = unifrnd(VarMin, VarMax, [1 dim]);
    end
    [Cost, FE] = calculate_fitness(Pop', problem, FE);
    Cost = Cost(:);

    [Cost, SortOrder] = sort(Cost);
    Pop = Pop(SortOrder, :);

    bsf = Cost(1);
    best_pos = Pop(1, :);
    GB = best_pos;                 % global best position

    for e = 1:nPop
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, Pop, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    HN = randi([1 ceil(nPop / 5)]);
    SP = mean(Pop, 1);
    FHPops = Pop(1:HN, :);
    Pop2 = Pop(HN + 1:end, :);
    PopNew = assign_territories(FHPops, Pop2, HN);

    while FE < maxFE
        PopTot = [];
        for i = 1:numel(PopNew)
            PR = PopNew{i};
            FHl = FHPops(i, :);
            SPl = mean(PR, 1);

            Ir = unifrnd(0, 1, 1, 2);
            FHnear = FHPops(randi(HN), :);
            FHl_new = FHl + (Ir(1) * GB - Ir(2) * FHnear);
            FHl_new = min(max(FHl_new, VarMin), VarMax);
            PopTot = [PopTot; FHl_new]; %#ok<AGROW>

            for q = 1:size(PR, 1)
                Ir = unifrnd(0, 1, 1, 2);
                PRq_new1 = PR(q, :) + (Ir(1) * FHl - Ir(2) * SPl);
                PRq_new1 = min(max(PRq_new1, VarMin), VarMax);
                PopTot = [PopTot; PRq_new1]; %#ok<AGROW>

                Ir = unifrnd(0, 1, 1, 2);
                FHAlter = FHPops(randi(HN), :);
                PRq_new2 = PR(q, :) + (Ir(1) * FHAlter - Ir(2) * SP);
                PRq_new2 = min(max(PRq_new2, VarMin), VarMax);
                PopTot = [PopTot; PRq_new2]; %#ok<AGROW>
            end
        end

        nNew = size(PopTot, 1);
        [CostTot, FE] = calculate_fitness(PopTot', problem, FE);
        CostTot = CostTot(:);

        for k = 1:nNew
            if CostTot(k) < bsf
                bsf = CostTot(k);
                best_pos = PopTot(k, :);
            end
            ec = FE - nNew + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Pop, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Select next generation
        [CostTot, SortOrder] = sort(CostTot);
        PopTot = PopTot(SortOrder, :);
        Pop = PopTot(1:nPop, :);
        Cost = CostTot(1:nPop);

        GB = best_pos;
        HN = randi([1 ceil(nPop / 5)]);
        SP = mean(Pop, 1);
        FHPops = Pop(1:HN, :);
        Pop2 = Pop(HN + 1:end, :);
        PopNew = assign_territories(FHPops, Pop2, HN);

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

% Assign each fire hawk a territory of nearby prey (Euclidean distance)
function PopNew = assign_territories(FHPops, Pop2, HN)
    PopNew = cell(0, 1);
    for i = 1:HN
        nPop2 = size(Pop2, 1);
        if nPop2 < HN
            break
        end
        Dist = sqrt(sum((FHPops(i, :) - Pop2).^2, 2));   % full Euclidean distance
        [~, b] = sort(Dist);
        alfa = randi(nPop2);
        PopNew{i, 1} = Pop2(b(1:alfa), :);
        Pop2(b(1:alfa), :) = [];
        if isempty(Pop2)
            break
        end
    end
    if isempty(PopNew)
        PopNew{1, 1} = Pop2;
    elseif ~isempty(Pop2)
        PopNew{end, 1} = [PopNew{end, 1}; Pop2];
    end
end

% ----------------------------------------------------------------------- %
% Octopus Optimization Algorithm (OOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop  = 30                       % Population size (octopuses)
%   pl    = 12                       % Safe mating distance threshold
%   wdamp = 1.5                      % Fluid velocity damping coefficient
%   beta1 = 0.5, beta2 = 0.25, beta3 = 0.75   % Perception coefficients
%   fai   = 0.75, u = 0.5, mu = 1.0  % Attraction coefficients
%   VelMax = 0.1*(ub-lb)
%
% Algorithm Concept:
%   - Recoil mechanism: the fluid resistance F, the recoil acceleration al
%     and the mode-switch parameter A drive three movement patterns
%       |A| > 1, rand > 0.5 : perception-guided spatial scanning (Eq. 1-2)
%       |A| > 1, rand <= 0.5: recoil-driven predator exclusion (Eq. 3-4)
%       |A| <= 1            : directed acceleration towards prey (Eq. 5-6)
%   - Mating behaviour: two random permutations are paired and, depending on
%     their index distance relative to pl, one of three arm strategies is
%     applied -- grasping (Eq. 7), extending (Eq. 8) or severing (Eq. 9)
%   - Offspring and parents are merged and truncated back to nPop
%
% Reference:
% Kaiguang Wang, Laith Abualigah, Aseel Smerat, Jiahang Li, Xiangjuan Wu,
% Hao Liu, Zhongshi Shao, Seyedali Mirjalili,
% A nature recoil mechanism-based Octopus Optimization Algorithm for solving
% the global and constraint optimization from engineering structural design
% problems,
% Journal of Computational Design and Engineering (2025) qwaf139.
% https://doi.org/10.1093/jcde/qwaf139
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ooa(problem)

    Dim    = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE  = problem.maxFe;

    nPop  = 30;
    MaxIt = max(1, ceil((maxFE - nPop) / (3 * nPop)));

    pl    = 12;
    wdamp = 1.5;
    beta1 = 0.5;
    beta2 = 0.25;
    beta3 = 0.75;
    fai   = 0.75;
    u     = 0.5;
    mu    = 1.0;

    VelMax = 0.1 * (VarMax - VarMin);
    VelMin = -VelMax;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    Pos  = initializationOOA(nPop, Dim, VarMax, VarMin);
    Vel  = rand(nPop, Dim);
    Vel0 = zeros(nPop, Dim);

    [Cost, FE] = calculate_fitness(Pos', problem, FE);
    Cost = Cost(:);

    BestPos  = Pos;
    BestCost = Cost;

    [GBestCost, gi] = min(BestCost);
    GBestPos = BestPos(gi, :);
    bsf = GBestCost;

    for eval_count = 1:min(nPop, maxFE)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Pos, Cost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Main loop
    for it = 1:MaxIt
        if FE >= maxFE, break; end

        a  = 2 .* (1 - it ./ MaxIt);                                % time-varying parameter
        al = 5000 * (it ./ MaxIt) .^ (1 - sqrt(it ./ MaxIt));       % recoil acceleration
        F  = 1 ./ 2 .* wdamp .* (1 - it ./ MaxIt) .^ (1 - exp(-it ./ MaxIt));   % fluid resistance

        for i = 1:nPop
            if FE >= maxFE, break; end

            A  = 2 * a * rand - a;
            A1 = rand() .* 2 .^ (u);
            A2 = rand() .* 2 .^ (mu);
            A3 = rand() .* 2 .^ (fai);

            if abs(A) > 1
                if rand > 0.5
                    % Searching for potential prey -- Eq. (1)-(2)
                    k0 = randperm(nPop, 1);
                    r0 = abs(Pos(k0, :) - Pos(i, :));
                    ribest = abs(BestPos(i, :) - Pos(i, :));
                    Vel(i, :) = F .* Vel(i, :) ...
                        + A1 .* exp(-beta2 .* r0 .^ (1./2) .* (it ./ MaxIt)) .* (Pos(k0, :) - Pos(i, :)) ...
                        + A3 .* exp(-beta1 .* ribest .^ (1./2) .* (1 - (it ./ MaxIt) .^ (1./2))) .* (BestPos(i, :) - Pos(i, :));

                    Vel(i, :) = max(Vel(i, :), VelMin);
                    Vel(i, :) = min(Vel(i, :), VelMax);

                    Pos(i, :) = Pos(i, :) + Vel(i, :);
                    Vel0(i, :) = Vel(i, :);
                else
                    % Avoiding predators -- Eq. (3)-(4)
                    k = randperm(nPop, 2);
                    r1 = abs(Pos(k(1), :) - Pos(i, :));
                    r2 = abs(Pos(k(2), :) - Pos(i, :));
                    Vel(i, :) = F .* Vel(i, :) ...
                        + A1 .* exp(-beta2 .* r1 .^ (1./2) .* (it ./ MaxIt)) .* (Pos(k(1), :) - Pos(i, :)) ...
                        + A1 .* exp(-beta2 .* r2 .^ (1./2) .* (it ./ MaxIt)) .* (Pos(k(2), :) - Pos(i, :));

                    Vel(i, :) = max(Vel(i, :), VelMin);
                    Vel(i, :) = min(Vel(i, :), VelMax);

                    Pos(i, :) = Pos(i, :) + Vel(i, :) - abs(A) .* ((Vel(i, :)) .^ 2 - (Vel0(i, :)) .^ 2) ./ (2 .* al);
                    Vel0(i, :) = Vel(i, :);
                end
            else
                % Attacking prey -- Eq. (5)-(6)
                ribest = abs(BestPos(i, :) - Pos(i, :));
                rgbest = abs(GBestPos - Pos(i, :));
                Vel(i, :) = F .* Vel(i, :) ...
                    + A3 .* exp(-beta1 .* ribest .^ (1./2) .* (1 - (it ./ MaxIt) .^ (1./2))) .* (BestPos(i, :) - Pos(i, :)) ...
                    + A2 .* exp(-beta3 .* rgbest .^ (1./2) .* (1 - it ./ MaxIt)) .* (GBestPos - Pos(i, :));

                Vel(i, :) = max(Vel(i, :), VelMin);
                Vel(i, :) = min(Vel(i, :), VelMax);

                Pos(i, :) = Pos(i, :) + Vel(i, :) + abs(A) .* ((Vel(i, :)) .^ 2 - (Vel0(i, :)) .^ 2) ./ (2 .* al);
                Vel0(i, :) = Vel(i, :);
            end

            % Velocity mirror effect
            IsOutside = (Pos(i, :) < VarMin | Pos(i, :) > VarMax);
            Vel(i, IsOutside) = -Vel(i, IsOutside);

            Pos(i, :) = max(Pos(i, :), VarMin);
            Pos(i, :) = min(Pos(i, :), VarMax);

            [Cost(i), FE] = calculate_fitness(Pos(i, :)', problem, FE);

            if Cost(i) < BestCost(i)
                BestPos(i, :) = Pos(i, :);
                BestCost(i)   = Cost(i);
                if BestCost(i) < GBestCost
                    GBestCost = BestCost(i);
                    GBestPos  = BestPos(i, :);
                end
            end

            if Cost(i) < bsf, bsf = Cost(i); end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pos, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Mating behaviour
        kk1 = randperm(nPop);
        kk2 = randperm(nPop);
        k1x = Pos(kk1, :);        % males
        k2x = Pos(kk2, :);        % females

        kurang_7 = abs(kk1(:) - kk2(:));
        id_int7  = find(kurang_7 > 0 & kurang_7 <= pl);   % arms grasping
        id_over7 = find(kurang_7 > pl);                   % arms extending
        id_same  = find(kurang_7 == 0);                   % arms severing

        SPos  = zeros(nPop, Dim);
        SCost = inf(nPop, 1);

        for q = 1:numel(id_int7)
            r = id_int7(q);
            p = randn(1, Dim);
            SPos(r, :) = p .* k1x(r, :) + (1 - p) .* k2x(r, :);      % Eq. (7)
        end
        for q = 1:numel(id_over7)
            r = id_over7(q);
            SPos(r, :) = rand(1, Dim) .* k2x(r, :);                  % Eq. (8)
        end
        for q = 1:numel(id_same)
            r = id_same(q);
            SPos(r, :) = rand(1, Dim) .* k2x(r, :) + k1x(r, :);      % Eq. (9)
        end

        for r = 1:nPop
            if FE >= maxFE, break; end
            SPos(r, :) = max(SPos(r, :), VarMin);
            SPos(r, :) = min(SPos(r, :), VarMax);
            [SCost(r), FE] = calculate_fitness(SPos(r, :)', problem, FE);

            if SCost(r) < GBestCost
                GBestCost = SCost(r);
                GBestPos  = SPos(r, :);
            end
            if SCost(r) < bsf, bsf = SCost(r); end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pos, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Merge parents and offspring, keep the best nPop
        allPos  = [Pos;  SPos];
        allCost = [Cost; SCost];
        allVel  = [Vel;  zeros(nPop, Dim)];
        allVel0 = [Vel0; zeros(nPop, Dim)];
        allBP   = [BestPos;  SPos];
        allBC   = [BestCost; SCost];

        [~, Sortindex] = sort(allCost);
        keep = Sortindex(1:nPop);
        Pos      = allPos(keep, :);
        Cost     = allCost(keep);
        Vel      = allVel(keep, :);
        Vel0     = allVel0(keep, :);
        BestPos  = allBP(keep, :);
        BestCost = allBC(keep);

        % Closing re-evaluation
        for i = 1:nPop
            if FE >= maxFE, break; end
            Pos(i, :) = max(Pos(i, :), VarMin);
            Pos(i, :) = min(Pos(i, :), VarMax);
            [Cost(i), FE] = calculate_fitness(Pos(i, :)', problem, FE);

            if Cost(i) < BestCost(i)
                BestCost(i)   = Cost(i);
                BestPos(i, :) = Pos(i, :);
                if BestCost(i) < GBestCost
                    GBestCost = BestCost(i);
                    GBestPos  = BestPos(i, :);
                end
            end

            if Cost(i) < bsf, bsf = Cost(i); end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pos, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = GBestCost;
    best_solution = GBestPos;
end

% Initialization
function Positions = initializationOOA(nPop, Dim, ub, lb)
    Positions = zeros(nPop, Dim);
    for i = 1:Dim
        Positions(:, i) = rand(nPop, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

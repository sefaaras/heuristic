% ----------------------------------------------------------------------- %
% Fitness-Distance Balance Cheetah Optimizer (FDB-CO)
% Variant of co: FDB picks the attack partner and the prey (case C3 of the study)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 6   % Population size (cheetahs)
%   m = 2   % Number of cheetahs in an active hunting group
%   T = ceil(D/10)*60   % Hunting time
%
% Algorithm Concept:
%   - CO's three hunting strategies (search, sit-and-wait, attack), chosen per
%     dimension, and its "leave the prey and go back home" restart, unchanged
%   - Attack step z_j = x_prey,j + r_check*(x_a,j - x_j): the partner a and the
%     prey are each an FDB pick over the whole population (cases C1 + C2)
%   - FDB score = normalised fitness + normalised L1 distance to the best
%   - The best position seen is still tracked, reported and drives the restart
%
% Reference:
% Sefa Aras,
% FDB-CO: Fitness-Distance Balance Selection in the Cheetah Optimizer for
% Unconstrained and Constrained Optimization,
% Duzce University Journal of Science and Technology (Duzce Universitesi Bilim
% ve Teknoloji Dergisi), submitted 2026, no DOI assigned yet.
% Components:
%   CO - Mohammad Amin Akbari, Mohsen Zare, Rasoul Azizipanah-Abarghooee,
%     Seyedali Mirjalili, Mohamed Deriche, The cheetah optimizer: a
%     nature-inspired metaheuristic algorithm for large-scale optimization
%     problems, Scientific Reports 12 (2022) 10953.
%     https://doi.org/10.1038/s41598-022-14338-z
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169.
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% co.m plus case C3, i.e. the C1 and C2 patches together and nothing else.
% Each makes its own fitnessDistanceBalance call on the population as it
% stands: C1 replaces the group partner a of beta = x_a - x, C2 the prey of the
% attack branch. The picks coincide unless the fitness is flat, where each call
% draws its own uniform index. Fallback: a C1 pick equal to the cheetah being
% updated would zero beta, so the partner stays the original group neighbour.
% Verified bit-for-bit against the study's runs (rng(run,'twister'),
% results/dubited/data): best error / objective and violation to 17 digits and
% the stored best_solution exactly, on CEC2022 D=10/20, CEC2020RW and CEDP.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_co(problem)

    D = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    n = 6;   % population size
    m = 2;   % group size

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initial population
    popPos = zeros(n, D);
    for i = 1:n
        popPos(i, :) = lb + rand(1, D) .* (ub - lb);
    end
    [popCost, FE] = calculate_fitness(popPos', problem, FE);
    popCost = popCost(:);

    [bsf, bidx] = min(popCost);
    best_pos = popPos(bidx, :);

    % Leader (BestSol) and prey (X_best)
    BestSolPos = popPos(bidx, :);  BestSolCost = popCost(bidx);
    X_bestPos = BestSolPos;        X_bestCost = BestSolCost;
    pop1Pos = popPos;              % initial home positions
    pop1Cost = popCost;           % initial home costs

    for e = 1:n
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, popPos, popCost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    t = 0;         % hunting time counter
    it = 1;        % iteration counter
    T = ceil(D / 10) * 60;
    BestCost = [];
    Globest = [];
    xd = randperm(D);

    while FE < maxFE
        i0 = randi(n, 1, m);
        for k = 1:m
            i = i0(k);
            if k == length(i0)
                a = i0(k - 1);
            else
                a = i0(k + 1);
            end
            fdb = fitnessDistanceBalance(popPos, popCost);
            if fdb ~= i
                a = fdb;   % C1; a self pick would zero beta, so the group partner is kept
            end

            X = popPos(i, :);
            X1 = popPos(a, :);
            Xb = BestSolPos;
            Xbest = popPos(fitnessDistanceBalance(popPos, popCost), :);   % C2: own pick, not the best seen

            kk = 0;
            if i <= 2 && t > 2 && t > ceil(0.2 * T + 1) && ...
                    abs(BestCost(t - 2) - BestCost(t - ceil(0.2 * T + 1))) <= 0.0001 * Globest(t - 1)
                X = X_bestPos;
                kk = 0;
            elseif i == 3
                X = BestSolPos;
                kk = -0.1 * rand * t / T;
            else
                kk = 0.25;
            end

            if mod(it, 100) == 0 || it == 1
                xd = randperm(D);
            end
            Z = X;

            for j = xd
                r_Hat = randn;
                r1 = rand;
                if k == 1
                    alpha = 0.0001 * t / T .* (ub(j) - lb(j));
                else
                    alpha = 0.0001 * t / T * abs(Xb(j) - X(j)) + 0.001 .* round(double(rand > 0.9));
                end
                r = randn;
                r_Check = abs(r).^exp(r / 2) .* sin(2 * pi * r);
                beta = X1(j) - X(j);
                h0 = exp(2 - 2 * t / T);
                H = abs(2 * r1 * h0 - h0);
                r2 = rand;
                r3 = kk + rand;
                if r2 <= r3
                    r4 = 3 * rand;
                    if H > r4
                        Z(j) = X(j) + r_Hat.^-1 .* alpha;      % search
                    else
                        Z(j) = Xbest(j) + r_Check .* beta;      % attack
                    end
                else
                    Z(j) = X(j);                                % sit & wait
                end
            end

            % Bound check (relocate out-of-range dims)
            xx1 = find(Z < lb); Z(xx1) = lb(xx1) + rand(1, numel(xx1)) .* (ub(xx1) - lb(xx1));
            xx1 = find(Z > ub); Z(xx1) = lb(xx1) + rand(1, numel(xx1)) .* (ub(xx1) - lb(xx1));

            [NewCost, FE] = calculate_fitness(Z', problem, FE);
            if NewCost < popCost(i)
                popPos(i, :) = Z;
                popCost(i) = NewCost;
                if popCost(i) < BestSolCost
                    BestSolPos = popPos(i, :);
                    BestSolCost = popCost(i);
                end
            end
            if NewCost < bsf
                bsf = NewCost;
                best_pos = Z;
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, popPos, popCost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        t = t + 1;

        % Leave the prey and go back home
        if FE < maxFE && t > T && t - round(T) - 1 >= 1 && t > 2
            if abs(BestCost(t - 1) - BestCost(t - round(T) - 1)) <= abs(0.01 * BestCost(t - 1))
                best = X_bestPos;
                j0 = randi(D, 1, ceil(D / 10 * rand));
                best(j0) = lb(j0) + rand(1, length(j0)) .* (ub(j0) - lb(j0));
                [bc, FE] = calculate_fitness(best', problem, FE);
                BestSolCost = bc;
                BestSolPos = best;
                if bc < bsf
                    bsf = bc;
                    best_pos = best;
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, popPos, popCost, population_history, fitness_history, ...
                        history_index, maxFE);
                end
                i0 = randi(n, 1, round(1 * n));
                popPos(i0(n - m + 1:n), :) = pop1Pos(i0(1:m), :);
                popCost(i0(n - m + 1:n)) = pop1Cost(i0(1:m));  % restore initial home costs
                popPos(i, :) = X_bestPos; popCost(i) = X_bestCost;
                t = 1;
            end
        end

        it = it + 1;

        % Update the prey (global best)
        if BestSolCost < X_bestCost
            X_bestPos = BestSolPos;
            X_bestCost = BestSolCost;
        end
        BestCost(t) = BestSolCost;
        Globest(1, t) = X_bestCost;
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

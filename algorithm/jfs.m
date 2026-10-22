% ----------------------------------------------------------------------- %
% Jellyfish Search optimizer (JS)
% Stored as jfs; the acronym JS, often written JSO, collides with jso (jSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop  = 50                       % Population size (jellyfish)
%   c(t)  = (1 - t/T)*(2*rand - 1)   % Time control, Eq. (17), drawn per jellyfish
%   C0    = 0.5                      % |c| >= C0 follows the ocean current
%   beta  = 3                        % Ocean-current distribution coefficient
%   gamma = 0.1                      % Passive-motion step as a fraction of ub - lb
%
% Algorithm Concept:
%   - Logistic-map initialization, x <- 4x(1 - x), from one random vector
%   - Ocean current: x + r.*(x* - beta*r*mean(X)), toward the best jellyfish x*
%   - Passive motion inside the swarm: x + gamma*r*(ub - lb)
%   - Active motion: x + r.*(x_j - x) toward a better random jellyfish x_j,
%     away from a worse one
%   - Out-of-box coordinates wrap around the box (periodic), Eq. (19)
%   - Greedy acceptance; the best is updated within the iteration
%
% Reference:
% Jui-Sheng Chou, Dinh-Nhat Truong,
% A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean,
% Applied Mathematics and Computation 389 (2021) 125535.
% https://doi.org/10.1016/j.amc.2020.125535
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 78961 v1.0.4 (js.m and
% initialization.m, 28 May 2022); nPop = 50 as in its main.m. MaxIt is
% counted from the budget, ceil((maxFe - nPop)/nPop) at one evaluation per
% jellyfish, so c(t) reaches 0 at maxFe. Kept as released: the passive or
% active test compares rand with 1 - c using the signed c, where the first
% test uses |c|, so passive motion only happens for c > 0, and its step is
% one-sided, left to the periodic wrap. The release's stagnation stop compares
% against 1e-350, which underflows to 0, so it can never fire; it is omitted.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jfs(problem)

    nd = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    nPop = 50;
    MaxIt = max(1, ceil((maxFE - nPop) / nPop));   % one evaluation per jellyfish per iteration

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Logistic-map initialization, Eq. (18)
    x = zeros(nPop, nd);
    x(1, :) = rand(1, nd);
    a = 4;
    for i = 1:(nPop - 1)
        x(i + 1, :) = a * x(i, :) .* (1 - x(i, :));
    end
    popi = VarMin + x .* (VarMax - VarMin);
    [popCost, FE] = calculate_fitness(popi', problem, FE);
    popCost = popCost(:)';

    [best_fitness, bidx] = min(popCost);
    best_solution = popi(bidx, :);
    for e = 1:min(nPop, maxFE)
        curve(e) = min(popCost(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, popi, popCost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    it = 0;
    while FE < maxFE
        it = it + 1;
        Meanvl = mean(popi, 1);
        [BestCost, bi] = min(popCost);   % the release's sort, first index of the minimum
        BestSol = popi(bi, :);

        for i = 1:nPop
            Ar = (1 - it * ((1) / MaxIt)) * (2 * rand - 1);   % Eq. (17)
            if abs(Ar) >= 0.5
                newsol = popi(i, :) + rand(1, nd) .* (BestSol - 3 * rand * Meanvl);   % ocean current, Eq. (11)
            elseif rand <= (1 - Ar)
                j = i;
                while j == i
                    j = randperm(nPop, 1);
                end
                Step = popi(i, :) - popi(j, :);   % direction, Eq. (15)
                if popCost(j) < popCost(i)
                    Step = -Step;
                end
                newsol = popi(i, :) + rand(1, nd) .* Step;   % active motion, Eq. (16)
            else
                newsol = popi(i, :) + 0.1 * (VarMax - VarMin) * rand;   % passive motion, Eq. (12)
            end
            newsol = simplebounds(newsol, VarMin, VarMax);   % Eq. (19)

            [newsolCost, FE] = calculate_fitness(newsol', problem, FE);
            if newsolCost < popCost(i)
                popi(i, :) = newsol;
                popCost(i) = newsolCost;
                if popCost(i) < BestCost
                    BestCost = popCost(i);
                    BestSol = popi(i, :);
                end
                if newsolCost < best_fitness
                    best_fitness = newsolCost;
                    best_solution = newsol;
                end
            end
            curve(FE) = best_fitness;   % one evaluation per step, so FE <= maxFE here
            [population_history, fitness_history, history_index] = record_history(...
                FE, popi, popCost, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Periodic boundary, Eq. (19): a coordinate past one bound re-enters from the other
function s = simplebounds(s, Lb, Ub)
    ns_tmp = s;
    I = ns_tmp < Lb;
    while sum(I) ~= 0
        ns_tmp(I) = Ub(I) + (ns_tmp(I) - Lb(I));
        I = ns_tmp < Lb;
    end
    J = ns_tmp > Ub;
    while sum(J) ~= 0
        ns_tmp(J) = Lb(J) + (ns_tmp(J) - Ub(J));
        J = ns_tmp > Ub;
    end
    s = ns_tmp;
end

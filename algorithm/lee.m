% ----------------------------------------------------------------------- %
% LangEvin Equation (LEE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP  = 20     % Population size (particles)
%   upr = 0.3    % Base probability of the dimension-mixing operator
%   inT = 1      % Initial temperature
%
% Algorithm Concept:
%   - Search Mechanism Operator: the Langevin equation drives a damped
%     velocity with a temperature-scaled random force,
%       V <- V - Lambda*V + sqrt(2*T*Lambda)*randn*(x - x_avg)
%   - Diversity Promoter: a binary mask mixes the Langevin move U with a
%     stepped move built from the distances to the best and to the pair mean
%   - Local Escaping Operator: with probability pr2 the particle jumps around
%     the best solution using two normal random scalings
%   - The temperature T and the friction Lambda both decay with the iteration
%
% Reference:
% Huiling Chen, Iman Ahmadianfar, Ali Asghar Heidari, Marjan Kordani,
% Arvin Samadi Koucheksaraee, Guoxi Liang,
% LEE: A physics-inspired optimizer based on LangEvin equation,
% Neurocomputing 666 (2026) 132288.
% https://doi.org/10.1016/j.neucom.2025.132288
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lee(problem)

    Dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    nP    = 20;
    MaxIt = max(1, ceil((maxFE - nP) / nP));

    upr = 0.3;
    inT = 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    X = repmat(LB, nP, 1) + repmat(UB - LB, nP, 1) .* rand(nP, Dim);   % Eq. (3)
    Z = rand(nP, Dim);
    V = zeros(nP, Dim);
    CostNew = zeros(nP, 1);
    T = inT;

    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:);

    [Best_Cost, ind] = min(Cost);
    Best_X = X(ind, :);
    bsf    = Best_Cost;

    for eval_count = 1:nP
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    for iter = 1:MaxIt
        if FE >= maxFE, break; end

        Lambda = exp(-(iter / MaxIt) ^ 2);                                         % Eq. (12)
        beta   = sin(((pi / 2) * iter + pi * (iter / MaxIt))) * (1 - iter / MaxIt) ...
                 .* exp(-2 * (iter / MaxIt));                                      % Eq. (14-2)

        [~, ind] = sort(Cost);

        a0 = 1:nP;
        [a1, a2] = Gna1a2(nP, a0);
        b = randn(nP, 1) .* beta;                                                  % Eq. (14-1)

        pr0 = upr + 0.1 .* rand(nP, 1);
        pr1 = pr0 + 0.1 .* sinh(randn(nP, 1));                                     % Eq. (18-6)
        pr2 = pr0 + 0.1 .* sinh(randn(nP, 1));                                     % Eq. (21)
        F   = 0.5 + 0.1 * sinh(randn(nP, 1));                                      % Eq. (17)
        Xpb = X(ind(randi(4, 1, nP)), :);

        for i = 1:nP
            if FE >= maxFE, break; end

            Xavg = (X(a1(i), :) + X(a2(i), :)) / 2;                                % Eq. (11)
            randomForce = sqrt(2 * T * Lambda) * randn .* (X(i, :) - Xavg);        % Eq. (10)
            V(i, :) = V(i, :) - Lambda * V(i, :) + randomForce;                    % Eq. (14)
            U = Xavg + F(i) .* (Xpb(i, :) - X(i, :)) + b(i) .* V(i, :);            % Eq. (16)

            Stp1 = Best_X - Xavg;                                                  % Eq. (18-2)
            Stp2 = X(i, :) - Xavg;                                                 % Eq. (18-3)

            UL  = rand * (UB - LB);
            Stp = Stp1 + (Stp2 - Stp1) ./ UL;                                      % Eq. (18-1)

            r = rand;
            L = rand(1, Dim) < pr1(i);
            Sigma = (randn * (r) + (1 - rand) ^ 2);
            Z(i, :) = L .* U + (1 - L) .* (X(i, :) + Stp .* Sigma);                % Eq. (18)

            if rand < pr2(i)
                rdn1 = randn; rdn2 = randn;
                if rand < (1 - iter / MaxIt)
                    Z(i, :) = X(i, :) + F(i) .* (rdn1) .* (Best_X - X(a1(i), :)) * Lambda;   % Eq. (19)
                else
                    Z(i, :) = Best_X + (F(i) .* rdn1 .* (b(i) .* Xpb(i, :) - X(i, :)) * Lambda + ...
                                        F(i) .* (rdn2) .* (b(i) .* Best_X - X(i, :)) * Lambda);  % Eq. (20)
                end
            end

            % Boundary handling
            Flag4ub = Z(i, :) > UB;
            Flag4lb = Z(i, :) < LB;
            Z(i, :) = (Z(i, :) .* (~(Flag4ub + Flag4lb))) + UB .* Flag4ub + LB .* Flag4lb;

            [CostNew(i), FE] = calculate_fitness(Z(i, :)', problem, FE);
            if CostNew(i) < Cost(i)
                X(i, :) = Z(i, :);
                Cost(i) = CostNew(i);
                if Cost(i) < Best_Cost
                    Best_X    = X(i, :);
                    Best_Cost = Cost(i);
                end
            end

            if CostNew(i) < bsf
                bsf = CostNew(i);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        T = inT * exp(-(iter / MaxIt));
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = Best_Cost;
    best_solution = Best_X;
end

% Generate two random index vectors distinct from r0 and from each other
function [a1, a2] = Gna1a2(Np1, r0)
    a1 = randperm(Np1, Np1);

    for i = 1:1000
        pos = (a1 == r0);
        if sum(pos) == 0
            break;
        else
            a1(pos) = floor(rand(1, sum(pos)) * Np1) + 1;
        end
    end

    a2 = floor(rand(1, Np1) * Np1) + 1;

    for i = 1:1000
        pos = ((a2 == a1) | (a2 == r0));
        if sum(pos) == 0
            break;
        else
            a2(pos) = floor(rand(1, sum(pos)) * Np1) + 1;
        end
    end
end

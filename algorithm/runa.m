% ----------------------------------------------------------------------- %
% RUNge Kutta optimizer (RUN)
% Stored as runa; run is a MATLAB builtin
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP = 50                          % Population size
%   f  = 20*exp(-12*t/T)             % Amplitude of the adaptive factor
%   SF = 2*(0.5 - rand)*f            % Adaptive factor, one per solution and iteration
%   mu = 0.5 + 0.1*randn             % Per-dimension weight of the difference term
%   ESQ probability = 0.5            % Chance a solution also takes the ESQ step
%
% Algorithm Concept:
%   - Search mechanism SM: a fourth-order Runge-Kutta slope (K1..K4) from the
%     better to the worse of x and the best of three random solutions
%   - Eq. (18): x' = p + r*SF*g*p + SF*SM + mu*d around p = xc (x or a random
%     solution per dimension) or p = xm (global or current best per dimension)
%   - Enhanced solution quality (ESQ): a point between the best and the mean of
%     three random solutions, perturbed by a weight w that shrinks with t/T
%   - A failed ESQ point may get one more Runge-Kutta move toward the best, Eq. (20)
%   - Greedy acceptance of every candidate; the best is refreshed per solution
%
% Reference:
% Iman Ahmadianfar, Ali Asghar Heidari, Amir H. Gandomi, Xuefeng Chu, Huiling Chen,
% RUN beyond the metaphor: An efficient optimization algorithm based on Runge
% Kutta method,
% Expert Systems with Applications 181 (2021) 115079.
% https://doi.org/10.1016/j.eswa.2021.115079
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' GitHub release (aliasgharheidaricom/RUN-Beyond-the-
% Metaphor-..., RUN.m of 13 Oct 2024, commit 2ddccad); that update pits x against
% the best of the random triple, where the 2021 code (also File Exchange 91075
% v1.0.4) used its position 1..3 as a population index. Kept as released: ESQ
% overwrites Xavg, so later solutions of the iteration use the last ESQ triple's
% mean, not the population mean, in Eq. (11.2). MaxIt is recounted every
% iteration as the iterations done plus the remaining budget over their mean FE
% cost (1.5*nP before the first), so t/T reaches 1 in the last iteration. The
% clamp is min/max (same as the release's for finite points); the best is
% reported at every evaluation, the guiding best refreshed as released.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = runa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    nP = 50;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(nP, dim, ub, lb);
    Xnew2 = zeros(1, dim);
    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:);

    [Best_Cost, ind] = min(Cost);
    Best_X = X(ind, :);
    best_fitness  = Best_Cost;
    best_solution = Best_X;
    for e = 1:min(nP, maxFE)
        curve(e) = min(Cost(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, Cost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    it = 1;
    while FE < maxFE
        it = it + 1;
        % MaxIt: iterations done plus the remaining budget at their mean FE cost
        if it == 2
            per_it = 1.5 * nP;
        else
            per_it = (FE - nP) / (it - 2);
        end
        MaxIt = it - 1 + ceil((maxFE - FE) / per_it);

        f = 20.*exp(-(12.*(it/MaxIt)));   % Eq. (17.6)
        Xavg = mean(X, 1);
        SF = 2.*(0.5-rand(1,nP)).*f;      % Eq. (17.5)

        for i = 1:nP
            [~, ind_l] = min(Cost);
            lBest = X(ind_l, :);

            [A, B, C] = RndX(nP, i);
            ABC = [A, B, C];
            [~, ind1] = min(Cost([A B C]));
            ind_ABC = ABC(ind1);

            % Delta X, Eqs. (11.1)-(11.3)
            gama = rand.*(X(i,:)-rand(1,dim).*(ub-lb)).*exp(-4*it/MaxIt);
            Stp = rand(1,dim).*((Best_X-rand.*Xavg)+gama);
            DelX = 2*rand(1,dim).*(abs(Stp));

            if Cost(i) < Cost(ind_ABC)
                Xb = X(i, :);
                Xw = X(ind_ABC, :);
            else
                Xb = X(ind_ABC, :);
                Xw = X(i, :);
            end

            SM = RungeKutta(Xb, Xw, DelX);

            L = rand(1,dim) < 0.5;
            Xc = L.*X(i,:)+(1-L).*X(A,:);   % Eq. (17.3)
            Xm = L.*Best_X+(1-L).*lBest;    % Eq. (17.4)

            vec = [1, -1];
            flag = floor(2*rand(1,dim)+1);
            r = vec(flag);

            g = 2*rand;
            mu = 0.5+.1*randn(1,dim);

            % Eq. (18)
            if rand < 0.5
                Xnew = (Xc+r.*SF(i).*g.*Xc) + SF(i).*(SM) + mu.*(Xm-Xc);
            else
                Xnew = (Xm+r.*SF(i).*g.*Xm) + SF(i).*(SM)+ mu.*(X(A,:)-X(B,:));
            end
            Xnew = min(max(Xnew, lb), ub);
            [CostNew, FE] = calculate_fitness(Xnew', problem, FE);
            if CostNew < Cost(i)
                X(i, :) = Xnew;
                Cost(i) = CostNew;
            end
            if CostNew < best_fitness
                best_fitness  = CostNew;
                best_solution = Xnew;
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, Cost, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            % Enhanced solution quality (ESQ), Eq. (19)
            if rand < 0.5
                EXP = exp(-5*rand*it/MaxIt);
                r = floor(Unifrnd(-1,2,1,1));

                u = 2*rand(1,dim);
                w = Unifrnd(0,2,1,dim).*EXP;               % Eq. (19.1)

                [A, B, C] = RndX(nP, i);
                Xavg = (X(A,:)+X(B,:)+X(C,:))/3;           % Eq. (19.2), also replaces the Eq. (11.2) mean
                beta = rand(1,dim);
                Xnew1 = beta.*(Best_X)+(1-beta).*(Xavg);   % Eq. (19.3)

                for j = 1:dim
                    if w(j) < 1
                        Xnew2(j) = Xnew1(j)+r*w(j)*abs((Xnew1(j)-Xavg(j))+randn);
                    else
                        Xnew2(j) = (Xnew1(j)-Xavg(j))+r*w(j)*abs((u(j).*Xnew1(j)-Xavg(j))+randn);
                    end
                end

                Xnew2 = min(max(Xnew2, lb), ub);
                [CostNew, FE] = calculate_fitness(Xnew2', problem, FE);
                accepted = CostNew < Cost(i);
                if accepted
                    X(i, :) = Xnew2;
                    Cost(i) = CostNew;
                end
                if CostNew < best_fitness
                    best_fitness  = CostNew;
                    best_solution = Xnew2;
                end
                curve(FE) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Cost, population_history, fitness_history, ...
                    history_index, maxFE);
                if FE >= maxFE, break; end

                if ~accepted
                    if rand < w(randi(dim))
                        SM = RungeKutta(X(i,:), Xnew2, DelX);
                        Xnew = (Xnew2-rand.*Xnew2)+ SF(i)*(SM+(2*rand(1,dim).*Best_X-Xnew2));   % Eq. (20)

                        Xnew = min(max(Xnew, lb), ub);
                        [CostNew, FE] = calculate_fitness(Xnew', problem, FE);
                        if CostNew < Cost(i)
                            X(i, :) = Xnew;
                            Cost(i) = CostNew;
                        end
                        if CostNew < best_fitness
                            best_fitness  = CostNew;
                            best_solution = Xnew;
                        end
                        curve(FE) = best_fitness;
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, X, Cost, population_history, fitness_history, ...
                            history_index, maxFE);
                        if FE >= maxFE, break; end
                    end
                end
            end

            % The guiding best, refreshed once per solution as released
            if Cost(i) < Best_Cost
                Best_X = X(i, :);
                Best_Cost = Cost(i);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Runge-Kutta search mechanism SM from the slopes K1..K4, as released
function SM = RungeKutta(XB, XW, DelX)
    dim = size(XB, 2);
    C = randi([1 2])*(1-rand);
    r1 = rand(1, dim);
    r2 = rand(1, dim);

    K1 = 0.5*(rand*XW-C.*XB);
    K2 = 0.5*(rand*(XW+r2.*K1.*DelX/2)-(C*XB+r1.*K1.*DelX/2));
    K3 = 0.5*(rand*(XW+r2.*K2.*DelX/2)-(C*XB+r1.*K2.*DelX/2));
    K4 = 0.5*(rand*(XW+r2.*K3.*DelX)-(C*XB+r1.*K3.*DelX));

    XRK = (K1+2.*K2+2.*K3+K4);
    SM = 1/6*XRK;
end

% Uniform numbers on [a, b], as the release writes unifrnd
function z = Unifrnd(a, b, c, dim)
    a2 = a/2;
    b2 = b/2;
    mu = a2+b2;
    sig = b2-a2;
    z = mu + sig .* (2*rand(c,dim)-1);
end

% Three distinct random indices other than i
function [A, B, C] = RndX(nP, i)
    Qi = randperm(nP); Qi(Qi==i) = [];
    A = Qi(1); B = Qi(2); C = Qi(3);
end

% Initialization, one column per dimension as in the release
function X = initialization(nP, dim, ub, lb)
    X = zeros(nP, dim);
    for i = 1:dim
        X(:, i) = rand(nP, 1).*(ub(i)-lb(i))+lb(i);
    end
end

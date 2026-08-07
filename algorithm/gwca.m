% ----------------------------------------------------------------------- %
% Great Wall Construction Algorithm (GWCA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N    = 50            % Population size (workers)
%   SL   = 1             % Slope length
%   T    = 8.3           % Traction
%   g    = 9.8           % Gravity
%   m    = 3             % Mass
%   e    = 0.1           % Elimination ratio (LNP = ceil(N*e))
%   P, Q = 9, 6          % Gamma shape / scale of the speed profile
%   Cmax, Cmin = e^3, e^2
%
% Algorithm Concept:
%   - Workers are ranked into three roles updated at random each turn:
%       engineer (Eq. 3-7): learns from the best worker with the physical
%         speed v = a*C*gampdf(it,P,Q)
%       soldier  (Eq. 9-10): improves towards the fitness-closest peer and
%         learns from the second-best worker
%       labourer (Eq. 11): follows the third-best worker and its own best
%   - Personnel elimination: the LNP worst workers are re-initialised at the
%     end of every iteration
%
% Reference:
% Ziyu Guan, Changxing Wang, et al.,
% Great Wall Construction Algorithm: A novel meta-heuristic algorithm for
% engineer problems,
% Expert Systems with Applications 233 (2023) 120905.
% https://doi.org/10.1016/j.eswa.2023.120905
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gwca(problem)

    dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    N       = 50;
    MaxIter = max(1, ceil((maxFE - N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Worker1 = zeros(1, dim);   Worker1_fit = inf;
    Worker2 = zeros(1, dim);   Worker2_fit = inf;
    Worker3 = zeros(1, dim);   Worker3_fit = inf;

    SL = 1; T = 8.3; g = 9.8; m = 3; e = 0.1; P = 9; Q = 6;
    Cmax = exp(3); Cmin = exp(2);
    LNP = ceil(N * e);

    % Initialisation
    Po = initial(N, dim, UB, LB);                       % Eq. (1)
    [Fitness, FE] = calculate_fitness(Po', problem, FE);
    Fitness = Fitness(:);

    for i = 1:N
        if Fitness(i) < Worker1_fit
            Worker1_fit = Fitness(i);  Worker1 = Po(i, :);
        elseif Fitness(i) > Worker1_fit && Fitness(i) < Worker2_fit
            Worker2_fit = Fitness(i);  Worker2 = Po(i, :);
        elseif Fitness(i) > Worker1_fit && Fitness(i) > Worker2_fit && Fitness(i) < Worker3_fit
            Worker3_fit = Fitness(i);  Worker3 = Po(i, :);
        end
    end
    Pbest    = Fitness;
    P_Pobest = Po;
    bsf      = Worker1_fit;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Po, Fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    for it = 1:MaxIter
        if FE >= maxFE, break; end

        for i = 1:N
            if FE >= maxFE, break; end

            Index = randi([1, 3]);
            if Index == 1
                % Engineer movement
                C = log((Cmax - Cmin) * (MaxIter - it) / MaxIter + Cmin);      % Eq. (7)
                H = (1 - it / MaxIter);                                        % Eq. (6)
                sitar = 80 * rand(1, dim);
                TL = SL - it / MaxIter + eps;
                a = (T .* TL) ./ m - g .* (H ./ sin(sitar));
                v = a .* C .* gampdf(it, P, Q);                                % Eq. (5)
                Study = (-1) ^ randi([0, 1]) * (Worker1 - Po(i, :)) .* rand(1, dim);   % Eq. (4)
                Po(i, :) = Worker1 + Study + Po(i, :) .* v .* rand(1, dim);    % Eq. (3)
            elseif Index == 2
                % Soldier movement
                C = log((Cmax - Cmin) * (MaxIter - it) / MaxIter + Cmin);
                H = (1 - it / MaxIter) + eps;
                sitar = 80 * rand(1, dim);
                v = m .* g .* (H ./ sin(sitar)) .* C .* gampdf(it / MaxIter, P, Q);    % Eq. (9)
                Idx = 1:N;
                Idx(Idx == i) = [];
                a = Fitness(Idx) - Fitness(i);
                [~, ide] = min(abs(a));
                improve = sign(Fitness(ide) - Fitness(i)) * (Po(ide, :) - Po(i, :)) .* v .* rand(1, dim);
                study = (Worker2 - Po(i, :)) .* rand(1, dim);
                Po(i, :) = Po(i, :) + improve + study;                          % Eq. (10)
            else
                % Labour movement
                Po(i, :) = Po(i, :) + 2 * (Worker3 - Po(i, :)) .* rand(1, dim) + ...
                           (P_Pobest(i, :) - Po(i, :)) .* gampdf(it / MaxIter, P, Q);  % Eq. (11)
            end

            % Constraint bound
            Flag4ub = Po(i, :) > UB;
            Flag4lb = Po(i, :) < LB;
            Po(i, :) = (Po(i, :) .* (~(Flag4ub + Flag4lb))) + UB .* Flag4ub + LB .* Flag4lb;

            [Fitness(i), FE] = calculate_fitness(Po(i, :)', problem, FE);

            % Update the local optimum
            if Fitness(i) < Pbest(i)
                Pbest(i)       = Fitness(i);
                P_Pobest(i, :) = Po(i, :);
            end

            % Update the leaders
            if Fitness(i) < Worker1_fit
                Worker1_fit = Fitness(i);  Worker1 = Po(i, :);
            elseif Fitness(i) > Worker1_fit && Fitness(i) < Worker2_fit
                Worker2_fit = Fitness(i);  Worker2 = Po(i, :);
            elseif Fitness(i) > Worker1_fit && Fitness(i) > Worker2_fit && Fitness(i) < Worker3_fit
                Worker3_fit = Fitness(i);  Worker3 = Po(i, :);
            end

            if Fitness(i) < bsf
                bsf = Fitness(i);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Po, Fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Personnel elimination mechanism
        [~, Idx] = sort(Fitness, 'descend');
        Po(Idx(1:LNP), :) = repmat(UB - LB, LNP, 1) .* rand(LNP, dim) + repmat(LB, LNP, 1);
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = Worker1_fit;
    best_solution = Worker1;
end

% Gamma density as shipped with the reference (shadows the builtin)
function Y = gampdf(x, a, b)
    Y = 1 / (gamma(x) * b ^ a) * x ^ (a - 1) * exp(-x / b);
end

% Chaotic (logistic-map) initialisation -- Eq. (1)
function Positions = initial(SearchAgents_no, dim, ub, lb)
    cxl = rand(SearchAgents_no, dim);
    for j = 1:dim
        if cxl(j) == 0,    cxl(j) = 0.1;  end
        if cxl(j) == 0.25, cxl(j) = 0.26; end
        if cxl(j) == 0.5,  cxl(j) = 0.51; end
        if cxl(j) == 0.75, cxl(j) = 0.76; end
        if cxl(j) == 1,    cxl(j) = 0.9;  end
    end
    for j = 1:dim
        cxl(j) = 4 * cxl(j) * (1 - cxl(j));
    end
    Positions = cxl .* (ub - lb) + lb;
end

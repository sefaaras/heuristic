% ----------------------------------------------------------------------- %
% Walrus Optimizer (WO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 50    % Population size (walruses)
%   P               = 0.4   % Proportion of females (males equal, rest children)
%
% Algorithm Concept:
%   - Two signals steer the herd: the danger signal A*R (A decays linearly
%     from 2) and the safety signal r2
%   - |Danger| >= 1: migration -- Beta*r3^2 scaled difference of two random
%     permutations of the herd
%   - |Danger| < 1 and safety >= 0.5: reproduction -- males are placed on a
%     Halton sequence, females interpolate between the male and the global
%     best, children take a Levy flight
%   - safety < 0.5 and |Danger| >= 0.5: feeding -- roaming towards the best
%   - safety < 0.5 and |Danger| < 0.5: gathering -- Cauchy-tangent steps
%     around the best and second-best walruses
%
% Reference:
% Muxuan Han, Zunfeng Du, Kum Fai Yuen, Haitao Zhu, Yancang Li, Qiuyu Yuan,
% Walrus Optimizer: A novel nature-inspired metaheuristic algorithm,
% Expert Systems with Applications 239 (2024) 122413.
% https://doi.org/10.1016/j.eswa.2023.122413
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 50;
    Max_iter = max(1, ceil(maxFE / SearchAgents_no));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Best_Pos = zeros(1, dim); Second_Pos = zeros(1, dim);
    Best_Score = inf;         Second_Score = inf;
    GBestX = repmat(Best_Pos, SearchAgents_no, 1);

    X = initialization(SearchAgents_no, dim, ub, lb);

    P = 0.4;
    F_number = round(SearchAgents_no * P);
    M_number = F_number;
    C_number = SearchAgents_no - F_number - M_number;

    Fvec = inf(SearchAgents_no, 1);

    t = 0;
    while t < Max_iter && FE < maxFE

        X = min(max(X, lb), ub);
        for i = 1:size(X, 1)
            if FE >= maxFE, break; end
            [fitness, FE] = calculate_fitness(X(i, :)', problem, FE);
            Fvec(i) = fitness;

            if fitness < Best_Score
                Best_Score = fitness;
                Best_Pos   = X(i, :);
            end
            if fitness > Best_Score && fitness < Second_Score
                Second_Score = fitness;
                Second_Pos   = X(i, :);
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = Best_Score;
            end
            % X is moved for the whole herd before this loop, so Fvec only
            % describes it again once the last walrus has been evaluated
            if i == size(X, 1) && FE >= 1 && FE <= maxFE
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, Fvec, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        Alpha = 1 - t / Max_iter;
        Beta  = 1 - 1 / (1 + exp((1 / 2 * Max_iter - t) / Max_iter * 10));
        A  = 2 * Alpha;
        r1 = rand();
        R  = 2 * r1 - 1;
        Danger_signal = A * R;
        r2 = rand();
        Safety_signal = r2;

        if abs(Danger_signal) >= 1
            r3 = rand();
            Rs = size(X, 1);
            Migration_step = (Beta * r3 ^ 2) * (X(randperm(Rs), :) - X(randperm(Rs), :));
            X = X + Migration_step;

        elseif abs(Danger_signal) < 1
            if Safety_signal >= 0.5
                i = 1;
                for i = 1:M_number
                    xy = zeros(M_number, 0);
                    base = 7;
                    xy(i, 1) = hal(i, base);
                    M = [];
                    m1 = xy(i, :);
                    m1 = lb + m1 .* (ub - lb);
                    M = [M; m1];
                    X(i, :) = M;
                end
                for j = M_number + 1:M_number + F_number
                    X(j, :) = X(j, :) + Alpha * (X(i, :) - X(j, :)) + (1 - Alpha) * (GBestX(j, :) - X(j, :));
                end
                for i = SearchAgents_no - C_number + 1:SearchAgents_no
                    Pc = rand;
                    o = GBestX(i, :) + X(i, :) .* levyFlight(dim);
                    X(i, :) = Pc * (o - X(i, :));
                end
            end

            if Safety_signal < 0.5 && abs(Danger_signal) >= 0.5
                for i = 1:SearchAgents_no
                    r4 = rand;
                    X(i, :) = X(i, :) * R - abs(GBestX(i, :) - X(i, :)) * r4 ^ 2;
                end
            end

            if Safety_signal < 0.5 && abs(Danger_signal) < 0.5
                for i = 1:size(X, 1)
                    for j = 1:size(X, 2)
                        theta1 = rand();
                        a1 = Beta * rand() - Beta;
                        b1 = tan(theta1 .* pi);
                        X1 = Best_Pos(j) - a1 * b1 * abs(Best_Pos(j) - X(i, j));

                        theta2 = rand();
                        a2 = Beta * rand() - Beta;
                        b2 = tan(theta2 .* pi);
                        X2 = Second_Pos(j) - a2 * b2 * abs(Second_Pos(j) - X(i, j));

                        X(i, j) = (X1 + X2) / 2;
                    end
                end
            end
        end

        t = t + 1;
    end

    curve(min(FE, maxFE):end) = Best_Score;

    best_fitness  = Best_Score;
    best_solution = Best_Pos;
end

% Halton sequence value
function halton = hal(index, base)
    result = 0;
    f = 1 / base;
    i = index;
    while (i > 0)
        result = result + f * mod(i, base);
        i = floor(i / base);
        f = f / base;
    end
    halton = result;
end

% Levy flight step
function o = levyFlight(d)
    beta = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    o = u ./ abs(v) .^ (1 / beta);
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

% ----------------------------------------------------------------------- %
% Information Acquisition Optimizer (IAO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50    % Population size (information units)
%
% Algorithm Concept:
%   - Information collection (Eq. 1): differential step between two randomly
%     chosen agents
%   - Information filtering (Eq. 2): chaotic coefficient delta built from a
%     cosine/arccos chaotic map (xi), a decaying phi and gamma
%   - Information analysis / organisation (Eq. 7): cosine-scaled contraction
%     towards the best information, with two variants selected by phi
%
% Reference:
% Xiao Wu, Shaobo Li, Xinghe Jiang, Yanqiu Zhou,
% Information acquisition optimizer: a new efficient algorithm for solving
% numerical and constrained engineering optimization problems,
% The Journal of Supercomputing (2024).
% https://doi.org/10.1007/s11227-024-06384-3
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = iao(problem)

    Dim   = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    N        = 50;
    Max_iter = max(1, ceil(maxFE / (4 * N)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Info     = initialization(N, Dim, UB, LB);
    New_Info = Info;
    Ffun     = inf(1, N);
    Ffun_new = zeros(1, N);

    Best_Pinfo = zeros(1, Dim);
    Best_Finfo = inf;
    bsf        = inf;

    iter = 1;
    while iter < Max_iter + 1 && FE < maxFE

        % Bounds check and best-information search
        for i = 1:size(Info, 1)
            if FE >= maxFE, break; end
            F_UB = Info(i, :) > UB;
            F_LB = Info(i, :) < LB;
            Info(i, :) = (Info(i, :) .* (~(F_UB + F_LB))) + UB .* F_UB + LB .* F_LB;

            [Ffun(1, i), FE] = calculate_fitness(Info(i, :)', problem, FE);

            if Ffun(1, i) < Best_Finfo
                Best_Finfo = Ffun(1, i);
                Best_Pinfo = Info(i, :);
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Ffun(1, i), bsf, curve, Info, Ffun, ...
                      population_history, fitness_history, history_index);
        end

        % Produce the candidate population
        rn = size(Info, 1);
        for i = 1:size(Info, 1)
            if FE >= maxFE, break; end
            r1 = randi([1, rn]);
            r2 = randi([1, rn]);

            theta = unifrnd(-1, +1);
            New_Info(i, :) = Info(i, :) + (Info(r1, :) - Info(r2, :)) * theta;   % Eq. (1)

            New_Info(i, :) = bound(New_Info(i, :), UB, LB);
            [Ffun_new(1, i), FE] = calculate_fitness(New_Info(i, :)', problem, FE);
            if Ffun_new(1, i) < Ffun(1, i)
                Info(i, :)  = New_Info(i, :);
                Ffun(1, i)  = Ffun_new(1, i);
            end
            if Ffun(1, i) < Best_Finfo
                Best_Finfo = Ffun(1, i);
                Best_Pinfo = Info(i, :);
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Ffun_new(1, i), bsf, curve, Info, Ffun, ...
                      population_history, fitness_history, history_index);
        end

        % Filter, evaluate and organise information
        for i = 1:size(Info, 1)
            if FE >= maxFE, break; end
            r3 = randi([1, rn]);
            r4 = randi([1, rn]);
            while r3 == r4
                r4 = randi([1, rn]);
            end

            xi     = 2 .* (mod(3.468 * rand() * (1 - rand() * cos(acos(rand() * 10 ^ 4))), 1));
            phi    = (cos(2 .* rand) + 1) .* (1 - (iter ./ Max_iter));
            gam    = sin((pi / 4) .^ (iter ./ Max_iter)) + phi + (log(iter ./ Max_iter)) ./ 8;
            delta  = cos(pi / 2 .* sqrt(abs(gam))) ./ xi;
            lambda = 2 .^ (sqrt(abs(gam)) - 2);

            if rand() < 0.5
                New_Info(i, :) = Info(i, :) - delta .* rand .* (Info(r3, :) - Info(i, :));   % Eq. (2)
            else
                New_Info(i, :) = Info(i, :) + delta .* rand .* (Info(r4, :) - Info(i, :));
            end
            New_Info(i, :) = bound(New_Info(i, :), UB, LB);
            [Ffun_new(1, i), FE] = calculate_fitness(New_Info(i, :)', problem, FE);
            if Ffun_new(1, i) < Ffun(1, i)
                Info(i, :) = New_Info(i, :);
                Ffun(1, i) = Ffun_new(1, i);
            end
            if Ffun(1, i) < Best_Finfo
                Best_Finfo = Ffun(1, i);
                Best_Pinfo = Info(i, :);
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Ffun_new(1, i), bsf, curve, Info, Ffun, ...
                      population_history, fitness_history, history_index);
            if FE >= maxFE, break; end

            % Analyse and organise information -- Eq. (7)
            if phi >= 0.5
                New_Info(i, :) = Best_Pinfo(1, :) .* cos((pi / 2) .* (sqrt(lambda .^ (1/3)))) - ...
                                 rand * (mean(Best_Pinfo(1, :)) - Info(i, :));
            else
                New_Info(i, :) = Best_Pinfo(1, :) .* cos((pi / 2) .* (sqrt(lambda .^ (1/3)))) - ...
                                 (rand * rand * (Best_Pinfo(1, :)) - (2 * rand - 1) * Info(i, :)) .* 0.8;
            end
            New_Info(i, :) = bound(New_Info(i, :), UB, LB);
            [Ffun_new(1, i), FE] = calculate_fitness(New_Info(i, :)', problem, FE);
            if Ffun_new(1, i) < Ffun(1, i)
                Info(i, :) = New_Info(i, :);
                Ffun(1, i) = Ffun_new(1, i);
            end
            if Ffun(1, i) < Best_Finfo
                Best_Finfo = Ffun(1, i);
                Best_Pinfo = Info(i, :);
            end
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, Ffun_new(1, i), bsf, curve, Info, Ffun, ...
                      population_history, fitness_history, history_index);
        end

        iter = iter + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = Best_Finfo;
    best_solution = Best_Pinfo;
end

% Bound handling
function x = bound(x, UB, LB)
    F_UB = x > UB;
    F_LB = x < LB;
    x = (x .* (~(F_UB + F_LB))) + UB .* F_UB + LB .* F_LB;
end

% Curve / history stamp for a single evaluation
function [bsf, curve, ph, fh, hi] = stamp(FE, maxFE, f, bsf, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        % +Inf is the not-yet-evaluated sentinel of Ffun, so the row waits for the
        % first full sweep; -Inf is a legitimate optimum and must not gate it
        if ~any(Fit == Inf)
            [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

% Initialization
function Info = initialization(N, Dim, UB, LB)
    Info = zeros(N, Dim);
    for i = 1:Dim
        Info(:, i) = rand(N, 1) .* (UB(i) - LB(i)) + LB(i);
    end
end

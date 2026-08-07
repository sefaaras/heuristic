% ----------------------------------------------------------------------- %
% Undivided Family Interaction Algorithm (UFIA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 67    % Population size (family members)
%   frequency    = 0.5   % Frequency of the oscillating coefficient beta
%
% Algorithm Concept:
%   - The family is split into a grandparent (global best) and three
%     subfamilies, each with its own parent (subfamily best)
%   - Subfamily 1: children are attracted to their parent and to the
%     grandparent through beta = 2*cos(2*pi*f*it)
%   - Subfamily 2: triangular interaction -- the centroid of three children
%     is combined with the parent and the grandparent
%   - Subfamily 3: cross-family learning -- a random parent and a random
%     child from any subfamily guide the move, switching to the grandparent
%     as the run progresses (pp = 1 - it/Max_iter)
%
% Reference:
% Vedik Basetti, Chandan Kumar Shiva, Sukriti Tiwari, Rama Rao Karri,
% Yogendra Arya, Sachidananda Sen,
% Solving nonlinear engineering problems using UFIA: A social inspired
% metaheuristic algorithm,
% Chaos, Solitons and Fractals 207 (2026) 117876.
% https://doi.org/10.1016/j.chaos.2026.117876
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ufia(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    NP        = 67;
    frequency = 0.5;
    n         = floor((NP - 1) / 3);
    Max_iter  = max(1, ceil((maxFE - NP) / (5 * n)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    X = initialization(NP, dim, ub, lb);
    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:);

    % Grandparent
    [GPS, idx] = min(fitness);
    GPP = X(idx, :);

    for eval_count = 1:NP
        if eval_count <= maxFE
            curve(eval_count) = GPS;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X(1:3*n, :), fitness(1:3*n), population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Divide into three subfamilies (grandparent removed first)
    X(idx, :)    = [];
    fitness(idx) = [];

    Children1 = X(1:n, :);
    Children2 = X(n+1:2*n, :);
    Children3 = X(2*n+1:3*n, :);

    fitnessC1 = fitness(1:n, :);
    fitnessC2 = fitness(n+1:2*n, :);
    fitnessC3 = fitness(2*n+1:3*n, :);

    [Parent_F1, i1] = min(fitnessC1); Parent_P1 = Children1(i1, :);
    [Parent_F2, i2] = min(fitnessC2); Parent_P2 = Children2(i2, :);
    [Parent_F3, i3] = min(fitnessC3); Parent_P3 = Children3(i3, :);

    BestGPS = GPS;
    BestGPP = GPP;
    bsf     = GPS;
    bsx     = GPP;

    % Main loop
    for it = 1:Max_iter
        if FE >= maxFE, break; end

        beta = 2 * cos(2 * pi * frequency * it);
        pp   = (1 - it / (Max_iter));

        % Subfamily 1
        for i = 1:n
            if FE >= maxFE, break; end
            rn = rand(1, 1);

            PChildren1 = Parent_P1 + rn * beta * abs(Parent_P1 - Children1(i, :));
            PChildren1 = bound(PChildren1, ub, lb);
            [PfitnessC1, FE] = calculate_fitness(PChildren1', problem, FE);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, PfitnessC1, PChildren1, bsf, bsx, curve, [Children1; Children2; Children3], ...
                      [fitnessC1; fitnessC2; fitnessC3], population_history, fitness_history, ...
                      history_index);

            GChildren1 = GPP + rn * beta * abs(GPP - Children1(i, :));
            GChildren1 = bound(GChildren1, ub, lb);
            if FE < maxFE
                [GfitnessC1, FE] = calculate_fitness(GChildren1', problem, FE);
            else
                GfitnessC1 = inf;
            end
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, GfitnessC1, GChildren1, bsf, bsx, curve, [Children1; Children2; Children3], ...
                      [fitnessC1; fitnessC2; fitnessC3], population_history, fitness_history, ...
                      history_index);

            if PfitnessC1 < fitnessC1(i, 1)
                fitnessC1(i, 1)  = PfitnessC1;
                Children1(i, :)  = PChildren1;
            elseif GfitnessC1 < fitnessC1(i, 1)
                fitnessC1(i, 1)  = GfitnessC1;
                Children1(i, :)  = GChildren1;
            end

            if fitnessC1(i, 1) < Parent_F1
                Parent_P1 = Children1(i, :);
                Parent_F1 = fitnessC1(i, 1);
            end
        end

        % Subfamily 2
        for i = 1:n
            if FE >= maxFE, break; end

            j = i;
            k = i;
            while i == j
                seed = randperm(n);
                j = seed(1);
                k = seed(2);
            end

            CI = mean([Children2(i, :); Children2(j, :); Children2(k, :)]);
            LF1 = round(1 + rand);
            LF2 = round(1 + rand);

            ChildrenNew1 = Children2(i, :) + rand(1, dim) .* (Parent_P2 - LF1 .* CI);
            ChildrenNew2 = Children2(j, :) + rand(1, dim) .* (GPP - LF2 .* CI);

            ChildrenNew1 = bound(ChildrenNew1, ub, lb);
            ChildrenNew2 = bound(ChildrenNew2, ub, lb);

            [f1, FE] = calculate_fitness(ChildrenNew1', problem, FE);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, f1, ChildrenNew1, bsf, bsx, curve, [Children1; Children2; Children3], ...
                      [fitnessC1; fitnessC2; fitnessC3], population_history, fitness_history, ...
                      history_index);
            if FE < maxFE
                [f2, FE] = calculate_fitness(ChildrenNew2', problem, FE);
            else
                f2 = inf;
            end
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, f2, ChildrenNew2, bsf, bsx, curve, [Children1; Children2; Children3], ...
                      [fitnessC1; fitnessC2; fitnessC3], population_history, fitness_history, ...
                      history_index);

            if f1 < fitnessC2(i, 1)
                fitnessC2(i, 1) = f1;
                Children2(i, :) = ChildrenNew1;
            end
            if f2 < fitnessC2(j, 1)
                fitnessC2(j, 1) = f2;
                Children2(j, :) = ChildrenNew2;
            end

            if fitnessC2(i, 1) < Parent_F2
                Parent_P2 = Children2(i, :);
                Parent_F2 = fitnessC2(i, 1);
            end
            if fitnessC2(j, 1) < Parent_F2
                Parent_P2 = Children2(j, :);
                Parent_F2 = fitnessC2(j, 1);
            end
        end

        % Subfamily 3
        for i = 1:n
            if FE >= maxFE, break; end

            kk = randi(3);
            if kk == 1
                Pk = Parent_P1;
            elseif kk == 2
                Pk = Parent_P2;
            else
                Pk = Parent_P3;
            end

            randFamily = randi(3);
            if randFamily == 1
                C = Children1(randi(n), :);
            elseif randFamily == 2
                C = Children2(randi(n), :);
            else
                C = Children3(randi(n), :);
            end

            rn = rand(1, 1);
            if rn < pp
                tempChildren3 = Children3(i, :) + rn * beta * abs(Pk - C);
            else
                tempChildren3 = Children3(i, :) + rn * beta * abs(GPP - C);
            end

            tempChildren3 = bound(tempChildren3, ub, lb);
            [tempfitnessC3, FE] = calculate_fitness(tempChildren3', problem, FE);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, tempfitnessC3, tempChildren3, bsf, bsx, curve, [Children1; Children2; Children3], ...
                      [fitnessC1; fitnessC2; fitnessC3], population_history, fitness_history, ...
                      history_index);

            if tempfitnessC3 < fitnessC3(i, 1)
                fitnessC3(i, 1) = tempfitnessC3;
                Children3(i, :) = tempChildren3;
            end

            if fitnessC3(i, 1) < Parent_F3
                Parent_P3 = Children3(i, :);
                Parent_F3 = fitnessC3(i, 1);
            end
        end

        % Grandparent update
        Parentscores = [Parent_F1; Parent_F2; Parent_F3];
        [GPS, idx] = min(Parentscores);
        Parents = [Parent_P1; Parent_P2; Parent_P3];
        GPP = Parents(idx, :);

        if GPS < BestGPS
            BestGPS = GPS;
            BestGPP = GPP;
        else
            GPS = BestGPS;
            GPP = BestGPP;
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Bound handling
function x = bound(x, ub, lb)
    Flag4ub = x > ub;
    Flag4lb = x < lb;
    x = (x .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
end

% Curve / history stamp for a single evaluation
function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, x, bsf, bsx, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

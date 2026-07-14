% ----------------------------------------------------------------------- %
% Forensic-Based Investigation (FBI) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50               % Population size
%
% Algorithm Concept:
%   - Inspired by the criminal-investigation process of police officers
%   - An investigation team (A1, A2 steps) locates suspects and a pursuit
%     team (B1, B2 steps) tracks them down; the two teams cooperate through
%     the shared best solution (Xbest)
%   - Parameter-free (no algorithm-specific tuning constants)
%
% Reference:
% Jui-Sheng Chou, Ngoc-Mai Nguyen,
% FBI inspired meta-optimization,
% Applied Soft Computing 93 (2020) 106339
% https://doi.org/10.1016/j.asoc.2020.106339
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fbi(problem)

    % Extract problem parameters
    D = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    NP = 50;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, NP, D);
    fitness_history = zeros(history_size, NP);
    history_index = 1;

    bf = inf;              % best-so-far cost (framework convention)
    bs = zeros(1, D);      % best-so-far position

    % Initialized population
    ObjVal = zeros(1, NP);
    Pop = rescale_matrix(rand(NP, D), LB, UB);
    for i = 1:NP
        [ObjVal(1, i), FE, bf, bs] = evalp(Pop(i, :), problem, FE, bf, bs);
    end

    % Memorize the best solution
    iBest = find(ObjVal == min(ObjVal));
    iBest = iBest(end);
    GlobalMin = ObjVal(iBest);
    Xbest = Pop(iBest, :);

    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = bf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Pop, ObjVal, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    % Optimization cycle
    PopA = Pop; ObjValA = ObjVal;
    PopB = Pop; ObjValB = ObjVal;

    while FE < maxFE
        FE_before = FE;

        %% Investigation team - team A, Step A1
        for i = 1:NP
            Change = fix(rand * D) + 1;
            nb1 = floor(rand * NP) + 1;
            while (nb1 == i)
                nb1 = floor(rand * NP) + 1;
            end
            nb2 = floor(rand * NP) + 1;
            while (nb1 == nb2 || nb2 == i)
                nb2 = floor(rand * NP) + 1;
            end
            solA = PopA(i, :);
            solA(Change) = PopA(i, Change) + (PopA(i, Change) - (PopA(nb1, Change) + PopA(nb2, Change)) / 2) * (rand - 0.5) * 2; % Eq.(2)
            for Change = 1:D
                if (solA(Change) <= LB(Change)) || (solA(Change) >= UB(Change))
                    solA(Change) = LB(Change) + (UB(Change) - LB(Change)) * rand();
                end
            end
            [f_a, FE, bf, bs] = evalp(solA, problem, FE, bf, bs);
            if f_a <= ObjValA(i)
                PopA(i, :) = solA;
                ObjValA(i) = f_a;
                if f_a <= GlobalMin
                    Xbest = solA;
                    GlobalMin = f_a;
                end
            end
        end

        %% Step A2
        if min(ObjValA) < max(ObjValA)
            prob = probability(ObjValA);
            for i = 1:NP
                if (rand > prob(i))
                    r(1) = floor(rand() * NP) + 1;
                    while r(1) == i
                        r(1) = floor(rand() * NP) + 1;
                    end
                    r(2) = floor(rand() * NP) + 1;
                    while (r(2) == r(1)) || (r(2) == i)
                        r(2) = floor(rand() * NP) + 1;
                    end
                    r(3) = floor(rand() * NP) + 1;
                    while (r(3) == r(2)) || (r(3) == r(1)) || (r(3) == i)
                        r(3) = floor(rand() * NP) + 1;
                    end
                    solA = PopA(i, 1:D);
                    Rnd = floor(rand() * D) + 1;
                    for j = 1:D
                        if (rand() < rand()) || (Rnd == j)
                            solA(j) = Xbest(j) + PopA(r(1), j) + rand() * (PopA(r(2), j) - PopA(r(3), j)); % Eq.(5)
                        else
                            solA(j) = PopA(i, j);
                        end
                    end
                    for j = 1:D
                        if (solA(j) <= LB(j)) || (solA(j) >= UB(j))
                            solA(j) = LB(j) + (UB(j) - LB(j)) * rand();
                        end
                    end
                    [f_a, FE, bf, bs] = evalp(solA, problem, FE, bf, bs);
                    if f_a <= ObjValA(i)
                        PopA(i, :) = solA;
                        ObjValA(i) = f_a;
                        if f_a <= GlobalMin
                            Xbest = solA;
                        end
                    end
                end
            end
        end

        %% Pursuing team - team B, Step B1
        for i = 1:NP
            SolB = PopB(i, 1:D);
            for j = 1:D
                SolB(j) = rand() * PopB(i, j) + rand() * (Xbest(j) - PopB(i, j)); % Eq.(6)
                if (SolB(j) < LB(j)) || (SolB(j) > UB(j))
                    SolB(j) = LB(j) + (UB(j) - LB(j)) * rand();
                end
            end
            [f_b, FE, bf, bs] = evalp(SolB, problem, FE, bf, bs);
            if f_b <= ObjValB(i)
                PopB(i, 1:D) = SolB;
                ObjValB(i) = f_b;
                if f_b <= GlobalMin
                    Xbest = SolB;
                    GlobalMin = f_b;
                end
            end
        end

        %% Step B2
        for i = 1:NP
            rr = floor(rand() * NP) + 1;
            while rr == i
                rr = floor(rand() * NP) + 1;
            end
            if ObjValB(i) > ObjValB(rr)
                SolB = PopB(rr, :) + rand(1, D) .* (PopB(rr, :) - PopB(i, :)) + rand(1, D) .* (Xbest - PopB(rr, :)); % Eq.(7)
                for j = 1:D
                    if (SolB(j) < LB(j)) || (SolB(j) > UB(j))
                        SolB(j) = LB(j) + (UB(j) - LB(j)) * rand();
                    end
                end
            else
                SolB = PopB(i, :) + rand(1, D) .* (PopB(i, :) - PopB(rr, :)) + rand(1, D) .* (Xbest - PopB(i, :)); % Eq.(8)
                for j = 1:D
                    if (SolB(j) < LB(j)) || (SolB(j) > UB(j))
                        SolB(j) = LB(j) + (UB(j) - LB(j)) * rand();
                    end
                end
            end
            [f_b, FE, bf, bs] = evalp(SolB, problem, FE, bf, bs);
            if f_b <= ObjValB(i)
                PopB(i, :) = SolB;
                ObjValB(i) = f_b;
                if f_b <= GlobalMin
                    Xbest = SolB;
                    GlobalMin = f_b;
                end
            end
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = bf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, PopA, ObjValA, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = bs;
    best_fitness = bf;

end

%% --- Objective evaluation: threads FE and best-so-far ---
function [z, FE, bf, bs] = evalp(pos, problem, FE, bf, bs)
    [z, FE] = calculate_fitness(pos', problem, FE);
    if z < bf
        bf = z;
        bs = pos;
    end
end

%% --- Rescale a [0,1] matrix into [LB,UB] ---
function f = rescale_matrix(X, LB, UB)
    [NP, D] = size(X);
    f = zeros(NP, D);
    for i = 1:D
        f(:, i) = LB(i) * ones(NP, 1) + (UB(i) - LB(i)) * X(:, i);
    end
end

%% --- Selection probability (Eq.3) ---
function prob = probability(fObjV)
    prob = (max(fObjV) - fObjV) / ((max(fObjV) - min(fObjV)));
end

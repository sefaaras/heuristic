% ----------------------------------------------------------------------- %
% Evolutionary Population Management based Adaptive Guided Differential Evolution (EPM-AGDE)
% Case-2 of the paper; its most frequent Proposed Case on CSBP-2 (3 of 12)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                  % Parents per epoch; as many trial children
%   F = 0.1 + 0.9*rand       % Scaling factor, per trial
%   CR = 0.05-0.15 | 0.9-1.0 % Two CR pools, chosen by their success-rate weights
%   nb = 5                   % Guide block size (AGDE's 10 % of NP)
%   phase switch = 0.5*maxFE % R3: random middle row before, the parent itself after
%
% Algorithm Concept:
%   - AGDE trial x_R3 + F*(x_R1 - x_R2) with adaptive two-pool CR and greedy
%     replacement; every trial is also kept as a child (Hypothesis-1)
%   - Guides (Hypothesis-3, Case-2): R1 cycles rows 1..5 (fitness elites), R2
%     rows 46..50 (tail of the FDB half); R3 random in 6..45, later x_j
%   - Epoch end: parents and children merged into U (2*NP)
%   - Next population (Hypothesis-2, Method-1): NP/2 fitness elites with no
%     parent-child pair, then each elite's best FDB mate measured from it
%
% Reference:
% Furkan Ustunsoy, Hamdi Tolga Kahraman, H. Huseyin Sayan, Yusuf Sonmez,
% Evolutionary population management for the design of metaheuristic search
% algorithms: Three improved algorithms, real-time charge scheduling problems,
% optimal solutions and stability analysis,
% Knowledge-Based Systems 328 (2025) 114221.
% https://doi.org/10.1016/j.knosys.2025.114221
% Components:
%   AGDE - Ali Wagdy Mohamed, Ali Khater Mohamed, Adaptive guided differential
%     evolution algorithm with novel mutation for numerical optimization,
%     International Journal of Machine Learning and Cybernetics 10(2) (2019)
%     253-277, https://doi.org/10.1007/s13042-017-0711-7
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' package (FX 181759, epm_agde-1.0.1, EPM_AGDE_V2.m +
% pop_productions.m) as a delta on this repository's agde.m. The paper names no
% single case: Table 2 defines Cases 1-6 and Table 10 lists a Proposed Case per
% CSBP-2 problem; Case-2 is the user's choice as the most frequent there.
% As released, AGDE's in-epoch greedy replacement is kept, so U holds the updated
% parents. Harness: the release's 5000*D switch is read as 0.5*maxFE (equal at
% maxFE = 10000*D) over main-loop evaluations as the release counts them, and the
% release's extra NP evaluations beyond maxFE are not spent. The production
% re-uses stored fitness and spends no evaluations.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = epm_agde(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    % AGDE Parameters
    NP = 50;                      % Population size
    nb = 5;                       % Leading/trailing guide block (rows 1..nb, NP-nb+1..NP)

    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);

    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize population
    Pop = initialization(NP, dim, ub, lb);

    % Evaluate initial population
    [Fit, FE] = calculate_fitness(Pop', problem, FE);
    Fit = Fit(:)';

    % Find initial best
    [best_fitness_current, iBest] = min(Fit);
    best_solution_current = Pop(iBest, :);

    % Record best fitness for each initial evaluation
    for eval_count = 1:NP
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Pop, Fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Adaptive CR parameters
    NW = [0.5, 0.5];  % Initial weights for CR pools

    YPop = zeros(NP, dim);
    fpop = zeros(1, NP);

    % Main loop
    Max_gen = ceil((maxFE - NP) / NP);
    g = 1;

    while FE < maxFE && g <= Max_gen

        CrPriods_Index = zeros(1, NP);
        Sr = zeros(1, 2);
        CrPriods_Count = zeros(1, 2);
        h = 1;

        for j = 1:NP
            % Adaptive CR Rule
            Ali = rand;
            if g <= 1
                if Ali <= 0.5
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            else
                if Ali <= NW(1)
                    CR = 0.05 + 0.1 * rand;
                    CrPriods_Index(j) = 1;
                else
                    CR = 0.9 + 0.1 * rand;
                    CrPriods_Index(j) = 2;
                end
            end
            CrPriods_Count(CrPriods_Index(j)) = CrPriods_Count(CrPriods_Index(j)) + 1;

            % Hypothesis-3, Case-2: R1/R2 walk the leading and trailing blocks of the EPM order
            if h > nb
                h = 1;
            end
            r1 = h;
            r2 = h + NP - nb;
            h = h + 1;
            if FE - NP <= 0.5 * maxFE        % release: nfeval <= 5000*D, nfeval excludes the initial NP
                r3 = randi([nb + 1, NP - nb]);
            else
                r3 = j;
            end

            % Adaptive scaling factor
            F = 0.1 + 0.9 * rand;

            % Mutation and Crossover
            X = zeros(1, dim);
            Rnd = randi(dim);
            for i = 1:dim
                if rand < CR || Rnd == i
                    X(i) = Pop(r3, i) + F * (Pop(r1, i) - Pop(r2, i));
                else
                    X(i) = Pop(j, i);
                end
            end

            % Boundary handling
            X = bound(X, ub, lb);

            % Evaluate trial vector
            [f_trial, FE] = calculate_fitness(X', problem, FE);
            YPop(j, :) = X;
            fpop(j) = f_trial;

            % Selection
            if f_trial <= Fit(j)
                Sr(CrPriods_Index(j)) = Sr(CrPriods_Index(j)) + 1;
                Pop(j, :) = X;
                Fit(j) = f_trial;

                if f_trial <= best_fitness_current
                    best_fitness_current = f_trial;
                    best_solution_current = X;
                end
            end

            % Record convergence curve
            if FE <= maxFE
                curve(FE) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Pop, Fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Hypothesis-2: next parents from the merged parents + children
        U = [Pop; YPop];
        FU = [Fit(:); fpop(:)];
        idx = epmProduceMethod1(U, FU);
        Pop = U(idx, :);
        Fit = FU(idx)';

        % Update CR pool weights
        CrPriods_Count(CrPriods_Count == 0) = 0.0001;
        Sr = Sr ./ CrPriods_Count;

        if sum(Sr) == 0
            W = [0.5, 0.5];
        else
            W = Sr / sum(Sr);
        end

        NW = (NW * (g - 1) + W) / g;
        g = g + 1;
    end

    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;

end

% EPM Hypothesis-2, Method-1 (paper Algorithm-2; released pop_productions.m)
function idx = epmProduceMethod1(U, FU)
    n2 = size(U, 1);
    [~, order] = sort(FU);
    FL = zeros(1, n2);
    idx = zeros(1, n2 / 2);
    c = 0;
    for i = 1:n2 / 2
        if ~any(FL == order(i))
            [idx, FL, c] = admit(idx, FL, c, order(i), n2);
        end
        if c == n2 / 4, break; end
    end
    for i = 1:n2 / 4
        mates = fdbRanking(U, FU, U(idx(i), :));
        for j = 1:n2
            if ~any(FL == mates(j))
                [idx, FL, c] = admit(idx, FL, c, mates(j), n2);
                break;
            end
        end
    end
end

% Takes row k into the next population and forbids it and its parent-child twin
function [idx, FL, c] = admit(idx, FL, c, k, n2)
    idx(c + 1) = k;
    FL(2 * c + 1) = k;
    if k > n2 / 2
        FL(2 * c + 2) = k - n2 / 2;
    else
        FL(2 * c + 2) = k + n2 / 2;
    end
    c = c + 1;
end

% Rows ordered by FDB score (normalised fitness + Manhattan distance to ref), best first
function order = fdbRanking(U, FU, ref)
    n = size(U, 1);
    if min(FU) == max(FU)
        order = randperm(n);
        return;
    end
    dist = zeros(n, 1);
    for j = 1:size(U, 2)
        dist = dist + abs(ref(j) - U(:, j));   % accumulated in the release's dimension order
    end
    minF = min(FU); rangeF = max(FU) - minF;
    minD = min(dist); rangeD = max(dist) - minD;
    score = (1 - (FU - minF) / rangeF) + (dist - minD) / rangeD;
    [~, order] = sort(score, 'descend');
end

% Initialization Function
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        X = zeros(SearchAgents_no, dim);
        for i = 1:dim
            X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

% Boundary Handling
function a = bound(a, ub, lb)
    % Random reinitialization for out-of-bound values
    for i = 1:length(a)
        if a(i) < lb(min(i, length(lb))) || a(i) > ub(min(i, length(ub)))
            lb_i = lb(min(i, length(lb)));
            ub_i = ub(min(i, length(ub)));
            a(i) = lb_i + (ub_i - lb_i) * rand;
        end
    end
end

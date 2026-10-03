% ----------------------------------------------------------------------- %
% Evolutionary Population Management based Teaching-Learning-based Artificial Bee Colony (EPM-TLABC)
% Case-5 of the paper; tied with Case-4 (4 of 12), see epm_tlabc_c4
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50           % Colony size; the onlooker children make U's second half
%   limit = 200            % Abandonment limit for the scout bee phase
%   CR = 0.5               % Crossover rate of the employed-phase DE donor
%   TF = round(1+rand)     % Teaching factor (1 or 2)
%   F = rand               % Scale factor of the employed-phase DE donor
%
% Algorithm Concept:
%   - Teaching-based employed bee phase as in TLABC, greedy per bee
%   - Onlooker phase, Eq. (14): child_k = x_a(k) + rand.*(x_r2 - x_r3), a = FDB order
%     of the colony relative to its best; greedy into slot k (Hypothesis-3)
%   - Case-5: (r2, r3) is the sequential pair (k, k + N/2) or (k - N/2, k),
%     one row from the fitness half and its FDB-half counterpart
%   - Generalized oppositional scout bee phase as in TLABC
%   - Epoch end: colony and onlooker children merged into U (2N); next colony =
%     N/2 fitness elites with no parent-child pair + each elite's FDB mate
%
% Reference:
% Furkan Ustunsoy, Hamdi Tolga Kahraman, H. Huseyin Sayan, Yusuf Sonmez,
% Evolutionary population management for the design of metaheuristic search
% algorithms: Three improved algorithms, real-time charge scheduling problems,
% optimal solutions and stability analysis,
% Knowledge-Based Systems 328 (2025) 114221.
% https://doi.org/10.1016/j.knosys.2025.114221
% Components:
%   TLABC - Xu Chen, Bin Xu, Congli Mei, Yuhan Ding, Kangji Li,
%     Teaching-learning-based artificial bee colony for solar photovoltaic
%     parameter estimation, Applied Energy 212 (2018) 1578-1588,
%     https://doi.org/10.1016/j.apenergy.2017.12.115
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' package (FX 181761, epm_tlabc-1.0.1, EPM_TLABC_V5.m +
% pop_productions.m) as a delta on this repository's tlabc.m. The paper names no
% single case: Table 4 defines Cases 1-5, Table 20 a Proposed Case per CSBP-2
% problem; Cases 4 and 5 tie as most frequent (4 of 12) and both are ported at
% the user's choice, this file and epm_tlabc_c4. Deliberate fix: the release's scout
% evaluates problem(newSol(i)), one scalar, and cannot run here; the two rows are
% evaluated as tlabc.m does. As released, the FDB order is recomputed for every
% onlooker on the colony updated so far, and trial counters stay with the slot,
% not the individual, across the reshuffle. The release's popsize extra
% evaluations are not spent; the production spends none.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = epm_tlabc_c5(problem)

    % Extract problem parameters
    dim = problem.dimension;
    low = problem.lb;
    up = problem.ub;
    maxIteration = problem.maxFe;
    
    % Algorithm parameters
    popsize = 50;
    trial = zeros(1, popsize);
    limit = 200;
    CR = 0.5;
    YPop = zeros(popsize, dim);
    fpop = zeros(1, popsize);
    
    FE = 0;                         % Function Evaluation Counter
    curve = zeros(1, maxIteration);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize population
    X = repmat(low, popsize, 1) + rand(popsize, dim) .* (repmat(up - low, popsize, 1));
    
    % Calculate initial fitness
    [val_X, FE] = calculate_fitness(X', problem, FE);
    
    [val_gBest, min_index] = min(val_X);
    gBest = X(min_index(1), :);
    
    % Record initial best fitness and store history
    for eval_count = 1:popsize
        if eval_count <= maxIteration
            curve(eval_count) = val_gBest;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, val_X, population_history, fitness_history, ...
                history_index, maxIteration);
        end
    end
    
    while FE < maxIteration
        % Teaching-based employed bee phase
        for i = 1:popsize
            [~, sortIndex] = sort(val_X);
            mean_result = mean(X);        % Calculate the mean
            Best = X(sortIndex(1), :);    % Identify the teacher
            TF = round(1 + rand * (1));
            Xi = X(i, :) + (Best - TF * mean_result) .* rand(1, dim);
            
            % Diversity learning
            r = generateR(popsize, i);
            F = rand;
            V = X(r(1), :) + F * (X(r(2), :) - X(r(3), :));
            flag = (rand(1, dim) <= CR);
            Xi(flag) = V(flag);
            Xi = boundary_repair(Xi, low, up, 'reflect');
            
            % Accept or Reject
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);
            
            if val_Xi < val_X(i)
                val_X(i) = val_Xi;
                X(i, :) = Xi;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
            
            % Record convergence curve and store history
            if FE <= maxIteration
                [val_gBest, gBest] = track_best(val_X, X, val_gBest, gBest);
                curve(FE) = val_gBest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, val_X, population_history, fitness_history, ...
                    history_index, maxIteration);
            end
            
            if FE >= maxIteration
                break;
            end
        end
        
        if FE >= maxIteration
            break;
        end
        
        % Learning-based onlooker bee phase, Eq. (14) (Hypothesis-3, Case-5)
        for k = 1:popsize
            % FDB order of the colony as updated so far, recomputed per bee as released
            a = fdbRanking(X, val_X(:), X(best_index(val_X), :));
            % Sequential pair (k, k + N/2) across the fitness and FDB halves
            if k <= popsize / 2
                r2 = k;
                r3 = k + popsize / 2;
            else
                r2 = k - popsize / 2;
                r3 = k;
            end
            Xi = X(a(k), :) + rand(1, dim) .* (X(r2, :) - X(r3, :));
            Xi = boundary_repair(Xi, low, up, 'reflect');

            % Accept or Reject into slot k; the child is kept for the merge either way
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);
            YPop(k, :) = Xi;
            fpop(k) = val_Xi;

            if val_Xi < val_X(k)
                val_X(k) = val_Xi;
                X(k, :) = Xi;
            end

            % Record convergence curve and store history
            if FE <= maxIteration
                [val_gBest, gBest] = track_best(val_X, X, val_gBest, gBest);
                curve(FE) = val_gBest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, val_X, population_history, fitness_history, ...
                    history_index, maxIteration);
            end

            if FE >= maxIteration
                break;
            end
        end

        if FE >= maxIteration
            break;
        end

        % Generalized oppositional scout bee phase
        ind = find(trial == max(trial));
        ind = ind(1);
        
        if (trial(ind) > limit)
            trial(ind) = 0;
            sol = (up - low) .* rand(1, dim) + low;
            solGOBL = (max(X) + min(X)) * rand - X(ind, :);
            newSol = [sol; solGOBL];
            newSol = boundary_repair(newSol, low, up, 'random');
            
            [val_sol, FE] = calculate_fitness(newSol', problem, FE);
            
            [~, min_index] = min(val_sol);
            X(ind, :) = newSol(min_index(1), :);
            val_X(ind) = val_sol(min_index(1));
            
            % Record convergence curve for scout phase evaluations
            for scout_idx = 1:2
                eval_count = FE - 2 + scout_idx;
                if eval_count <= maxIteration
                    [val_gBest, gBest] = track_best(val_X, X, val_gBest, gBest);
                    curve(eval_count) = val_gBest;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, X, val_X, population_history, fitness_history, ...
                        history_index, maxIteration);
                end
            end
        end
        
        % Hypothesis-2: next colony from the merged colony + onlooker children
        U = [X; YPop];
        FU = [val_X(:); fpop(:)];
        idx = epmProduceMethod1(U, FU);
        X = U(idx, :);
        val_X = FU(idx)';

        % The best food source is memorized
        if min(val_X) < val_gBest
            [val_gBest, min_index] = min(val_X);
            gBest = X(min_index(1), :);
        end
    end
    
    % Final best solution
    best_fitness = val_gBest;
    best_solution = gBest;

end

% Running best over all evaluations; the scout phase can make min(val_X) rise
function [vb, xb] = track_best(val_X, X, vb, xb)
    [m, i] = min(val_X);
    if m < vb
        vb = m;
        xb = X(i(1), :);
    end
end

function r = generateR(popsize, i)
    % Generate index r = [r1 r2 r3 r4 r5]
    r1 = randi(popsize);
    while r1 == i
        r1 = randi(popsize);
    end
    r2 = randi(popsize);
    while r2 == r1 || r2 == i
        r2 = randi(popsize);
    end
    r3 = randi(popsize);
    while r3 == r2 || r3 == r1 || r3 == i
        r3 = randi(popsize);
    end
    r4 = randi(popsize);
    while r4 == r3 || r4 == r2 || r4 == r1 || r4 == i
        r4 = randi(popsize);
    end
    r5 = randi(popsize);
    while r5 == r4 || r5 == r3 || r5 == r2 || r5 == r1 || r5 == i
        r5 = randi(popsize);
    end
    r = [r1 r2 r3 r4 r5];
end

function u = boundary_repair(v, low, up, str)
    [NP, D] = size(v);
    u = v;
    
    if strcmp(str, 'absorb')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j)
                    u(i, j) = up(j);
                elseif v(i, j) < low(j)
                    u(i, j) = low(j);
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
    
    if strcmp(str, 'random')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j) || v(i, j) < low(j)
                    u(i, j) = low(j) + rand * (up(j) - low(j));
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
    
    if strcmp(str, 'reflect')
        for i = 1:NP
            for j = 1:D
                if v(i, j) > up(j)
                    u(i, j) = max(2 * up(j) - v(i, j), low(j));
                elseif v(i, j) < low(j)
                    u(i, j) = min(2 * low(j) - v(i, j), up(j));
                else
                    u(i, j) = v(i, j);
                end
            end
        end
    end
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

% First index of the minimum, as the release's min(fitness)
function i = best_index(v)
    [~, i] = min(v);
end

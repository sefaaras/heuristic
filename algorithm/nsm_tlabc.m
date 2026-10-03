% ----------------------------------------------------------------------- %
% Natural Survivor Method based Teaching-Learning-based Artificial Bee Colony (NSM-TLABC)
% Variant of tlabc: NSM survivor selection
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50, limit = 200, CR = 0.5   % Population, abandonment limit, crossover (tlabc)
%   w = [0.9882 0.6862 0.6868]            % NSM weights: fitness, from best, from centre
%   n_centre = 50                         % Centre = mean of the whole population
%   p_NSM = 1 - sigmoid(15*FE/maxFE - 9.9) % NSM vs greedy test, ~1 early -> 0.006 at end
%
% Algorithm Concept:
%   - Teaching-based employed bees, learning-based onlookers and generalized
%     oppositional scouts, as in tlabc
%   - Employed bee phase: with probability p_NSM the greedy survivor test is
%     replaced by comparing the NSM scores of the parent and the candidate
%   - NSM score = w1*fitness + w2*distance from the best + w3*distance from the
%     population centre, each min-max normalised, L1 distances
%   - A candidate worse than the population's worst is rejected and one better
%     than its best is accepted; onlookers keep the greedy test
%
% Reference:
% Hamdi Tolga Kahraman, Mehmet Kati, Sefa Aras, Durdane Ayse Tasci,
% Development of the Natural Survivor Method (NSM) for designing an updating
% mechanism in metaheuristic search algorithms,
% Engineering Applications of Artificial Intelligence 122 (2023) 106121.
% https://doi.org/10.1016/j.engappai.2023.106121
% Components: base algorithm TLABC --
% Xu Chen, Bin Xu, Congli Mei, Yuhan Ding, Kangji Li,
% Teaching-learning-based artificial bee colony for solar photovoltaic parameter estimation,
% Applied Energy 212 (2018) 1578-1588
% https://doi.org/10.1016/j.apenergy.2017.12.115
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' group working copy (tlabcNSM.m with its NSM score file
% and the sigmoid switch of the group's toolbox), applied as a delta onto the
% pool's tlabc, whose per-evaluation val_gBest refresh (see its note) is kept. NSM
% costs no extra evaluation. Kept as written: in the NSM branch a candidate worse
% than the population's worst leaves trial(i) unchanged, where a greedy rejection
% would count it towards abandonment. The sigmoid is indexed by this harness's FE,
% which includes the initial population; the copy counts from 1 after it. Against
% the copy run in its own harness (CEC2014 D=10, 1e5 FE, 3 runs) the mean errors
% agree: F1 1.2e5 vs 1.1e5, F11 939 vs 879, F17 486 vs 700, all worse than tlabc.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nsm_tlabc(problem)

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

    % NSM parameters of the group's tlabcNSM.m
    nsm_w = [0.9882 0.6862 0.6868];   % fitness, distance from best, from centre
    n_centre = popsize;

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

            % Accept or Reject: NSM score test with probability p_NSM, else greedy
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);

            if nsm_switch(maxIteration, FE)
                [f_worst, i_worst] = max(val_X);
                [f_best, i_best] = min(val_X);
                % Worse than the worst: rejected without touching trial(i), as in the copy
                if f_worst > val_Xi
                    if nsm_accept(X(i, :), val_X(i), Xi, val_Xi, X(i_worst, :), f_worst, ...
                            X(i_best, :), f_best, X, val_X, n_centre, nsm_w)
                        val_X(i) = val_Xi;
                        X(i, :) = Xi;
                        trial(i) = 0;
                    else
                        trial(i) = trial(i) + 1;
                    end
                end
            elseif val_Xi < val_X(i)
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

        % Learning-based onlooker bee phase
        Fitness = calculateFitnessABC(val_X);
        prob = Fitness / sum(Fitness);
        cum_prob = cumsum(prob);

        for k = 1:popsize
            i = find(rand < cum_prob, 1);
            if isempty(i), i = popsize; end
            j = randi(popsize);
            while j == i
                j = randi(popsize);
            end

            if val_X(i) < val_X(j)
                Xi = X(i, :) + rand(1, dim) .* (X(i, :) - X(j, :));
            else
                Xi = X(i, :) + rand(1, dim) .* (X(j, :) - X(i, :));
            end
            Xi = boundary_repair(Xi, low, up, 'reflect');

            % Accept or Reject
            [val_Xi, FE] = calculate_fitness(Xi', problem, FE);

            if val_Xi < val_X(i)
                val_X(i) = val_Xi;
                X(i, :) = Xi;
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

function fFitness = calculateFitnessABC(fObjV)
    fFitness = zeros(size(fObjV));
    ind = find(fObjV >= 0);
    fFitness(ind) = 1 ./ (fObjV(ind) + 1);
    ind = find(fObjV < 0);
    fFitness(ind) = 1 + abs(fObjV(ind));
end

% Sigmoid_Func_2_Increase of the group's toolbox: true selects the NSM test
function use = nsm_switch(maxFE, fe)
    if maxFE < fe
        use = false;
    else
        use = 1 / (1 + exp(-(15 * fe / maxFE - 9.9))) < rand;
    end
end

% NSM survivor test (FDDBPopulasyonGuncelleTest of the group's copy): true keeps xn over xo
function accept = nsm_accept(xo, fo, xn, fn, xw, fw, xb, fb, X, fit, n_centre, w)
    if fb == fw || fb > fn
        accept = true;
    elseif fb == fo
        accept = false;
    else
        [~, order] = sort(fit);
        n_centre = max(n_centre, 2);
        centre = sum(X(order(1:n_centre), :), 1) / n_centre;
        cand = [xo; xn; xw; xb];
        % Distance from the centre is min-max normalised over old, new, worst and best
        dc = sum(abs(cand - centre), 2)';
        dc = (dc - min(dc)) / (max(dc) - min(dc));
        % Distance from the best is divided by its maximum over old, new and worst
        db = sum(abs(cand(1:3, :) - xb), 2)';
        db = db / max(db);
        nf = 1 - ([fo, fn] - fb) / (fw - fb);
        score = w(1) * nf + w(2) * db(1:2) + w(3) * dc(1:2);
        accept = score(2) > score(1);
    end
end

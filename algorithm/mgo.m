% ----------------------------------------------------------------------- %
% Mountain Gazelle Optimizer (MGO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                            % Herd size; each gazelle spawns 4 candidates
%   a = -1 -> -2 (linear)             % Scales the Cof rows (a+1)+rand and a*randn(1,D)
%   F = randn(1,D)*exp(2 - 2*t/T)     % TSM step amplitude, e^2 -> 1 over the budget
%   ceil(N/3) = 10                    % Gazelles averaged into the herd centre BH
%
% Algorithm Concept:
%   - Each gazelle x spawns four candidates; every difference term is scaled by
%     one of four coefficient rows Cof, picked at random per term
%   - TSM (territorial males): best - |(r*BH - r*x).*F|.*Cof, r in {1,2}, with
%     BH the mean of ceil(N/3) random gazelles
%   - MH (maternity herds): BH + Cof + (r*best - r*x_rand).*Cof
%   - BMH (bachelor herds): x - D + (r*best - r*BH).*Cof, D = (|x|+|best|)*(2r-1)
%   - MSF (migration for food): lb + r7*(ub - lb) with one scalar r7
%   - Candidates join the herd and the best gazelle is re-taken after every
%     spawn; at the end of the iteration the herd is sorted and the best N survive
%
% Reference:
% Benyamin Abdollahzadeh, Farhad Soleimanian Gharehchopogh, Nima Khodadadi,
% Seyedali Mirjalili,
% Mountain Gazelle Optimizer: A new Nature-inspired Metaheuristic Algorithm for
% Global Optimization Problems,
% Advances in Engineering Software 174 (2022) 103282.
% https://doi.org/10.1016/j.advengsoft.2022.103282
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's File Exchange 118680 v1.0.1 (MGO.m and helpers, all
% dated 2022-10-06); a seeded run matches it bit for bit (cec14). Kept as
% released, as in the paper: MSF's single scalar r7 puts the random gazelle on
% the box diagonal, and BH's X_ra term is multiplied by floor(rand) = 0.
% MaxIter = ceil((maxFE - N)/(4N)): initialisation costs N and an iteration 4N,
% so a and F reach their final values in the last, budget-truncated iteration.
% The release refreshes the reported best only at iteration end; here it is
% tracked per evaluation, and the history records the best N of the herd so
% far (the survivors if the iteration ended there). An all-NaN initial herd,
% where the release's best stays empty and it crashes, starts from gazelle 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mgo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    MaxIter = max(1, ceil((maxFE - N) / (4 * N)));   % four evaluations per gazelle per iteration

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = rand(N, dim) .* (ub - lb) + lb;
    [Sol_Cost, FE] = calculate_fitness(X', problem, FE);
    Sol_Cost = Sol_Cost(:);

    [bsf, ib] = min(Sol_Cost);
    bsx = X(ib, :);
    for e = 1:min(N, maxFE)
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, Sol_Cost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % The release's sequential <= scan keeps the LAST of tied minima
    ib = find(Sol_Cost == min(Sol_Cost), 1, 'last');
    if isempty(ib)
        ib = 1;
    end
    BestX = X(ib, :);

    Iter = 0;
    while FE < maxFE
        Iter = Iter + 1;
        for i = 1:N
            % Rows 1..N stay the parents until the end of the iteration; offspring are appended
            RandomSolution = randperm(N, ceil(N / 3));
            M = X(randi([(ceil(N / 3)), N]), :) * floor(rand) + mean(X(RandomSolution, :)) .* ceil(rand);   % BH

            cofi = coefficient_vector(dim, Iter, MaxIter);
            A = randn(1, dim) .* exp(2 - Iter * (2 / MaxIter));   % F
            D = (abs(X(i, :)) + abs(BestX)) * (2 * rand - 1);

            NewX = solution_imp(X, BestX, lb, ub, N, cofi, M, A, D, i);
            NewX = min(max(NewX, lb), ub);

            nE = min(4, maxFE - FE);   % the last spawn is cut to the remaining budget
            NewX = NewX(1:nE, :);
            [fNew, FE] = calculate_fitness(NewX', problem, FE);
            fNew = fNew(:);
            nH = numel(Sol_Cost);
            X = [X; NewX]; %#ok<AGROW>
            Sol_Cost = [Sol_Cost; fNew]; %#ok<AGROW>
            for k = 1:nE
                if fNew(k) < bsf
                    bsf = fNew(k);
                    bsx = NewX(k, :);
                end
                ec = FE - nE + k;
                curve(ec) = bsf;
                [~, s] = sort(Sol_Cost(1:nH + k));
                keep = s(1:N);
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X(keep, :), Sol_Cost(keep), population_history, fitness_history, ...
                    history_index, maxFE);
            end
            [~, idbest] = min(Sol_Cost);
            BestX = X(idbest, :);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        [Sol_Cost, SortOrder] = sort(Sol_Cost);
        X = X(SortOrder(1:N), :);
        Sol_Cost = Sol_Cost(1:N);
        BestX = X(1, :);
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsx;
end

% Coefficient rows Cof; a runs from -1 to -2 over the iterations
function cofi = coefficient_vector(dim, Iter, MaxIter)
    a2 = -1 + Iter * ((-1) / MaxIter);
    u = randn(1, dim);
    v = randn(1, dim);
    cofi = zeros(4, dim);
    cofi(1, :) = rand(1, dim);
    cofi(2, :) = (a2 + 1) + rand;
    cofi(3, :) = a2 .* randn(1, dim);
    cofi(4, :) = u .* v .^ 2 .* cos((rand * 2) * u);
end

% The four candidates of gazelle i: MSF, TSM, MH, BMH (draw order as released)
function NewX = solution_imp(X, BestX, lb, ub, N, cofi, M, A, D, i)
    NewX = zeros(4, size(X, 2));
    NewX(1, :) = (ub - lb) * rand + lb;
    NewX(2, :) = BestX - abs((randi(2) * M - randi(2) * X(i, :)) .* A) .* cofi(randi(4), :);
    NewX(3, :) = (M + cofi(randi(4), :)) + (randi(2) * BestX - randi(2) .* X((randi(N)), :)) .* cofi(randi(4), :);
    NewX(4, :) = (X(i, :) - D) + (randi(2) * BestX - randi(2) * M) .* cofi(randi(4), :);
end

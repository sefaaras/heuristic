% ----------------------------------------------------------------------- %
% Adaptive Opposition Slime Mould Algorithm (AOSMA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N   = 30     % Population size (slime mould)
%   del = 0.03   % Random re-initialisation probability
%
% Algorithm Concept:
%   - Slime Mould Algorithm base (smell-index weights, oscillating vb/vc)
%   - Adaptive opposition: for an agent whose new fitness is worse than its
%     old fitness, an opposite solution min(X)+max(X)-X is tried and kept
%     only if it improves (Eq. 14-16).
%
% Reference:
% Manoj Kumar Naik, Rutuparna Panda, Ajith Abraham,
% Adaptive opposition slime mould algorithm,
% Soft Computing 25 (22) (2021) 14297-14313.
% https://doi.org/10.1007/s00500-021-06140-2
% ----------------------------------------------------------------------- %
% Implementation Note:
% Eq.(6) divides by (best - worst) and the inherited "+ eps" covers only a zero
% span: RC25 returns Inf near its interior pole, and one Inf smell makes every
% ratio Inf/Inf and the population NaN in a generation. The span is taken over
% the finite smells, a non-finite one getting the worst ratio, as in the SMA
% base. An accepted opposite solution now carries its fitness; the reference kept
% the pre-opposition value, surviving as oldfitness to order Eq.(7) and set p.
% ceil(maxFE/(N*1.25)) assumes Eq.(14-16) re-evaluates a quarter of the swarm,
% but it fires only for an agent that got worse, so where nothing worsens the run
% ended at 82 % of the budget. It is now the horizon of Eq.(11-12) alone, FE
% terminates, and t is clamped at 1, so only runs that reached it change.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = aosma(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    Max_iter = ceil(maxFE / (N * 1.25));

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    bestPositions = zeros(1, dim);
    Destination_fitness = inf;
    AllFitness = inf * ones(N, 1);
    weight = ones(N, dim);

    X = initialization(N, dim, ub, lb);
    it = 1;
    lb = ones(1, dim) .* lb;
    ub = ones(1, dim) .* ub;
    del = 0.03;

    X = min(max(X, lb), ub);
    for i = 1:N
        [AllFitness(i), FE] = calculate_fitness(X(i, :)', problem, FE);
    end

    % True best-so-far (for curve and returned result)
    [bsf, bidx] = min(AllFitness);
    bsf_sol = X(bidx, :);
    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, AllFitness', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        oldfitness = AllFitness;
        [SmellOrder, SmellIndex] = sort(oldfitness);   % Eq. (7)
        worstFitness = SmellOrder(N);
        bestFitness = SmellOrder(1);

        % The span is taken over the finite smells; a non-finite worst makes every ratio NaN
        finiteSmell = SmellOrder(isfinite(SmellOrder));
        if ~isempty(finiteSmell)
            worstFitness = finiteSmell(end);
        else
            worstFitness = bestFitness;
        end
        S = bestFitness - worstFitness + eps;

        ratio = (bestFitness - SmellOrder) / S;
        ratio(~isfinite(SmellOrder)) = 1;   % a non-finite smell is the worst one, not a NaN

        for i = 1:N
            for j = 1:dim
                if i <= (N / 2)   % Eq. (6)
                    weight(SmellIndex(i), j) = 1 + rand() * log10(ratio(i) + 1);
                else
                    weight(SmellIndex(i), j) = 1 - rand() * log10(ratio(i) + 1);
                end
            end
        end

        if bestFitness < Destination_fitness
            bestPositions = X(SmellIndex(1), :);
            Destination_fitness = bestFitness;
        end

        t = min(it / Max_iter, 1);
        a = atanh(-t + 1);   % Eq. (11)
        b = 1 - t;           % Eq. (12)

        for i = 1:N   % Eq. (13)
            if rand < del
                X(i, :) = (ub - lb) * rand + lb;   % Eq. (13c)
            else
                p = tanh(abs(oldfitness(i) - Destination_fitness));   % Eq. (4)
                vb = unifrnd(-a, a, 1, dim);
                vc = unifrnd(-b, b, 1, dim);
                A = randi([1, N]);
                r2 = rand();
                for j = 1:dim
                    if r2 < p
                        X(i, j) = bestPositions(j) + vb(j) * (weight(i, j) * bestPositions(j) - X(A, j));   % Eq. (13a)
                    else
                        X(i, j) = vc(j) * X(i, j);   % Eq. (13b)
                    end
                end
            end
        end

        X = min(max(X, lb), ub);
        for i = 1:N
            [AllFitness(i), FE] = calculate_fitness(X(i, :)', problem, FE);
            if AllFitness(i) < bsf
                bsf = AllFitness(i); bsf_sol = X(i, :);
            end
            if FE <= maxFE
                curve(FE) = bsf;
            end
            % X is moved for the whole population before this loop, so AllFitness
            % only describes it again once the last agent has been re-evaluated
            if i == N && FE <= maxFE
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, AllFitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        if FE >= maxFE, break; end

        % Adaptive opposition (Eq. 14-16)
        for i = 1:N
            if AllFitness(i) > oldfitness(i)
                D(i, :) = min(X(i, :)) + max(X(i, :)) - X(i, :);   % Eq. (14)
                Flag4ub = D(i, :) > ub;
                Flag4lb = D(i, :) < lb;
                D(i, :) = (D(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
                [Dfit, FE] = calculate_fitness(D(i, :)', problem, FE);   % Eq. (15)
                if Dfit < AllFitness(i)
                    X(i, :) = D(i, :);
                    AllFitness(i) = Dfit;
                end
                if Dfit < bsf
                    bsf = Dfit; bsf_sol = D(i, :);
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, AllFitness', population_history, fitness_history, ...
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
            end
        end

        it = it + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_solution = bsf_sol;
    best_fitness = bsf;
end

% Initialization
function [X] = initialization(N, dim, up, down)
    for i = 1:dim
        high = up(i); low = down(i);
        X(:, i) = rand(1, N) .* (high - low) + low;
    end
end

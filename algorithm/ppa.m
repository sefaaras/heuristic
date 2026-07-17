% ----------------------------------------------------------------------- %
% Parasitism-Predation Algorithm (PPA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 30   % Population size (nests)
%
% Algorithm Concept:
%   - Ecosystem of three species with time-varying growth rates:
%       Crows (host)   -> nesting phase (Levy flights)
%       Cuckoos        -> parasitism phase (brood parasitism, ranking sel.)
%       Cats           -> predation phase (tracking-mode velocity)
%
% Reference:
% Al-Attar A. Mohamed, S. A. Hassan, A. M. Hemeida, Salem Alkhalaf,
% M. M. M. Mahmoud, Ayman M. Baha Eldin,
% Parasitism - Predation algorithm (PPA): A novel approach for feature
% selection,
% Ain Shams Engineering Journal 11 (2) (2020) 293-308.
% https://doi.org/10.1016/j.asej.2019.10.004
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ppa(problem)

    % Extract problem parameters
    d = problem.dimension;
    ub = problem.ub;
    lb = problem.lb;
    maxFE = problem.maxFe;

    n = 30;
    maxiter = ceil(maxFE / n);

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, n, d);
    fitness_history = zeros(history_size, n);
    history_index = 1;

    fitness = inf * ones(n, 1);

    if length(lb) < d
        lb = lb * ones(1, d); ub = ub * ones(1, d);
    end

    % Random initial solutions
    nest = initialization(n, ub, lb);

    % Evaluate initial nests
    [fnew_all, FE] = calculate_fitness(nest', problem, FE);
    fnew_all = fnew_all(:);
    for j = 1:n
        if fnew_all(j) <= fitness(j)
            fitness(j) = fnew_all(j);
        end
    end

    [fitness, I] = sort(fitness);
    nest = nest(I, :);
    fmin = fitness(1);
    bestnest = nest(1, :);
    bsf = fmin;

    for eval_idx = 1:n
        eval_count = FE - n + eval_idx;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, nest, fitness', population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    wMax = 0.9;
    wMin = 0.3;
    w = linspace(wMax, wMin, maxiter);
    Cats_v = 0.25 * nest;

    % Growth rates (Section 3.1.2.1)
    [GrowthRateCrows, GrowthRateCats, GrowthRateCuckoos] = Growth_rate(n, maxiter);

    for t = 2:maxiter
        if FE >= maxFE, break; end

        nCats = GrowthRateCats(t);
        nCrows = GrowthRateCrows(t);
        nCuckoos = GrowthRateCuckoos(t);

        % Nesting phase: Crows
        new_Crows = get_Crows(nest(1:nCrows, :), lb, ub);

        % Parasitism phase: crow-cuckoo subsystem
        Rolette_index = RankingSelection(n, nCuckoos);
        parasitized_nests = nest(Rolette_index(1:nCuckoos), :);
        new_Cuckoos = get_Cuckoos(parasitized_nests, lb, ub, t, maxiter);

        % Predation phase: crow-cat subsystem
        non_parasitized_nests = randperm(n);
        non_parasitized_nests(Rolette_index(1:nCuckoos)) = [];
        Cats_nests = nest(non_parasitized_nests(1:nCats), :);
        new_Cats_v = Cats_v(non_parasitized_nests(1:nCats), :);
        [new_Cats, new_Cats_v] = get_Cats(new_Cats_v, Cats_nests, bestnest, t, maxiter, w(t), lb, ub);
        Cats_v(non_parasitized_nests(1:nCats), :) = new_Cats_v;

        % Evaluation
        newnest = [new_Crows; new_Cuckoos; new_Cats];
        [fnew_all, FE] = calculate_fitness(newnest', problem, FE);
        fnew_all = fnew_all(:);
        for j = 1:size(nest, 1)
            if fnew_all(j) <= fitness(j)
                fitness(j) = fnew_all(j);
                nest(j, :) = newnest(j, :);
            end
            if fnew_all(j) < bsf
                bsf = fnew_all(j);
            end
        end

        [fitness, I] = sort(fitness);
        nest = nest(I, :);
        fmin = fitness(1);
        bestnest = nest(1, :);

        for eval_idx = 1:size(newnest, 1)
            eval_count = FE - size(newnest, 1) + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, nest, fitness', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_solution = bestnest;
    best_fitness = fmin;
end

%% --- Initialization ---
function nest = initialization(n, ub, lb)
    for i = 1:n
        nest(i, :) = lb + (ub - lb) .* rand(size(lb));
    end
end

%% --- Growth rates ---
function [GrowthRateCrows, GrowthRateCats, GrowthRateCuckoos] = Growth_rate(n, maxiter)
    GrowthRateCrows = round(n * linspace(2 / 3, 1 / 2, maxiter));
    GrowthRateCats = round(n * linspace(0.01, 1 / 3, maxiter));
    GrowthRateCuckoos = n - GrowthRateCrows - GrowthRateCats;
end

%% --- Crows nesting (Levy flights) ---
function nest = get_Crows(nest, lb, ub)
    n = size(nest, 1);
    beta = 3 / 2; sigma = 0.6965745; %#ok<NASGU>
    num = ceil(n * rand(1, n));
    for i = 1:n
        s = nest(i, :);
        u = randn(size(s)) * sigma;
        v = randn(size(s));
        step = u ./ abs(v).^(1 / beta);       % Eq. (7)
        stepsize = 0.1 * step .* randn(size(s)); % Eq. (8)
        s = s + stepsize .* (nest(num(i), :) - s); % Eq. (6)
        out = s < lb | s > ub;
        s(:, out) = lb(out) + (ub(out) - lb(out)) .* rand(1, nnz(out)); % Eq. (9)
        nest(i, :) = s;
    end
end

%% --- Ranking selection ---
function choice = RankingSelection(n, N)
    ranking = (1:n);
    weights = ranking / sum(ranking);
    Select_Fitness = cumsum(weights);
    choice = [];
    while length(choice) < N
        Random_Fitness = rand(1, n);
        selected = ranking(Select_Fitness <= Random_Fitness);
        choice = union(choice, selected, 'stable')';
    end
end

%% --- Cuckoos parasitism ---
function new_nest = get_Cuckoos(nest, lb, ub, t, maxiter)
    pa = t / 2 / maxiter;
    n = size(nest, 1);
    K = rand(size(nest)) > pa;   % Eq. (12)
    stepsize = rand * (nest(randperm(n), :) - nest(randperm(n), :));
    new_nest = nest + stepsize .* K;   % Eq. (10)
    new_nest = simplebounds(new_nest, lb, ub);
end

%% --- Simple bounds ---
function Sol = simplebounds(Sol, lb, ub)
    for i = 1:size(Sol, 1)
        Vub = Sol(i, :) > ub;
        Vlb = Sol(i, :) < lb;
        Sol(i, :) = (Sol(i, :) .* (~(Vub + Vlb))) + (lb + rand(size(lb)) .* (ub - lb)) .* (Vub + Vlb);
    end
end

%% --- Cats predation (tracking mode) ---
function [Cats, Cats_v] = get_Cats(Cats_v, Cats, gx, t, maxiter, w, lb, ub)
    perCnt = 1 - t / maxiter / 4;
    vlb = perCnt * lb;
    vUp = perCnt * ub;
    c = 2 - t / maxiter;
    for iSz2 = 1:size(Cats, 1)
        V_vec = w * Cats_v(iSz2, :) + c * rand * (gx - Cats(iSz2, :));   % Eq. (13)
        V_vec = max(V_vec, vlb);
        V_vec = min(V_vec, vUp);
        Cats_v(iSz2, :) = V_vec;
    end
    Cats = Cats + Cats_v;   % Eq. (14)
    Cats = simplebounds(Cats, lb, ub);
end

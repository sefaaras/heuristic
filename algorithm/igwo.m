% ----------------------------------------------------------------------- %
% Improved Grey Wolf Optimizer (I-GWO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50   % Population size (number of wolves)
%
% Algorithm Concept:
%   - Extends GWO with a Dimension Learning-based Hunting (DLH) strategy
%   - Each wolf builds a neighbourhood; neighbours share dimension info
%   - GWO candidate and DLH candidate compete; the better is selected
%   - Improves population diversity and exploration/exploitation balance
%   - Pairwise distances use plain matrix algebra (no Statistics Toolbox)
%
% Reference:
% Mohammad H. Nadimi-Shahraki, Shokooh Taghian, Seyedali Mirjalili,
% An improved grey wolf optimizer for solving engineering problems,
% Expert Systems with Applications 166 (2021) 113917
% https://doi.org/10.1016/j.eswa.2020.113917
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = igwo(problem)

    % Extract problem parameters
    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N  = 50;
    lu = [lb; ub];

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    % Leaders
    Alpha_pos = zeros(1, dim); Alpha_score = inf;
    Beta_pos  = zeros(1, dim); Beta_score  = inf;
    Delta_pos = zeros(1, dim); Delta_score = inf;

    % Initialize the positions of wolves
    Positions = initialization(N, dim, ub, lb);
    Positions = boundConstraint(Positions, Positions, lu);

    % Evaluate initial population
    [Fit, FE] = calculate_fitness(Positions', problem, FE);
    Fit = Fit(:)';

    pBestScore = Fit;
    pBest = Positions;

    best_so_far = min(Fit);
    for eval_count = 1:N
        curve(eval_count) = best_so_far;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Positions, Fit', population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    % Main loop
    while FE < maxFE

        % Update Alpha, Beta, and Delta
        for i = 1:N
            fitness = Fit(i);
            if fitness < Alpha_score
                Alpha_score = fitness; Alpha_pos = Positions(i, :);
            end
            if fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness; Beta_pos = Positions(i, :);
            end
            if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
                Delta_score = fitness; Delta_pos = Positions(i, :);
            end
        end

        % a decreases linearly from 2 to 0 over the FE budget
        a = 2 - FE * (2 / maxFE);

        % --- Candidate position Xi-GWO ---
        X_GWO = zeros(N, dim);
        for i = 1:N
            for j = 1:dim
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a; C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;

                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a; C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;

                r1 = rand(); r2 = rand();
                A3 = 2 * a * r1 - a; C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;

                X_GWO(i, j) = (X1 + X2 + X3) / 3;
            end
            X_GWO(i, :) = boundConstraint(X_GWO(i, :), Positions(i, :), lu);
        end
        [Fit_GWO, FE] = calculate_fitness(X_GWO', problem, FE);
        Fit_GWO = Fit_GWO(:)';

        % --- Candidate position Xi-DLH ---
        radius = sqrt(sum((Positions - X_GWO).^2, 2));      % Eq. (10), per-wolf radius
        dist_Position = pairwise_dist(Positions);           % all-pair euclidean distances
        r1p = randperm(N, N);

        X_DLH = zeros(N, dim);
        for t = 1:N
            neighbor_t = (dist_Position(t, :) <= radius(t));   % Eq. (11)
            Idx = find(neighbor_t == 1);
            random_Idx_neighbor = randi(numel(Idx), 1, dim);
            for d = 1:dim
                X_DLH(t, d) = Positions(t, d) + rand .* (Positions(Idx(random_Idx_neighbor(d)), d) ...
                    - Positions(r1p(t), d));                   % Eq. (12)
            end
            X_DLH(t, :) = boundConstraint(X_DLH(t, :), Positions(t, :), lu);
        end
        [Fit_DLH, FE] = calculate_fitness(X_DLH', problem, FE);
        Fit_DLH = Fit_DLH(:)';

        % --- Selection (Eq. 13) ---
        tmp = Fit_GWO < Fit_DLH;
        tmp_rep = repmat(tmp', 1, dim);
        tmpFit = tmp .* Fit_GWO + (1 - tmp) .* Fit_DLH;
        tmpPositions = tmp_rep .* X_GWO + (1 - tmp_rep) .* X_DLH;

        % --- Updating personal bests ---
        tmp = pBestScore <= tmpFit;
        tmp_rep = repmat(tmp', 1, dim);
        pBestScore = tmp .* pBestScore + (1 - tmp) .* tmpFit;
        pBest = tmp_rep .* pBest + (1 - tmp_rep) .* tmpPositions;

        Fit = pBestScore;
        Positions = pBest;

        best_so_far = min(best_so_far, min(Fit));

        % Record convergence curve and history for this generation (2N evals)
        for eval_idx = 1:(2 * N)
            eval_count = FE - 2 * N + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_so_far;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Positions, Fit', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    [val, index] = min(Fit);
    best_fitness  = val;
    best_solution = Positions(index, :);
end

%% --- All-pair euclidean distance matrix (replaces squareform(pdist(X))) ---
function D = pairwise_dist(X)
    G = X * X';
    sq = diag(G);
    D2 = bsxfun(@plus, sq, sq') - 2 * G;
    D2(D2 < 0) = 0;
    D = sqrt(D2);
end

%% --- Initialization Function ---
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(SearchAgents_no, dim);
        for i = 1:dim
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Boundary Handling (reflect to midpoint) ---
function vi = boundConstraint(vi, pop, lu)
    [NP, ~] = size(pop);
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

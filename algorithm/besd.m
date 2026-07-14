% ----------------------------------------------------------------------- %
% Bezier Search Differential Evolution (BeSD) for unconstrained benchmark
% problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                % Sub-pattern size (population per cycle)
%   K = 5                 % Pattern-matrix multiplier (K*N patterns)
%
% Algorithm Concept:
%   - A universal Differential Evolution variant that builds mutation
%     vectors from cubic Bezier curves (Bernstein basis) over four control
%     points, combined with bijective and top-N-best vectors
%   - Requires the Symbolic Math Toolbox (bernsteinMatrix)
%
% Reference:
% Pinar Civicioglu, Erkan Besdok,
% Bezier Search Differential Evolution Algorithm for numerical function
% optimization: A comparative study with CRMLSP, MVO, WA, SHADE and LSHADE,
% Expert Systems with Applications 165 (2021) 113875
% https://doi.org/10.1016/j.eswa.2020.113875
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = besd(problem)

    % Extract problem parameters
    D = problem.dimension;
    low = problem.lb;
    up = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, D);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    maxcycle = ceil(maxFE / N);

    % Initialization
    K = 5;
    S = zeros(K * N, D);
    for i = 1:K * N
        S(i, :) = rand(1, D) .* ((up - low) + low);   % LINE 2 (kept as in the reference)
    end
    [fitS, FE] = calculate_fitness(S', problem, FE);
    fitS = fitS(:)';
    [gmin, ind] = min(fitS);          % LINE 4
    gbest = S(ind, :);                % LINE 4

    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = gmin;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, S(1:N, :), fitS(1:N), population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    % Iterative search phase
    for epk = 1:maxcycle    % LINE 5
        if FE >= maxFE, break; end
        FE_before = FE;

        % Generation of j0, j1, and j2
        while 1, j1 = randperm(K * N);  j2 = randperm(K * N);  if sum(j1 == j2) == 0; break; end, end   % LINE 7
        j1 = j1(1:N);  j2 = j2(1:N);   % LINE 8
        j0 = j1;                       % LINE 9

        % Sub-pattern matrix P and fitP
        P = S(j1, :);  fitP = fitS(j1);   % LINE 11

        % Bijective vectors dv1
        dv1 = S(j2, :);   % LINE 13

        % Top-N-Best pattern vectors
        [~, index] = sort(fitS, 'ascend');  H = S(index, :);   % LINE 16
        tbest = P;
        for i = 1:N,  tbest(i, :) = H(ceil(rand^3 * K * N), :);  end   % LINE 16

        % Bezier mutation vectors dv2 (LINES 18-19)
        while 1
            j1 = randperm(N);  j2 = randperm(N);
            if sum(j1 == 1:N) == 0 && sum(j2 == 1:N) == 0 && sum(j1 == j2) == 0; break; end
        end
        dv2 = P;
        for i = 1:N
            X = [P(i, :); P(j1(i), :); P(j2(i), :); tbest(i, :)];
            X = X(randperm(4), :);
            B = bernsteinMatrix(3, rand);
            dv2(i, :) = B * X;
        end

        % Evolutionary step size (LINE 21)
        a = randn(N, 1);  b = 1 + rand(N, 1);  c = randn(N, 1).^randi(7, N, 1);  F = a .* b.^c;

        % Crossover control matrix map (LINES 23-27)
        [map1, map2] = genmap(N, D);
        if rand < rand, map = map1; else, map = map2; end

        % Trial pattern vectors T (LINE 29-30)
        w1 = randn(N, 1);  w2 = randn(N, 1);
        T = P;
        for iii = 1:N
            T(iii, :) = P(iii, :) + map(iii, :) .* F(iii) .* (w1(iii) .* (dv2(iii, :) - P(iii, :)) + w2(iii) .* (dv1(iii, :) - P(iii, :)));
        end

        % Boundary control mechanism (LINE 32)
        for i = 1:N
            for index = 1:D
                if T(i, index) < low, T(i, index) = rand * (up(index) - low(index)) + low(index); end
                if T(i, index) > up,  T(i, index) = rand * (up(index) - low(index)) + low(index); end
            end
        end

        % Update the sub-pattern matrix (LINES 34-35)
        [fitT, FE] = calculate_fitness(T', problem, FE);
        fitT = fitT(:)';
        ind = fitT < fitP;
        P(ind, :) = T(ind, :);
        fitP(ind) = fitT(ind);

        % Update the global solution gbest
        [BestVal, index] = min(fitP);   % LINE 37
        BestP = P(index, :);
        if BestVal < gmin,  gmin = BestVal;  gbest = BestP;  end   % LINE 38

        % Update the pattern matrix S and fitS
        S(j0, :) = P;  fitS(j0) = fitP;   % LINE 40

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = gmin;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, P, fitP, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = gbest;
    best_fitness = gmin;

end

%% --- Generation of crossover control matrix map (LINES 23-27) ---
function [map1, map2] = genmap(N, D)
    map1 = zeros(N, D);
    map2 = zeros(N, D);
    for j = 1:N
        h = randperm(D);  w = rand.^randi(7);
        map1(j, h(1:ceil(w * D))) = 1;
        h = randperm(D);  w = 1 - rand.^randi(7);
        map2(j, h(1:ceil(w * D))) = 1;
    end
end

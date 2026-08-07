% ----------------------------------------------------------------------- %
% Attraction-Repulsion Optimization Algorithm (AROA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N   = 50                       % Population size
%   c   = 0.95                     % Attraction-repulsion strength
%   fr1 = 0.15, fr2 = 0.6          % Local-search step scales
%   p1  = 0.2,  p2  = 0.8          % Operator-selection probabilities
%   Ef  = 0.4                      % Escape probability
%   tr1 = 0.9, tr2 = 0.85, tr3 = 0.9   % Per-dimension activation thresholds
%
% Algorithm Concept:
%   - Attraction-Repulsion operator (Eq. 6-9): every candidate is pulled
%     towards better and pushed away from worse neighbours among its k
%     nearest peers, weighted by the normalised distance I
%   - Attraction to the best (Eq. 10) with the tanh-scheduled amplitude m
%   - Local search (Eq. 12-15): Gaussian noise, a sine/cosine step around a
%     roulette-selected peer, or a uniform kick
%   - Escape operator (Eq. 17) and a memory operator (Eq. 18) that restores
%     the better of the current and the memorised position
%
% Reference:
% Karol Cymerys, Mariusz Oszust,
% Attraction-Repulsion Optimization Algorithm for global optimization problems,
% Swarm and Evolutionary Computation 84 (2024) 101459.
% https://doi.org/10.1016/j.swevo.2023.101459
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = aroa(problem)

    dim      = problem.dimension;
    lb       = problem.lb;
    ub       = problem.ub;
    maxEvals = problem.maxFe;

    N   = 50;
    c   = 0.95;
    fr1 = 0.15;
    fr2 = 0.6;
    p1  = 0.2;
    p2  = 0.8;
    Ef  = 0.4;
    tr1 = 0.9;
    tr2 = 0.85;
    tr3 = 0.9;

    tmax = max(1, ceil((maxEvals - N) / (2 * N)));
    FE   = 0;

    curve = zeros(1, maxEvals);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Xmin = repmat(lb, N, 1);
    Xmax = repmat(ub, N, 1);

    % Random initialisation -- Eq. (3)
    X = rand(N, dim) .* (ub - lb) + lb;
    [X, F, FE, nAdd] = evaluate_population(X, problem, ub, lb, FE, maxEvals);
    [fbest, ibest] = min(F);
    xbest = X(ibest, :);
    bsf   = fbest;

    for eval_count = 1:min(N, maxEvals)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X, F, population_history, fitness_history, ...
            history_index, maxEvals);
    end

    X_memory = X;
    F_memory = F;

    % Main loop
    for t = 1:tmax
        if FE >= maxEvals, break; end

        D = squareform(pdist(X, 'squaredeuclidean'));   % Eq. (4)
        m = tanh(t, tmax, [-2, 7]);                     % Eq. (11)

        for i = 1:N
            Dimax = max(D(i, :));
            k = floor((1 - t / tmax) * N) + 1;          % Eq. (9)
            [~, neighbors] = sort(D(i, :));

            % Attraction-repulsion operator -- Eq. (6)
            delta_ni = zeros(1, dim);
            for j = neighbors(1:k)
                I = 1 - (D(i, j) / Dimax);              % Eq. (7)
                s = sign(F(j) - F(i));                  % Eq. (8)
                delta_ni = delta_ni + c * (X_memory(i, :) - X_memory(j, :)) * I * s;
            end
            ni = delta_ni / N;

            % Attraction to the best solution -- Eq. (10)
            if rand < p1
                bi = m * c .* (rand(1, dim) .* xbest - X_memory(i, :));
            else
                bi = m * c .* (xbest - X_memory(i, :));
            end

            % Local search operators -- Eq. (15)
            if rand < p2
                if rand > 0.5 * t / tmax + 0.25
                    u1 = rand(1, dim) > tr1;
                    ri = u1 .* random('Normal', zeros(1, dim), fr1 * (1 - t / tmax) * (ub - lb));   % Eq. (12)
                else
                    u2 = rand(1, dim) > tr2;
                    w = index_roulette_wheel_selection(F, k);
                    Xw = X_memory(w, :);
                    if rand < 0.5
                        ri = fr2 * u2 .* (1 - t / tmax) .* sin(2 * pi * rand(1, dim)) .* abs(rand(1, dim) .* Xw - X_memory(i, :));
                    else
                        ri = fr2 * u2 .* (1 - t / tmax) .* cos(2 * pi * rand(1, dim)) .* abs(rand(1, dim) .* Xw - X_memory(i, :));
                    end
                end
            else
                u3 = rand(1, dim) > tr3;
                ri = u3 .* (2 * rand(1, dim) - ones(1, dim)) .* (ub - lb);   % Eq. (14)
            end

            X(i, :) = X(i, :) + ni + bi + ri;           % Eq. (16)
        end

        [X, F, FE, nAdd] = evaluate_population(X, problem, ub, lb, FE, maxEvals);
        [fbest_candidate, ibest_candidate] = min(F);
        if fbest_candidate < fbest
            fbest = fbest_candidate;
            xbest = X(ibest_candidate, :);
        end
        if fbest < bsf, bsf = fbest; end
        [curve, population_history, fitness_history, history_index] = ...
            stampN(FE, maxEvals, nAdd, bsf, curve, X, F, population_history, ...
                   fitness_history, history_index);

        [X, F] = memory_operator(X, F, X_memory, F_memory);   % Eq. (18)
        X_memory = X;
        F_memory = F;

        % Escape operator -- Eq. (17)
        CF = (1 - t / tmax) ^ 3;
        if rand < Ef
            u4 = rand(N, dim) < Ef;
            X = X + CF * (u4 .* (rand(N, dim) .* (Xmax - Xmin) + Xmin));
        else
            r7 = rand();
            X = X + (CF * (1 - r7) + r7) * (X(randperm(N), :) - X(randperm(N), :));
        end

        if FE >= maxEvals, break; end
        [X, F, FE, nAdd] = evaluate_population(X, problem, ub, lb, FE, maxEvals);
        [fbest_candidate, ibest_candidate] = min(F);
        if fbest_candidate < fbest
            fbest = fbest_candidate;
            xbest = X(ibest_candidate, :);
        end
        if fbest < bsf, bsf = fbest; end
        [curve, population_history, fitness_history, history_index] = ...
            stampN(FE, maxEvals, nAdd, bsf, curve, X, F, population_history, ...
                   fitness_history, history_index);

        [X, F] = memory_operator(X, F, X_memory, F_memory);   % Eq. (18)
        X_memory = X;
        F_memory = F;
    end

    curve(min(FE, maxEvals):end) = bsf;

    best_fitness  = fbest;
    best_solution = xbest;
end

% Evaluate the whole population (bounded by the FE budget)
function [X, F, FE, nEval] = evaluate_population(X, problem, ub, lb, FE, maxEvals)
    N = size(X, 1);
    F = Inf(N, 1);
    X = max(lb, min(ub, X));                 % check space bounds
    nEval = min(N, max(0, maxEvals - FE));
    if nEval > 0
        [f, FE] = calculate_fitness(X(1:nEval, :)', problem, FE);
        F(1:nEval) = f(:);
    end
end

% Memory operator -- Eq. (18)
function [X, F] = memory_operator(X, F, X_memory, F_memory)
    dim = size(X, 2);
    Inx  = F_memory < F;
    Indx = repmat(Inx, 1, dim);
    X = Indx .* X_memory + ~Indx .* X;
    F = Inx  .* F_memory + ~Inx  .* F;
end

% Scheduled amplitude -- Eq. (11) (shadows the builtin, as in the source)
function y = tanh(t, tmax, range)
    z = 2 * (t / tmax * (range(2) - range(1)) + range(1));
    y = 0.5 * ((exp(z) - 1) / (exp(z) + 1) + 1);
end

% Fitness-proportional peer selection among the k nearest
function selected_index = index_roulette_wheel_selection(F, k)
    fitness = F(1:k);
    weights = max(fitness) - fitness;
    weights = cumsum(weights / sum(weights));
    selected_index = roulette_wheel_selection(weights);
end

function selected_index = roulette_wheel_selection(weights)
    r = rand();
    selected_index = 1;
    for index = size(weights, 1)
        if r <= weights(index)
            selected_index = index;
            break;
        end
    end
end

% Curve / history stamp for a batch of N evaluations
function [curve, ph, fh, hi] = stampN(FE, maxFE, n, bsf, curve, X, Fit, ph, fh, hi)
    for k = 1:n
        ec = FE - n + k;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

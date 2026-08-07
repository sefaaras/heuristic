% ----------------------------------------------------------------------- %
% Artificial Lemming Algorithm (ALA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50    % Population size (lemmings)
%
% Algorithm Concept:
%   - Four behaviours selected by the energy factor E = 2*log(1/rand)*theta,
%     with theta = 2*atan(1-FEs/MaxFEs)
%   - E > 1, long-distance migration: Brownian-driven mix of the best and a
%     random peer (r1 blending)
%   - E > 1, digging holes: r2 = rand*(1+sin(0.5*FEs)) step towards the best
%     from a random peer
%   - E <= 1, foraging: spiral around the best with radius = ||best - x||
%   - E <= 1, predator evasion: Levy flight towards the best, scaled by the
%     linearly decaying G
%
% Reference:
% Yaning Xiao, Hao Cui, Ruba Abu Khurma, Pedro A. Castillo,
% Artificial lemming algorithm: a novel bionic meta-heuristic technique for
% solving real-world engineering optimization problems,
% Artificial Intelligence Review 58, 84 (2025).
% https://doi.org/10.1007/s10462-024-11023-7
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ala(problem)

    dim    = problem.dimension;
    lb     = problem.lb;
    ub     = problem.ub;
    MaxFEs = problem.maxFe;

    N = 50;

    FE    = 0;
    curve = zeros(1, MaxFEs);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N, dim, ub, lb);
    Position = zeros(1, dim);
    Score    = inf;
    vec_flag = [1, -1];

    [fitness, FE] = calculate_fitness(X', problem, FE);
    fitness = fitness(:)';
    for i = 1:N
        if fitness(1, i) < Score
            Position = X(i, :);
            Score    = fitness(1, i);
        end
    end
    bsf = Score;

    for eval_count = 1:N
        if eval_count <= MaxFEs
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, fitness, population_history, fitness_history, ...
                history_index, MaxFEs);
        end
    end

    Xnew = zeros(N, dim);

    % Main optimisation loop
    while FE < MaxFEs
        RB = randn(N, dim);                      % Brownian motion
        F  = vec_flag(floor(2 * rand() + 1));    % random directional flag
        theta = 2 * atan(1 - FE / MaxFEs);       % time-varying parameter

        for i = 1:N
            E = 2 * log(1 / rand) * theta;
            if E > 1
                if rand < 0.3
                    r1 = 2 * rand(1, dim) - 1;
                    Xnew(i, :) = Position + F .* RB(i, :) .* ...
                                 (r1 .* (Position - X(i, :)) + (1 - r1) .* (X(i, :) - X(randi(N), :)));
                else
                    r2 = rand() * (1 + sin(0.5 * FE));
                    Xnew(i, :) = X(i, :) + F .* r2 * (Position - X(randi(N), :));
                end
            else
                if rand < 0.5
                    radius = sqrt(sum((Position - X(i, :)) .^ 2));
                    r3 = rand();
                    spiral = radius * (sin(2 * pi * r3) + cos(2 * pi * r3));
                    Xnew(i, :) = Position + F .* X(i, :) .* spiral * rand;
                else
                    G = 2 * (sign(rand - 0.5)) * (1 - FE / MaxFEs);
                    Xnew(i, :) = Position + F .* G * Levy(dim) .* (Position - X(i, :));
                end
            end
        end

        % Boundary check and evaluation
        for i = 1:N
            if FE >= MaxFEs, break; end
            Flag4ub = Xnew(i, :) > ub;
            Flag4lb = Xnew(i, :) < lb;
            Xnew(i, :) = (Xnew(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;

            [newPopfit, FE] = calculate_fitness(Xnew(i, :)', problem, FE);
            if newPopfit < fitness(i)
                X(i, :)       = Xnew(i, :);
                fitness(1, i) = newPopfit;
            end
            if fitness(1, i) < Score
                Position = X(i, :);
                Score    = fitness(1, i);
            end

            if newPopfit < bsf
                bsf = newPopfit;
            end
            if FE <= MaxFEs
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fitness, population_history, fitness_history, ...
                    history_index, MaxFEs);
            end
        end
    end

    curve(min(FE, MaxFEs):end) = bsf;

    best_fitness  = Score;
    best_solution = Position;
end

% Levy flight step
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v) .^ (1 / beta);
    o = step;
end

% Initialization
function X = initialization(N, Dim, UB, LB)
    X = zeros(N, Dim);
    for i = 1:Dim
        X(:, i) = rand(N, 1) .* (UB(i) - LB(i)) + LB(i);
    end
end

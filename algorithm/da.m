% ----------------------------------------------------------------------- %
% Dragonfly Algorithm (DA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                       % Dragonflies
%   r  = (ub-lb)/4 + (ub-lb)*2*(iter/Max_iter)   % Neighbourhood radius, GROWS
%   w = 0.9 -> 0.4 (linear)      % Inertia weight
%   my_c = 0.1 -> 0 (linear)     % Master swarming coefficient, zero at half-run
%   s = a = c = 2*rand*my_c      % Separation / alignment / cohesion weights
%   f = 2*rand,  e = my_c        % Food attraction and enemy distraction
%   Delta_max = (ub-lb)/10       % Step (velocity) clamp
%
% Algorithm Concept:
%   - Reynolds' three boid primitives plus two survival terms, each weighted:
%       Separation  S = -sum(x_j - x_i)  over neighbours j
%       Alignment   A = mean(step_j),    Cohesion C = mean(x_j) - x_i
%       Food        F = x_food - x_i,    Enemy    E = x_enemy + x_i
%     giving DeltaX <- (a*A + c*C + s*S + f*F + e*E) + w*DeltaX, a velocity
%   - The behavioural switch is the NEIGHBOURHOOD RADIUS, which GROWS over the
%     run: early on few neighbours and Levy flights (static swarming, pure
%     exploration), later everyone is a neighbour and the cohesive terms take
%     over (dynamic swarming towards the food source)
%   - my_c decays to zero at half the budget, leaving only food and inertia
%   - Bound handling is a WRAP: a component past a bound reappears at the
%     opposite bound with a fresh random step component
%
% Reference:
% Seyedali Mirjalili,
% Dragonfly algorithm: a new meta-heuristic optimization technique for solving
% single-objective, discrete, and multi-objective problems,
% Neural Computing and Applications, vol. 27, pp. 1053-1073, 2016.
% https://doi.org/10.1007/s00521-015-1920-1
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own MATLAB release ("source codes demo version 1.0"),
% including the SEQUENTIAL update in which dragonfly i sees a population where
% 1..i-1 have already moved this generation.
% Two reference properties kept as written: distance.m is commented "Euclidean"
% but returns the PER-DIMENSION absolute difference as a vector, which is what
% the `all(Dist <= r)` tests actually need; and `all(Dist ~= 0)`, meant to
% exclude a dragonfly from its own neighbourhood, also excludes any other that
% shares even one coordinate exactly.
% Max_iteration is derived as floor(maxFe/N) so the run spends exactly the budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = da(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    N         = 40;
    Max_iter  = max(2, floor(maxFE / N));
    Delta_max = (ub - lb) / 10;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    X      = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    DeltaX = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);

    Food_fitness  = inf;
    Food_pos      = zeros(1, dim);
    Enemy_fitness = -inf;
    Enemy_pos     = zeros(1, dim);

    bsf  = inf;
    bsfx = X(1, :);

    LBm = repmat(lb, N, 1);
    UBm = repmat(ub, N, 1);

    % Main loop
    for iter = 1:Max_iter
        if FE >= maxFE
            break;
        end

        r  = (ub - lb) / 4 + (ub - lb) * (iter / Max_iter) * 2;
        w  = 0.9 - iter * ((0.9 - 0.4) / Max_iter);
        my_c = max(0, 0.1 - iter * (0.1 / (Max_iter / 2)));

        s = 2 * rand * my_c;      % separation
        a = 2 * rand * my_c;      % alignment
        c = 2 * rand * my_c;      % cohesion
        f = 2 * rand;             % food
        e = my_c;                 % enemy

        % Evaluate the whole swarm, update food and enemy
        [Fitness, FE] = calculate_fitness(X', problem, FE);
        Fitness = Fitness(:)';

        for i = 1:N
            if Fitness(i) < Food_fitness
                Food_fitness = Fitness(i);
                Food_pos     = X(i, :);
            end
            if Fitness(i) > Enemy_fitness && all(X(i, :) < ub) && all(X(i, :) > lb)
                Enemy_fitness = Fitness(i);
                Enemy_pos     = X(i, :);
            end
            if Fitness(i) < bsf
                bsf  = Fitness(i);
                bsfx = X(i, :);
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, Fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Move each dragonfly, in place
        for i = 1:N
            % Per-dimension "distance"; see the header note
            Dall = abs(X - X(i, :));
            nb   = all(Dall <= r, 2) & all(Dall ~= 0, 2);
            neighbours_no = sum(nb);

            if neighbours_no > 1
                S = -sum(X(nb, :) - X(i, :), 1);            % Eq. (3.1)
                A =  sum(DeltaX(nb, :), 1) / neighbours_no; % Eq. (3.2)
                C =  sum(X(nb, :), 1) / neighbours_no - X(i, :);   % Eq. (3.3)
            else
                S = zeros(1, dim);
                A = DeltaX(i, :);
                C = zeros(1, dim);
            end

            Dist2Food = abs(X(i, :) - Food_pos);            % Eq. (3.4)
            if all(Dist2Food <= r)
                F = Food_pos - X(i, :);
            else
                F = 0;
            end

            Dist2Enemy = abs(X(i, :) - Enemy_pos);          % Eq. (3.5)
            if all(Dist2Enemy <= r)
                Enemy = Enemy_pos + X(i, :);
            else
                Enemy = zeros(1, dim);
            end

            % Wrap a violating component to the opposite bound, fresh step
            hi = X(i, :) > ub;
            lo = X(i, :) < lb;
            if any(hi)
                X(i, hi)      = lb(hi);
                DeltaX(i, hi) = rand(1, sum(hi));
            end
            if any(lo)
                X(i, lo)      = ub(lo);
                DeltaX(i, lo) = rand(1, sum(lo));
            end

            if any(Dist2Food > r)
                if neighbours_no > 1
                    DeltaX(i, :) = w * DeltaX(i, :) + rand(1, dim) .* A ...
                                 + rand(1, dim) .* C + rand(1, dim) .* S;
                    DeltaX(i, :) = min(max(DeltaX(i, :), -Delta_max), Delta_max);
                    X(i, :)      = X(i, :) + DeltaX(i, :);
                else
                    X(i, :)      = X(i, :) + levyStep(dim) .* X(i, :);   % Eq. (3.8)
                    DeltaX(i, :) = 0;
                end
            else
                % Eq. (3.6)
                DeltaX(i, :) = (a * A + c * C + s * S + f * F + e * Enemy) + w * DeltaX(i, :);
                DeltaX(i, :) = min(max(DeltaX(i, :), -Delta_max), Delta_max);
                X(i, :)      = X(i, :) + DeltaX(i, :);
            end
        end

        X = min(max(X, LBm), UBm);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function o = levyStep(d)
% Mantegna's Levy step, as in the release's Levy.m (Eqs. 3.9 and 3.10).
    beta  = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    o = 0.01 * (u ./ abs(v) .^ (1 / beta));
end

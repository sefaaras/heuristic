% ----------------------------------------------------------------------- %
% Crayfish Optimization Algorithm (COA)
% Stored as crayfishoa; the acronym COA collides with coa (Coyote)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 30       % Population size (paper's experiments; the demo uses 100)
%   C1    = 0.2      % Food-intake scale, Eq. (4)
%   mu    = 25       % Most suitable temperature, Eq. (4)
%   sigma = 3        % Temperature spread of the intake, Eq. (4)
%   C3    = 3        % Food factor (largest food), Eq. (11)
%   C2    = 2 -> 1   % Summer-resort step coefficient (linear), Eq. (7)
%
% Algorithm Concept:
%   - temp = 20 + 15*rand per iteration; above 30 the crayfish explore
%     (summer resort or competition), otherwise they forage
%   - Summer resort: step C2*rand toward the cave Xshade, the midpoint of the
%     best-so-far and the best offspring of the previous iteration
%   - Competition: per dimension X_ij - X_zj + Xshade_j with a random crayfish z
%   - Foraging: food = best; size Q = C3*rand*f_i/f_food; a big food (Q > 2)
%     is torn, Xfood = exp(-1/Q)*Xfood, and eaten with a cos - sin step
%   - Small food is eaten directly, the step scaled by the intake p(temp)
%   - Greedy one-to-one replacement of each crayfish by its offspring
%
% Reference:
% Heming Jia, Honghua Rao, Changsheng Wen, Seyedali Mirjalili,
% Crayfish optimization algorithm,
% Artificial Intelligence Review 56 (Suppl 2) (2023) 1919-1979.
% https://doi.org/10.1007/s10462-023-10567-4
% ----------------------------------------------------------------------- %
% Implementation Note:
%   Ported from the authors' code, github.com/rao12138/COA-s-code (2023-07-27,
%   identical to COA.zip on Mirjalili's site). The reference evaluates the food
%   for every foraging crayfish and re-evaluates Xnew(1) once per iteration;
%   here the food is evaluated only after a tear has moved it (same values,
%   fewer FEs) and the duplicate is dropped. As in the reference, the torn food
%   accumulates over the crayfish of one iteration and is never a population
%   member, but a food better than the incumbent is reported. Eq. (11) is a raw
%   fitness ratio, so negative or zero fitness changes which branch is taken;
%   kept, since exp(-1/Q) is only taken for Q > 2 and every step stays finite.
%   t/T in Eq. (7) is read off the budget as FE/maxFE.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = crayfishoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N     = 30;
    C1    = 0.2;
    mu    = 25;
    sigma = 3;
    C3    = 3;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    % fbest/xbest is the reference's Best_fitness/best_position; frep/xrep also sees the food probes
    fbest = inf;
    xbest = X(1, :);
    for e = 1:N
        if fit(e) < fbest
            fbest = fit(e);
            xbest = X(e, :);
        end
        if e <= maxFE
            curve(e) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end
    frep = fbest;
    xrep = xbest;

    % X_L of Eq. (5): best offspring of the previous iteration (global_position in the reference)
    xlocal = xbest;

    while FE < maxFE
        C2   = 2 - FE / maxFE;                                    % Eq. (7)
        temp = rand * 15 + 20;                                    % Eq. (3)
        p    = C1 / (sqrt(2 * pi) * sigma) * exp(-(temp - mu)^2 / (2 * sigma^2));  % Eq. (4)
        xshade = (xbest + xlocal) / 2;                            % Eq. (5)

        % Eq. (10); f(Xfood) is known until a tear moves the food
        xfood  = xbest;
        ffood  = fbest;
        moved  = false;
        out    = false;
        Xnew   = X;
        for i = 1:N
            if temp > 30
                if rand < 0.5
                    Xnew(i, :) = X(i, :) + C2 * rand(1, dim) .* (xshade - X(i, :));    % Eq. (6)
                else
                    z = round(rand(1, dim) * (N - 1)) + 1;                             % Eq. (9)
                    Xnew(i, :) = X(i, :) - X(z + (0:dim - 1) * N) + xshade;            % Eq. (8)
                end
            else
                if moved
                    if FE >= maxFE
                        out = true;
                        break;
                    end
                    % calculate_fitness clamps on the constrained suites, and 0 lies in every CEC box
                    xprobe = min(max(xfood, lb), ub);
                    [ffood, FE] = calculate_fitness(xprobe', problem, FE);
                    ffood = ffood(1);
                    moved = false;
                    if ffood < frep
                        frep = ffood;
                        xrep = xprobe;
                    end
                    curve(FE) = frep;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fit, population_history, fitness_history, ...
                        history_index, maxFE);
                end
                Q = C3 * rand * fit(i) / ffood;                                        % Eq. (11)
                if Q > (C3 + 1) / 2
                    torn = exp(-1 / Q) * xfood;                                        % Eq. (12)
                    moved = moved || any(torn ~= xfood);
                    xfood = torn;
                    Xnew(i, :) = X(i, :) + xfood * p .* (cos(2 * pi * rand(1, dim)) - sin(2 * pi * rand(1, dim)));  % Eq. (13)
                else
                    Xnew(i, :) = (X(i, :) - xfood) * p + p * rand(1, dim) .* X(i, :);  % Eq. (14)
                end
            end
        end
        if out
            break;
        end

        Xnew = min(max(Xnew, lb), ub);
        m = min(N, maxFE - FE);
        [fnew, FE] = calculate_fitness(Xnew(1:m, :)', problem, FE);
        fnew = fnew(:)';

        [~, k] = min(fnew);
        xlocal = Xnew(k, :);

        for i = 1:m
            if fnew(i) < fit(i)
                X(i, :) = Xnew(i, :);
                fit(i) = fnew(i);
                if fit(i) < fbest
                    fbest = fit(i);
                    xbest = X(i, :);
                    if fbest < frep
                        frep = fbest;
                        xrep = xbest;
                    end
                end
            end
            ec = FE - m + i;
            curve(ec) = frep;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = frep;
    best_fitness  = frep;
    best_solution = xrep;
end

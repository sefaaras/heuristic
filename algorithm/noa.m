% ----------------------------------------------------------------------- %
% Nutcracker Optimization Algorithm (NOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                      % Population size
%   Alpha = 0.05                % Chance of the box-scaled jump in late exploration
%   Pa1 = 1 - t                 % Chance a foraging move explores, t the spent budget
%   Pa2 = 0.2                   % Chance of the recovery stage over the cache search
%   Prb = 0.2                   % Chance a coordinate of the second reference point moves
%   RL = 0.05*Levy(1.5)         % Levy step matrix, redrawn every sweep
%
% Algorithm Concept:
%   - Every sweep picks one of two strategies by a coin flip: foraging-and-storage
%     or cache-search-and-recovery
%   - Foraging explores with probability 1-t: early around the population mean plus a
%     Levy-weighted random difference and a box-scaled jump, later around a random member
%   - Otherwise it exploits: a Levy pull towards the best, a random difference added to
%     the best, or the best shrunk towards the origin by |l|, l = rand*(1-t)
%   - Cache search rotates two reference points off the member by a random angle and
%     evaluates both; the better replaces the member if it improves it (Eq. 17)
%   - Recovery (probability Pa2) moves towards the best and one reference point
%   - Each member keeps its own best (Eq. 20): a move that does not improve is undone
%   - Out-of-box coordinates are redrawn uniformly or clamped, by a coin flip per move
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Mohammed Jameel, Mohamed Abouhawwash,
% Nutcracker optimizer: A novel nature-inspired metaheuristic algorithm for
% global optimization and engineering design problems,
% Knowledge-Based Systems 262 (2023) 110248.
% https://doi.org/10.1016/j.knosys.2022.110248
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' function-evaluation-based CEC2020 release
% (github.com/redamohamed8/Function-evaluation-based-NOA-for-CEC2020, NOA.m),
% whose search is line for line the main repository's NOA.m; N = 30 is its
% driver's (the classic-function demo uses 25). Its clock t already counts
% evaluations, so t/T is FE/maxFe, now including the initial sweep. Kept as
% released: a = (t/T)^(2/t) on the absolute count, the always-true (rand < 5)
% factor on the early jump, and a second reference point offset by
% a*cos(theta)*ub (written (ub-lb)+lb, no random factor). The release's
% theta == pi/2 branch (probability 2^-53) is left out.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = noa(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    N = 30;
    Alpha = 0.05;
    Pa2 = 0.2;
    Prb = 0.2;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    fit = inf(N, 1);
    bsf = inf;
    bsf_solution = X(1, :);

    n = min(N, maxFE);
    [fv, FE] = calculate_fitness(X(1:n, :)', problem, FE);
    fit(1:n) = fv(:);
    for k = 1:n
        if fit(k) < bsf
            bsf = fit(k);
            bsf_solution = X(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, X(1:n, :), fit(1:n), population_history, fitness_history, history_index, maxFE);
    end

    % X doubles as the local-best memory: Eq. (20) undoes every move that does not improve
    best_sc = bsf;
    best_nc = bsf_solution;

    while FE < maxFE
        RL = 0.05 * levy(N, dim, 1.5);
        t = FE / maxFE;
        l = rand * (1 - t);                                   % Eq. (3)
        if rand < rand                                        % Eq. (11)
            a = t ^ (2 / FE);
        else
            a = (1 - t) ^ (2 * t);
        end

        if rand < rand
            mo = mean(X, 1);
            for i = 1:N
                if FE >= maxFE
                    break;
                end
                if rand < rand                                % Eq. (2)
                    mu = rand;
                elseif rand < rand
                    mu = randn;
                else
                    mu = RL(1, 1);
                end
                cv = randi(N);
                cv1 = randi(N);
                t = FE / maxFE;
                x = X(i, :);
                if rand < 1 - t                               % exploration, Pa1 = (T - t)/T
                    cv2 = randi(N);
                    r2 = rand;
                    move = rand(1, dim) > rand(1, dim);
                    if t < 0.5
                        cand = mo + RL(i, :) .* (X(cv, :) - X(cv1, :)) + mu * (r2 * r2 * ub - lb);   % Eq. (1)
                    else
                        cand = X(cv2, :) + mu * (X(cv, :) - X(cv1, :)) ...
                               + mu * (rand(1, dim) < Alpha) .* (r2 * r2 * ub - lb);              % Eq. (1)
                    end
                    x(move) = cand(move);
                else
                    mu = rand;
                    if rand < rand
                        r1 = rand;
                        x = x + mu * abs(RL(i, :)) .* (best_nc - x) + r1 * (X(cv, :) - X(cv1, :));  % Eq. (3)
                    elseif rand < rand
                        move = rand(1, dim) > rand(1, dim);
                        cand = best_nc + mu * (X(cv, :) - X(cv1, :));                              % Eq. (3)
                        x(move) = cand(move);
                    else
                        x = best_nc * abs(l);                                                      % Eq. (3)
                    end
                end
                x = return_to_box(x, lb, ub);

                [fv, FE] = calculate_fitness(x', problem, FE);
                fx = fv(1);
                if fx < bsf
                    bsf = fx;
                    bsf_solution = x;
                end
                curve(FE) = bsf;
                if fx < fit(i)
                    X(i, :) = x;
                    fit(i) = fx;
                end
                if fit(i) < best_sc
                    best_sc = fit(i);
                    best_nc = X(i, :);
                end
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, history_index, maxFE);
            end
        else
            for i = 1:N
                if FE >= maxFE
                    break;
                end
                ang = pi * rand;
                cv = randi(N);
                cv1 = randi(N);
                x = X(i, :);
                RP1 = x + a * cos(ang) * (X(cv, :) - X(cv1, :));                        % Eq. (9)
                RP2 = x + a * cos(ang) * ((ub - lb) + lb) .* (rand(1, dim) < Prb);      % Eq. (10)
                RP2 = return_to_box(RP2, lb, ub);
                RP1 = return_to_box(RP1, lb, ub);

                if rand < Pa2                                 % recovery stage
                    cv = randi(N);
                    if rand < rand
                        ref = RP1;                            % Eq. (13)
                    else
                        ref = RP2;                            % Eq. (15)
                    end
                    move = rand(1, dim) > rand(1, dim);
                    cand = x + rand(1, dim) .* (best_nc - x) + rand(1, dim) .* (ref - X(cv, :));
                    x(move) = cand(move);
                    x = return_to_box(x, lb, ub);

                    [fv, FE] = calculate_fitness(x', problem, FE);
                    fx = fv(1);
                    if fx < bsf
                        bsf = fx;
                        bsf_solution = x;
                    end
                    curve(FE) = bsf;
                    if fx < fit(i)
                        X(i, :) = x;
                        fit(i) = fx;
                    end
                else                                          % cache-search stage
                    [fv, FE] = calculate_fitness(RP1', problem, FE);
                    f1 = fv(1);
                    if f1 < bsf
                        bsf = f1;
                        bsf_solution = RP1;
                    end
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fit, population_history, fitness_history, history_index, maxFE);
                    if FE >= maxFE
                        break;
                    end

                    [fv, FE] = calculate_fitness(RP2', problem, FE);
                    f2 = fv(1);
                    if f2 < bsf
                        bsf = f2;
                        bsf_solution = RP2;
                    end
                    curve(FE) = bsf;
                    if f2 < f1 && f2 < fit(i)                 % Eq. (17)
                        X(i, :) = RP2;
                        fit(i) = f2;
                    elseif f1 < f2 && f1 < fit(i)
                        X(i, :) = RP1;
                        fit(i) = f1;
                    end
                end
                if fit(i) < best_sc
                    best_sc = fit(i);
                    best_nc = X(i, :);
                end
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, fit, population_history, fitness_history, history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness = bsf;
    best_solution = bsf_solution;
end

% Non-finite coordinates are redrawn first, then a coin flip picks redraw or clamp
function x = return_to_box(x, lb, ub)
    bad = ~isfinite(x);
    if any(bad)
        r = lb + rand(1, numel(x)) .* (ub - lb);
        x(bad) = r(bad);
    end
    if rand < rand
        out = x > ub | x < lb;
        r = lb + rand(1, numel(x)) .* (ub - lb);
        x(out) = r(out);
    else
        x = min(max(x, lb), ub);
    end
end

% Mantegna's Levy draw, randn in place of the Statistics-toolbox random()
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
    sigma_u = (num / den) ^ (1 / beta);
    u = sigma_u * randn(n, m);
    v = randn(n, m);
    z = u ./ (abs(v) .^ (1 / beta));
end

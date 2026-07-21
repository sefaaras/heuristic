% ----------------------------------------------------------------------- %
% Smell Agent Optimization (SAO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters (author's canonical values):
%   N   = 50               % Population size (smell molecules)
%   T   = 3                % Temperature (constant)
%   K   = 1.38064852e-23   % Boltzmann constant
%   m   = 2.4              % Molecular mass
%   olf = 3.5              % Olfaction capacity
%   SN  = 2.5              % Random-mode step
%
% Algorithm Concept:
%   - Sniffing mode: accumulate a kinetic-gas velocity v and move x = x + v
%   - Trailing mode: move toward the best (agent) and away from the worst
%   - Random mode: if trailing did not improve on sniffing, escape with a
%     random step + re-sniff + re-trail
%
% Reference:
% Abdulkarim T. Salawudeen, Muhammed B. Mu'azu, Yusuf A. Sha'aban,
% Adedokun E. Adedokun, A Novel Smell Agent Optimization (SAO): An
% extensive CEC study and engineering application,
% Knowledge-Based Systems 232 (2021) 107486.
% https://doi.org/10.1016/j.knosys.2021.107486
% Reference code: github.com/SALAWUDEEN/Smell-Agent-Optimization
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sao(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 50;
    T = 3;
    K = 1.38064852e-23;
    m = 2.4;
    olf = 3.5;
    SN = 2.5;
    smell_step = sqrt(3 * K * T / m);

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    molecules = initialization(N, dim, ub, lb);
    v = molecules * 0.1;   % initial velocity (10% of positions)

    [y, FE] = calculate_fitness(molecules', problem, FE);
    y = y(:)';

    [bsf, bidx] = min(y);
    best_pos = molecules(bidx, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, molecules, y, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    while FE < maxFE
        % --- Sniffing mode ---
        v = v + rand(N, dim) * smell_step;
        molecules = molecules + v;
        for i = 1:N
            molecules(i, :) = bound(molecules(i, :), ub, lb);
        end
        [ys, FE] = calculate_fitness(molecules', problem, FE);
        ys = ys(:)';
        for i = 1:N
            if ys(i) < bsf, bsf = ys(i); best_pos = molecules(i, :); end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, molecules, ys, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE, break; end

        % --- Select smell source (best) and worst molecule ---
        [~, ai] = min(ys); x_agent = molecules(ai, :);
        [~, wi] = max(ys); x_worst = molecules(wi, :);

        % --- Trailing mode (toward agent, away from worst) ---
        molecules = molecules + rand(N, dim) * olf .* (repmat(x_agent, N, 1) - molecules) ...
                              - rand(N, dim) * olf .* (repmat(x_worst, N, 1) - molecules);
        for i = 1:N
            molecules(i, :) = bound(molecules(i, :), ub, lb);
        end
        [yt, FE] = calculate_fitness(molecules', problem, FE);
        yt = yt(:)';
        for i = 1:N
            if yt(i) < bsf, bsf = yt(i); best_pos = molecules(i, :); end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, molecules, yt, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        if FE >= maxFE, break; end

        % --- Random mode: escape when trailing failed to improve on sniffing ---
        for i = 1:N
            if yt(i) > ys(i)
                molecules(i, :) = molecules(i, :) + rand(1, dim) * SN;
                molecules(i, :) = molecules(i, :) + (v(i, :) + rand(1, dim) * smell_step);
                molecules(i, :) = molecules(i, :) + rand(1, dim) * olf .* (x_agent - molecules(i, :)) ...
                                                  - rand(1, dim) * olf .* (x_worst - molecules(i, :));
                molecules(i, :) = bound(molecules(i, :), ub, lb);
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

%% --- Initialization ---
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

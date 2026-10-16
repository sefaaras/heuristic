% ----------------------------------------------------------------------- %
% White Shark Optimizer (WSO)
% Not wsoa (War Strategy Optimization): a different method
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   whiteSharks = 30               % Population size
%   fmin, fmax  = 0.07, 0.75       % Wavy motion steps v/f, f = fmin + (fmax-fmin)/(fmax+fmin)
%   tau         = 4.11             % Constriction mu = 2/|2 - tau - sqrt(tau^2 - 4*tau)| = 0.7188
%   rr          = U(1, 3)          % Velocity gain wr = |2*r1 - (r2 + r3)|/rr
%   a0, a1      = 6.25, 100        % mv = 1/(a0 + exp((T/2 - t)/a1)), ~0 -> 0.16
%   a2          = 0.0005           % s_s = 1 - exp(-a2*t/T), 0 -> 5e-4
%
% Algorithm Concept:
%   - Velocity v = mu*v + wr*(wbest_nu - x): each shark is pulled toward the
%     personal best of a randomly drawn shark nu, with no global-best term
%   - Wavy motion x = x + v/f; with probability mv instead, only the coordinates
%     outside the box are reset to the violated bound
%   - Fishing school: with probability s_s per coordinate, a jump around gbest
%     by a random fraction of |gbest - x|, averaged with the previous shark
%   - Only sharks inside the box are evaluated; a better position replaces the
%     shark's personal best wbest
%
% Reference:
% Malik Braik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar,
% Mohammed A. Awadallah,
% White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for
% global optimization problems,
% Knowledge-Based Systems 243 (2022) 108457.
% https://doi.org/10.1016/j.knosys.2022.108457
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's File Exchange 107365 v1.0.0 (WSO.m, 2022-02-28),
% which departs from the paper: the paper's velocity (p1, p2, a gbest term) is
% commented out in favour of the wr form, so p1/p2 are never used, and the
% fishing-school divisor 2*rand is coded /2*rand. Kept: gbest is refreshed to
% wbest(index), index being the INITIAL best shark. Out-of-box sharks are not
% evaluated, so T = ceil((maxFE - N)/N) is a plan: the loop runs to the budget
% with t clamped at T (as clpso). An in-box shark that did not move is not
% re-evaluated (the release repeats the same value: 9-11 % of its spend on cec14,
% 45 % on D=2 F23), so a run takes 1.10-1.83 T iterations. The history records
% the sharks sitting at the position their fitness was measured at.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = wso(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    whiteSharks = 30;
    itemax = max(1, ceil((maxFE - whiteSharks) / whiteSharks));   % planned horizon at one evaluation per shark
    fmax = 0.75;
    fmin = 0.07;
    tau = 4.11;
    mu = 2 / abs(2 - tau - sqrt(tau^2 - 4 * tau));
    a0 = 6.250;
    a1 = 100;
    a2 = 0.0005;
    rmin = 1;
    rmax = 3.0;
    f = fmin + (fmax - fmin) / (fmax + fmin);

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    WSO_Positions = rand(whiteSharks, dim) .* (ub - lb) + lb;
    v = 0.0 * WSO_Positions;
    [fit, FE] = calculate_fitness(WSO_Positions', problem, FE);
    fit = fit(:);
    fitness = fit;
    [fmin0, index] = min(fit);
    wbest = WSO_Positions;
    gbest = WSO_Positions(index, :);
    lastX = WSO_Positions;   % where each fit(i) was measured

    bsf = fmin0;
    bsx = gbest;
    for e = 1:min(whiteSharks, maxFE)
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, WSO_Positions, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    ite = 0;
    while FE < maxFE
        ite = ite + 1;
        t = min(ite, itemax);   % past the planned horizon the schedules hold their final values
        mv = 1 / (a0 + exp((itemax / 2.0 - t) / a1));
        s_s = abs((1 - exp(-a2 * t / itemax)));

        nu = floor((whiteSharks) .* rand(1, whiteSharks)) + 1;
        for i = 1:whiteSharks
            rr = rmin + rand() * (rmax - rmin);
            wr = abs(((2 * rand()) - (1 * rand() + rand())) / rr);
            v(i, :) = mu * v(i, :) + wr * (wbest(nu(i), :) - WSO_Positions(i, :));
        end

        for i = 1:whiteSharks
            a = sign(WSO_Positions(i, :) - ub) > 0;
            b = sign(WSO_Positions(i, :) - lb) < 0;
            if rand < mv
                % The release's x.*(~xor(a,b)) + ub.*a + lb.*b, without its Inf*0 = NaN
                WSO_Positions(i, a) = ub(a);
                WSO_Positions(i, b) = lb(b);
            else
                WSO_Positions(i, :) = WSO_Positions(i, :) + v(i, :) / f;
            end
        end

        for i = 1:whiteSharks
            for j = 1:dim
                if rand < s_s
                    Dist = abs(rand * (gbest(j) - 1 * WSO_Positions(i, j)));
                    if i == 1
                        WSO_Positions(i, j) = gbest(j) + rand * Dist * sign(rand - 0.5);
                    else
                        WSO_Pos = gbest(j) + rand * Dist * sign(rand - 0.5);
                        WSO_Positions(i, j) = (WSO_Pos + WSO_Positions(i - 1, j)) / 2 * rand;   % paper: / (2*rand)
                    end
                end
            end
        end

        inbox = all(WSO_Positions >= lb & WSO_Positions <= ub, 2);
        moved = any(WSO_Positions ~= lastX, 2);
        ev = find(inbox & moved);
        ev = ev(1:min(end, maxFE - FE));
        if isempty(ev)
            continue;
        end
        nev = numel(ev);
        [fev, FE] = calculate_fitness(WSO_Positions(ev, :)', problem, FE);
        fit(ev) = fev(:);
        lastX(ev, :) = WSO_Positions(ev, :);
        rec = all(WSO_Positions == lastX, 2);   % moved out-of-box sharks have no fitness here
        for k = 1:nev
            i = ev(k);
            if fit(i) < fitness(i)
                wbest(i, :) = WSO_Positions(i, :);
                fitness(i) = fit(i);
            end
            if fitness(i) < fmin0
                fmin0 = fitness(i);
                gbest = wbest(index, :);   % index is the initial best shark's, as released
            end
            if fit(i) < bsf
                bsf = fit(i);
                bsx = WSO_Positions(i, :);
            end
            ec = FE - nev + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, WSO_Positions(rec, :), fit(rec), population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsx;
end

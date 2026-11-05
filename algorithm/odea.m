% ----------------------------------------------------------------------- %
% Opposition-Based Differential Evolution (ODE)
% Stored as odea; ode is a MATLAB builtin (funfun/ode.m)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Np = 100   % Population size
%   F  = 0.5   % Differential amplification factor
%   Cr = 0.9   % Crossover probability constant
%   Jr = 0.3   % Jumping rate, probability of a generation jump
%
% Algorithm Concept:
%   - DE/rand/1/bin: v = x_r1 + F*(x_r2 - x_r3) with i, r1, r2, r3 distinct,
%     binomial crossover, and the trial replaces its target if f(u) <= f(x)
%   - Opposition-based initialisation: a random population and its opposite
%     a + b - x through the box are evaluated, the Np fittest of both start the run
%   - Generation jumping: with probability Jr the opposite min_j + max_j - x through
%     the population's own range is evaluated and the Np fittest of both survive
%
% Reference:
% Shahryar Rahnamayan, Hamid R. Tizhoosh, Magdy M. A. Salama,
% Opposition-Based Differential Evolution,
% IEEE Transactions on Evolutionary Computation 12(1) (2008) 64-79.
% https://doi.org/10.1109/TEVC.2007.894200
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Table III, parameters of Sec. V); no author code is public.
% Crossover is the paper's Eq. (3), i.e. with the j = k index that the Table III
% pseudocode leaves out. The paper gives no bound handling, so violating donor
% components are clamped to the box. Both opposites mirror through a centre (the
% box's at initialisation, the population's range when jumping), which pulls the
% search towards it; kept as published.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = odea(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NP = 100;
    F  = 0.5;
    Cr = 0.9;
    Jr = 0.3;

    rows    = (1:NP)';
    diagIdx = sub2ind([NP NP], rows, rows);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Table III steps 1-5: P0 and its opposite through the box, Eq. (6)
    P0 = lb + rand(NP, D) .* (ub - lb);
    C  = [P0; lb + ub - P0];
    nEval = min(2 * NP, maxFE);
    C  = C(1:nEval, :);
    [fc, FE] = calculate_fitness(C', problem, FE);
    fc = fc(:);

    best_fitness  = inf;
    best_solution = C(1, :);
    for e = 1:nEval
        if fc(e) < best_fitness
            best_fitness  = fc(e);
            best_solution = C(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, C, fc, population_history, fitness_history, history_index, maxFE);
    end
    [~, ord] = sort(fc);
    keep = ord(1:min(NP, nEval));
    pop  = C(keep, :);
    fit  = fc(keep);

    while FE < maxFE
        % Eq. (1): r1, r2, r3 mutually different and different from i
        K = rand(NP);
        K(diagIdx) = inf;
        [~, ord] = sort(K, 2);
        V = pop(ord(:, 1), :) + F * (pop(ord(:, 2), :) - pop(ord(:, 3), :));
        V = min(max(V, lb), ub);

        % Eq. (3)
        mask = rand(NP, D) <= Cr;
        mask(sub2ind([NP D], rows, randi(D, NP, 1))) = true;
        U = pop;
        U(mask) = V(mask);

        nEval = min(NP, maxFE - FE);   % the last generation is cut to the remaining budget
        [fu, FE] = calculate_fitness(U(1:nEval, :)', problem, FE);
        fu = fu(:);
        for e = 1:nEval
            if fu(e) < best_fitness
                best_fitness  = fu(e);
                best_solution = U(e, :);
            end
            ec = FE - nEval + e;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, pop, fit, population_history, fitness_history, history_index, maxFE);
        end

        % Table III lines 20-25; a NaN target is replaced by any non-NaN trial
        fp = fit(1:nEval);
        repl = find(fu <= fp | (isnan(fp) & ~isnan(fu)));
        pop(repl, :) = U(repl, :);
        fit(repl) = fu(repl);

        % Table III steps 26-32: generation jumping through the population's range, Eq. (7)
        if FE < maxFE && rand < Jr
            OP = min(pop, [], 1) + max(pop, [], 1) - pop;
            nEval = min(NP, maxFE - FE);
            OP = OP(1:nEval, :);
            [fo, FE] = calculate_fitness(OP', problem, FE);
            fo = fo(:);
            for e = 1:nEval
                if fo(e) < best_fitness
                    best_fitness  = fo(e);
                    best_solution = OP(e, :);
                end
                ec = FE - nEval + e;
                curve(ec) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fit, population_history, fitness_history, history_index, maxFE);
            end
            C  = [pop; OP];
            fc = [fit; fo];
            [~, ord] = sort(fc);
            pop = C(ord(1:NP), :);
            fit = fc(ord(1:NP));
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

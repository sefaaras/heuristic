% ----------------------------------------------------------------------- %
% Differential Evolution with Adaptive Hill-Climbing Simplex Crossover (DEahcSPX)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = max(30, D)   % Population size P
%   F  = 0.9          % Scaling factor of the differential mutation
%   Cr = 0.9          % Binomial crossover probability
%   n_p = 3           % SPX parents: the best individual and n_p - 1 random members
%   epsilon = 1.0     % SPX expansion rate
%
% Algorithm Concept:
%   - DE/rand/1/bin: v = x_j + F*(x_k - x_l) with i, j, k, l distinct, binomial
%     crossover, and the trial replaces its target if f(u) <= f(x)
%   - Each generation, before reproduction, AHCXLS refines the best individual
%     by simplex crossover (SPX) with n_p - 1 randomly chosen partners
%   - SPX: y_i = O + epsilon*(x_i - O) about the parents' centroid O, chained as
%     C_i = r_{i-1}*(y_{i-1} - y_i + C_{i-1}), r_i = u^(1/(i+1)); child y_np + C_np
%   - Hill climbing: while the child beats the first parent it takes its place,
%     and the best individual's (Lamarckian), and SPX runs again on the same partners
%
% Reference:
% Nasimul Noman, Hitoshi Iba,
% Accelerating Differential Evolution Using an Adaptive Local Search,
% IEEE Transactions on Evolutionary Computation 12(1) (2008) 107-125.
% https://doi.org/10.1109/TEVC.2007.895272
% Components:
% DE -- R. Storn, K. Price, J. Global Optim. 11(4) (1997) 341-359, doi:10.1023/A:1008202821328
% SPX -- S. Tsutsui, M. Yamamura, T. Higuchi, Proc. GECCO 1999, 657-664 (no DOI)
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Figs. 2-3, Secs. V-B and V-E); no author code is public.
% It runs P = 30 at N = 10 and P = N at N = 30..200; max(30, D) reproduces both and
% extends to D < 10 (P = N alone leaves DE/rand/1 short of four members at D < 4).
% Left open, chosen here: the SPX partners are distinct and exclude the best, and a
% trial wins a tie. SPX is Fig. 3 as printed (r_i = u^(1/(i+1)) from i = 1), so the
% child leans to the best parent; epsilon = 1 keeps it inside the parents' simplex.
% No bound handling is given; violating mutant components are redrawn uniformly in
% the box, as clamping traps the paper's own DE baseline in a corner of Ackley
% (N = 30, 3e5 FE, 50 seeds, runs below 1e-6; the paper has 50/50 for both):
%    clamp:   DE  0/50 (all 19.96)   DEahcSPX 46/50
%    redraw:  DE 50/50 (7.5e-10)     DEahcSPX 50/50 (3.1e-15)
%    CEC2020RW D <= 30, 38 problems x 3 seeds, median best feasible f: redraw 8, clamp 8, tie 22
% The budget is checked before every SPX child, so a run can end inside a climb.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = deahcspx(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NP      = max(30, D);
    F       = 0.9;
    Cr      = 0.9;
    n_p     = 3;
    epsilon = 1.0;

    rows    = (1:NP)';
    diagIdx = sub2ind([NP NP], rows, rows);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Fig. 2(b) steps 1-2
    pop = lb + rand(NP, D) .* (ub - lb);
    n0  = min(NP, maxFE);
    [f0, FE] = calculate_fitness(pop(1:n0, :)', problem, FE);
    fit = inf(NP, 1);
    fit(1:n0) = f0(:);

    best_fitness  = inf;
    best_solution = pop(1, :);
    for e = 1:n0
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = pop(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, pop(1:n0, :), fit(1:n0), population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        % Fig. 2(b) steps 3-4, Fig. 2(a) steps 1-4: the best and n_p - 1 random partners
        [~, B] = min(fit);
        mates = randperm(NP - 1, n_p - 1);
        mates = mates + (mates >= B);
        par   = [pop(B, :); pop(mates, :)];
        f1    = fit(B);

        % Fig. 2(a) steps 5-8: climb while the child beats P[1]; partners stay fixed
        while FE < maxFE
            C = spx(par, epsilon);
            C = min(max(C, lb), ub);   % a convex combination of in-box parents; absorbs rounding only
            [fc, FE] = calculate_fitness(C', problem, FE);
            fc = fc(1);
            if fc < best_fitness
                best_fitness  = fc;
                best_solution = C;
            end
            curve(FE) = best_fitness;
            improved = fc < f1;
            if improved
                par(1, :) = C;
                f1 = fc;
                pop(B, :) = C;
                fit(B)    = fc;
            end
            [population_history, fitness_history, history_index] = record_history(...
                FE, pop, fit, population_history, fitness_history, history_index, maxFE);
            if ~improved
                break;
            end
        end
        if FE >= maxFE
            break;
        end

        % Fig. 2(b) steps 5-7, Eq. (1): j, k, l mutually different and different from i
        K = rand(NP);
        K(diagIdx) = inf;
        [~, ord] = sort(K, 2);
        V = pop(ord(:, 1), :) + F * (pop(ord(:, 2), :) - pop(ord(:, 3), :));
        % Violating components redrawn uniformly in the box, per dimension (Implementation Note)
        bad = V < lb | V > ub;
        R = lb + rand(NP, D) .* (ub - lb);
        V(bad) = R(bad);

        % Eq. (2)
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

        % Knockout selection into P^(G+1); a NaN target is replaced by any non-NaN trial
        fp = fit(1:nEval);
        repl = find(fu <= fp | (isnan(fp) & ~isnan(fu)));
        pop(repl, :) = U(repl, :);
        fit(repl) = fu(repl);
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

function C = spx(par, epsilon)
% Fig. 3, with par(1, :) the individual AHCXLS refines
    n = size(par, 1);
    O = mean(par, 1);
    Y = O + epsilon * (par - O);
    r = rand(n - 1, 1) .^ (1 ./ ((1:n - 1)' + 1));
    Cc = zeros(1, size(par, 2));
    for i = 2:n
        Cc = r(i - 1) * (Y(i - 1, :) - Y(i, :) + Cc);
    end
    C = Y(n, :) + Cc;
end

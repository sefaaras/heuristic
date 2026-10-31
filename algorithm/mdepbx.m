% ----------------------------------------------------------------------- %
% Modified Differential Evolution with p-Best Crossover (MDE_pBX)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP  = 100                        % Population size, the same for every D
%   q   = 0.15                       % Group size of current-to-gr_best/1, share of NP
%   p   = NP/2 -> 1 (linear, ceil)   % Top-ranked crossover partners, Eq. (7)
%   Fm  = 0.5                        % Initial Cauchy location of F
%   CRm = 0.6                        % Initial Gaussian mean of CR
%   wF  = 0.8 + 0.2*rand             % Weight of the old Fm, drawn per generation
%   wCr = 0.9 + 0.1*rand             % Weight of the old CRm, drawn per generation
%   n   = 1.5                        % Exponent of the power mean of successful F, CR
%
% Algorithm Concept:
%   - DE/current-to-gr_best/1: v = x + F*(x_grbest - x + x_r1 - x_r2), x_grbest the
%     best of a fresh random group of q*NP other members drawn for each target
%   - p-best crossover: binomial crossover of the donor with a random member of the
%     current top p instead of the target; p shrinks linearly from NP/2 to 1
%   - F ~ Cauchy(Fm, 0.1) and CR ~ N(CRm, 0.1) per individual, redrawn until inside
%     (0, 1] and [0, 1]; the trial replaces its target if f(u) <= f(x)
%   - Fm and CRm move towards a power mean of the generation's strictly successful
%     F and CR, with the weights wF and wCr redrawn every generation
%
% Reference:
% Sk. Minhazul Islam, Swagatam Das, Saurav Ghosh, Subhrajit Roy,
% Ponnuthurai Nagaratnam Suganthan,
% An Adaptive Differential Evolution Algorithm With Novel Mutation and Crossover
% Strategies for Global Numerical Optimization,
% IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 42(2)
% (2012) 482-500.
% https://doi.org/10.1109/TSMCB.2011.2167966
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Sec. III, Eqs. (6)-(13), NP from Sec. IV-B); no author code
% is public. Changed: (10)/(13) print the power mean as sum over S of (x^n/|S|)^(1/n),
% i.e. |S|^(1/3)*mean(S); it drives CRm to ~4, where the redraw of (11) never ends
% (capped, CR = 1 on 99.8 % of draws and the p-best crossover is inert). The textbook
% power mean (sum(x^n)/|S|)^(1/n) of the cited [26] is used instead, as in PaGMO, and a
% generation without a success leaves Fm, CRm unchanged. Neither form reproduces Table I
% (D=30, 3e5 FE, f1: paper 1.34e-62, printed 2.0e3, textbook 0.59). Paper silent, chosen:
% group from the NP-1 others so the four vectors of (6) differ; the partner may be the
% target; Gmax = generations the budget allows; no bound handling in the paper, so
% violating donor components are clamped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mdepbx(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NP  = 100;
    q   = 0.15;
    n   = 1.5;
    Fm  = 0.5;
    CRm = 0.6;

    qn   = round(q * NP);
    Gmax = ceil((maxFE - NP) / NP);
    rows = (1:NP)';
    diagIdx = sub2ind([NP NP], rows, rows);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Eq. (2)
    pop = lb + rand(NP, D) .* (ub - lb);
    [fit, FE] = calculate_fitness(pop', problem, FE);
    fit = fit(:);

    best_fitness  = inf;
    best_solution = pop(1, :);
    for i = 1:NP
        if fit(i) < best_fitness
            best_fitness  = fit(i);
            best_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    G = 0;
    while FE < maxFE
        G = G + 1;

        % Eq. (8): Cauchy(Fm, 0.1), redrawn while F <= 0 or F > 1
        F = Fm + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
        F = redraw(F, @(m) Fm + 0.1 * tan(pi * (rand(m, 64) - 0.5)), @(v) v > 0 & v <= 1, inf);
        % Eq. (11): N(CRm, 0.1), redrawn while outside [0, 1]
        CR = CRm + 0.1 * randn(NP, 1);
        CR = redraw(CR, @(m) CRm + 0.1 * randn(m, 64), @(v) v >= 0 & v <= 1, inf);

        % Group of qn distinct others per target; its best is x_gr_best of Eq. (6)
        K = rand(NP);
        K(diagIdx) = inf;
        [~, ord] = sort(K, 2);
        grp = ord(:, 1:qn);
        [~, k] = min(reshape(fit(grp), NP, qn), [], 2);
        gr = grp(sub2ind([NP qn], rows, k));

        % r1 ~= r2, both outside {i, gr_best}
        K = rand(NP);
        K(diagIdx) = inf;
        K(sub2ind([NP NP], rows, gr)) = inf;
        [~, r1] = min(K, [], 2);
        K(sub2ind([NP NP], rows, r1)) = inf;
        [~, r2] = min(K, [], 2);

        % Eq. (6), then the clamp: the paper specifies no bound handling
        V = pop + F .* (pop(gr, :) - pop + pop(r1, :) - pop(r2, :));
        V = min(max(V, lb), ub);

        % Eq. (7) and the p-best crossover, Eq. (4) with a top-p member in place of x_i
        p = ceil(NP / 2 * (1 - (G - 1) / Gmax));
        [~, rnk] = sort(fit);
        partner = rnk(randi(p, NP, 1));
        mask = rand(NP, D) <= CR;
        mask(sub2ind([NP D], rows, randi(D, NP, 1))) = true;
        U = pop(partner, :);
        U(mask) = V(mask);

        nEval = min(NP, maxFE - FE);   % the last generation is cut to the remaining budget
        [fu, FE] = calculate_fitness(U(1:nEval, :)', problem, FE);
        fu = fu(:);

        for i = 1:nEval
            if fu(i) < best_fitness
                best_fitness  = fu(i);
                best_solution = U(i, :);
            end
            ec = FE - nEval + i;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, pop, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end

        % Eq. (5); a NaN target is replaced by any non-NaN trial
        fp = fit(1:nEval);
        better = find(fu < fp);
        repl = find(fu <= fp | (isnan(fp) & ~isnan(fu)));
        pop(repl, :) = U(repl, :);
        fit(repl) = fu(repl);

        % Eqs. (9a)-(13) with the textbook power mean (see the note)
        if ~isempty(better)
            wF  = 0.8 + 0.2 * rand;
            Fm  = wF * Fm + (1 - wF) * mean(F(better) .^ n) ^ (1 / n);
            wCr = 0.9 + 0.1 * rand;
            CRm = wCr * CRm + (1 - wCr) * mean(CR(better) .^ n) ^ (1 / n);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

% Rejection sampling, 64 candidates per pending entry per pass, at most `passes` passes
function v = redraw(v, draw, inside, passes)
    bad = find(~inside(v));
    pass = 0;
    while ~isempty(bad) && pass < passes
        pass = pass + 1;
        C = draw(numel(bad));
        ok = inside(C);
        hit = any(ok, 2);
        [~, col] = max(ok, [], 2);
        got = find(hit);
        v(bad(got)) = C(sub2ind(size(C), got, col(got)));
        bad = bad(~hit);
    end
end

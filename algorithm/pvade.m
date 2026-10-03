% ----------------------------------------------------------------------- %
% Population's Variance-based Adaptive Differential Evolution (PVADE)
% CEC 2013 competition -- 18th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 25 + 2.5*D               % Population size (50/100/150 at D = 10/30/50)
%   F = L_j (p 0.8) or Fdec (0.2) % Per gene; L_j = max(min(0.6, 1 - dn_j), 0.3)
%   Fdec = 0.2 -> 0.9 (linear)    % Over itermax = maxFe/NP generations
%   CR = max_j(1 - dn_j)          % One rate per generation; x0.1 after 90 stalled ones
%   p_opp = D/33.33 * dn_j        % Quasi-opposition chance per gene (D/60 when stalled)
%   stall = 90                    % Generations without a new best that switch both
%
% Algorithm Concept:
%   - dn_j = (D/20) var_j(pop) / max over past generations of var_j: normalised
%     diversity of each coordinate drives F, CR and the opposition chance
%   - One coin per generation picks rand-to-best/1 (x + F(best - x) + F(r1 - r2))
%     or rand/1 (r3 + F(r1 - r2)); donors from rotated random permutations
%   - Exponential-type crossover: a block of the sorted CR mask, rotated by a
%     random offset, so no mutant gene is guaranteed
%   - Quasi-opposition per gene against the trial population's column range
%   - Trials clipped to the box; greedy one-to-one selection (trial <= parent)
%
% Reference:
% Leandro dos Santos Coelho, Helon V. H. Ayala, Roberto Zanetti Freire,
% Population's variance-based Adaptive Differential Evolution for real parameter
% optimization,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, 1672-1677.
% https://doi.org/10.1109/CEC.2013.6557762
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from ADE_sub.m (tipoF = tipoCR = opposition = 2, as main_ADE_sub.m runs it)
% in "PVADE codes and results.zip" of the organiser archive (P-N-Suganthan/CEC2013).
% The driver sets NP = 50/100/150 at D = 10/30/50 and itermax = 10000*D/NP; the line
% NP = 25 + 2.5*D gives all three. Kept as released: in the first generation max()
% runs along the one-row variance history, so dn is scaled by the largest coordinate
% variance; the opposed gene is mirrored twice, so the new value lands between the
% original and the column midpoint, not the opposite point. Removed: the stop at the
% known optimum. Non-finite trial coordinates are redrawn uniformly (the release
% evaluates NaN). Same rng seed gives the release's run exactly.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = pvade(problem)

    D = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    NP = round(25 + 2.5 * D);
    itermax = maxFE / NP;
    XVmin = repmat(lb, NP, 1);
    XVmax = repmat(ub, NP, 1);

    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    FE = 0;
    bsf = inf;

    pop = XVmin + (XVmax - XVmin) .* rand(NP, D);
    bsf_solution = pop(1, :);
    val = inf(NP, 1);
    n_eval = min(NP, maxFE);
    [fv, FE] = calculate_fitness(pop(1:n_eval, :)', problem, FE);
    val(1:n_eval) = fv(:);
    track(pop(1:n_eval, :), val(1:n_eval));

    [bestval, indice] = min(val);
    bestmem = pop(indice, :);
    bestval_ant = bestval;
    without_improve = 0;
    rot = 0:NP - 1;
    rotd = 0:D - 1;
    iter = 1;
    rand;  % CR_ant in the release: drawn, never used; kept for the rng stream
    var_max = [];

    while FE < maxFE
        var_now = var(pop);
        if iter == 1
            var_max = var_now;
            dn = (D / 20) * (var_now ./ max(var_now));
        else
            var_max = max(var_max, var_now);
            dn = (D / 20) * (var_now ./ var_max);
        end
        Lj = max(min(0.6, 1 - dn), 0.3);

        Fdec = 0.7 * (iter / itermax) + 0.2;
        % The release draws F(i,j) row by row; rand(D, NP)' is that order
        R = rand(D, NP)';
        F = repmat(Lj, NP, 1);
        F(R <= 0.2) = Fdec;

        if without_improve > 90
            CR = 0.1 * max(1 - dn);
        else
            CR = max(1 - dn);
        end

        popold = pop;
        ind = randperm(2);
        a1 = randperm(NP);
        rt = rem(rot + ind(1), NP);
        a2 = a1(rt + 1);
        rt = rem(rot + ind(2), NP);
        a3 = a2(rt + 1);
        pm1 = popold(a1, :);
        pm2 = popold(a2, :);
        pm3 = popold(a3, :);
        bm = repmat(bestmem, NP, 1);

        mui = rand(NP, D) < CR;
        mui = sort(mui');
        for i = 1:NP
            n = floor(rand * D);
            rtd = rem(rotd + n, D);
            mui(:, i) = mui(rtd + 1, i);
        end
        mui = mui';
        mpo = mui < 0.5;

        if rand > 0.5
            ui = popold + F .* (bm - popold) + F .* (pm1 - pm2);
        else
            ui = pm3 + F .* (pm1 - pm2);
        end
        ui = popold .* mpo + ui .* mui;

        if without_improve > 90
            prob_op = (D / 60) * dn;
        else
            prob_op = (D / 33.3333334) * dn;
        end
        % Column extremes kept incrementally; recomputed only when an extreme may have moved
        cmin = min(ui, [], 1);
        cmax = max(ui, [], 1);
        for i = 1:NP
            for j = 1:D
                if rand < prob_op(j)
                    miu = cmin(j);
                    mau = cmax(j);
                    orig = ui(i, j);
                    opp = miu + mau - orig;
                    op2 = miu + mau - opp;
                    M = (miu + mau) / 2;
                    if opp < M
                        nv = M + (op2 - M) * rand;
                    else
                        nv = op2 + (M - op2) * rand;
                    end
                    ui(i, j) = nv;
                    if ~(orig > miu && orig < mau && nv > miu && nv < mau)
                        cmin(j) = min(ui(:, j));
                        cmax(j) = max(ui(:, j));
                    end
                end
            end
        end

        bad = ~isfinite(ui);
        if any(bad(:))
            ui(bad) = XVmin(bad) + rand(nnz(bad), 1) .* (XVmax(bad) - XVmin(bad));
        end
        ui = min(max(ui, XVmin), XVmax);

        n_eval = min(NP, maxFE - FE);
        tempval = inf(NP, 1);
        [fv, FE] = calculate_fitness(ui(1:n_eval, :)', problem, FE);
        tempval(1:n_eval) = fv(:);

        [best_tempval, indice] = min(tempval);
        if best_tempval < bestval
            bestval = best_tempval;
            bestmem = ui(indice, :);
        end
        for i = 1:n_eval
            if tempval(i) <= val(i)
                pop(i, :) = ui(i, :);
                val(i) = tempval(i);
            end
        end
        track(ui(1:n_eval, :), tempval(1:n_eval));

        if bestval == bestval_ant
            without_improve = without_improve + 1;
        else
            without_improve = 0;
        end
        bestval_ant = bestval;
        iter = iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsf_solution;

    % Best-so-far per evaluation, then one history sample per evaluation with the settled population
    function track(X, f)
        n = size(X, 1);
        for q = 1:n
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = X(q, :);
            end
            ec = FE - n + q;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, val, population_history, fitness_history, history_index, maxFE);
            end
        end
    end
end

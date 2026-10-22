% ----------------------------------------------------------------------- %
% Tree Physiology Optimization (TPO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   branches = 30     % Branches, each keeping the best of its own shoots
%   leaves   = 30     % Shoots per branch: 900 evaluations per iteration
%   alpha    = 0.3    % Root absorption constant, Eq. (3)
%   beta     = 50     % Shoot extension factor, Eq. (1)
%   theta    = 0.9    % Randomness decay, theta^i at iteration i
%
% Algorithm Concept:
%   - Carbon gain (Eq. 2): C = theta^i*(S_branchbest - S), each shoot's offset
%     from the best shoot of its own branch
%   - Root elongation (Eq. 3): r = r + alpha*C*eps, eps ~ U(-0.5, 0.5) per coordinate
%   - Nutrient uptake (Eq. 5): N = theta^i*(r - r_prev), so every step shrinks
%     as theta^(2i) and the shoots contract onto the global best
%   - Shoot extension (Eq. 1): S = S + (S_gbest - S) + beta*N, every shoot jumps
%     to the global best shoot plus its own nutrient step
%   - Branch bests replaced greedily; shoots clamped to the box after each move
%
% Reference:
% A. Hanif Halim, I. Ismail,
% Tree Physiology Optimization in Benchmark Function and Traveling Salesman
% Problem,
% Journal of Intelligent Systems 28(5) (2019) 849-871.
% https://doi.org/10.1515/jisys-2017-0156
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 63982 v1.0.0.0
% (TPO_UnconstrainedOptim.m, 3 Aug 2017), written for two variables x, y; every
% operation there is coordinate-wise, so it runs per coordinate here with the
% random draws in the release's x-then-y order (bit-identical to it at D = 2).
% The release draws its initial shoots in [0, 1] whatever the range; here they
% are drawn in [lb, ub]. Kept as released: beta = 50 (Table 3 prints 0.7, the
% text recommends 20-60), the first growth step cos(pi/3)*5*rand in absolute
% units, the shoot loop of which only the last pass survives (cos(30*pi/3) = 1),
% and theta^i counted in iterations, not scaled to the budget. The last
% iteration evaluates only the shoots the budget covers.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tpo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    pop   = 30;   % branches
    n     = 30;   % shoots per branch
    alpha = 0.3;
    beta  = 50;
    theta = 0.9;
    NS    = pop * n;

    % Shoots and roots are pop x n x dim: page d is the release's x (d = 1) or y (d = 2) matrix
    lb3 = reshape(lb, 1, 1, dim);
    ub3 = reshape(ub, 1, 1, dim);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    [S, ~] = init_tree(pop, n, dim, lb3, ub3);   % the first roots are drawn, then redrawn unused

    % Shoots are evaluated in linear order of the pop x n grid, branch index fastest
    Srec = reshape(S, NS, dim);
    nev = min(NS, maxFE);
    [zrec, FE] = calculate_fitness(Srec(1:nev, :)', problem, FE);
    zrec = zrec(:)';

    fbest = Inf;
    Xbest = Srec(1, :);
    for e = 1:nev
        if zrec(e) < fbest
            fbest = zrec(e);
            Xbest = Srec(e, :);
        end
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, Srec(1:nev, :), zrec, population_history, fitness_history, ...
            history_index, maxFE);
    end

    if FE < maxFE
        % Initial best shoot of each branch
        zn = reshape(zrec, pop, n);
        z0 = zeros(pop, 1);
        x0 = zeros(pop, dim);
        for jj = 1:pop
            [z0(jj), Imin] = min(zn(jj, :));
            x0(jj, :) = reshape(S(jj, Imin, :), 1, dim);
        end

        % The release draws the shoots and roots a second time before the first growth
        [S, rn] = init_tree(pop, n, dim, lb3, ub3);
        x00 = repmat(reshape(x0, pop, 1, dim), 1, n);   % each branch's own best
        S = S + (x00 - S) + cos(pi / 3) .* 5 .* rand(pop, n, dim);   % first shoot growth
        S = range_tree(S, lb3, ub3);
    end

    i = 0;
    while FE < maxFE
        i = i + 1;
        Theta = theta^(1 * i);

        Snew = reshape(S, NS, dim);
        nev = min(NS, maxFE - FE);
        [fnew, FE] = calculate_fitness(Snew(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for e = 1:nev
            Srec(e, :) = Snew(e, :);   % position at which each entry of zrec was measured
            zrec(e) = fnew(e);
            if fnew(e) < fbest
                fbest = fnew(e);
                Xbest = Snew(e, :);
            end
            ec = FE - nev + e;
            curve(ec) = fbest;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Srec, zrec, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        zn = reshape(fnew, pop, n);
        for ij = 1:pop
            [zb, Imin] = min(zn(ij, :));   % current best shoot of the branch
            if zb < z0(ij)
                z0(ij) = zb;
                x0(ij, :) = reshape(S(ij, Imin, :), 1, dim);
            end
        end
        [~, I00] = min(z0);   % current global best branch
        x00 = repmat(reshape(x0(I00, :), 1, 1, dim), pop, n);
        x0l = repmat(reshape(x0, pop, 1, dim), 1, n);

        C = Theta * (x0l - S);                                  % Eq. (2), carbon gain
        r0 = rn;
        rn = rn + (rand(pop, n, dim) - 0.5) .* alpha .* C;      % Eq. (3), root elongation
        Nu = Theta .* (rn - r0);                                % Eq. (5), nutrient uptake
        % Eq. (1); every pass restarts from x00, so only ii = pop counts, kept for its rounding
        for ii = 1:pop
            S = S + (x00 - S) + Nu .* cos(ii * pi / 3) .* beta;
        end
        S = range_tree(S, lb3, ub3);   % the release's clamp of x00 here is overwritten unused
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Shoots in the box (the release: rand in [0, 1]), roots in [0, 1]; x pages before y pages
function [S, rn] = init_tree(pop, n, dim, lb3, ub3)
    S  = lb3 + rand(pop, n, dim) .* (ub3 - lb3);
    rn = rand(pop, n, dim);
end

% Clamp to the box, lower bound first as in the release
function S = range_tree(S, lb3, ub3)
    S = min(max(S, lb3), ub3);
end

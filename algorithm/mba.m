% ----------------------------------------------------------------------- %
% Mine Blast Algorithm (MBA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Ns    = 50                       % Shrapnel pieces per explosion (Npop = Ns)
%   mu    = 10                       % Exploration factor: iterations k <= mu explore
%   d0    = ub - lb                  % Initial distance of every piece, per coordinate
%   theta = 360/Ns                   % Angular spacing; piece j flies at j*theta deg
%   K     = ceil((maxFe - 1)/Ns)     % Iterations, Ns FE each after the first shot
%   alpha = sum(mu+1:K)/log(1/eps)   % Reduction constant: d reaches d0*eps at k = K
%
% Algorithm Concept:
%   - One first shot point, uniform in the box, is the first mine; each iteration
%     it explodes into Ns pieces, x = Xe + exp(-sqrt(|m/d|)).*x_mine, Eq. (3)
%   - Exploration (k <= mu): d = d.*|randn|.^2, Xe = d*cos(theta_j), Eqs. (7)-(8)
%   - Exploitation: Xe = d.*rand*cos(theta_j), Eq. (4); every piece's d is then
%     cut to d/exp(k/alpha), Eq. (9)
%   - m and d in the exponent are the slope and the (x, f) distance of the piece's
%     previous throw from its mine, Eqs. (5)-(6)
%   - The best piece, if better than the mine, exchanges places with it
%   - A piece leaving the box is clamped and its d cut once more by Eq. (9)
%
% Reference:
% Ali Sadollah, Ardeshir Bahreininejad, Hadi Eskandar, Mohd Hamdi,
% Mine blast algorithm: A new population based algorithm for solving constrained
% engineering optimization problems,
% Applied Soft Computing 13(5) (2013) 2592-2612.
% https://doi.org/10.1016/j.asoc.2012.11.026
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Sec. 2, Eqs. (1)-(9), Steps 1-12); no author code is public.
% Silent points read as: Eq. (3) takes m, d of the previous throw (Steps 10 -> 4 -> 5),
% m = 0 at the first; theta_j = j*360/Ns deg (Sec. 2.2 spreads pieces over 360 deg);
% rand, randn per coordinate; d is a persistent radius, Eq. (5)'s distance feeds only
% Eq. (3) (Table 2's alpha puts sum(k)/alpha at 25-45 ~ log(1/eps) on 8 of 16 problems).
% Changed: sqrt(|m/d|), the printed root is complex for m < 0; alpha from the budget by
% that eps rule; the first shot costs 1 FE; feasibility rules and the d cut of infeasible
% pieces are left out. Kept as printed: one cos(theta_j) for every coordinate of a piece,
% and exp(.) <= 1 shrinks the mine toward x = 0, so the search homes on the ORIGIN, not
% the box centre, and the paper's results are not reproduced:
%     sphere D=10, 1e5 FE, median of 5     opt at 0      opt random     [0,200], opt 100
%     as printed                           6.6e-38       4.9e3          1.4e4
%     exp(.) set to 1 (control only)       5.1e-32       0              0
%     CEDP, Table 2 settings, 20 runs      welded        spring         speed reducer  3-bar
%     paper mean, Tables 21/19/23/15       1.7249        0.012713       2996.77        263.898
%     this port, mean (feasible runs)      3.79 (16)     0.0255 (14)    5462 (3)       265.65 (20)
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mba(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    Ns    = 50;
    mu    = 10;
    K     = max(1, ceil((maxFE - 1) / Ns));
    alpha = max(1, sum(mu + 1:K)) / log(1 / eps);   % Sec. 2.2: d ~ eps at the final iteration
    c     = cosd((1:Ns)' * 360 / Ns);   % Sec. 2.2: theta = 360/Ns spaces the pieces over 360 deg

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % First shot point, Eq. (1)
    xm = lb + rand(1, dim) .* (ub - lb);
    [fm, FE] = calculate_fitness(xm', problem, FE);
    fm = fm(1);
    best_fitness  = fm;
    best_solution = xm;
    curve(1) = fm;
    [population_history, fitness_history, history_index] = record_history(...
        1, xm, fm, population_history, fitness_history, history_index, maxFE);

    d  = repmat(ub - lb, Ns, 1);   % d0 per piece and coordinate
    m  = zeros(Ns, dim);           % no previous throw: exp(0) = 1 in Eq. (3)
    dt = ones(Ns, dim);
    Xrec = [xm; zeros(Ns, dim)];   % row 1 the mine, row 1 + j piece j, as measured
    frec = [fm, zeros(1, Ns)];
    np   = 0;                      % pieces thrown so far: rows 2..np+1 are live

    k = 0;
    while FE < maxFE
        k = k + 1;
        if k <= mu
            d  = d .* abs(randn(Ns, dim)) .^ 2;   % Eq. (7)
            Xe = d .* c;                          % Eq. (8)
        else
            Xe = d .* rand(Ns, dim) .* c;         % Eq. (4)
        end
        g = exp(-sqrt(abs(m ./ dt)));   % Eq. (3); |.| because m < 0 makes the root complex
        g(isnan(g)) = 1;                % 0/0: a throw that moved neither x nor f
        X = Xe + g .* xm;               % Eq. (3)

        out = any(~(X >= lb & X <= ub), 2);
        X = min(max(X, lb), ub);              % Sec. 2.5: placed back in the box
        d(out, :) = d(out, :) / exp(k / alpha);   % Sec. 2.5: its distance cut by Eq. (9)

        nt = min(Ns, maxFE - FE);
        FE0 = FE;
        [f, FE] = calculate_fitness(X(1:nt, :)', problem, FE);
        f = f(:);

        % Slope and distance of this throw from its mine, Eqs. (5)-(6), for the next one
        dX = X(1:nt, :) - xm;
        dF = f - fm;
        dt(1:nt, :) = sqrt(dX .^ 2 + dF .^ 2);
        m(1:nt, :)  = dF ./ dX;

        % Steps 7-9: the best piece exchanges places with the mine if it is better
        [fb, ib] = min(f);
        swap = fb < fm || (isnan(fm) && ~isnan(fb));
        for e = 1:nt
            Xrec(1 + e, :) = X(e, :);
            frec(1 + e)    = f(e);
            np = max(np, e);
            if e == nt && swap
                Xrec([1, 1 + ib], :) = Xrec([1 + ib, 1], :);
                frec([1, 1 + ib])    = frec([1 + ib, 1]);
            end
            if f(e) < best_fitness || (isnan(best_fitness) && ~isnan(f(e)))
                best_fitness  = f(e);
                best_solution = X(e, :);
            end
            curve(FE0 + e) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE0 + e, Xrec(1:1 + np, :), frec(1:1 + np), population_history, ...
                fitness_history, history_index, maxFE);
        end
        if swap
            xm = X(ib, :);
            fm = fb;
        end

        if k > mu
            d = d / exp(k / alpha);   % Eq. (9), Step 11
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

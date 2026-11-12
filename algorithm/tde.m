% ----------------------------------------------------------------------- %
% Trigonometric Mutation Differential Evolution (TDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 30            % Population size
%   F  = 0.99          % Scaling factor of the DE/rand/1 mutation
%   CR = 0.85          % Binomial crossover probability
%   Mt = 0.05          % Probability of the trigonometric mutation, per trial
%
% Algorithm Concept:
%   - DE/rand/1/bin with one-to-one generational selection: the trial replaces
%     its target if f(u) <= f(x); r1, r2, r3 are mutually different and not i
%   - With probability Mt the donor is instead the trigonometric mutation of the
%     triangle x_r1, x_r2, x_r3, weighted by p_k = |f(x_rk)| / sum_j |f(x_rj)|
%   - v = (x_r1+x_r2+x_r3)/3 + (p2-p1)(x_r1-x_r2) + (p3-p2)(x_r2-x_r3) + (p1-p3)(x_r3-x_r1)
%   - The donor leaves the centroid along each leg toward the vertex with the
%     smaller |f|, a greedy local step; binomial crossover follows either donor
%
% Reference:
% Hui-Yuan Fan, Jouni Lampinen,
% A Trigonometric Mutation Operation to Differential Evolution,
% Journal of Global Optimization 27(1) (2003) 105-129.
% https://doi.org/10.1023/A:1024653025686
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Secs. 2-5, Eqs. (1)-(4)); no author code is public. NP, F, CR, Mt
% are the Sec. 5.1 benchmark set (Sec. 6 changes only NP). The paper stops on CPU
% time; maxFe replaces it, the last generation cut to the remaining budget. Its
% bounds only seed the population; violating mutant components are redrawn in the
% box, as clamping traps runs (Ackley D = 30 on [-20, 30], 2e5 FE, 50 runs, mean f):
%    DE  (Mt = 0)      none 0.068   clamp 7.8    redraw 0.037
%    TDE (Mt = 0.05)   none 0.565   clamp 0.683  redraw 0.561
% p_k uses |f| as printed, which assumes f >= 0 with its optimum at 0: on negative f
% the pull reverses, on an offset f (CEC bias) the step shrinks toward the centroid.
% A zero or non-finite sum of |f| (0/0) gets equal weights, i.e. the centroid.
% Paper Fig. 5 vs port without repair, as there, at the FE where its DE (Mt = 0) matches:
%    Mt                             0      0.05   0.3    1
%    Ackley D = 30, 25k FE          16.5   0.4    5.4    14.6    port 16.5  0.57  4.6  13.5
%    Rastrigin A = 2, D = 20, 27k   34.8   5.7    9.1    34.4    port 33.8  2.4   7.8  24.4
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tde(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NP = 30;
    F  = 0.99;
    CR = 0.85;
    Mt = 0.05;

    rows    = (1:NP)';
    diagIdx = sub2ind([NP NP], rows, rows);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    pop = lb + rand(NP, D) .* (ub - lb);
    n0  = min(NP, maxFE);
    [f0, FE] = calculate_fitness(pop(1:n0, :)', problem, FE);
    fit = inf(NP, 1);
    fit(1:n0) = f0(:);

    bsf          = fit(1);
    bsf_solution = pop(1, :);
    for e = 1:n0
        if fit(e) < bsf || (isnan(bsf) && ~isnan(fit(e)))
            bsf          = fit(e);
            bsf_solution = pop(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, pop(1:n0, :), fit(1:n0), population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        % Eqs. (1) and (4): r1, r2, r3 mutually different and different from i
        K = rand(NP);
        K(diagIdx) = inf;
        [~, ord] = sort(K, 2);
        r1 = ord(:, 1);
        r2 = ord(:, 2);
        r3 = ord(:, 3);
        X1 = pop(r1, :);
        X2 = pop(r2, :);
        X3 = pop(r3, :);

        % Eq. (1)
        V = X3 + F * (X1 - X2);

        % Sec. 4: the trigonometric donor replaces it when U(0,1) < Mt
        t = find(rand(NP, 1) < Mt);
        if ~isempty(t)
            a = abs(fit(r1(t)));
            b = abs(fit(r2(t)));
            c = abs(fit(r3(t)));
            s = a + b + c;
            p1 = a ./ s;
            p2 = b ./ s;
            p3 = c ./ s;
            % 0/0 (all three f = 0) or a non-finite f: equal weights (Implementation Note)
            flat = ~(s > 0 & isfinite(s));
            p1(flat) = 1/3;
            p2(flat) = 1/3;
            p3(flat) = 1/3;
            % Eq. (4)
            V(t, :) = (X1(t, :) + X2(t, :) + X3(t, :)) / 3 + (p2 - p1) .* (X1(t, :) - X2(t, :)) ...
                    + (p3 - p2) .* (X2(t, :) - X3(t, :)) + (p1 - p3) .* (X3(t, :) - X1(t, :));
        end

        % Violating components redrawn uniformly in the box (Implementation Note)
        bad = ~(V >= lb & V <= ub);
        R = lb + rand(NP, D) .* (ub - lb);
        V(bad) = R(bad);

        % Eq. (2): rand_j <= CR or j = k
        mask = rand(NP, D) <= CR;
        mask(sub2ind([NP D], rows, randi(D, NP, 1))) = true;
        U = pop;
        U(mask) = V(mask);

        nEval = min(NP, maxFE - FE);   % the last generation is cut to the remaining budget
        [fu, FE] = calculate_fitness(U(1:nEval, :)', problem, FE);
        fu = fu(:);
        for e = 1:nEval
            % Eq. (3); every donor above came from generation G, so this is generational
            if fu(e) <= fit(e) || (isnan(fit(e)) && ~isnan(fu(e)))
                pop(e, :) = U(e, :);
                fit(e)    = fu(e);
            end
            if fu(e) < bsf || (isnan(bsf) && ~isnan(fu(e)))
                bsf          = fu(e);
                bsf_solution = U(e, :);
            end
            ec = FE - nEval + e;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, pop, fit, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

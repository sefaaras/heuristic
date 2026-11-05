% ----------------------------------------------------------------------- %
% Global-best Harmony Search (GHS)
% Variant of IHS (ihs): the bandwidth step is replaced by a copy from the best
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   HMS  = 5                        % Harmony memory size
%   HMCR = 0.9                      % Harmony memory considering rate
%   PAR  = 0.01 -> 0.99 (linear)    % Chance a copied variable is taken from the best
%   NI   = maxFe - HMS              % Improvisations, one evaluation each
%
% Algorithm Concept:
%   - Each variable of the new harmony is copied, with probability HMCR, from a
%     random memory row (its own row per variable), else drawn uniformly in the box
%   - Pitch adjustment, with probability PAR(gn): the copied variable i is
%     replaced by x_best(k), k a random dimension of the best harmony in memory
%   - PAR rises linearly with the improvisation count gn, as in IHS; there is
%     no bandwidth parameter
%   - One harmony per iteration; it replaces the worst memory row if better
%
% Reference:
% Mahamed G.H. Omran, Mehrdad Mahdavi,
% Global-best harmony search,
% Applied Mathematics and Computation 198 (2) (2008) 643-656.
% https://doi.org/10.1016/j.amc.2007.09.004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Checked against the paper; no author code exists. Step 3 is the paper's (Sec.
% 4, p. 645), PAR(t) its Eq. (1), and HMS 5, HMCR 0.9, PARmin 0.01, PARmax 0.99
% its Sec. 5 settings (p. 647); NI comes from the budget (the paper ran 50,000
% evaluations). Kept as published: k is drawn over all D dimensions, so on a box
% whose dimensions differ (CEC2020RW) the copied value can lie outside dimension
% i's range; it is clamped onto the bound, as is every out-of-box variable (the
% paper is silent on bounds, its boxes are uniform). Table 4 rerun (D = 30,
% Camel-Back 2, 50,000 FE, 30 runs): 9 of 10 means within the paper's SD. Sphere
% is open: the paper's HS settings miss it here too (7.9 vs 1.9e-4), and a 15-bit
% RNG, which the value grid of the authors' example tables implies, does not close it.
%     mean, port / paper   Sphere 9.3e-3 / 1.0e-5   Rastrigin 5.9e-3 / 8.6e-3
%     Ackley 1.8e-2 / 2.1e-2   Griewank 2.4e-2 / 0.10   Rosenbrock 66 / 50
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ghs(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    HMS    = 5;
    HMCR   = 0.9;
    PARmin = 0.01;
    PARmax = 0.99;
    NI     = max(1, maxFE - HMS);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    HM = lb + rand(HMS, D) .* span;
    [f, FE] = calculate_fitness(HM', problem, FE);
    f = f(:);

    bsf  = inf;
    bsfx = HM(1, :);
    for i = 1:HMS
        if f(i) < bsf
            bsf  = f(i);
            bsfx = HM(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, HM, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    while FE < maxFE
        gn  = FE - HMS + 1;              % improvisation number, 1..NI
        PAR = PARmin + (PARmax - PARmin) * gn / NI;

        % Random selection everywhere, then memory consideration overwrites its share
        x = lb + rand(1, D) .* span;
        idx = find(rand(1, D) < HMCR);
        if ~isempty(idx)
            src = randi(HMS, 1, numel(idx));
            x(idx) = HM(sub2ind([HMS D], src, idx));
            adj = idx(rand(1, numel(idx)) < PAR);
            if ~isempty(adj)
                [~, ib] = min(f);        % best harmony in memory; NaN rows are skipped
                x(adj) = HM(ib, randi(D, 1, numel(adj)));
            end
        end
        x = min(max(x, lb), ub);

        [fx, FE] = calculate_fitness(x', problem, FE);
        fx = fx(1);

        if fx < bsf
            bsf  = fx;
            bsfx = x;
        end

        [fworst, iworst] = max(f);
        if fx < fworst
            HM(iworst, :) = x;
            f(iworst)     = fx;
        end

        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            FE, HM, f, population_history, fitness_history, ...
            history_index, maxFE);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

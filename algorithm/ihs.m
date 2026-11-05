% ----------------------------------------------------------------------- %
% Improved Harmony Search (IHS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   HMS  = 7                                   % Harmony memory size
%   HMCR = 0.95                                % Harmony memory considering rate
%   PAR  = 0.35 -> 0.99 (linear)               % Pitch adjusting rate
%   bw   = 0.04*(ub-lb) -> 1e-8*(ub-lb) (exp.) % Bandwidth per dimension, 4 -> 1e-6 on [-50, 50]
%   NI   = maxFe - HMS                         % Improvisations, one evaluation each
%
% Algorithm Concept:
%   - Each variable of the new harmony is copied, with probability HMCR, from a
%     random memory row (its own row per variable), else drawn uniformly in the box
%   - A copied variable is pitch-adjusted with probability PAR(gn) to
%     x +/- rand*bw(gn); a step that would leave the box is not taken
%   - PAR rises linearly with the improvisation count gn, so late in the run
%     nearly every copied variable is perturbed
%   - bw(gn) = bwmax*exp(gn*ln(bwmin/bwmax)/NI) falls exponentially from a
%     coarse step to a fine one
%   - One harmony per iteration; it replaces the worst memory row if better
%
% Reference:
% M. Mahdavi, M. Fesanghary, E. Damangir,
% An improved harmony search algorithm for solving optimization problems,
% Applied Mathematics and Computation 188 (2) (2007) 1567-1579.
% https://doi.org/10.1016/j.amc.2006.11.033
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author code exists. The 2007 paper tunes HMS, PAR and bw per problem (Table 1);
% used here are the settings of its two unconstrained functions (both rows agree:
% HMS 7, HMCR 0.95, PAR 0.35-0.99, bw 4 -> 1e-6 on the [-50, 50] box), scaled to
% each dimension's range; NI comes from the budget. Not used: the IHS settings of
% the authors' GHS study (Omran and Mahdavi, 10.1016/j.amc.2007.09.004, Sec. 5:
% HMS 5, HMCR 0.9, PAR 0.01-0.99, bw 1e-4 to a printed bwmax 1/(20(UB-LB))). That
% bwmax lies below bwmin on its +/-500 and +/-600 boxes and contradicts the coarse-
% to-fine bandwidth this paper argues for (Sec. 2.2), although its Table 4 IHS column
% fits it better than (UB-LB)/20. The best is the best harmony evaluated, every FE.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ihs(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    HMS    = 7;
    HMCR   = 0.95;
    PARmin = 0.35;
    PARmax = 0.99;
    bwmax  = 0.04 * span;                % Table 1: 4 on the [-50, 50] box
    bwRat  = 1e-6 / 4;                   % bwmin/bwmax, Table 1's 1e-6 -> 4 in every dimension
    NI     = max(1, maxFE - HMS);
    c      = log(bwRat) / NI;            % bw(gn) = bwmax*exp(c*gn) reaches bwmin at gn = NI

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
        bw  = bwmax * exp(c * gn);

        % Random selection everywhere, then memory consideration overwrites its share
        x = lb + rand(1, D) .* span;
        idx = find(rand(1, D) < HMCR);
        if ~isempty(idx)
            src = randi(HMS, 1, numel(idx));
            x(idx) = HM(sub2ind([HMS D], src, idx));
            adj = idx(rand(1, numel(idx)) < PAR);
            t = x(adj) + bw(adj) .* (2 * rand(1, numel(adj)) - 1);   % +/- rand*bw
            in = t >= lb(adj) & t <= ub(adj);   % Fig. 1: a step out of the box is not taken
            x(adj(in)) = t(in);
        end
        x = min(max(x, lb), ub);            % rounding residue of lb + rand*span only

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

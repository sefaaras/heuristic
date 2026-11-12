% ----------------------------------------------------------------------- %
% Invasive Weed Optimization (IWO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N0 = 5                      % Initial weeds
%   p_max = 10                  % Colony size kept by competitive exclusion
%   S_min = 0, S_max = 2        % Seeds of the worst and of the best weed
%   n = 3                       % Nonlinear modulation index of the sigma schedule
%   sigma = 0.375 -> 5e-5 of (ub - lb)  % Per-dimension SD, initial -> final
%
% Algorithm Concept:
%   - Reproduction: seeds per weed grow linearly with fitness, from S_min for
%     the worst weed of the colony to S_max for the best, rounded down (Fig. 1)
%   - Spatial dispersal: seeds are normal around their parent with
%     sigma = (1 - t)^n * (sigma_init - sigma_final) + sigma_final, Eq. (1)
%   - Competitive exclusion: once parents and seeds exceed p_max, all are
%     ranked together and only the p_max best survive to reproduce
%
% Reference:
% A. R. Mehrabian, C. Lucas,
% A novel numerical optimization algorithm inspired from weed colonization,
% Ecological Informatics 1(4) (2006) 355-366.
% https://doi.org/10.1016/j.ecoinf.2006.07.003
% ----------------------------------------------------------------------- %
% Implementation Note:
% Settings are the paper's general-purpose ones: Sec. 4.4.1 takes n = 3, p_max = 10 and
% S_max = 2 "from experiences obtained from previous sections" and Sec. 6 recommends
% them; N0 = 5, S_min = 0 and sigma follow Tables 10 and 12, whose sigma_init and
% sigma_final are 0.375 and 5e-5 of the initial-area width, here of ub - lb. The paper
% lets weeds leave the initial area; seeds are clipped to the box here. iter/iter_max
% of Eq. (1) is the FE fraction spent by the end of the generation, so the last one
% reaches sigma_final. Guards: with no finite fitness spread every weed counts as best;
% NaN gets S_min seeds. Replica of these operators on the paper's protocol (200
% generations, unbounded), paper / here:
%      Table 11, Griewank d = 6, Table 12 setting, 100 runs: 100 % below 1e-9 in 1996 FE /
%          1 % (mean 1.8e-7) in 948 FE                                  (not reproduced)
%          algorithm/iwo.m on [-1, 1] at 1996 FE: 3 % (mean 3.9e-9); old settings 39 % (8.4e-5)
%      Table 7, Griewank d = 10/20/50/100, Table 6 setting (p_max 20, n 3), 20 runs, mean:
%          0.0373 0.0494 0 0 / 0.131 1.04 37.8 548                      (not reproduced)
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = iwo(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N0       = 5;
    p_max    = 10;
    S_min    = 0;
    S_max    = 2;
    n_mod    = 3;
    % Tables 10 and 12: 0.375 and 5e-5 of the initial-area width
    sig_init = 0.375 * span;
    sig_fin  = 5e-5 * span;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    P  = lb + rand(N0, D) .* span;
    m0 = min(N0, maxFE);
    [fv, FE] = calculate_fitness(P(1:m0, :)', problem, FE);
    P  = P(1:m0, :);
    fP = fv(:);

    bsf          = inf;
    bsf_solution = P(1, :);
    for i = 1:m0
        % A NaN incumbent could never be beaten, so it is overwritten
        if fP(i) < bsf || (isnan(bsf) && ~isnan(fP(i)))
            bsf          = fP(i);
            bsf_solution = P(i, :);
        end
        curve(i) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            i, P, fP, population_history, fitness_history, history_index, maxFE);
    end
    fe0 = FE;

    while FE < maxFE
        % Reproduction: linear in fitness between the colony's worst and best
        fin   = isfinite(fP);
        ratio = zeros(size(fP));
        if any(fin)
            f_best  = min(fP(fin));
            f_worst = max(fP(fin));
            if f_worst > f_best
                ratio(fin) = (fP(fin) - f_worst) / (f_best - f_worst);
            else
                ratio(fin) = 1;
            end
        end
        ratio(fP == -inf) = 1;
        S = floor(S_min + (S_max - S_min) * ratio);
        if sum(S) == 0
            S(:) = S_max;
        end

        ns = min(sum(S), maxFE - FE);
        parent = repelem((1:numel(S))', S);
        parent = parent(1:ns);

        % iter/iter_max of the paper, counted in FE spent by the end of this generation
        t = min(1, (FE + ns - fe0) / max(1, maxFE - fe0));
        sigma = (1 - t)^n_mod * (sig_init - sig_fin) + sig_fin;

        seeds = P(parent, :) + sigma .* randn(ns, D);
        seeds = min(max(seeds, lb), ub);
        [fv, FE] = calculate_fitness(seeds', problem, FE);
        fs = fv(:);

        P  = [P; seeds];
        fP = [fP; fs];
        for i = 1:ns
            if fs(i) < bsf || (isnan(bsf) && ~isnan(fs(i)))
                bsf          = fs(i);
                bsf_solution = seeds(i, :);
            end
            fe_i = FE - ns + i;
            curve(fe_i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                fe_i, P, fP, population_history, fitness_history, history_index, maxFE);
        end

        % Competitive exclusion; sort ranks NaN last, so NaN weeds go first
        if numel(fP) > p_max
            [fP, order] = sort(fP);
            fP = fP(1:p_max);
            P  = P(order(1:p_max), :);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% ----------------------------------------------------------------------- %
% Thermal Exchange Optimization (TEO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n      = 30                 % Objects (Sec. 4.1; even: the halves pair up)
%   nTM    = 4                  % Thermal memory size (Table 3, Scenario IV)
%   pro    = 0.15               % Chance to redraw one coordinate (Table 3)
%   c1, c2 = 1, 1               % Environment-perturbation switches, Eq. (10)
%   t      = iter/MaxIter       % Eq. (9), MaxIter = ceil(maxFe/n) rounds
%
% Algorithm Concept:
%   - Objects sorted by cost; the first half is the environment of the second
%     and vice versa, T_i paired with T_(i+n/2)
%   - Thermal memory: the nTM best points ever evaluated replace the nTM worst
%     objects before the pairing
%   - The environment is rescaled about the origin, coordinate-wise:
%     T_env = (1 - (c1 + c2*(1 - t))*rand) .* T_env'
%   - Newton cooling: T_new = T_env + (T_old - T_env)*exp(-beta*t), beta =
%     cost/worst cost, so better objects move less
%   - With probability pro one coordinate of an object is redrawn uniformly
%     in its range; every new object replaces the old one (no selection)
%
% Reference:
% Ali Kaveh, Armin Dadras,
% A novel meta-heuristic optimization algorithm: Thermal exchange optimization,
% Advances in Engineering Software 110 (2017) 69-84.
% https://doi.org/10.1016/j.advengsoft.2017.03.014
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper: Steps 1-10 of Sec. 2.2.2, Eqs. (7)-(12), Figs. 2-3.
% Sec. 4.1 fixes n = 30 but not c1, c2, pro, TM; Table 3 Scenario IV (both
% controls on) is used. pro = 0.05 fits Table 8's origin-centred rows better,
% no Table 2 setting reaches its F9/F10; median CEC2014 error, 5 seeds:
%      D=10, 1e5 FE    F1     F3     F4    F9   F10
%      pro = 0.15   2.0e7   81.2    103  44.4  1109
%      pro = 0.05   2.7e7  434.4    132  58.8  1293
% Fig. 2 counts the initial evaluation as iteration 1, so evaluated updates
% use t < 1; the last round is cut to the FE budget; best updated per eval.
% Paper silent, chosen here: clamping to the box, as the second author's
% MOTEO release does; Eq. (8) assumes positive costs, so beta is kept when
% negative costs flip its sign, an undefined ratio is set to 1, and a zero
% temperature difference leaves T_env (0*Inf guard).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = teo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    n   = 30;
    nTM = 4;
    pro = 0.15;
    c1  = 1;
    c2  = 1;
    MaxIter = max(1, ceil(maxFE / n));   % evaluation rounds, the initial one included
    half = n / 2;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    T = lb + rand(n, dim) .* (ub - lb);   % Steps 1-2, Eq. (7)
    [fit, FE] = calculate_fitness(T', problem, FE);
    fit = fit(:)';

    best_fitness  = inf;
    best_solution = T(1, :);
    for e = 1:min(n, maxFE)
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = T(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, T, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    TMx = zeros(0, dim);
    TMf = zeros(1, 0);
    iter = 0;
    while FE < maxFE
        iter = iter + 1;

        % Step 3: thermal memory, then it replaces the nTM worst objects
        [TMf, ord] = sort([TMf, fit], 'ascend');
        pool = [TMx; T];
        k = min(nTM, numel(ord));
        TMf = TMf(1:k);
        TMx = pool(ord(1:k), :);
        [fit, ord] = sort(fit, 'ascend');
        T = T(ord, :);
        T(n - k + 1:n, :) = TMx;
        fit(n - k + 1:n) = TMf;

        % Step 4: sort and pair the halves (Fig. 3)
        [fit, ord] = sort(fit, 'ascend');
        T = T(ord, :);
        Tenv0 = T([half + 1:n, 1:half], :);

        % Steps 5-6, Eqs. (8)-(9): iter counts the evaluation rounds done so far
        beta = fit(:) / max(fit);
        beta(~isfinite(beta)) = 1;   % zero or non-finite worst cost
        tt = iter / MaxIter;

        % Step 7, Eq. (10): perturb the environment
        Tenv = (1 - (c1 + c2 * (1 - tt)) * rand(n, dim)) .* Tenv0;

        % Step 8, Eq. (11): Newton's law of cooling
        d = T - Tenv;
        Tnew = Tenv + d .* exp(-beta * tt);
        Tnew(d == 0) = Tenv(d == 0);   % no temperature difference, no exchange (avoids 0*Inf)

        % Step 9, Eq. (12): regenerate one coordinate with probability pro
        for i = 1:n
            if rand < pro
                j = randi(dim);
                Tnew(i, j) = lb(j) + rand * (ub(j) - lb(j));
            end
        end
        Tnew = min(max(Tnew, lb), ub);

        m = min(n, maxFE - FE);
        [fnew, FE] = calculate_fitness(Tnew(1:m, :)', problem, FE);
        fnew = fnew(:)';
        for i = 1:m
            T(i, :) = Tnew(i, :);
            fit(i)  = fnew(i);
            if fnew(i) < best_fitness
                best_fitness  = fnew(i);
                best_solution = Tnew(i, :);
            end
            ec = FE - m + i;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, T, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

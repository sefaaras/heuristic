% ----------------------------------------------------------------------- %
% Ensemble of Strategies and Perturbation Parameter in SOMA (ESP-SOMA)
% CEC 2019 100-Digit Challenge -- 18th (last) place of 18 (score 51.92)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 10                            % Population size
%   gap = 7                            % Failed generations before PRT and strategy are re-drawn
%   step = 0.11, pathLength = 3.0      % Path t = 0.11, 0.22, ..., 2.97 (27 points per leader)
%   PRT in {0.1, 0.3, 0.5, 0.7, 0.9}   % Per-individual perturbation rate
%   strategy in {AllToOne, AllToAll, AllToRandom}  % Per-individual migration strategy
%
% Algorithm Concept:
%   - Each individual carries its own migration strategy and PRT, both drawn
%     uniformly at the start
%   - AllToOne walks towards the population best, AllToRandom towards a random
%     other individual, AllToAll towards every other individual at each step
%   - A jump moves the coordinates whose uniform draw is below PRT, plus one random
%     coordinate that always moves
%   - The best point on the path replaces the individual even when it is worse
%     (generational update); the best-so-far is kept separately
%   - Every gap-th failed generation (a success does not reset the count), PRT and
%     strategy are re-drawn by a roulette weighted by each value's holders + 1
%   - A coordinate that leaves the box is put halfway between the bound and the
%     individual's own value
%
% Reference:
% Tomas Kadavy, Michal Pluhacek, Roman Senkerik, Adam Viktorin,
% The Ensemble of Strategies and Perturbation Parameter in Self-organizing Migrating
% Algorithm Solving CEC 2019 100-Digit Challenge, 2019 IEEE Congress on Evolutionary
% Computation (CEC), 2019, pp. 372-375.
% https://doi.org/10.1109/CEC.2019.8790012
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from main.py in github.com/TBU-AILab/ESP_SOMA_python (author group; its
% README cites this paper). The paper is paywalled and the organisers report NP and
% adaptivePRT tuned per function at 1e7 FE, so the README's usage settings are used:
% NP = 10, gap = 7, step = 0.11, pathLength = 3.0, adaptive PRT (the script's own
% tail runs NP = 100 with gap = 2; Skanderova's 2022 SOMA review also used gap = 7).
% Kept as released: the best path point replaces its individual unconditionally,
% and an AllToOne individual that is the population best does not move but still
% counts a failed generation. Each path is built in full and evaluated as one batch
% cut at maxFe (its points do not depend on each other's fitness).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = esp_soma(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NP         = 10;
    gap        = 7;
    step       = 0.11;
    pathLength = 3.0;
    prt_pool   = [0.1, 0.3, 0.5, 0.7, 0.9];

    % np.arange(step, pathLength, step)
    nT   = ceil((pathLength - step) / step);
    tvec = step + (0:nT - 1) * step;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    NP = max(2, min(NP, maxFE));

    % Strategy codes follow the release's enum order: 1 AllToOne, 2 AllToAll, 3 AllToRandom
    PRT      = zeros(NP, 1);
    Strategy = zeros(NP, 1);
    Counter  = zeros(NP, 1);
    for i = 1:NP
        PRT(i)      = prt_pool(randi(numel(prt_pool)));
        Strategy(i) = randi(3);
    end

    pop = lb + rand(dim, NP)' .* span;
    [fv, FE] = calculate_fitness(pop', problem, FE);
    fit = fv(:);
    bsf = inf;
    bsf_solution = pop(1, :);
    for q = 1:NP
        if fit(q) < bsf
            bsf = fit(q);
            bsf_solution = pop(q, :);
        end
        curve(q) = bsf;
    end
    [population_history, fitness_history, history_index] = record_history( ...
        FE, pop, fit, population_history, fitness_history, history_index, maxFE);

    while FE < maxFE
        % newPop: rows 1..i-1 hold this generation's replacements, the rest the old population
        newPop = pop;
        newFit = fit;
        for i = 1:NP
            if FE >= maxFE
                break;
            end
            others = [1:i - 1, i + 1:NP];
            switch Strategy(i)
                case 1
                    b = find(fit == min(fit), 1, 'last');  % pickBest keeps the last of ties
                    if isempty(b) || b == i
                        leaders = zeros(1, 0);
                    else
                        leaders = b;
                    end
                case 3
                    leaders = others(randi(NP - 1));
                otherwise
                    leaders = others;
            end

            nL = numel(leaders);
            if nL == 0
                ind  = pop(i, :);
                indf = fit(i);
            else
                % Columns run step-major, leader-minor: the release's loop order
                N  = nT * nL;
                x  = pop(i, :)';
                XL = pop(leaders(repmat(1:nL, 1, nT)), :)';
                tt = repelem(tvec, nL);
                PV = rand(dim, N) < PRT(i);
                PV(sub2ind([dim, N], randi(dim, 1, N), 1:N)) = true;
                Y  = x + (XL - x) .* tt .* PV;
                lo = Y < lb';
                hi = Y > ub';
                mid_lo = repmat((lb' + x) / 2, 1, N);
                mid_hi = repmat((ub' + x) / 2, 1, N);
                Y(lo) = mid_lo(lo);
                Y(hi & ~lo) = mid_hi(hi & ~lo);

                n = min(N, maxFE - FE);
                Y = Y(:, 1:n);
                FE0 = FE;
                [fv, FE] = calculate_fitness(Y, problem, FE);
                f = fv(:);
                for q = 1:n
                    if f(q) < bsf
                        bsf = f(q);
                        bsf_solution = Y(:, q)';
                    end
                    curve(FE0 + q) = bsf;
                end
                for q = FE0 + 1:FE - 1
                    [population_history, fitness_history, history_index] = record_history( ...
                        q, newPop, newFit, population_history, fitness_history, history_index, maxFE);
                end
                % The release returns as soon as the budget is spent, before any update
                if FE >= maxFE
                    [population_history, fitness_history, history_index] = record_history( ...
                        FE, newPop, newFit, population_history, fitness_history, history_index, maxFE);
                    break;
                end
                idx = find(f == min(f), 1, 'last');  % ties go to the later point (>=)
                if isempty(idx)
                    idx = 1;
                end
                ind  = Y(:, idx)';
                indf = f(idx);
            end

            if indf >= fit(i)
                Counter(i) = Counter(i) + 1;
                if Counter(i) >= gap
                    Counter(i)  = 0;
                    PRT(i)      = roulette(PRT, prt_pool, 0.3);
                    Strategy(i) = roulette(Strategy, 1:3, 2);  % a None strategy falls to AllToAll
                end
            end
            newPop(i, :) = ind;
            newFit(i)    = indf;
            if nL > 0
                [population_history, fitness_history, history_index] = record_history( ...
                    FE, newPop, newFit, population_history, fitness_history, history_index, maxFE);
            end
        end
        pop = newPop;
        fit = newFit;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Counter(vals).update(pool): keys in first-appearance order, each weighted by its count + 1
function v = roulette(vals, pool, fallback)
    keys = unique(vals(:)', 'stable');
    keys = [keys, pool(~ismember(pool, keys))];
    w = sum(vals(:) == keys, 1) + ismember(keys, pool);
    pick = rand * sum(w);
    k = find(cumsum(w) > pick, 1);
    if isempty(k)
        v = fallback;
    else
        v = keys(k);
    end
end

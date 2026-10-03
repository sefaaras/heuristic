% ----------------------------------------------------------------------- %
% Ecogeography-Based Optimization with a Tuned Maturity Model (TEBO)
% CEC 2015 learning-based track -- 9th place (organiser's draft ranking)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                     % Habitats (population size); not set by the release
%   eta = 0.95 - 0.9*t^ke       % Maturity: chance of the DE-style (global) moves
%   ke = 1                      % Maturity exponent; 1 is the untuned linear model
%   F ~ N(0.5, 0.3)             % Scale factor, drawn per habitat
%   CR = 0.9                    % Crossover rate of the DE-style moves
%   k = 3                       % Random local topology, rebuilt after 3 stagnant sweeps
%
% Algorithm Concept:
%   - Each habitat makes one candidate per sweep and keeps it only if better
%   - With probability eta: rand/1 or current-to-best/1 DE mutation with binomial
%     crossover, the two chosen with equal chance
%   - Otherwise local migration over a random neighbourhood graph: a coordinate is
%     immigrated with the habitat's immigration rate (worse habitats immigrate more)
%   - The donor neighbour is picked in proportion to its emigration rate; the
%     coordinate is copied or moved by F towards it, again with equal chance
%   - eta falls from 0.95 to 0.05 over the budget, so the search turns local
%
% Reference:
% Yu-Jun Zheng, Xiao-Bei Wu,
% Tuning maturity model of ecogeography-based optimization on CEC 2015
% single-objective optimization test problems,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 1018-1024.
% https://doi.org/10.1109/CEC.2015.7257001
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (TEBO.m, compintell.cn/downloads/TEBO.zip).
% The release takes NP and ke as arguments with no default: ke was tuned per CEC 2015
% function, so the untuned linear model ke = 1 is used, and NP = 50 is this port's
% choice (the paper was not available to check it). Kept as released: the strategy
% success rates are updated into a misspelt variable (successRates), so the selection
% stays 50/50 throughout; a new best does not reset the stagnation counter. The
% release does not count its NP initial evaluations; here they are counted and eta
% runs over the remaining maxFe - NP. unidrnd/normrnd are replaced by the same base
% draws (ceil(n*rand), 0.5 + 0.3*randn). A neighbour draw that misses by rounding
% takes the last neighbour (the release would index 0 and crash).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tebo(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NP     = 50;
    ke     = 1;
    eps_r  = 0.00000001;
    eta    = 0.95;
    n_init = min(NP, maxFE);

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;
    bsf = inf;
    bsf_solution = lb + 0.5 * span;

    % Row by row, coordinate by coordinate, as the release's Init
    X = lb + rand(D, NP)' .* span;
    f = inf(NP, 1);
    f(1:n_init) = evaluate(X(1:n_init, :));
    [population_history, fitness_history, history_index] = record_history( ...
        FE, X(1:n_init, :), f(1:n_init), population_history, fitness_history, history_index, maxFE);

    [minI, maxI] = min_max(f);
    optValue = f(minI);
    [emg, img] = calc_rates(f, f(minI), f(maxI), eps_r);
    top  = rand_top(NP, 3);
    nimp = 0;

    while FE < maxFE
        for i = 1:NP
            if FE >= maxFE
                break;
            end
            u = zeros(1, D);
            j = ceil(D * rand);
            scalingF = randn * 0.3 + 0.5;
            if rand < eta
                sel = select_strategy();
                [r1, r2, r3] = three_indices(NP, i);
                for d = 1:D
                    if rand < 0.9 || j == d
                        if sel == 1
                            u(d) = X(r1, d) + scalingF * (X(r2, d) - X(r3, d));
                        else
                            u(d) = X(i, d) + scalingF * (X(minI, d) - X(i, d)) + scalingF * (X(r1, d) - X(r2, d));
                        end
                        if ~(u(d) >= lb(d) && u(d) <= ub(d))
                            u(d) = lb(d) + rand * (ub(d) - lb(d));
                        end
                    else
                        u(d) = X(i, d);
                    end
                end
            else
                sel = 2 + select_strategy();
                nbs = find(top(i, :));
                nbs = nbs(nbs ~= i);
                % Sequential sums, the release's running total over the neighbours
                cum_emg = cumsum(emg(nbs));
                for d = 1:D
                    if rand < img(i) || j == d
                        nIndex = select_neighbor(nbs, cum_emg);
                        if sel == 3
                            u(d) = X(i, d) + scalingF * (X(nIndex, d) - X(i, d));
                            if ~(u(d) >= lb(d) && u(d) <= ub(d))
                                u(d) = lb(d) + rand * (ub(d) - lb(d));
                            end
                        else
                            u(d) = X(nIndex, d);
                        end
                    else
                        u(d) = X(i, d);
                    end
                end
            end
            fu = evaluate(u);
            if fu < f(i)
                X(i, :) = u;
                f(i) = fu;
            end
            [population_history, fitness_history, history_index] = record_history( ...
                FE, X, f, population_history, fitness_history, history_index, maxFE);
        end
        if FE >= maxFE
            break;
        end
        [minI, maxI] = min_max(f);
        if f(minI) < optValue
            optValue = f(minI);
        else
            nimp = nimp + 1;
            if nimp >= 3
                top  = rand_top(NP, 3);
                nimp = 0;
            end
        end
        [emg, img] = calc_rates(f, f(minI), f(maxI), eps_r);
        % Maturity over the evaluations after initialisation, the release's nfe / nfes
        eta = 0.95 - 0.9 * ((FE - n_init) / (maxFE - n_init)) ^ ke;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of Xr (already cut to the budget), tracking the best per evaluation
    function fr = evaluate(Xr)
        [fv, FE_new] = calculate_fitness(Xr', problem, FE);
        fr = fv(:);
        for q = 1:numel(fr)
            if fr(q) < bsf
                bsf = fr(q);
                bsf_solution = Xr(q, :);
            end
            curve(FE + q) = bsf;
        end
        FE = FE_new;
    end
end

% Best and worst index; the worst is not checked on a row that sets a new best, as released
function [minI, maxI] = min_max(f)
    minI = 1;
    maxI = 1;
    bv = f(1);
    wv = bv;
    for i = 2:numel(f)
        if f(i) < bv
            minI = i;
            bv = f(i);
        elseif f(i) > wv
            maxI = i;
            wv = f(i);
        end
    end
end

% Emigration rises and immigration falls with habitat quality
function [emg, img] = calc_rates(f, minV, maxV, eps_r)
    den = maxV - minV + eps_r;
    emg = (maxV - f' + eps_r) / den;
    img = (f' - minV + eps_r) / den;
end

% Random symmetric graph, each ordered pair linked with 1-(1-1/NP)^k; no node left isolated
function top = rand_top(NP, k)
    prob = 1.0 - power(1.0 - 1.0 / NP, k);
    top = zeros(NP, NP);
    for i = 1:NP
        iso = 1;
        for j = 1:NP
            if i ~= j && rand < prob
                top(i, j) = 1;
                top(j, i) = 1;
                iso = 0;
            end
        end
        if iso == 1
            j = ceil(NP * rand);
            while i == j
                j = ceil(NP * rand);
            end
            top(i, j) = 1;
            top(j, i) = 1;
        end
    end
end

function [r1, r2, r3] = three_indices(NP, index)
    r1 = ceil(NP * rand);
    while r1 == index
        r1 = ceil(NP * rand);
    end
    r2 = ceil(NP * rand);
    while r2 == index || r2 == r1
        r2 = ceil(NP * rand);
    end
    r3 = ceil(NP * rand);
    while r3 == index || r3 == r2 || r3 == r1
        r3 = ceil(NP * rand);
    end
end

% Neighbour drawn in proportion to its emigration rate
function nIndex = select_neighbor(nbs, cum_emg)
    s = cum_emg(end);
    if s == 0
        nIndex = nbs(1);
        return;
    end
    r = rand * s;
    pick = find(r < cum_emg, 1);
    if isempty(pick)
        pick = numel(nbs);
    end
    nIndex = nbs(pick);
end

% The release's success rates never leave [0.5 0.5], so this is a fair coin over two strategies
function sel = select_strategy()
    if rand < 0.5
        sel = 1;
    else
        sel = 2;
    end
end

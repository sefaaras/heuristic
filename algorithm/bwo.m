% ----------------------------------------------------------------------- %
% Beluga Whale Optimization (BWO)
% ----------------------------------------------------------------------- %
% NOTE: This is Beluga Whale Optimization (Zhong et al., 2022), which shares
% the "BWO" acronym with Black Widow Optimization (stored separately as bwoa).
%
% Algorithm Parameters:
%   Npop = 50   % Population size (beluga whales)
%
% Algorithm Concept:
%   - Exploration: paired swimming with mirrored sin/cos position updates
%   - Exploitation: prey capture using Levy flight toward the best whale
%   - Whale fall: self-adaptive re-initialization whose probability WF
%     decays over the run
%
% Reference:
% Changting Zhong, Gang Li, Zeng Meng,
% Beluga whale optimization: A novel nature-inspired metaheuristic algorithm,
% Knowledge-Based Systems 251 (2022) 109215.
% https://doi.org/10.1016/j.knosys.2022.109215
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bwo(problem)

    % Extract problem parameters
    nD = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    Npop = 50;
    Max_it = ceil((maxFE / Npop) * 0.95);

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    fit = inf * ones(Npop, 1);
    newfit = fit;

    if size(ub, 2) == 1
        lb = lb * ones(1, nD); ub = ub * ones(1, nD);
    end

    pos = zeros(Npop, nD);
    for i = 1:Npop
        pos(i, :) = rand(1, nD) .* (ub - lb) + lb;
    end

    [fit(:), FE] = calculate_fitness(pos', problem, FE);
    fit = fit(:);

    [fvalbest, index] = min(fit);
    xposbest = pos(index, :);

    bsf = fvalbest;
    for eval_count = 1:Npop
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, pos, fit', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    T = 1;
    while T <= Max_it
        if FE >= maxFE, break; end

        newpos = pos;
        WF = 0.1 - 0.05 * (T / Max_it);              % probability of whale fall
        kk = (1 - 0.5 * T / Max_it) * rand(Npop, 1);  % exploration/exploitation split

        for i = 1:Npop
            if kk(i) > 0.5 % exploration phase
                r1 = rand(); r2 = rand();
                RJ = ceil(Npop * rand);
                while RJ == i
                    RJ = ceil(Npop * rand);
                end
                if nD <= Npop / 5
                    params = randperm(nD, 2);
                    newpos(i, params(1)) = pos(i, params(1)) + (pos(RJ, params(1)) - pos(i, params(2))) * (r1 + 1) * sin(r2 * 360);
                    newpos(i, params(2)) = pos(i, params(2)) + (pos(RJ, params(1)) - pos(i, params(2))) * (r1 + 1) * cos(r2 * 360);
                else
                    params = randperm(nD);
                    for j = 1:floor(nD / 2)
                        newpos(i, 2 * j - 1) = pos(i, params(2 * j - 1)) + (pos(RJ, params(1)) - pos(i, params(2 * j - 1))) * (r1 + 1) * sin(r2 * 360);
                        newpos(i, 2 * j) = pos(i, params(2 * j)) + (pos(RJ, params(1)) - pos(i, params(2 * j))) * (r1 + 1) * cos(r2 * 360);
                    end
                end
            else  % exploitation phase
                r3 = rand(); r4 = rand(); C1 = 2 * r4 * (1 - T / Max_it);
                RJ = ceil(Npop * rand);
                while RJ == i
                    RJ = ceil(Npop * rand);
                end
                alpha = 3 / 2;
                sigma = (gamma(1 + alpha) * sin(pi * alpha / 2) / (gamma((1 + alpha) / 2) * alpha * 2^((alpha - 1) / 2)))^(1 / alpha);
                u = randn(1, nD) .* sigma;
                v = randn(1, nD);
                S = u ./ abs(v).^(1 / alpha);
                KD = 0.05;
                LevyFlight = KD .* S;
                newpos(i, :) = r3 * xposbest - r4 * pos(i, :) + C1 * LevyFlight .* (pos(RJ, :) - pos(i, :));
            end
            % boundary
            Flag4ub = newpos(i, :) > ub;
            Flag4lb = newpos(i, :) < lb;
            newpos(i, :) = (newpos(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            [newfit(i, 1), FE] = calculate_fitness(newpos(i, :)', problem, FE);
            if newfit(i, 1) < fit(i, 1)
                pos(i, :) = newpos(i, :);
                fit(i, 1) = newfit(i, 1);
            end
            if newfit(i, 1) < bsf
                bsf = newfit(i, 1);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, pos, fit', population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        if FE >= maxFE, break; end

        for i = 1:Npop
            % whale falls
            if kk(i) <= WF
                RJ = ceil(Npop * rand); r5 = rand(); r6 = rand(); r7 = rand();
                C2 = 2 * Npop * WF;
                stepsize2 = r7 * (ub - lb) * exp(-C2 * T / Max_it);
                newpos(i, :) = (r5 * pos(i, :) - r6 * pos(RJ, :)) + stepsize2;
                % boundary
                Flag4ub = newpos(i, :) > ub;
                Flag4lb = newpos(i, :) < lb;
                newpos(i, :) = (newpos(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
                [newfit(i, 1), FE] = calculate_fitness(newpos(i, :)', problem, FE);
                if newfit(i, 1) < fit(i, 1)
                    pos(i, :) = newpos(i, :);
                    fit(i, 1) = newfit(i, 1);
                end
                if newfit(i, 1) < bsf
                    bsf = newfit(i, 1);
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, pos, fit', population_history, fitness_history, ...
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
            end
        end

        [fval, index] = min(fit);
        if fval < fvalbest
            fvalbest = fval;
            xposbest = pos(index, :);
        end
        T = T + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness = fvalbest;
    best_solution = xposbest;
end

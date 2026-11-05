% ----------------------------------------------------------------------- %
% Gbest-guided Artificial Bee Colony (GABC)
% Variant of ABC (abc): a gbest term is added to the neighbour move
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SN    = 40        % Food sources: colony of 80, half employed, half onlookers
%   limit = SN*D      % Trials without improvement before a source is abandoned
%   C     = 1.5       % psi ~ U(0, C): weight of the pull toward the global best
%
% Algorithm Concept:
%   - Move (Eq. 2): one random dimension j, v_ij = x_ij + phi*(x_ij - x_kj)
%     + psi*(y_j - x_ij), phi ~ U(-1, 1), k a random other source, y the global best
%   - Greedy selection on fitness 1/(1+f) for f >= 0, 1+|f| for f < 0; a failed
%     move adds one to the source's trial counter, a success resets it
%   - Employed bees: one move from every source per cycle
%   - Onlookers: SN further moves, each on a source drawn by roulette with
%     P_i = fit_i/sum(fit), computed once after the employed bees
%   - Global best memorised after the onlookers; then the source with the most
%     trials, once above limit, is redrawn uniformly in the box (one per cycle)
%
% Reference:
% Guopu Zhu, Sam Kwong,
% Gbest-guided artificial bee colony algorithm for numerical function optimization,
% Applied Mathematics and Computation 217 (7) (2010) 3166-3173.
% https://doi.org/10.1016/j.amc.2010.08.049
% ----------------------------------------------------------------------- %
% Implementation Note:
% Checked against the paper; no GABC code was released. The loop is Karaboga's
% basic ABC MATLAB release (artificialbeecolony/MATLABv2_ABC runABC.m, 2009) with
% the paper's Eq. (2) and its onlooker rule P_i = fit_i/sum(fit) (Sec. 2), one
% roulette draw per onlooker in place of the release's 0.9*fit/max(fit)+0.1 sweep.
% SN = 40: the paper's population 80 and 5000 generations for 400,000 FE (Sec.
% 4.2) is 2*SN evaluations per cycle. The paper is silent on limit, kept at SN*D
% (Karaboga and Akay 2009, as abc), and on the fitness transform, the phi/psi draw
% order, bound handling (clamped) and when y is refreshed: all kept from the
% release, y memorised after init and after each onlooker stage. Reported: the
% best point evaluated. All-zero or infinite fitness: onlooker weights guarded.
%     Table 7 rerun, 400,000 FE, 30 runs, mean port / paper (below 1e-20 = 0, as there)
%     GABC: Rosenbrock D=2 1.6e-4 / 1.7e-4, D=3 8.8e-4 / 2.7e-3; Schaffer D=3 0 / 1.9e-18
%     GABC D=30: Sphere 2.8e-16 / 4.2e-16, Ackley 3.0e-14 / 3.2e-14, Griewank 0 / 3.0e-17
%     C = 0 against the paper's ABC: Schaffer D=3 6.1e-6 / 5.8e-6, Rosenbrock D=2 4.4e-3 / 9.9e-3
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gabc(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    SN = 40;
    limit = SN * dim;
    C = 1.5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Foods = rand(SN, dim) .* (ub - lb) + lb;
    [ObjVal, FE] = calculate_fitness(Foods', problem, FE);
    ObjVal = ObjVal(:)';
    Fitness = abc_fitness(ObjVal);
    trial = zeros(1, SN);

    % Memorised best as the release keeps it: the last index among ties
    [GlobalMin, GlobalParams] = memorize(ObjVal, Foods, inf, Foods(1, :));

    best_fitness = Inf;
    best_solution = Foods(1, :);
    for e = 1:min(SN, maxFE)
        if ObjVal(e) < best_fitness
            best_fitness = ObjVal(e);
            best_solution = Foods(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, Foods, ObjVal, population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE
        % Employed bees
        for i = 1:SN
            if FE >= maxFE, break; end
            forage(i);
            [population_history, fitness_history, history_index] = record_history(...
                FE, Foods, ObjVal, population_history, fitness_history, ...
                history_index, maxFE);
        end

        % Onlookers: one roulette draw each on P_i = fit_i/sum(fit), fixed for the stage
        cw = cumsum(onlooker_weights(Fitness));
        for t = 1:SN
            if FE >= maxFE, break; end
            forage(find(cw >= rand * cw(end), 1));
            [population_history, fitness_history, history_index] = record_history(...
                FE, Foods, ObjVal, population_history, fitness_history, ...
                history_index, maxFE);
        end

        [GlobalMin, GlobalParams] = memorize(ObjVal, Foods, GlobalMin, GlobalParams);

        % Scout; ties go to the last index, as the release's ind(end)
        s = find(trial == max(trial), 1, 'last');
        if trial(s) > limit && FE < maxFE
            trial(s) = 0;
            sol = (ub - lb) .* rand(1, dim) + lb;
            [f, FE] = calculate_fitness(sol', problem, FE);
            Foods(s, :) = sol;
            ObjVal(s) = f;
            Fitness(s) = abc_fitness(f);
            if f < best_fitness
                best_fitness = f;
                best_solution = sol;
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, Foods, ObjVal, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % One employed or onlooker move of source i, drawn in the release's order
    function forage(i)
        j = fix(rand * dim) + 1;
        k = fix(rand * SN) + 1;
        while k == i
            k = fix(rand * SN) + 1;
        end
        phi = (rand - 0.5) * 2;
        psi = rand * C;
        sol = Foods(i, :);
        sol(j) = Foods(i, j) + phi * (Foods(i, j) - Foods(k, j)) ...
                 + psi * (GlobalParams(j) - Foods(i, j));
        sol = min(max(sol, lb), ub);   % shifted onto the violated bound, as released
        [f, FE] = calculate_fitness(sol', problem, FE);
        fs = abc_fitness(f);
        if fs > Fitness(i)
            Foods(i, :) = sol;
            Fitness(i) = fs;
            ObjVal(i) = f;
            trial(i) = 0;
        else
            trial(i) = trial(i) + 1;
        end
        if f < best_fitness
            best_fitness = f;
            best_solution = sol;
        end
        curve(FE) = best_fitness;
    end
end

% The release's MemorizeBestSource: keep the colony's best if it beats the memory
function [gmin, gx] = memorize(ObjVal, Foods, gmin, gx)
    b = find(ObjVal == min(ObjVal), 1, 'last');
    if ~isempty(b) && ObjVal(b) < gmin
        gmin = ObjVal(b);
        gx = Foods(b, :);
    end
end

% The release's calculateFitness: f = -1 maps to 2, NaN and +Inf to 0, only f = -Inf to Inf
function fit = abc_fitness(f)
    fit = zeros(size(f));
    pos = f >= 0;
    fit(pos) = 1 ./ (f(pos) + 1);
    neg = f < 0;
    fit(neg) = 1 + abs(f(neg));
end

% Weights proportional to fit_i; divided by max(fit) so their sum cannot overflow
function w = onlooker_weights(fit)
    top = max(fit);
    if isinf(top)
        w = double(isinf(fit));   % an f = -Inf source takes every onlooker
    elseif top > 0
        w = fit / top;
    else
        w = ones(size(fit));      % all NaN or +Inf: uniform
    end
end

% ----------------------------------------------------------------------- %
% Modified Artificial Bee Colony (MABC)
% Variant of ABC (abc); not the MABC of Gao and Liu, Comput. Oper. Res. 2012
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SN    = max(2, ceil(D/2))   % Food sources: colony size D (Table 6), half employed
%   limit = 200                 % Trials without improvement before a source is abandoned
%   MR    = 0.4                 % Modification rate: chance that a dimension moves
%   SF    = 1                   % Scaling factor: phi ~ U(-SF, SF)
%   ASF   = false               % Adaptive SF by the 1/5 rule, Eq. (6); Table 6 runs fixed
%   m     = 10                  % Cycles per 1/5-rule window (ASF only)
%
% Algorithm Concept:
%   - Move: v_ij = x_ij + phi_ij*(x_ij - x_kj), k a random other source, on each
%     dimension with probability MR plus one random dimension that always moves
%   - Greedy selection on fitness 1/(1+f) for f >= 0, 1+|f| for f < 0; a failed
%     move adds one to the source's trial counter, a success resets it
%   - Employed bees: one move from every source per cycle
%   - Onlookers: SN further moves, a source taken with probability fit_i/sum(fit)
%     by a cyclic sweep over the colony
%   - Scout: the source with the most trials, once above limit, is redrawn
%     uniformly in the box; at most one per cycle
%   - ASF: every m cycles SF is multiplied by 0.85 if under 1/5 of the moves
%     succeeded and divided by 0.85 if over 1/5
%
% Reference:
% Bahriye Akay, Dervis Karaboga,
% A modified Artificial Bee Colony algorithm for real-parameter optimization,
% Information Sciences 192 (2012) 120-142.
% https://doi.org/10.1016/j.ins.2010.07.015
% ----------------------------------------------------------------------- %
% Implementation Note:
% No MABC code was released: the skeleton is the authors' basic ABC MATLAB
% release (runABCv2.m, 2009), the parameters the paper's Table 6. Colony size D
% counts employed plus onlooker bees, so SN = D/2, and maxFe = 10000*D gives the
% paper's 10000 cycles; ceil and the floor of 2 are crash fixes for odd D and D=2.
% Onlookers follow Eq. (4), fit/sum(fit), not the release's 0.9*fit/max + 0.1.
% The forced dimension is how the paper reads MR = 0 as basic ABC (Sec. 5.1),
% as the authors' constrained Delphi release does. ASF: m and the start SF are
% not given; m = 10 cycles (about 10D moves, Schwefel's window) from SF = 1.
% Kept: greedy selection on the transformed fitness, blind near f = 0 to gains
% under 1e-16 of 1+f. The best is the best point evaluated, at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mabc(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    SN = max(2, ceil(dim / 2));
    limit = 200;
    MR = 0.4;
    SF = 1;
    ASF = false;
    m = 10;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Eq. (1)
    Foods = lb + rand(SN, dim) .* (ub - lb);
    [ObjVal, FE] = calculate_fitness(Foods', problem, FE);
    ObjVal = ObjVal(:)';
    Fitness = abc_fitness(ObjVal);
    trial = zeros(1, SN);

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

    nTry = 0;
    nWin = 0;
    cycle = 0;
    while FE < maxFE
        % Employed bees
        for i = 1:SN
            if FE >= maxFE, break; end
            forage(i);
        end

        % Onlookers: the pseudo-code's sweep, i advancing whether or not it is chosen
        prob = selection_prob(Fitness);
        i = 1;
        t = 0;
        while t < SN && FE < maxFE
            if rand < prob(i)
                t = t + 1;
                forage(i);
            end
            i = mod(i, SN) + 1;
        end

        % Scout; ties go to the last index, as the release's ind(end)
        maxTrial = max(trial);
        if maxTrial > limit && FE < maxFE
            s = find(trial == maxTrial, 1, 'last');
            trial(s) = 0;
            sol = lb + rand(1, dim) .* (ub - lb);
            [f, FE] = calculate_fitness(sol', problem, FE);
            Foods(s, :) = sol;
            ObjVal(s) = f;
            Fitness(s) = abc_fitness(f);
            stamp(sol, f);
        end

        cycle = cycle + 1;
        if ASF && mod(cycle, m) == 0
            % Eq. (6); a success is an employed or onlooker move the greedy step accepted
            rate = nWin / max(nTry, 1);
            if rate < 1/5
                SF = SF * 0.85;
            elseif rate > 1/5
                SF = SF / 0.85;
            end
            nTry = 0;
            nWin = 0;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % One employed or onlooker move of source i, Eq. (5) with phi in [-SF, SF]
    function forage(i)
        k = randi(SN - 1);
        k = k + (k >= i);
        mask = rand(1, dim) < MR;
        mask(randi(dim)) = true;   % MR = 0 must still move one dimension, as basic ABC
        phi = SF * (2 * rand(1, dim) - 1);
        sol = Foods(i, :);
        sol(mask) = sol(mask) + phi(mask) .* (Foods(i, mask) - Foods(k, mask));
        sol = min(max(sol, lb), ub);   % set to the violated bound (Sec. 2.2)
        [f, FE] = calculate_fitness(sol', problem, FE);
        fs = abc_fitness(f);
        nTry = nTry + 1;
        if fs > Fitness(i)
            Foods(i, :) = sol;
            ObjVal(i) = f;
            Fitness(i) = fs;
            trial(i) = 0;
            nWin = nWin + 1;
        else
            trial(i) = trial(i) + 1;
        end
        stamp(sol, f);
    end

    % Best-so-far at this evaluation, then one history sample of the colony
    function stamp(x, f)
        if f < best_fitness
            best_fitness = f;
            best_solution = x;
        end
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            FE, Foods, ObjVal, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

% Eq. (3) as released: f = -1 maps to 2, NaN and +Inf to 0, only f = -Inf to Inf
function fit = abc_fitness(f)
    fit = zeros(size(f));
    pos = f >= 0;
    fit(pos) = 1 ./ (f(pos) + 1);
    neg = f < 0;
    fit(neg) = 1 + abs(f(neg));
end

% Eq. (4), fit_i / sum(fit), scaled by the max first so the sum cannot overflow
function p = selection_prob(fit)
    top = max(fit);
    if isinf(top)
        r = double(isinf(fit));   % an f = -Inf source takes every onlooker
    elseif top > 0
        r = fit / top;
    else
        r = ones(size(fit));      % all NaN or +Inf: uniform, so the sweep still ends
    end
    p = r / sum(r);
end

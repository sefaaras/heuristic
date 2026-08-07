% ----------------------------------------------------------------------- %
% Composite Differential Evolution (CoDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 30                 % Population size
%   F  pool = [1.0 1.0 0.8]      % Three fixed control parameter settings,
%   CR pool = [0.1 0.9 0.2]      % each (F, CR) pair used as a unit
%
% Algorithm Concept:
%   - For every target, EACH of three strategies builds one trial with an
%     independently drawn parameter setting, and only the best of the three
%     competes with the parent -- three evaluations per individual per generation
%   - The strategies are complementary rather than individually strong:
%       rand/1/bin  unbiased exploration;  rand/2/bin  wider perturbation;
%       current-to-rand/1  rotation invariant, no crossover at all
%   - The (F, CR) settings are spread just as deliberately: (1.0, 0.1) for
%     separable and rotated problems, (1.0, 0.9) for non-separable ones, and
%     (0.8, 0.2) as a compromise
%   - Nothing is adapted. CoDE's point is that a fixed, well-spread ensemble
%     matches the adaptive schemes of its time without any learning machinery
%   - Violating components are reflected about the bound, then clamped
%
% Reference:
% Yong Wang, Zixing Cai, Qingfu Zhang,
% Differential Evolution With Composite Trial Vector Generation Strategies
% and Control Parameters,
% IEEE Transactions on Evolutionary Computation, vol. 15, no. 1, pp. 55-66, 2011.
% https://doi.org/10.1109/TEVC.2010.2087271
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (CoDE.m and generator.m by
% Y. Wang), including the deviation the reference documents in a comment: the
% current-to-rand/1 strategy draws its three donors UNIFORMLY WITH REPLACEMENT
% from the whole population, target included, unlike the other two. The authors
% state they found this improves performance.
% Named codede because `code` is not usable as a MATLAB function name.
% BUDGET GRANULARITY: a generation costs 3*popsize = 90 evaluations and an
% individual's three trials are only meaningful together, so the last generation
% is not truncated and a run can overshoot maxFe by up to 89, as the reference does.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = codede(problem)

    n     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    popsize = 30;
    Fpool   = [1.0 1.0 0.8];
    CRpool  = [0.1 0.9 0.2];

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    p = repmat(lb, popsize, 1) + rand(popsize, n) .* repmat(ub - lb, popsize, 1);

    [fit, FE] = calculate_fitness(p', problem, FE);
    fit = fit(:);

    bsf          = inf;
    bsf_solution = p(1, :);
    for i = 1:popsize
        if fit(i) < bsf
            bsf          = fit(i);
            bsf_solution = p(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, p, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Exclusion table: ISET(i, :) lists every index other than i
    ISET = zeros(popsize, popsize - 1);
    for i = 1:popsize
        ISET(i, :) = [1:i-1, i+1:popsize];
    end
    rows = (1:popsize)';

    % Main loop
    while FE < maxFE
        % Independent uniform parameter setting per strategy per individual
        paraIndex = floor(rand(popsize, 3) * numel(Fpool)) + 1;
        F1 = Fpool(paraIndex(:, 1))';   CR1 = CRpool(paraIndex(:, 1))';
        F2 = Fpool(paraIndex(:, 2))';
        F3 = Fpool(paraIndex(:, 3))';   CR3 = CRpool(paraIndex(:, 3))';

        % Strategy 1: rand/1/bin
        r = drawDistinct(ISET, popsize, 3);
        v1 = p(r(:, 1), :) + F1(:, ones(1, n)) .* (p(r(:, 2), :) - p(r(:, 3), :));
        v1 = repairReflect(v1, lb, ub);
        u1 = binCross(v1, p, CR1, popsize, n, rows);

        % Strategy 2: current-to-rand/1 (indices with replacement)
        q  = floor(rand(popsize, 3) * popsize) + 1;
        K  = rand(popsize, 1);
        v2 = p + K(:, ones(1, n)) .* (p(q(:, 1), :) - p) ...
               + F2(:, ones(1, n)) .* (p(q(:, 2), :) - p(q(:, 3), :));
        v2 = repairReflect(v2, lb, ub);
        u2 = v2;                                  % no crossover for this strategy

        % Strategy 3: rand/2/bin
        s  = drawDistinct(ISET, popsize, 5);
        K3 = rand(popsize, 1);
        v3 = p(s(:, 1), :) + K3(:, ones(1, n)) .* (p(s(:, 2), :) - p(s(:, 3), :)) ...
                           + F3(:, ones(1, n)) .* (p(s(:, 4), :) - p(s(:, 5), :));
        v3 = repairReflect(v3, lb, ub);
        u3 = binCross(v3, p, CR3, popsize, n, rows);

        % Evaluate the three trials of every individual
        uSet = zeros(3 * popsize, n);
        uSet(1:3:end, :) = u1;
        uSet(2:3:end, :) = u2;
        uSet(3:3:end, :) = u3;

        [fitSet, FE] = calculate_fitness(uSet', problem, FE);
        fitSet = fitSet(:);

        nEval = 3 * popsize;
        for k = 1:nEval
            if fitSet(k) < bsf
                bsf          = fitSet(k);
                bsf_solution = uSet(k, :);
            end
            ec = FE - nEval + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, p, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Best of the three trials competes with the parent
        F3set = reshape(fitSet, 3, popsize)';        % popsize x 3
        [bestFit, bestID] = min(F3set, [], 2);
        bestRow = 3 * (rows - 1) + bestID;

        win = (fit >= bestFit);                      % ties replace, as in the reference
        p(win, :) = uSet(bestRow(win), :);
        fit(win)  = bestFit(win);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function idx = drawDistinct(ISET, NP, k)
% k distinct indices per row, drawn without replacement from ISET(i,:), which excludes i
    [~, ord] = sort(rand(NP, size(ISET, 2)), 2);
    lin = (1:NP)' + (ord(:, 1:k) - 1) * NP;
    idx = ISET(lin);
end

function u = binCross(v, p, CR, NP, n, rows)
    t = rand(NP, n) < CR(:, ones(1, n));
    t(sub2ind([NP n], rows, floor(rand(NP, 1) * n) + 1)) = true;
    u = p;
    u(t) = v(t);
end

function v = repairReflect(v, lb, ub)
% Reflect about the violated bound, clamping at the opposite bound if the reflection overshoots
    NP = size(v, 1);
    L  = repmat(lb, NP, 1);
    U  = repmat(ub, NP, 1);

    low = v < L;
    v(low) = 2 .* L(low) - v(low);
    over = low & (v > U);
    v(over) = U(over);

    high = v > U;
    v(high) = 2 .* U(high) - v(high);
    under = high & (v < L);
    v(under) = L(under);
end

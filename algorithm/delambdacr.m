% ----------------------------------------------------------------------- %
% Hybrid Differential Evolution with Adaptive Crossover Operator (DE-LambdaCr)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Xpop        = 5*D                % Population size
%   XpopRefresh = 2*D                % Best members kept by a population refresh
%   GammaVar    = 3                  % Refresh once every IQR < (init range)/GammaVar
%   minVarPop   = 0.05*(ub - lb)     % Smallest half-width of a refresh range
%   F  ~ Tri(0.3, 0.4, 0.5)          % Scaling factor (min, mode, max), never adapted
%   CR ~ Tri(0.2, 0.5, 1.0)          % Initial crossover-rate triangle, refitted (CRsuccess = 15)
%   Recomb = 0.75                    % Linear recombination factor
%   pLS = 1/(100*D)                  % Per-child probability of a local search
%
% Algorithm Concept:
%   - DE/rand/1 with F drawn per member from a triangle; the three donors are any
%     three distinct members, the target included
%   - Binomial crossover for CR < 0.95, linear recombination x + 0.75*(v - x) for
%     CR >= 0.95 or when a CR >= 0.5 draw took every coordinate from the donor
%   - The CR triangle (min, median, max) is refitted from successful trials' CRs once
%     15 have gathered; a side under 0.1 is widened to 0.1 inside [0.2, 1]
%   - Refresh: when every coordinate's IQR is below 1/3 of its current init range,
%     3D members are redrawn around the median in a shrunk range, the best 2D kept
%   - A child is refined by an fmincon active-set (SQP) search with probability pLS
%
% Reference:
% Gilberto Reynoso-Meza, Javier Sanchis, Xavier Blasco, Juan M. Herrero,
% Hybrid DE algorithm with adaptive crossover operator for solving real-world
% numerical optimization problems,
% IEEE Congress on Evolutionary Computation (CEC 2011), New Orleans, 2011, 1551-1556.
% https://doi.org/10.1109/CEC.2011.5949800
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release DE_TCR (File Exchange 39217, 19 April 2013 version).
% Kept as released: both refresh tests are `if` on a vector (all coordinates must pass);
% the new upper refresh bound uses the moved lower one; the CR success list is never
% cleared (surplus rows join every refit); kept best members and a searched child are
% re-evaluated; 5D and 2D lack the paper's caps [50, 300] and [20, 120]; the target may
% be its own donor (the paper excludes it). Changed: MAXGEN = 200*D is dropped (tuned to
% 1e4*D FE, it ends a D = 2 run at 4 % of the budget); the budget is cut exactly, also
% inside fmincon; fmincon's points are clamped to the box (active-set left it on up to
% 0.4 % of evaluations, by <= 9e-3) and an fmincon error keeps the child. Seeded runs
% match the release bit for bit to the first such point, on CEC2020RW for the whole run.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = delambdacr(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    Xpop        = 5 * D;
    XpopRefresh = 2 * D;
    GammaVar    = 3;
    minVarPop   = 0.05 * (ub - lb);
    MinF = 0.3;   MedF = 0.4;   MaxF = 0.5;
    MinCr = 0.2;  MedCr = 0.5;  MaxCr = 1.0;
    Recomb    = 0.75;
    CRsuccess = 15;
    RateLS    = 1 - 1 / (100 * D);

    initLo = lb;   % (re)initialisation range, moved by every refresh
    initHi = ub;
    ParamEVO = zeros(CRsuccess, 1);
    SuccessEvoCr = 0;
    Child = zeros(Xpop, D);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % rand(D, n)' draws row by row, in the order of the release's nested loops
    pop = initLo + (initHi - initLo) .* rand(D, Xpop)';
    nEval = min(Xpop, maxFE);
    [fit, FE] = calculate_fitness(pop(1:nEval, :)', problem, FE);
    fit = fit(:);

    best_fitness  = inf;
    best_solution = pop(1, :);
    for e = 1:nEval
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = pop(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, pop(1:nEval, :), fit, population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        % Population refresh; both `if`s on a vector are all() as in the release
        VarPop = iqr(pop);
        if all(VarPop <= (initHi - initLo) / GammaVar)
            LookingBest = sortrows([fit pop]);
            med = median(pop);
            if all(VarPop < minVarPop)
                initLo = med - minVarPop;
                initHi = med + minVarPop;
            else
                initLo = med - (initHi - initLo) / GammaVar;
                initHi = med + (initHi - initLo) / GammaVar;   % release: uses the moved initLo
            end
            initLo = max(initLo, lb);
            initHi = min(initHi, ub);

            fresh = initLo + (initHi - initLo) .* rand(D, Xpop - XpopRefresh)';
            newPop = [fresh; LookingBest(1:XpopRefresh, 2:end)];
            nEval = min(Xpop, maxFE - FE);
            newPop = newPop(1:nEval, :);
            [newFit, FE] = calculate_fitness(newPop', problem, FE);
            newFit = newFit(:);
            for e = 1:nEval
                if newFit(e) < best_fitness
                    best_fitness  = newFit(e);
                    best_solution = newPop(e, :);
                end
                ec = FE - nEval + e;
                curve(ec) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fit, population_history, fitness_history, history_index, maxFE);
            end
            if nEval < Xpop
                break;
            end
            pop = newPop;
            fit = newFit;
        end

        % Inverse-CDF triangular draws, F then CR for each member as the release interleaves them
        R  = rand(2, Xpop);
        Fv = triangular(R(1, :)', MinF, MedF, MaxF);
        CRv = triangular(R(2, :)', MinCr, MedCr, MaxCr);

        Parent   = pop;
        JxParent = fit;
        for xp = 1:Xpop
            rev = randperm(Xpop);
            Mutant = Parent(rev(1), :) + Fv(xp) * (Parent(rev(2), :) - Parent(rev(3), :));
            Mutant = min(max(Mutant, lb), ub);

            CRrate = CRv(xp);
            xpar = Parent(xp, :);
            if CRrate >= 0.95
                child = xpar + Recomb .* (Mutant - xpar);
                fromParent = false(1, D);
                flagCr = true;
            else
                fromParent = rand(1, D) > CRrate;
                child = Mutant;
                child(fromParent) = xpar(fromParent);
                flagCr = any(fromParent);
            end
            % All coordinates from the donor: give one back (low CR) or recombine (CR >= 0.5)
            if ~flagCr && CRrate < 0.5
                j = randperm(D);
                child(j(1)) = xpar(j(1));
            end
            if ~flagCr && CRrate >= 0.5 && CRrate < 0.95
                child = xpar + Recomb .* (Mutant - xpar);
            end
            % All coordinates from the parent: take one from the donor
            if all(fromParent)
                j = randperm(D);
                child(j(1)) = Mutant(j(1));
            end
            child = min(max(child, lb), ub);

            if rand > RateLS
                [child, flog, xls, fls] = local_search(child, problem, lb, ub, maxFE - FE);
                running = best_fitness;
                for t = 1:numel(flog)
                    FE = FE + 1;
                    if flog(t) < running
                        running = flog(t);
                    end
                    curve(FE) = running;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Parent, JxParent, population_history, fitness_history, history_index, maxFE);
                end
                if fls < best_fitness
                    best_fitness  = fls;
                    best_solution = xls;
                end
                if FE >= maxFE
                    break;
                end
            end
            Child(xp, :) = child;
        end
        if FE >= maxFE
            break;
        end

        nEval = min(Xpop, maxFE - FE);   % the last generation is cut to the remaining budget
        [fc, FE] = calculate_fitness(Child(1:nEval, :)', problem, FE);
        fc = fc(:);
        for e = 1:nEval
            if fc(e) < best_fitness
                best_fitness  = fc(e);
                best_solution = Child(e, :);
            end
            ec = FE - nEval + e;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Parent, JxParent, population_history, fitness_history, history_index, maxFE);
        end
        if nEval < Xpop
            break;
        end

        % Greedy selection; every success logs its CR, a NaN parent loses to any number
        for xp = 1:Xpop
            if fc(xp) <= JxParent(xp) || (isnan(JxParent(xp)) && ~isnan(fc(xp)))
                SuccessEvoCr = SuccessEvoCr + 1;
                ParamEVO(SuccessEvoCr, 1) = CRv(xp);
                Parent(xp, :) = Child(xp, :);
                JxParent(xp) = fc(xp);
            end
        end
        pop = Parent;
        fit = JxParent;

        % CR triangle refit over the whole success list (stale rows included, as released)
        if SuccessEvoCr >= CRsuccess
            SuccessEvoCr = 0;
            DataCR = sortrows(ParamEVO, 1);
            MaxCr = DataCR(end, 1);
            MinCr = DataCR(1, 1);
            MedCr = median(DataCR);
            if abs(MedCr - MaxCr) < 0.1
                MaxCr = min(MedCr + 0.1, 1.0);
            end
            if abs(MedCr - MinCr) < 0.1
                MinCr = max(MedCr - 0.1, 0.2);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

% Inverse CDF of the triangular distribution (lo, mode, hi), written as the release
function v = triangular(u, lo, mode, hi)
    v = hi - sqrt((1 - u) * (hi - lo) * (hi - mode));
    low = u < (mode - lo) * (2 / (hi - lo)) * 0.5;
    v(low) = lo + sqrt(u(low) * (hi - lo) * (mode - lo));
end

% The release's LocalSearch.m: fmincon active-set from x0, at most `cap` evaluations
function [x, flog, xb, fb] = local_search(x0, problem, lb, ub, cap)
    n    = 0;
    flog = zeros(1, min(cap, 11000));
    xb   = x0;
    fb   = inf;
    options = optimset('MaxFunEvals', 1e4, 'Display', 'off', ...
                       'algorithm', 'active-set', 'UseParallel', 'never');
    x = x0;
    try
        xs = fmincon(@objective, x0, [], [], [], [], lb(:), ub(:), [], options);
        if ~any(isnan(xs))
            x = min(max(xs, lb), ub);
        end
    catch
        % budget reached inside fmincon, or fmincon refused the point
    end
    flog = flog(1:n);

    function f = objective(xx)
        if n >= cap
            error('delambdacr:budget', 'FE budget exhausted inside the local search');
        end
        n = n + 1;
        xx = min(max(xx(:), lb(:)), ub(:));   % active-set steps may leave the box
        fv = calculate_fitness(xx, problem, 0);
        f = fv(1);
        flog(n) = f;
        if f < fb
            fb = f;
            xb = xx(:)';
        end
    end
end

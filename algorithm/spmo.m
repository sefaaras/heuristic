% ----------------------------------------------------------------------- %
% Spider Monkey Optimization (SMO)
% Stored as spmo; smo is Starling Murmuration Optimizer
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N           = 50             % Spider monkeys
%   MG          = 5              % Maximum number of groups (max_part)
%   LocalLimit  = D*N            % Local leader stagnation limit
%   GlobalLimit = N              % Global leader stagnation limit
%   pr          = 0.1 -> 0.5     % Perturbation rate, linear in evaluations spent
%   acc_err     = 1e-5           % Stagnation threshold (release: per problem, mostly 1e-5)
%
% Algorithm Concept:
%   - Local leader phase: each coordinate, with probability 1 - pr, moves by
%     U(0,1)(LL - x) + U(-1,1)(x_r - x), x_r from the own group; greedy
%   - Global leader phase: monkeys chosen with prob 0.9*fit/max fit + 0.1 move
%     one random coordinate the same way toward GL; group size - 1 trials
%   - Learning: GL and each LL take the best value seen; a change below
%     acc_err counts one stagnation
%   - Local leader decision: a group stagnant over LocalLimit is redirected,
%     each coordinate redrawn with prob 1 - pr, else x + U(GL - x) + U(x - LL)
%   - Global leader decision: GL stagnant over GlobalLimit splits the swarm
%     into one group more (fission), and back into one group after MG (fusion)
%
% Reference:
% Jagdish Chand Bansal, Harish Sharma, Shimpi Singh Jadon, Maurice Clerc,
% Spider Monkey Optimization algorithm for numerical optimization,
% Memetic Computing 6(1) (2014) 31-47.
% https://doi.org/10.1007/s12293-013-0128-0
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' C++ release, originalsmo.cpp of 14 Mar 2016 (smo.scrs.in,
% SMO_CppCode.zip); seeded runs are bit-identical to it, run as a MEX on the same
% rand stream with the changes below. Kept as released: create_group cuts N/part in
% doubles, so with 4 or 5 groups the last two overlap (members 40-50 or 45-50); the
% partner draw never picks a group's last member, nor does the global phase; an LL
% slot is never reset, so after a redirection or a regrouping it keeps its old
% leader until a member beats it; pr ends at 0.5 (paper Eq. (6): 0.4). Changed: pr
% is linear in evaluations (release: +0.4/5000 per iteration); the run starts as one
% group led by its best, unused slots empty (the release inherits both from the last
% run); no stop at an acceptable error; NaN counts as +Inf, an undefined ratio as 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = spmo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    N           = 50;
    maxPart     = 5;
    LocalLimit  = dim * N;
    GlobalLimit = N;
    accErr      = 1e-5;
    pr          = 0.1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % initialize(): monkey by monkey, coordinate by coordinate
    X = rand(dim, N)' .* (ub - lb) + lb;
    m = min(N, maxFE);
    fval = inf(1, N);
    [f0, FE] = calculate_fitness(X(1:m, :)', problem, FE);
    f0 = f0(:)';
    f0(isnan(f0)) = inf;
    fval(1:m) = f0;
    fit = calc_fitness(fval);

    best_fitness  = inf;
    best_solution = X(1, :);
    for e = 1:m
        if fval(e) < best_fitness
            best_fitness  = fval(e);
            best_solution = X(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fval, population_history, fitness_history, ...
            history_index, maxFE);
    end
    done = FE >= maxFE;

    % Leaders; slots 2..MG hold no leader until a split elects one
    GlobalMin = fval(1);
    GLpos = X(1, :);
    GlobalLimitCount = 0;
    LocalMin = inf(1, maxPart);
    LLpos = zeros(maxPart, dim);
    LocalLimitCount = zeros(1, maxPart);
    LocalMin(1) = fval(1);
    LLpos(1, :) = X(1, :);
    part = 1;
    gpoint = create_group(N, part);   % 0-based [lo hi] rows, as in the release
    group = size(gpoint, 1);
    global_learning();
    local_learning();
    prob = zeros(1, N);

    while ~done
        for k = 1:group
            local_leader_phase(k);
            if done, break; end
        end
        if done, break; end
        calculate_probabilities();
        for k = 1:group
            global_leader_phase(k);
            if done, break; end
        end
        if done, break; end
        global_learning();
        local_learning();
        local_leader_decision();
        if done, break; end
        global_leader_decision();
        pr = 0.1 + 0.4 * (FE - N) / (maxFE - N);   % the release's cr += 0.4/Max_iterations
    end

    curve(min(FE, maxFE):end) = best_fitness;

    function local_leader_phase(k)
        lo = gpoint(k, 1);
        hi = gpoint(k, 2);
        for i0 = lo:hi
            i = i0 + 1;
            r = i;
            while r == i
                r = fix(rand * (hi - lo) + lo) + 1;   % lo..hi-1, never the last member
            end
            newp = X(i, :);
            for j = 1:dim
                if rand >= pr
                    newp(j) = X(i, j) + (LLpos(k, j) - X(i, j)) * rand + (X(r, j) - X(i, j)) * (rand - 0.5) * 2;
                end
                if newp(j) < lb(j), newp(j) = lb(j); end
                if newp(j) > ub(j), newp(j) = ub(j); end
            end
            fnew = evaluate(newp);
            fitSol = calc_fitness(fnew);
            if fitSol > fit(i)
                X(i, :) = newp;
                fval(i) = fnew;
                fit(i) = fitSol;
            end
            stamp(newp, fnew);
            if done, return; end
        end
    end

    function calculate_probabilities()
        ratio = fit / max(fit);
        ratio(isnan(ratio)) = 1;   % Inf/Inf or 0/0: that monkey is the maximum
        prob = 0.9 * ratio + 0.1;
    end

    function global_leader_phase(k)
        lo = gpoint(k, 1);
        hi = gpoint(k, 2);
        i0 = lo;
        l = lo;
        while l < hi
            i = i0 + 1;
            if rand < prob(i)
                l = l + 1;
                r = i;
                while r == i
                    r = fix(rand * (hi - lo) + lo) + 1;
                end
                p = fix(rand * dim) + 1;
                newp = X(i, :);
                newp(p) = X(i, p) + (GLpos(p) - X(i, p)) * rand + (X(r, p) - X(i, p)) * (rand - 0.5) * 2;
                if newp(p) < lb(p), newp(p) = lb(p); end
                if newp(p) > ub(p), newp(p) = ub(p); end
                fnew = evaluate(newp);
                fitSol = calc_fitness(fnew);
                if fitSol > fit(i)
                    X(i, :) = newp;
                    fval(i) = fnew;
                    fit(i) = fitSol;
                end
                stamp(newp, fnew);
                if done, return; end
            end
            i0 = i0 + 1;
            if i0 == hi, i0 = lo; end   % wraps before the last member, as released
        end
    end

    function global_learning()
        Gtrial = GlobalMin;
        [mn, idx] = min(fval);
        if mn < GlobalMin
            GlobalMin = mn;
            GLpos = X(idx, :);
        end
        if abs(Gtrial - GlobalMin) < accErr
            GlobalLimitCount = GlobalLimitCount + 1;
        else
            GlobalLimitCount = 0;
        end
    end

    function local_learning()
        OldMin = LocalMin(1:group);
        for g = 1:group
            lo = gpoint(g, 1) + 1;
            [mn, idx] = min(fval(lo:gpoint(g, 2) + 1));
            if mn < LocalMin(g)
                LocalMin(g) = mn;
                LLpos(g, :) = X(lo + idx - 1, :);
            end
        end
        for g = 1:group
            if abs(OldMin(g) - LocalMin(g)) < accErr
                LocalLimitCount(g) = LocalLimitCount(g) + 1;
            else
                LocalLimitCount(g) = 0;
            end
        end
    end

    function local_leader_decision()
        for g = 1:group
            if LocalLimitCount(g) > LocalLimit
                for i = gpoint(g, 1) + 1:gpoint(g, 2) + 1
                    for j = 1:dim
                        if rand >= pr
                            X(i, j) = lb(j) + rand * (ub(j) - lb(j));
                        else
                            X(i, j) = X(i, j) + (GLpos(j) - X(i, j)) * rand + (X(i, j) - LLpos(g, j)) * rand;
                        end
                        if X(i, j) < lb(j), X(i, j) = lb(j); end
                        if X(i, j) > ub(j), X(i, j) = ub(j); end
                    end
                    fval(i) = evaluate(X(i, :));
                    fit(i) = calc_fitness(fval(i));
                    stamp(X(i, :), fval(i));
                    if done, return; end
                end
                LocalLimitCount(g) = 0;
            end
        end
    end

    function global_leader_decision()
        if GlobalLimitCount > GlobalLimit
            GlobalLimitCount = 0;
            if part < maxPart
                part = part + 1;
            else
                part = 1;
            end
            gpoint = create_group(N, part);
            group = size(gpoint, 1);
            local_learning();
        end
    end

    function f = evaluate(x)
        [f, FE] = calculate_fitness(x', problem, FE);
        f = f(1);
        if isnan(f), f = inf; end
    end

    % Best-so-far at this evaluation, then one history sample of (X, fval)
    function stamp(x, f)
        if f < best_fitness
            best_fitness  = f;
            best_solution = x;
        end
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            FE, X, fval, population_history, fitness_history, ...
            history_index, maxFE);
        done = FE >= maxFE;
    end
end

% CalculateFitness of the release: 1/(1+f) for f >= 0, else 1 + |f|
function fit = calc_fitness(f)
    fit = 1 ./ (f + 1);
    neg = f < 0;
    fit(neg) = 1 + abs(f(neg));
end

% create_group of the release: hi = lo + int(N/part), the remainder test in doubles
function gpoint = create_group(N, part)
    gpoint = zeros(0, 2);
    lo = 0;
    while lo < N
        hi = lo + fix(N / part);
        gpoint(end + 1, :) = [lo, hi]; %#ok<AGROW>
        if (N - hi) < (N / part)
            gpoint(end, 2) = N - 1;
        end
        lo = hi + 1;
    end
end

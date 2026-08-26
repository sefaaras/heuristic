function [fitness, FE, is_feasible, violation, objective, max_violation] = calculate_fitness(solutions, problem, FE, count_this)
% The only evaluator. solutions is D x ps (one column per candidate).
%
%   fitness       value the SEARCH minimises; on a constrained suite
%                 (CEC2020RW, CEDP) this folds in the constraints. Row for
%                 the numeric suites, column for those two -- each suite's own
%                 convention is preserved
%   FE            updated counter, returned UNCHANGED when count_this is false
%   is_feasible   true where violation == 0
%   violation     mean constraint violation v(x), 0 for unconstrained suites
%   objective     raw f(x) before any constraint folding
%   max_violation largest single-constraint violation, 0 when unconstrained
%
% count_this = false keeps an audit call out of the run counters (default true).
% calculate_fitness('reset') drops the cached problem state and must run once per
% run (main.m does it), or a parfor worker carries one job's state into the next.
% calculate_fitness('stats') returns the counters accumulated since that reset.

    persistent lastFhd lastNum lastDim lastIsCon worstObjective nIneq nEq runStats

    if ischar(solutions) || isstring(solutions)
        switch lower(char(solutions))
            case 'reset'
                lastFhd = []; lastNum = []; lastDim = []; lastIsCon = [];
                worstObjective = []; nIneq = []; nEq = [];
                runStats = new_stats();
                clear('cec20rw_func');   % drop its persistent G/B/P/Q/initial_flags
                fitness = []; FE = 0; is_feasible = []; violation = []; objective = [];
                max_violation = [];
                return;
            case 'stats'
                if isempty(runStats)
                    runStats = new_stats();
                end
                runStats.n_ineq = nIneq;
                runStats.n_eq   = nEq;
                runStats.is_rw  = ~isempty(lastIsCon) && lastIsCon;
                fitness = runStats;
                FE = 0; is_feasible = []; violation = []; objective = [];
                max_violation = [];
                return;
            otherwise
                error('calculate_fitness:badCommand', 'Unknown command "%s".', char(solutions));
        end
    end

    if nargin < 4
        count_this = true;
    end
    if isempty(runStats)
        runStats = new_stats();
    end

    num_evaluations = size(solutions, 2);
    D = size(solutions, 1);

    firstForThisProblem = isempty(lastNum) || problem.number ~= lastNum || ...
                          D ~= lastDim || ~isequal(problem.fhd, lastFhd);

    if firstForThisProblem
        % The CEC routine reads input_data from the current folder, so cd there
        % for the first evaluation of this (function, dimension).
        fname = func2str(problem.fhd);
        isRW = contains(fname, 'cec20rw');
        isCedp = contains(fname, 'cedp');
        lastIsCon = isRW || isCedp;   % suites that return [f, g, h]

        originalDir = pwd;
        restoreDir = onCleanup(@() cd(originalDir));  % restore cwd on return/error

        % CEDP is absent here on purpose: it is pure MATLAB and reads no
        % input_data, so it needs no cd -- unlike the MEX suites, which load
        % their shift and rotation data from the current folder.
        if contains(fname, 'cec14')
            cd('problem/CEC2014');
        elseif contains(fname, 'cec17')
            cd('problem/CEC2017');
        elseif isRW
            cd('problem/CEC2020RW');
        elseif contains(fname, 'cec20')
            cd('problem/CEC2020');
        elseif contains(fname, 'cec21')
            cd('problem/CEC2021');
        elseif contains(fname, 'cec22')
            cd('problem/CEC2022');
        end

        if isRW
            clear('cec20rw_func');

            par = Cal_par(problem.number);
            nIneq = par.g;
            nEq   = par.h;
        elseif isCedp
            par = cedp_par(problem.number);
            nIneq = par.g;
            nEq   = par.h;
        end
        worstObjective = [];   % new problem => new reference
        refBefore = worstObjective;

        [fitness, is_feasible, violation, objective, worstObjective, max_violation] = ...
            eval_core(solutions, problem, lastIsCon, worstObjective, nIneq, nEq);

        lastFhd = problem.fhd; lastNum = problem.number; lastDim = D;
    else
        refBefore = worstObjective;
        [fitness, is_feasible, violation, objective, worstObjective, max_violation] = ...
            eval_core(solutions, problem, lastIsCon, worstObjective, nIneq, nEq);
    end

    if count_this
        runStats = accumulate(runStats, is_feasible, violation, objective, num_evaluations);
        FE = FE + num_evaluations;
    else
        % An audit call leaves the evaluator as it found it. The CEC2020RW
        % reference is a running max, so raising it here would change the fitness
        % the search then sees for every infeasible point.
        worstObjective = refBefore;
    end
end

function s = new_stats()
% Per-run counters. n_eval is the ground truth for budget spend; the rest feeds
% the CEC2020RW report.
    s = struct('n_eval', 0, ...            % counted evaluations since the reset
               'n_feasible', 0, ...        % of those, how many were feasible
               'viol_sum', 0, ...          % sum of v(x) over evaluated points
               'best_obj', NaN, ...        % best raw f(x) seen, feasible or not
               'fe_best_obj', NaN, ...     % FE index at which it was evaluated
               'best_feas_obj', NaN, ...   % best raw f(x) among feasible points
               'fe_best_feas_obj', NaN, ...% FE index at which it was evaluated
               'first_fe_feasible', NaN, ...% FE index of the first feasible point
               'n_ineq', [], ...           % filled in by the 'stats' command
               'n_eq', [], ...
               'is_rw', false);
end

function s = accumulate(s, is_feasible, violation, objective, num_evaluations)
% Fold one evaluated batch into the run counters. FE indices are 1-based over the
% whole run, so they line up with the convergence curve.
    base = s.n_eval;
    s.n_eval = base + num_evaluations;

    v = double(violation(:));
    s.viol_sum = s.viol_sum + sum(v(isfinite(v)));

    feas = logical(is_feasible(:));
    s.n_feasible = s.n_feasible + sum(feas);

    obj = double(objective(:));
    [bo, kb] = min(obj);
    if ~isempty(bo) && ~isnan(bo) && (isnan(s.best_obj) || bo < s.best_obj)
        s.best_obj    = bo;
        s.fe_best_obj = base + kb;
    end

    fi = find(feas);
    if ~isempty(fi)
        if isnan(s.first_fe_feasible)
            s.first_fe_feasible = base + fi(1);
        end
        [bf, kf] = min(obj(fi));
        if ~isnan(bf) && (isnan(s.best_feas_obj) || bf < s.best_feas_obj)
            s.best_feas_obj    = bf;
            s.fe_best_feas_obj = base + fi(kf);
        end
    end
end

function [fitness, is_feasible, violation, objective, worstObjective, max_violation] = ...
        eval_core(solutions, problem, isCon, worstObjective, nIneq, nEq)
% Assumes the working directory is already correct if the routine still has data
% to read.
    ps = size(solutions, 2);

    if ~isCon
        fitness = feval(problem.fhd, solutions, problem.number);
        is_feasible = true(size(fitness));
        violation = zeros(size(fitness));
        objective = fitness;
        max_violation = zeros(size(fitness));
        return;
    end

    % A constrained suite returns [f, g, h] and expects solutions as row vectors.
    if isfield(problem, 'lb') && isfield(problem, 'ub') ...
            && ~isempty(problem.lb) && ~isempty(problem.ub)
        lb = problem.lb(:);   % D x 1
        ub = problem.ub(:);   % D x 1
        solutions = min(max(solutions, lb), ub);  % D x ps (implicit expansion)
    end
    [f, g, h] = feval(problem.fhd, solutions', problem.number);

    shape = size(f);      % preserve the suite's native orientation (column)
    f = f(:);

    EQ_TOL = 1e-4;        % delta fixed by the CEC2020RW guidelines, Eq. (2)
    G = pick_constraints(g, nIneq, ps);    % p x ps
    H = pick_constraints(h, nEq,   ps);    % q x ps
    m = size(G, 1) + size(H, 1);           % m = p + q, as coded by the suite

    % 12 of the 57 problems return a non-finite constraint for legal points -- the
    % RC34-RC43 power-flow solvers do it at the box centre, where a converging
    % population sits. Such a point is reported as maximally infeasible, because
    % max(0, NaN) is 0 and would read as "no violation".
    FAILED_EVAL = 1e12;
    failed = any(~isfinite(G), 1).' | any(~isfinite(H), 1).' | ~isfinite(f);

    gviol = sum(max(0, G), 1).';           % ps x 1
    Habs  = abs(H);
    Habs(Habs - EQ_TOL <= 0) = 0;          % within tolerance => no violation
    hviol = sum(Habs, 1).';

    if m > 0
        violation = (gviol + hviol) / m;
    else
        violation = zeros(ps, 1);
    end
    violation(failed) = FAILED_EVAL;
    is_feasible = (violation <= 0);

    % Reported next to the mean v(x) so a near-feasible solution can be told from
    % one that misses a single constraint badly.
    Vall = [max(0, G); Habs];              % m x ps
    if isempty(Vall)
        max_violation = zeros(ps, 1);
    else
        max_violation = max(Vall, [], 1).';
    end
    max_violation(failed) = FAILED_EVAL;

    % Scalarisation, Eq. (6). The reference is built from the genuinely finite
    % objectives only, so a failed evaluation cannot drag it up and change the
    % fitness every other infeasible point is scored against.
    finite_f = f(isfinite(f));
    if ~isempty(finite_f)
        worstObjective = max([worstObjective; finite_f]);
    end
    if isempty(worstObjective)
        ref = 0;
    else
        ref = worstObjective;
    end

    f(~isfinite(f)) = FAILED_EVAL;

    fitness = f;
    fitness(~is_feasible) = ref + violation(~is_feasible);

    fitness       = reshape(fitness, shape);
    objective     = reshape(f, shape);
    violation     = reshape(violation, shape);
    is_feasible   = reshape(is_feasible, shape);
    max_violation = reshape(max_violation, shape);
end

function C = pick_constraints(c, n, ps)
% Return the real constraints as nc x ps.
%
% Every suite here hands them over that way already -- cec20rw_func ends on
% `g=g'; h=h';` and cedp_func does the same -- so the nc x ps reading is
% tried first and the transpose is only a fallback for a block that does not
% match it. The order matters exactly when a batch holds as many candidates as
% the problem has constraints: both readings fit a square block, and reading it
% the wrong way hands every candidate another one's constraints. That case is
% not rare, it is where the LPSR families spend their last generations.
%
% n is the count from the par table and is not always right -- CEC2020RW's own
% Cal_par miscounts RC17, RC21 and RC44 -- so it is used to confirm a reading,
% never as the only test.
    if n == 0 || isempty(c)
        C = zeros(0, ps);
    elseif size(c, 2) == ps
        C = c;
    elseif size(c, 1) == ps
        C = c.';
    else
        error('calculate_fitness:constraintShape', ...
              'Constraint block is %dx%d but there are %d solutions.', ...
              size(c, 1), size(c, 2), ps);
    end
end

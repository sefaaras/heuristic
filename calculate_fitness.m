
%% --- Fitness Calculation ---

function [fitness, FE, is_feasible, violation, objective] = calculate_fitness(solutions, problem, FE)
    % Calculate fitness using problem structure and update FE counter
    % Input:
    %   solutions: matrix where each column is a solution vector
    %   problem: problem structure containing fhd and number
    %   FE: current function evaluation count
    % Output:
    %   fitness:     value the SEARCH minimises (for CEC2020RW this folds in the
    %                constraints, see below); row for the numeric CEC suites,
    %                column for CEC2020RW -- the original per-suite convention
    %   FE:          updated function evaluation count
    %   is_feasible: logical array, true when violation == 0
    %   violation:   mean constraint violation v(x), 0 for unconstrained suites
    %   objective:   raw objective f(x) before any constraint folding
    %
    % Call calculate_fitness('reset') once per run (main.m does this) to drop the
    % cached problem state -- otherwise a parfor worker carries one job's state
    % into the next.

    persistent lastFhd lastNum lastDim lastIsRW worstObjective nIneq nEq

    if ischar(solutions) || isstring(solutions)
        if strcmpi(solutions, 'reset')
            lastFhd = []; lastNum = []; lastDim = []; lastIsRW = [];
            worstObjective = []; nIneq = []; nEq = [];
            clear('cec20rw_func');   % drop its persistent G/B/P/Q/initial_flags
            fitness = []; FE = 0; is_feasible = []; violation = []; objective = [];
            return;
        end
        error('calculate_fitness:badCommand', 'Unknown command "%s".', char(solutions));
    end

    num_evaluations = size(solutions, 2);  % Number of solutions to evaluate
    D = size(solutions, 1);                % Problem dimension (rows)

    firstForThisProblem = isempty(lastNum) || problem.number ~= lastNum || ...
                          D ~= lastDim || ~isequal(problem.fhd, lastFhd);

    if firstForThisProblem
        % First evaluation for this (function, dimension): the CEC routine will
        % read input_data from the current folder, so cd there for this call.
        fname = func2str(problem.fhd);
        lastIsRW = contains(fname, 'cec20rw');

        originalDir = pwd;
        restoreDir = onCleanup(@() cd(originalDir));  % restore cwd on return/error

        if contains(fname, 'cec14')
            cd('problem/CEC2014');
        elseif contains(fname, 'cec17')
            cd('problem/CEC2017');
        elseif lastIsRW
            cd('problem/CEC2020RW');
        elseif contains(fname, 'cec20')
            cd('problem/CEC2020');
        elseif contains(fname, 'cec21')
            cd('problem/CEC2021');
        elseif contains(fname, 'cec22')
            cd('problem/CEC2022');
        end

        if lastIsRW
            clear('cec20rw_func');

            par = Cal_par(problem.number);
            nIneq = par.g;
            nEq   = par.h;
        end
        worstObjective = [];   % new problem => new reference

        [fitness, is_feasible, violation, objective, worstObjective] = ...
            eval_core(solutions, problem, lastIsRW, worstObjective, nIneq, nEq);

        lastFhd = problem.fhd; lastNum = problem.number; lastDim = D;
    else
        [fitness, is_feasible, violation, objective, worstObjective] = ...
            eval_core(solutions, problem, lastIsRW, worstObjective, nIneq, nEq);
    end

    FE = FE + num_evaluations;  % Update FE counter
end

function [fitness, is_feasible, violation, objective, worstObjective] = ...
        eval_core(solutions, problem, isRW, worstObjective, nIneq, nEq)
    % Evaluate fitness (and, for CEC2020RW, feasibility). Assumes the working
    % directory is already correct if the routine still needs to read data.
    ps = size(solutions, 2);

    if ~isRW
        % Other CEC functions return only fitness (always feasible)
        fitness = feval(problem.fhd, solutions, problem.number);
        is_feasible = true(size(fitness));
        violation = zeros(size(fitness));
        objective = fitness;
        return;
    end

    % CEC2020RW returns [f, g, h]. It expects solutions as row vectors.
    if isfield(problem, 'lb') && isfield(problem, 'ub') ...
            && ~isempty(problem.lb) && ~isempty(problem.ub)
        lb = problem.lb(:);   % D x 1
        ub = problem.ub(:);   % D x 1
        solutions = min(max(solutions, lb), ub);  % D x ps (implicit expansion)
    end
    [f, g, h] = feval(problem.fhd, solutions', problem.number);

    shape = size(f);      % preserve the suite's native orientation (column)
    f = f(:);

    % --- Mean constraint violation, Eq. (2) of CEC2020RW ------------

    EQ_TOL = 1e-4;        % delta fixed by the competition guidelines
    G = pick_constraints(g, nIneq, ps);    % p x ps
    H = pick_constraints(h, nEq,   ps);    % q x ps
    m = size(G, 1) + size(H, 1);           % m = p + q, as coded by the suite

    gviol = sum(max(0, G), 1).';           % ps x 1
    Habs  = abs(H);
    Habs(Habs - EQ_TOL <= 0) = 0;          % within tolerance => no violation
    hviol = sum(Habs, 1).';

    if m > 0
        violation = (gviol + hviol) / m;
    else
        violation = zeros(ps, 1);
    end
    is_feasible = (violation <= 0);

    % --- Fold the constraints into the scalar the search minimises -----------
    finite_f = f(isfinite(f));
    if ~isempty(finite_f)
        worstObjective = max([worstObjective; finite_f]);
    end
    if isempty(worstObjective)
        ref = 0;
    else
        ref = worstObjective;
    end

    fitness = f;
    fitness(~is_feasible) = ref + violation(~is_feasible);

    fitness     = reshape(fitness, shape);
    objective   = reshape(f, shape);
    violation   = reshape(violation, shape);
    is_feasible = reshape(is_feasible, shape);
end

function C = pick_constraints(c, n, ps)
    % Return the real constraints as nc x ps.
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


%% --- Fitness Calculation ---

function [fitness, FE, is_feasible] = calculate_fitness(solutions, problem, FE)
    % Calculate fitness using problem structure and update FE counter
    % Input:
    %   solutions: matrix where each column is a solution vector
    %   problem: problem structure containing fhd and number
    %   FE: current function evaluation count
    % Output:
    %   fitness: calculated fitness values
    %   FE: updated function evaluation count
    %   is_feasible: logical array indicating feasibility (for CEC2020RW)

    persistent lastFhd lastNum lastDim lastIsRW

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

        [fitness, is_feasible] = eval_core(solutions, problem, lastIsRW);

        lastFhd = problem.fhd; lastNum = problem.number; lastDim = D;
    else
        [fitness, is_feasible] = eval_core(solutions, problem, lastIsRW);
    end

    FE = FE + num_evaluations;  % Update FE counter
end

function [fitness, is_feasible] = eval_core(solutions, problem, isRW)
    % Evaluate fitness (and, for CEC2020RW, feasibility). Assumes the working
    % directory is already correct if the routine still needs to read data.
    if isRW
        % CEC2020RW returns [f, g, h] where g<=0 and h=0 for feasibility.
        % CEC2020RW expects solutions as row vectors (each row is a solution).
        if isfield(problem, 'lb') && isfield(problem, 'ub') ...
                && ~isempty(problem.lb) && ~isempty(problem.ub)
            lb = problem.lb(:);   % D x 1
            ub = problem.ub(:);   % D x 1
            solutions = min(max(solutions, lb), ub);  % D x ps (implicit expansion)
        end
        [fitness, g, h] = feval(problem.fhd, solutions', problem.number);

        tolerance = 1e-6;
        if isempty(g)
            g_feasible = true(size(fitness));
        elseif size(g, 2) == size(fitness, 1)   % g is ng x ps
            g_feasible = all(g <= 0, 1)';
        else                                    % g is ps x ng
            g_feasible = all(g <= 0, 2);
        end

        if isempty(h)
            h_feasible = true(size(fitness));
        elseif size(h, 2) == size(fitness, 1)   % h is nh x ps
            h_feasible = all(abs(h) <= tolerance, 1)';
        else                                    % h is ps x nh
            h_feasible = all(abs(h) <= tolerance, 2);
        end

        is_feasible = g_feasible & h_feasible;
    else
        % Other CEC functions return only fitness (always feasible)
        fitness = feval(problem.fhd, solutions, problem.number);
        is_feasible = true(size(fitness));  % All solutions are feasible
    end
end

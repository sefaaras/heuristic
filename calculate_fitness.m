
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
    
    num_evaluations = size(solutions, 2);  % Number of solutions to evaluate
    
    % Handle different CEC functions - need to be in correct directory for input data
    originalDir = pwd;
    try
        % Change to appropriate CEC directory based on function type
        if contains(func2str(problem.fhd), 'cec14')
            cd('problem/CEC2014');
        elseif contains(func2str(problem.fhd), 'cec17')
            cd('problem/CEC2017');
        elseif contains(func2str(problem.fhd), 'cec20rw')
            cd('problem/CEC2020RW');
        elseif contains(func2str(problem.fhd), 'cec20')
            cd('problem/CEC2020');
        elseif contains(func2str(problem.fhd), 'cec21')
            cd('problem/CEC2021');
        elseif contains(func2str(problem.fhd), 'cec22')
            cd('problem/CEC2022');
        end
        
        % Calculate fitness and constraints
        if contains(func2str(problem.fhd), 'cec20rw')
            % CEC2020RW returns [f, g, h] where g<=0 and h=0 for feasibility
            % CEC2020RW expects solutions as row vectors (each row is a solution)
            [fitness, g, h] = feval(problem.fhd, solutions', problem.number);
            
            % Check feasibility for CEC2020RW problems
            tolerance = 1e-6;  % Tolerance for equality constraints
            
            if isempty(g)
                g_feasible = true(size(fitness));
            else
                % g can be ng×ps (constraints x solutions)
                % Need to check if all constraints are satisfied for each solution
                if size(g, 2) == size(fitness, 1)  % g is ng×ps
                    g_feasible = all(g <= 0, 1)';  % Check across constraints, transpose to match fitness
                else  % g is ps×ng  
                    g_feasible = all(g <= 0, 2);  % Check across constraints for each solution
                end
            end
            
            if isempty(h)
                h_feasible = true(size(fitness));
            else
                % h can be nh×ps (constraints x solutions)  
                % Need to check if all constraints are satisfied for each solution
                if size(h, 2) == size(fitness, 1)  % h is nh×ps
                    h_feasible = all(abs(h) <= tolerance, 1)';  % Check across constraints, transpose to match fitness
                else  % h is ps×nh
                    h_feasible = all(abs(h) <= tolerance, 2);  % Check across constraints for each solution
                end
            end
            
            is_feasible = g_feasible & h_feasible;
        else
            % Other CEC functions return only fitness (always feasible)
            fitness = feval(problem.fhd, solutions, problem.number);
            is_feasible = true(size(fitness));  % All solutions are feasible
        end
        
    catch ME
        % Return to original directory on error
        cd(originalDir);
        rethrow(ME);
    end
    
    % Return to original directory
    cd(originalDir);
    
    FE = FE + num_evaluations;  % Update FE counter
end

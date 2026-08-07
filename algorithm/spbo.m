% ----------------------------------------------------------------------- %
% Student Psychology Based Optimization (SPBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   student = 20          % Population size (number of students)
%
% Algorithm Concept:
%   - Models the psychology of students trying to improve their performance
%     in a class to become the best student
%   - Best student, good students, average students and students who improve
%     randomly follow different update rules (Eqs. 1-4)
%   - Updates are performed one subject (dimension) at a time
%
% Reference:
% Bikash Das, Vaskar Mukherjee, Debapriya Das,
% Student psychology based optimization algorithm: A new population based
% optimization algorithm for solving optimization problems,
% Advances in Engineering Software 146 (2020) 102804
% https://doi.org/10.1016/j.advengsoft.2020.102804
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = spbo(problem)

    % Extract problem parameters
    dim = problem.dimension;
    mini = problem.lb;
    maxi = problem.ub;
    maxFE = problem.maxFe;

    student = 20;
    variable = dim;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the set of random solutions
    X = initialization(student, variable, maxi, mini);
    sol = X;

    % Evaluate the initial set
    [Objective_values, FE] = calculate_fitness(X', problem, FE);
    Objective_values = Objective_values(:);
    fitness = Objective_values;

    Best_fitness = min(fitness);
    idxb = find(fitness == Best_fitness, 1, 'last');
    Best_student = sol(idxb, :);

    for eval_count = 1:student
        curve(eval_count) = Best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, sol, fitness, population_history, fitness_history, ...
            history_index, maxFE);
    end

    while FE < maxFE

        for do = 1:variable

            % Column means of the current solution set
            meanv = mean(sol, 1);   % 1 x variable

            par = sol;
            par1 = sol;

            check = rand(student, 1);
            mid = rand(student, 1);

            for dw = 1:student
                if Best_fitness == fitness(dw, 1)
                    % Best Student (Eq. 1)
                    jg = fitness(randperm(numel(fitness), 1));
                    lk = find(fitness == jg, 1, 'last');
                    par1(dw, do) = par(dw, do) + (((-1)^(round(1 + rand))) * rand * (par(dw, do) - par(lk, do)));
                else
                    if check(dw, 1) < mid(dw, 1)
                        % Good Student (Eqs. 2a / 2b)
                        rta = rand;
                        if rta > rand
                            par1(dw, do) = Best_student(1, do) + (rand * (Best_student(1, do) - par(dw, do)));
                        else
                            par1(dw, do) = par(dw, do) + (rand * (Best_student(1, do) - par(dw, do))) + ((rand * (par(dw, do) - meanv(1, do))));
                        end
                    else
                        an = rand;
                        if rand > an
                            % Average Student (Eq. 3)
                            par1(dw, do) = par(dw, do) + (rand * (meanv(1, do) - par(dw, do)));
                        else
                            % Student who improves randomly (Eq. 4)
                            par1(dw, do) = mini(do) + (rand * (maxi(do) - mini(do)));
                        end
                    end
                end
            end

            % Boundary checking on the updated subject (column do)
            par1(:, do) = min(max(par1(:, do), mini(do)), maxi(do));

            X = par1;

            % Evaluate the whole class
            FE_before = FE;
            [fun1, FE] = calculate_fitness(X', problem, FE);
            fun1 = fun1(:);

            % Update the solution if there is a better one
            for vt = 1:student
                if fitness(vt, 1) > fun1(vt, 1)
                    fitness(vt, 1) = fun1(vt, 1);
                    sol(vt, :) = par1(vt, :);
                end
            end

            % Update the best student
            [Best_fitness1, fo] = min(fitness);
            Best_student1 = sol(fo, :);
            if Best_fitness > Best_fitness1
                Best_fitness = Best_fitness1;
                Best_student = Best_student1;
            end

            % Record convergence curve and history
            for eval_count = (FE_before + 1):FE
                if eval_count <= maxFE
                    curve(eval_count) = Best_fitness;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, sol, fitness, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end

            if FE >= maxFE
                break;
            end
        end
    end

    best_solution = Best_student;
    best_fitness = Best_fitness;

end

% Initialization Function
function X = initialization(student, variable, maxi, mini)
    Boundary_no = size(maxi, 2);
    if Boundary_no == 1
        X = mini + rand(student, variable) .* (maxi - mini);
    end
    if Boundary_no > 1
        for i = 1:variable
            maxi_i = maxi(i);
            mini_i = mini(i);
            X(:, i) = mini_i + rand(student, 1) .* (maxi_i - mini_i);
        end
    end
end

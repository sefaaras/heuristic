% ----------------------------------------------------------------------- %
% Moth Search (MS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50           % Population size
%   max_step_size = 1.0    % Maximum step size for Levy flights
%
% Algorithm Concept:
%   - Best moth is viewed as the light source (phototaxis)
%   - Moths close to best: Levy flights for local search (exploitation)
%   - Moths far from best: Direct movement towards best (exploration)
%
% Reference:
% Gai-Ge Wang,
% Moth search algorithm: a bio-inspired metaheuristic algorithm
% for global optimization problems,
% Memetic Computing 10 (2018) 151-164
% https://doi.org/10.1007/s12293-016-0212-3
% ----------------------------------------------------------------------- %
% Implementation Note:
% The greedy selection reverts the position as well as the fitness. Keeping the
% old fitness while leaving the moth at its new, worse position made fitness stop
% describing Moths for every moth that did not improve, and that pair is what the
% next iteration sorts, records and measures distances against.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ms(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    % Algorithm parameters
    popsize = 50;                 % Total population size
    max_step_size = 1.0;          % Maximum step size for Levy flights
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize moth positions
    Moths = initialization(popsize, dim, ub, lb);
    
    % Evaluate initial population
    [fitness, FE] = calculate_fitness(Moths', problem, FE);
    
    % Find the best moth (light source)
    [best_fitness, best_idx] = min(fitness);
    best_position = Moths(best_idx, :);
    
    % Record initial population
    for eval_count = 1:popsize
        curve(eval_count) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Moths, fitness, population_history, fitness_history, ...
            history_index, maxFE);
    end
    
    % Main loop
    iteration = 1;
    Max_iter = ceil((maxFE - popsize) / popsize);
    
    while FE < maxFE && iteration <= Max_iter
        
        % Calculate distances from all moths to the best moth (light source)
        distances = zeros(popsize, 1);
        for i = 1:popsize
            distances(i) = norm(Moths(i, :) - best_position);
        end
        
        % Calculate mean distance to determine threshold
        mean_distance = mean(distances);
        
        old_Moths = Moths;

        % Update each moth's position based on distance to best
        for i = 1:popsize
            
            if distances(i) <= mean_distance
                % Close to the light source: Levy flights, scale shrinking with the iteration
                scale = max_step_size / (iteration^2);
                levy_step = LevyWalk(dim);
                Moths(i, :) = Moths(i, :) + scale * levy_step;
                
            else
                % Far from the light source: phototaxis, a big step towards the best moth
                step_size = rand();  % Random step size
                Moths(i, :) = Moths(i, :) + step_size * (best_position - Moths(i, :));
            end
            
            % Apply boundary constraints
            Moths(i, :) = bound(Moths(i, :), ub, lb);
        end
        
        % Evaluate new population
        [new_fitness, FE] = calculate_fitness(Moths', problem, FE);
        
        % Update fitness and best solution
        for i = 1:popsize
            % Greedy selection: keep better solution
            if new_fitness(i) < fitness(i)
                fitness(i) = new_fitness(i);

                % Update global best if improved
                if fitness(i) < best_fitness
                    best_fitness = fitness(i);
                    best_position = Moths(i, :);
                end
            else
                Moths(i, :) = old_Moths(i, :);
            end
        end
        
        % Record convergence curve and history
        for eval_idx = 1:popsize
            eval_count = FE - popsize + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Moths, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        iteration = iteration + 1;
    end
    
    % Return best solution
    best_solution = best_position;
    
end

% Helper Functions

function Positions = initialization(popsize, dim, ub, lb)
    Boundary_no = size(ub, 2);
    
    if Boundary_no == 1
        Positions = rand(popsize, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(popsize, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

function deltaX = LevyWalk(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * (beta - 1) / 2) / ...
             (gamma((beta) / 2) * (beta - 1) * 2^((beta - 2) / 2)))^(1 / (beta - 1));
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / (beta - 1));
    deltaX = 0.01 * step;
end


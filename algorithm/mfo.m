% ----------------------------------------------------------------------- %
% Moth-Flame Optimization (MFO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                  % Population size (number of moths)
%   b = 1                   % Shape constant for logarithmic spiral
%
% Algorithm Concept:
%   - Inspired by moth navigation using moon (transverse orientation)
%   - Moths spiral around flames (best solutions found so far)
%   - Number of flames decreases over iterations
%   - Logarithmic spiral flight path
%
% Reference:
% S. Mirjalili,
% Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm,
% Knowledge-Based Systems, Volume 89, 2015, Pages 228-249
% https://doi.org/10.1016/j.knosys.2015.07.006
%
% Implementation Note:
%   curve and best_fitness track the best point the run has evaluated. The flame
%   set is built from the previous generation, as the reference has it, so it
%   lags the moths by one generation and never sees the final one; reporting it
%   directly would leave the last generation's evaluations out of the result.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mfo(problem)
    
    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    N = 30;                       % Population size (number of moths)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize the positions of moths
    Moth_pos = initialization(N, dim, ub, lb);
    
    % Evaluate initial population
    [Moth_fitness, FE] = calculate_fitness(Moth_pos', problem, FE);
    
    % Sort the first population of moths
    [fitness_sorted, I] = sort(Moth_fitness);
    sorted_population = Moth_pos(I, :);
    
    % Initialize flames (best solutions)
    best_flames = sorted_population;
    best_flame_fitness = fitness_sorted;
    
    % Best solution so far
    Best_flame_score = fitness_sorted(1);
    Best_flame_pos = sorted_population(1, :);

    % Best point evaluated so far; the flames lag it by one generation by design
    bsf_fit = Best_flame_score;
    bsf_x = Best_flame_pos;

    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:N
        curve(eval_count) = bsf_fit;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Moth_pos, Moth_fitness, population_history, fitness_history, ...
            history_index, maxFE);
    end
    
    % Main loop
    Max_iteration = ceil((maxFE - N) / N);
    Iteration = 1;
    
    while FE < maxFE && Iteration <= Max_iteration
        
        % Number of flames Eq. (3.14) in the paper
        Flame_no = round(N - Iteration * ((N - 1) / Max_iteration));
        
        % Store previous population
        previous_population = Moth_pos;
        previous_fitness = Moth_fitness;
        
        % a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + Iteration * ((-1) / Max_iteration);
        
        % Update moth positions
        for i = 1:N
            for j = 1:dim
                if i <= Flame_no
                    % Update the position of the moth with respect to its corresponding flame
                    distance_to_flame = abs(sorted_population(i, j) - Moth_pos(i, j));
                    b = 1;
                    t = (a - 1) * rand + 1;
                    
                    % Eq. (3.12) - Logarithmic spiral
                    Moth_pos(i, j) = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + sorted_population(i, j);
                else
                    % Update the position of the moth with respect to one flame (best flame)
                    distance_to_flame = abs(sorted_population(Flame_no, j) - Moth_pos(i, j));
                    b = 1;
                    t = (a - 1) * rand + 1;
                    
                    % Eq. (3.12)
                    Moth_pos(i, j) = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + sorted_population(Flame_no, j);
                end
            end
            
            % Apply boundary constraints
            Moth_pos(i, :) = bound(Moth_pos(i, :), ub, lb);
        end
        
        % Evaluate new moth positions
        [Moth_fitness, FE] = calculate_fitness(Moth_pos', problem, FE);
        [m_best, m_idx] = min(Moth_fitness);
        if m_best < bsf_fit
            bsf_fit = m_best;
            bsf_x = Moth_pos(m_idx, :);
        end

        % Combine previous population with current flames
        double_population = [previous_population; best_flames];
        double_fitness = [previous_fitness(:)', best_flame_fitness(:)'];
        
        % Sort the combined population
        [double_fitness_sorted, I] = sort(double_fitness);
        double_sorted_population = double_population(I, :);
        
        % Select top N solutions as new flames
        fitness_sorted = double_fitness_sorted(1:N);
        sorted_population = double_sorted_population(1:N, :);
        
        % Update the flames
        best_flames = sorted_population;
        best_flame_fitness = fitness_sorted;
        
        % Update the position of best flame obtained so far
        Best_flame_score = fitness_sorted(1);
        Best_flame_pos = sorted_population(1, :);
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxFE
                curve(eval_count) = bsf_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, Moth_pos, Moth_fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        
        Iteration = Iteration + 1;
    end
    
    % Return best solution
    best_fitness = bsf_fit;
    best_solution = bsf_x;
    
end

% Initialization Function
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);  % Number of boundaries
    
    % If the boundaries of all variables are equal
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    
    % If each variable has a different lb and ub
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

% Boundary Handling
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end


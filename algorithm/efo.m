% ----------------------------------------------------------------------- %
% Electromagnetic Field Optimization (EFO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N_emp = 50       % Population size (number of electromagnets)
%   R_rate = 0.3     % Randomization rate
%   Ps_rate = 0.2    % Probability of selecting from positive field
%   P_field = 0.1    % Positive field ratio (top 10%)
%   N_field = 0.45   % Negative field ratio (bottom 45%)
%   phi = golden ratio = (1 + sqrt(5))/2
%   
% Algorithm Concept:
%   - Population divided into positive, neutral, and negative fields
%   - New particles generated using electromagnetic force equations
%   - Golden ratio guides the search process
%   - Sorted population maintained throughout
%
% Reference:
% H. Abedinpourshotorban, S.M. Shamsuddin, Z. Beheshti, D.N.A. Jawawi,
% Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm,
% Swarm and Evolutionary Computation, 2016, Vol. 26, pp. 8-22.
% https://doi.org/10.1016/j.swevo.2015.07.002
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds  
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = efo(problem)
    
    % Extract problem parameters
    N_var = problem.dimension;     % Problem dimension
    minval = problem.lb;          % Lower bounds
    maxval = problem.ub;          % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    N_emp = 50;                   % Population size
    R_rate = 0.3;                 % Randomization rate
    Ps_rate = 0.2;                % Probability of selecting from positive field
    P_field = 0.1;                % Positive field ratio (top 10%)
    N_field = 0.45;               % Negative field ratio (bottom 45%)
    phi = (1 + sqrt(5))/2;        % Golden ratio
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N_emp, N_var);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N_emp);            % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize population
    em_pop = zeros(N_emp, N_var);
    for i = 1:N_emp
        em_pop(i, :) = rand(1, N_var) .* (maxval - minval) + minval;
    end
    
    % Evaluate initial population
    [fit, FE] = calculate_fitness(em_pop', problem, FE);
    em_pop = [em_pop, fit'];  % Attach fitness as last column (transpose to column vector)
    em_pop = sortpop(em_pop, N_var + 1);  % Sort by fitness
    
    % Record initial population
    for eval_count = 1:N_emp
        curve(eval_count) = em_pop(1, N_var + 1);  % Best fitness
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, em_pop(:, 1:N_var), em_pop(:, N_var + 1), ...
            population_history, fitness_history, history_index, sampling_interval, history_size);
    end
    
    % Pre-calculate random vectors to increase speed
    Max_gen = maxFE;
    r_index1 = randi([1 (round(N_emp * P_field))], [N_var Max_gen]);  % From positive field
    r_index2 = randi([(round(N_emp * (1 - N_field))) N_emp], [N_var Max_gen]);  % From negative field
    r_index3 = randi([(round(N_emp * P_field) + 1) (round(N_emp * (1 - N_field)) - 1)], [N_var Max_gen]);  % From neutral field
    ps = rand(N_var, Max_gen);  % Probability of selecting from positive field
    r_force = rand(1, Max_gen);  % Random force
    rp = rand(1, Max_gen);  % Random numbers for randomization probability
    randomization = rand(1, Max_gen);  % Randomization coefficients
    
    RI = 1;  % Index for random electromagnet replacement
    new_emp = zeros(1, N_var + 1);  % Temporary array for new particle
    
    % Main optimization loop
    while FE < maxFE
        
        r = r_force(1, FE);
        
        % Generate new electromagnet
        for i = 1:N_var
            if (ps(i, FE) > Ps_rate)
                % Use electromagnetic force equation
                new_emp(i) = em_pop(r_index3(i, FE), i) + ...
                             phi * r * (em_pop(r_index1(i, FE), i) - em_pop(r_index3(i, FE), i)) + ...
                             r * (em_pop(r_index3(i, FE), i) - em_pop(r_index2(i, FE), i));
            else
                % Select from positive field
                new_emp(i) = em_pop(r_index1(i, FE), i);
            end
            
            % Boundary checking
            if (new_emp(i) >= maxval(i) || new_emp(i) <= minval(i))
                new_emp(i) = minval(i) + (maxval(i) - minval(i)) * randomization(1, FE);
            end
        end
        
        % Replacement of one electromagnet with random number (for diversity)
        if (rp(1, FE) < R_rate)
            new_emp(RI) = minval(RI) + (maxval(RI) - minval(RI)) * randomization(1, FE);
            RI = RI + 1;
            
            if (RI > N_var)
                RI = 1;
            end
        end
        
        % Evaluate new electromagnet
        [new_emp(N_var + 1), FE] = calculate_fitness(new_emp(1:N_var)', problem, FE);
        
        % Update population if new particle is better than worst
        if (new_emp(N_var + 1) < em_pop(N_emp, N_var + 1))
            position = find(em_pop(:, N_var + 1) > new_emp(N_var + 1), 1);
            em_pop = insert_in_pop(em_pop, new_emp, position);
        end
        
        % Record convergence curve and history
        if FE <= maxFE
            curve(FE) = em_pop(1, N_var + 1);  % Best fitness
            [population_history, fitness_history, history_index] = record_history(...
                FE, em_pop(:, 1:N_var), em_pop(:, N_var + 1), ...
                population_history, fitness_history, history_index, sampling_interval, history_size);
        end
    end
    
    % Return best solution
    best_fitness = em_pop(1, N_var + 1);
    best_solution = em_pop(1, 1:N_var);
    
end

%% --- Helper Functions ---

% Insert new particle in sorted position
function [newpop] = insert_in_pop(cpopulation, nparticle, position)
    newpop = [cpopulation(1:position-1, :); nparticle; cpopulation(position:end-1, :)];
end

% Sort population by fitness column
function [sorted] = sortpop(unsorted, column)
    [~, I] = sort(unsorted(:, column));
    sorted = unsorted(I, :);
end


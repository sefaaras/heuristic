% ----------------------------------------------------------------------- %
% Weighted Differential Evolution (WDE) Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Population size (uses 2*N internally)
%
% Algorithm Concept:
%   - Based on Differential Evolution with weighted recombination
%   - Uses cubic random weights for population combination
%   - Dual population strategy (2N individuals)
%   - Adaptive scaling factor F with cubic random distribution
%   - Crossover mask generation with variable dimensions
%
% Reference:
% Civicioglu, P., Besdok, E., Gunen, M.A. and Atasever, U.H. (2020),
% Weighted differential evolution algorithm for numerical function 
% optimization: a comparative study with cuckoo search, artificial bee 
% colony, adaptive differential evolution, and backtracking search 
% optimization algorithms,
% Neural Computing and Applications, 32, 3923-3937.
% https://doi.org/10.1007/s00521-018-3822-5
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = wde(problem)
    
    % Extract problem parameters
    D = problem.dimension;        % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % WDE Parameters
    N = 50;                       % Base population size (uses 2*N internally)
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, 2*N, D);
    fitness_history = zeros(history_size, 2*N);
    history_index = 1;
    
    % Ensure bounds are vectors
    if numel(lb) == 1
        lb = lb * ones(1, D);
        ub = ub * ones(1, D);
    end
    
    % INITIALIZATION - Generate population of size 2*N (see Eq.1 in reference)
    P = GenP(2*N, D, lb, ub);
    
    % Evaluate initial population
    [fitP, FE] = calculate_fitness(P', problem, FE);
    
    % Find initial best
    [best_fitness_current, best_idx] = min(fitP);
    best_solution_current = P(best_idx, :);
    
    % Record best fitness for each initial evaluation
    for eval_count = 1:min(2*N, maxFE)
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, P, fitP, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Calculate maximum epochs
    MaxEpk = ceil((maxFE - 2*N) / N);
    
    % Main loop
    epk = 1;
    while FE < maxFE && epk <= MaxEpk
        
        % Random permutation for selection
        j = randperm(2*N);
        k = j(1:N);
        l = j(N+1:2*N);
        trialP = P(k, :);
        fitTrialP = fitP(k);
        
        % Weighted combination (memory)
        temp = trialP;
        for index = 1:N
            w = rand(N, 1).^3;
            w = w ./ sum(w);
            tempP = P(l, :);
            res = zeros(N, D);
            for i = 1:N
                res(i, :) = w(i) * tempP(i, :);
            end
            temp(index, :) = sum(res);
        end
        
        % Generate mutation indices
        m = randperm(N);
        while sum((1:N) == m) > 0
            m = randperm(N);
        end
        
        % Difference vector
        E = temp - trialP(m, :);
        
        % Recombination mask
        M = GenM(N, D);
        
        % Scaling factor F (cubic random distribution)
        if rand < rand
            F = randn(1, D).^3;
        else
            F = randn(N, 1).^3;
        end
        
        % Generate trial vectors
        Trial = zeros(N, D);
        if numel(F) == N
            for i = 1:N
                Trial(i, :) = trialP(i, :) + F(i) .* M(i, :) .* E(i, :);
            end
        else
            for i = 1:D
                Trial(:, i) = trialP(:, i) + F(i) .* M(:, i) .* E(:, i);
            end
        end
        
        % Boundary control
        Trial = BoundaryControl(Trial, lb, ub);
        
        % Evaluate trial vectors
        [fitT, FE] = calculate_fitness(Trial', problem, FE);
        
        % Selection - keep better solutions
        ind = fitT < fitTrialP;
        trialP(ind, :) = Trial(ind, :);
        fitTrialP(ind) = fitT(ind);
        
        % Update population
        fitP(k) = fitTrialP;
        P(k, :) = trialP;
        
        % Update best solution
        [min_fit, min_idx] = min(fitP);
        if min_fit < best_fitness_current
            best_fitness_current = min_fit;
            best_solution_current = P(min_idx, :);
        end
        
        % Record convergence curve for each evaluation
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, P, fitP, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        epk = epk + 1;
    end
    
    % Fill remaining curve values
    for i = FE+1:maxFE
        curve(i) = best_fitness_current;
    end
    
    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;
    
end

%% --- Crossover Mask Generation ---
function M = GenM(N, D)
    M = zeros(N, D);
    for i = 1:N
        if rand < rand
            k = rand^3;
        else
            k = 1 - rand^3;
        end
        V = randperm(D);
        j = V(1:ceil(k*D));
        M(i, j) = 1;
    end
end

%% --- Population Initialization ---
function pop = GenP(N, D, low, up)
    pop = zeros(N, D);
    for i = 1:N
        for j = 1:D
            pop(i, j) = rand * (up(j) - low(j)) + low(j);
        end
    end
end

%% --- Boundary Control ---
function pop = BoundaryControl(pop, low, up)
    [popsize, dim] = size(pop);
    for i = 1:popsize
        for j = 1:dim
            F = rand^3;
            if pop(i, j) < low(j)
                pop(i, j) = low(j) + F * (up(j) - low(j));
            end
            if pop(i, j) > up(j)
                pop(i, j) = up(j) + F * (low(j) - up(j));
            end
        end
    end
end


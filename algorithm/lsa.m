% ----------------------------------------------------------------------- %
% Lightning Search Algorithm (LSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50  % Population size (number of projectiles)
%   max_ch_time = 10  % Maximum channel time before elimination
%   
% Algorithm Concept:
%   - Lightning propagates through the sky
%   - Channels are created and eliminated based on energy
%   - Direction updated based on fitness improvement
%   - Focking procedure for exploration
%
% Reference:
% H. Shareef, A.A. Ibrahim, A.H. Mutlag,
% Lightning search algorithm,
% Applied Soft Computing 36 (2015) 315-333
% http://dx.doi.org/10.1016/j.asoc.2015.07.028
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = lsa(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    N = 50;                       % Population size/number of agents
    D = dim;                      % Number of dimensions
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);  % Store population at sampled FEs
    fitness_history = zeros(history_size, N);          % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize population
    Dpoint = zeros(N, D);
    for d = 1:D
        Dpoint(:, d) = rand(N, 1) * (ub(d) - lb(d)) + lb(d);
    end
    
    ch_time = 0;              % Reset channel time
    max_ch_time = 10;         % Maximum channel time
    direct = sign(unifrnd(-1, 1, 1, dim));  % Random direction
    
    % Evaluate initial population
    [Ec, FE] = calculate_fitness(Dpoint', problem, FE);
    
    % Record best fitness for each initial evaluation
    [best_val, ~] = min(Ec);
    for eval_count = 1:N
        curve(eval_count) = best_val;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Dpoint, Ec, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main optimization loop
    while FE < maxFE
        
        % Update channel - eliminate worst if time exceeded
        ch_time = ch_time + 1;
        if ch_time >= max_ch_time
            [~, ds] = sort(Ec, 'ascend');
            Dpoint(ds(N), :) = Dpoint(ds(1), :);  % Eliminate the worst channel
            Ec(ds(N)) = Ec(ds(1));                % Update
            ch_time = 0;                           % Reset
        end
        
        % Ranking the fitness value
        [~, ds] = sort(Ec, 'ascend');
        best = Ec(ds(1));
        
        % Calculate iteration progress for energy update
        current_iter = FE / N;
        total_iter = maxFE / N;
        Energy = 2.05 - 2 * exp(-5 * (total_iter - current_iter) / total_iter);
        
        % Update direction
        for d = 1:D
            if FE >= maxFE, break; end
            
            Dpoint_test = Dpoint(ds(1), :);
            Dpoint_test(d) = Dpoint_test(d) + direct(d) * 0.005 * (ub(d) - lb(d));
            Dpoint_test = bound(Dpoint_test, ub, lb);
            
            [fv_test, FE] = calculate_fitness(Dpoint_test', problem, FE);
            
            if fv_test < best  % If better, keep positive direction
                direct(d) = direct(d);
            else
                direct(d) = -1 * direct(d);  % Otherwise, reverse direction
            end
            
            % Update best if improved
            if fv_test < best
                best = fv_test;
                Ec(ds(1)) = fv_test;
                Dpoint(ds(1), :) = Dpoint_test;
            end
            
            % Record in curve
            if FE <= maxFE
                curve(FE) = min(Ec);
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Dpoint, Ec, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        % Update position for each agent
        for i = 1:N
            if FE >= maxFE, break; end
            
            dist = Dpoint(i, :) - Dpoint(ds(1), :);
            Dpoint_temp = zeros(1, D);
            
            for d = 1:D
                if isequal(Dpoint(i, :), Dpoint(ds(1), :))
                    Dpoint_temp(d) = Dpoint(i, d) + direct(d) * abs(normrnd(0, Energy));
                else
                    if dist(d) < 0
                        Dpoint_temp(d) = Dpoint(i, d) + exprnd(abs(dist(d)));
                    else
                        Dpoint_temp(d) = Dpoint(i, d) - exprnd(dist(d));
                    end
                end
                
                % Re-initialize if out of bounds
                if (Dpoint_temp(d) > ub(d)) || (Dpoint_temp(d) < lb(d))
                    Dpoint_temp(d) = rand(1) * (ub(d) - lb(d)) + lb(d);
                end
            end
            
            [fv, FE] = calculate_fitness(Dpoint_temp', problem, FE);
            
            if fv < Ec(i)
                Dpoint(i, :) = Dpoint_temp;
                Ec(i) = fv;
                
                % Focking procedure (exploration)
                if rand < 0.01
                    Dpoint_fock = zeros(1, D);
                    for d = 1:D
                        Dpoint_fock(d) = ub(d) + lb(d) - Dpoint_temp(d);  % Focking
                    end
                    
                    [fock_fit, FE] = calculate_fitness(Dpoint_fock', problem, FE);
                    
                    if fock_fit < Ec(i)
                        Dpoint(i, :) = Dpoint_fock;  % Replace the channel
                        Ec(i) = fock_fit;
                    end
                    
                    % Record after focking evaluation
                    if FE <= maxFE
                        curve(FE) = min(Ec);
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, Dpoint, Ec, population_history, fitness_history, ...
                            history_index, sampling_interval, history_size);
                    end
                    if FE >= maxFE, break; end
                end
            end
            
            % Record after each agent update
            if FE <= maxFE
                curve(FE) = min(Ec);
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Dpoint, Ec, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
    end
    
    % Select the optimal value
    [best_fitness, best_idx] = min(Ec);
    best_solution = Dpoint(best_idx, :);
    
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end


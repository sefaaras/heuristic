% ----------------------------------------------------------------------- %
% Memetic Frog Leaping Algorithm (MFLA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   populationSize = 50    % Total population size
%   m = 5                  % Number of memeplexes
%   beta = 1.2             % Lévy flight parameter
%
% Algorithm Concept:
%   - Based on Shuffled Frog Leaping Algorithm (SFLA)
%   - Uses memeplex structure (dividing population into groups)
%   - Local search with Lévy flights
%   - Mass-based gravitational attraction
%   - Global shuffling phase with memetic operators
%
% Reference:
% Tang, D., Liu, Z., Yang, J., & Zhao, J. (2019). 
% Memetic frog leaping algorithm for global optimization. 
% Soft Computing, 23(21), 11077-11105.
% https://doi.org/10.1007/s00500-018-3662-3
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = mfla(problem)
    
    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    para = 1.2;                   % Lévy flight beta parameter
    parap2 = 5;                   % Number of memeplexes
    m = parap2;
    populationSize = 50;          % Total population size
    n = round(populationSize / m); % Number of frogs per memeplex
    le = n;                       % Number of local evolution steps
    q = m * n;                    % Actual population size
    beta = para;
    M = dim;
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, q, dim);
    fitness_history = zeros(history_size, q);
    history_index = 1;
    
    % Initialize frog population
    frog = struct('fitness', {}, 'center', {});
    Positions = zeros(q, M);
    
    for i = 1:q
        data = lb + (ub - lb) .* rand(1, M);
        Positions(i, :) = data;
        frog(i).center = data;
    end
    
    % Evaluate initial population
    [fitness_vals, FE] = calculate_fitness(Positions', problem, FE);
    
    for i = 1:q
        frog(i).fitness = fitness_vals(i);
    end
    
    % Find best frog
    [~, best_idx] = min(fitness_vals);
    tempFrog = frog(best_idx);
    
    % Record initial population
    for eval_count = 1:min(q, maxFE)
        curve(eval_count) = tempFrog.fitness;
        current_positions = reshape([frog.center], M, q)';
        current_fitness = [frog.fitness];
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, current_positions, current_fitness, ...
            population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    while FE < maxFE
        
        % Sort frogs by fitness (descending order - best first)
        for i = 1:q-1
            for j = 1:q-i
                if frog(j).fitness > frog(j+1).fitness
                    temp = frog(j+1);
                    frog(j+1) = frog(j);
                    frog(j) = temp;
                end
            end
        end
        
        % Partition into memeplexes and perform local search
        for k = 1:le
            for i = 1:m
                % Find worst and best in this memeplex
                Xw = frog(i);
                XwNo = i;
                locXwNO = 1;
                Xb = frog(i);
                XbNo = i;
                Xwb = frog(i);
                
                locMean = zeros(n, M);
                fitMeans = zeros(1, n);
                
                % Identify memeplex members
                for tt = 1:n
                    idx = i + m * (tt - 1);
                    if frog(idx).fitness < Xb.fitness
                        Xb = frog(idx);
                        XbNo = idx;
                    end
                    if frog(idx).fitness > Xw.fitness
                        Xw = frog(idx);
                        XwNo = idx;
                        locXwNO = tt;
                    end
                    fitMeans(tt) = frog(idx).fitness;
                    locMean(tt, :) = frog(idx).center;
                end
                
                % Calculate weighted center
                w11 = rand;
                v11 = rand;
                Xwb.center = (w11 / (w11 + v11)) .* Xb.center + (v11 / (w11 + v11)) .* Xw.center;
                
                % Calculate distances and masses
                dis = zeros(1, n);
                for tt2 = 1:n
                    dis(tt2) = norm(frog(i + m * (tt2 - 1)).center - frog(XbNo).center, 2)^2 + eps;
                end
                
                temploc = locXwNO;
                
                % Mass calculation
                [Mass] = massCalculation(fitMeans, 1);
                
                % Remove worst frog from calculation
                tempDis = dis;
                tempDis(temploc) = [];
                
                tempMass = Mass;
                tempMass(temploc) = [];
                
                % Gravitational attraction
                Gm = tempMass ./ tempDis;
                totalGm = sum(Gm);
                Gm = Gm ./ totalGm;
                
                tempLocalMeans = locMean;
                tempLocalMeans(temploc, :) = [];
                
                % Calculate two types of mean
                mbest1 = sum(tempLocalMeans, 1) / n;
                mbest2 = sum(Gm' * ones(1, M) .* tempLocalMeans, 1);
                
                % Lévy flight step
                sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
                         (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
                u = randn(size(frog(1).center)) * sigma;
                vw = randn(size(frog(1).center));
                step = u ./ abs(vw).^(1 / beta);
                stepsize = rand .* (randn) .* step;
                
                % Choose mean type randomly
                if rand < 0.5
                    mbest = mbest1;
                else
                    mbest = mbest2;
                end
                
                % Update worst frog
                temp = Xwb.center + rand * (stepsize .* mbest - frog(XwNo).center);
                
                % Boundary constraints
                temp(temp > ub) = ub(temp > ub);
                temp(temp < lb) = lb(temp < lb);
                
                % Evaluate new position
                [fitness_new, FE] = calculate_fitness(temp', problem, FE);
                
                % Update if improved
                if fitness_new < frog(XwNo).fitness
                    frog(XwNo).center = temp;
                    frog(XwNo).fitness = fitness_new;
                end
                
                % Update global best
                if fitness_new < tempFrog.fitness
                    tempFrog.center = temp;
                    tempFrog.fitness = fitness_new;
                end
                
                % Record convergence
                if FE <= maxFE
                    curve(FE) = tempFrog.fitness;
                    current_positions = reshape([frog.center], M, q)';
                    current_fitness = [frog.fitness];
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, current_positions, current_fitness, ...
                        population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                
                if FE >= maxFE
                    break;
                end
            end % m
            
            if FE >= maxFE
                break;
            end
        end % le
        
        if FE >= maxFE
            break;
        end
        
        % Global shuffling phase
        frogA = zeros(q, M);
        for iii = 1:q
            frogA(iii, :) = frog(iii).center;
        end
        
        % Random mapping matrix
        map = zeros(q, M);
        for i = 1:q
            u = randperm(M);
            map(i, u(1:ceil(rand * M))) = 1;
        end
        
        % Generate new positions
        K = rand(size(frogA)) > 0;
        stepsize = rand .* map .* (frogA(randperm(q), :) - frogA(randperm(q), :));
        
        new_frogA = frogA + stepsize .* K;
        
        % Evaluate shuffled population
        for jjj = 1:q
            data = new_frogA(jjj, :);
            data(data > ub) = ub(data > ub);
            data(data < lb) = lb(data < lb);
            
            [fitness_new, FE] = calculate_fitness(data', problem, FE);
            
            if fitness_new < frog(jjj).fitness
                frog(jjj).center = data;
                frog(jjj).fitness = fitness_new;
            end
            
            if fitness_new < tempFrog.fitness
                tempFrog.center = data;
                tempFrog.fitness = fitness_new;
            end
            
            % Record convergence
            if FE <= maxFE
                curve(FE) = tempFrog.fitness;
                current_positions = reshape([frog.center], M, q)';
                current_fitness = [frog.fitness];
                [population_history, fitness_history, history_index] = record_history(...
                    FE, current_positions, current_fitness, ...
                    population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            
            if FE >= maxFE
                break;
            end
        end
        
    end % while
    
    % Return best solution
    best_solution = tempFrog.center;
    best_fitness = tempFrog.fitness;
    
end

%% --- Helper Functions ---

function [M] = massCalculation(fit, min_flag)
    % Mass calculation based on fitness values
    Fmax = max(fit);
    Fmin = min(fit);
    [~, N] = size(fit);
    
    if Fmax == Fmin
        M = ones(1, N);
    else
        if min_flag == 1  % for minimization
            best = Fmin;
            worst = Fmax;
        else  % for maximization
            best = Fmax;
            worst = Fmin;
        end
        M = 1 * (fit - worst) ./ (best - worst) + 0;
    end
    
    M = M ./ sum(M);
end


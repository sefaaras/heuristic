% ----------------------------------------------------------------------- %
% Coyote Optimization Algorithm (COA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_packs = 20      % Number of packs
%   n_coy = 5         % Number of coyotes per pack
%   p_leave = 0.005*n_coy^2  % Probability of leaving a pack
%   Ps = 1/D          % Probability of birth
%   
% Algorithm Concept:
%   - Social organization: Pack structure with alpha leaders
%   - Social condition update: Based on alpha and pack tendency
%   - Birth mechanism: Pups born from random parents
%   - Pack exchange: Coyotes can leave and join other packs
%
% Reference:
% Julio César Pierezan, Leandro dos Santos Coelho,
% Coyote Optimization Algorithm: A New Metaheuristic for Global Optimization Problems,
% 2018 IEEE Congress on Evolutionary Computation (CEC), 2018, pp. 1-8
% https://doi.org/10.1109/CEC.2018.8477769
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = coa(problem)
    
    % Extract problem parameters
    D = problem.dimension;        % Problem dimension
    VarMin = problem.lb;          % Lower bounds
    VarMax = problem.ub;          % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    n_coy = 5;                    % Number of coyotes per pack
    n_packs = 20;                 % Number of packs
    p_leave = 0.005 * n_coy^2;    % Probability of leaving a pack
    Ps = 1 / D;                   % Probability for pup generation
    
    pop_total = n_packs * n_coy;  % Total population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;             % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, pop_total, D);  % Store population at sampled FEs
    fitness_history = zeros(history_size, pop_total);        % Store fitness values at sampled FEs
    history_index = 1;                % Current index in history arrays
    
    % Initialize coyotes population
    coyotes = repmat(VarMin, pop_total, 1) + rand(pop_total, D) .* ...
              (repmat(VarMax, pop_total, 1) - repmat(VarMin, pop_total, 1));
    
    ages = zeros(pop_total, 1);       % Ages of coyotes
    packs = reshape(randperm(pop_total), n_packs, []);  % Pack organization
    coypack = repmat(n_coy, n_packs, 1);  % Number of coyotes per pack
    
    % Evaluate initial population
    [costs, FE] = calculate_fitness(coyotes', problem, FE);
    costs = costs';  % Convert to column vector for consistency
    
    % Find initial best solution
    [GlobalMin, ibest] = min(costs);
    GlobalParams = coyotes(ibest, :);
    
    % Record best fitness for each initial evaluation and store history
    for eval_count = 1:pop_total
        curve(eval_count) = GlobalMin;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, coyotes, costs, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % Main loop
    year = 0;
    while FE < maxFE
        % Update the years counter
        year = year + 1;
        
        % Execute the operations inside each pack
        for p = 1:n_packs
            % Get the coyotes that belong to each pack
            pack_indices = packs(p, :);
            coyotes_aux = coyotes(pack_indices, :);
            costs_aux = costs(pack_indices, :);
            ages_aux = ages(pack_indices, 1);
            n_coy_aux = coypack(p, 1);
            
            % Detect alphas according to the costs (Eq. 5)
            [costs_aux, inds] = sort(costs_aux, 'ascend');
            coyotes_aux = coyotes_aux(inds, :);
            ages_aux = ages_aux(inds, :);
            c_alpha = coyotes_aux(1, :);
            
            % Compute the social tendency of the pack (Eq. 6)
            tendency = median(coyotes_aux, 1);
            
            % Update coyotes' social condition
            new_coyotes = zeros(n_coy_aux, D);
            for c = 1:n_coy_aux
                if FE >= maxFE
                    break;
                end
                
                % Select two random coyotes different from c
                rc1 = c;
                while rc1 == c
                    rc1 = randi(n_coy_aux);
                end
                rc2 = c;
                while rc2 == c || rc2 == rc1
                    rc2 = randi(n_coy_aux);
                end
                
                % Try to update the social condition according to the alpha and
                % the pack tendency (Eq. 12)
                new_c = coyotes_aux(c, :) + rand * (c_alpha - coyotes_aux(rc1, :)) + ...
                                            rand * (tendency - coyotes_aux(rc2, :));
                
                % Keep the coyotes in the search space (boundary control)
                new_coyotes(c, :) = min(max(new_c, VarMin), VarMax);
                
                % Evaluate the new social condition (Eq. 13)
                [new_cost, FE] = calculate_fitness(new_coyotes(c, :)', problem, FE);
                new_cost = new_cost(1);  % Extract scalar from potential array
                
                % Record convergence curve and history
                if FE <= maxFE
                    [GlobalMin, ~] = min(costs);
                    curve(FE) = GlobalMin;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, coyotes, costs, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                
                % Adaptation (Eq. 14)
                if new_cost < costs_aux(c, 1)
                    costs_aux(c, 1) = new_cost;
                    coyotes_aux(c, :) = new_coyotes(c, :);
                end
            end
            
            if FE >= maxFE
                break;
            end
            
            %% Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = randperm(n_coy_aux, 2);
            prob1 = (1 - Ps) / 2;
            prob2 = prob1;
            pdr = randperm(D);
            p1 = zeros(1, D);
            p2 = zeros(1, D);
            p1(pdr(1)) = 1; % Guarantee 1 characteristic per individual
            p2(pdr(2)) = 1; % Guarantee 1 characteristic per individual
            r = rand(1, D - 2);
            p1(pdr(3:end)) = r < prob1;
            p2(pdr(3:end)) = r > 1 - prob2;
            
            % Eventual noise 
            n = ~(p1 | p2);
            
            % Generate the pup considering intrinsic and extrinsic influence
            pup = p1 .* coyotes_aux(parents(1), :) + ...
                  p2 .* coyotes_aux(parents(2), :) + ...
                  n .* (VarMin + rand(1, D) .* (VarMax - VarMin));
            
            % Verify if the pup will survive
            [pup_cost, FE] = calculate_fitness(pup', problem, FE);
            pup_cost = pup_cost(1);  % Extract scalar from potential array
            
            % Record convergence curve and history
            if FE <= maxFE
                [GlobalMin, ~] = min(costs);
                curve(FE) = GlobalMin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, coyotes, costs, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            
            % Replace the worst coyote if pup is better
            worst = find(pup_cost < costs_aux);
            if ~isempty(worst)
                [~, older] = sort(ages_aux(worst), 'descend');
                which = worst(older);
                coyotes_aux(which(1), :) = pup;
                costs_aux(which(1), 1) = pup_cost;
                ages_aux(which(1), 1) = 0;
            end
            
            %% Update the pack information
            coyotes(pack_indices, :) = coyotes_aux;
            costs(pack_indices, :) = costs_aux;
            ages(pack_indices, 1) = ages_aux;
        end
        
        if FE >= maxFE
            break;
        end
        
        %% A coyote can leave a pack and enter another pack (Eq. 4)
        if n_packs > 1
            if rand < p_leave
                rp = randperm(n_packs, 2);
                rc = [randperm(coypack(rp(1), 1), 1) ...
                      randperm(coypack(rp(2), 1), 1)];
                aux = packs(rp(1), rc(1));
                packs(rp(1), rc(1)) = packs(rp(2), rc(2));
                packs(rp(2), rc(2)) = aux;
            end
        end
        
        %% Update coyotes ages
        ages = ages + 1;
        
        %% Update global best (best alpha coyote among all alphas)
        [GlobalMin, ibest] = min(costs);
        GlobalParams = coyotes(ibest, :);
    end
    
    % Return best solution
    best_fitness = GlobalMin;
    best_solution = GlobalParams;
    
end


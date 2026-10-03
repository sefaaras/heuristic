% ----------------------------------------------------------------------- %
% Levy Roulette Fitness-Distance Balance based Coyote Optimization Algorithm (LRFDB-COA)
% Variant of coa: roulette-FDB guide replaces the pack median; Levy draws pick pup genes
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_packs = 20      % Number of packs
%   n_coy = 5         % Number of coyotes per pack
%   p_leave = 0.005*n_coy^2  % Probability of leaving a pack
%   Ps = 1/D          % Probability of birth
%   beta = 1.5        % Levy index of the birth draw (Mantegna), magnitude capped at 1
%
% Algorithm Concept:
%   - Social organization: Pack structure with alpha leaders
%   - Social condition update: alpha term plus a tendency term whose guide is
%     drawn by roulette FDB from the whole population, not the pack median
%   - Birth: each pup gene comes from parent 1, parent 2 or noise by comparing
%     a Levy magnitude min(|step|, 1) with (1-Ps)/2, not a uniform draw
%   - Pack exchange: Coyotes can leave and join other packs
%
% Reference:
% Serhat Duman, Hamdi T. Kahraman, Ugur Guvenc, Sefa Aras,
% Development of a Levy flight and FDB-based coyote optimization algorithm for
% global optimization and real-world ACOPF problems,
% Soft Computing 25(8) (2021) 6577-6617.
% https://doi.org/10.1007/s00500-021-05654-z
% Components:
%   COA - Juliano Pierezan, Leandro dos Santos Coelho, Coyote Optimization
%     Algorithm: A New Metaheuristic for Global Optimization Problems, 2018 IEEE
%     Congress on Evolutionary Computation (CEC), 2018, pp. 1-8,
%     https://doi.org/10.1109/CEC.2018.8477769
%   FDB - Hamdi Tolga Kahraman, Sefa Aras, Eyup Gedikli, Fitness-distance
%     balance (FDB): A new selection method for meta-heuristic search
%     algorithms, Knowledge-Based Systems 190 (2020) 105169,
%     https://doi.org/10.1016/j.knosys.2019.105169
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' working copy (l_rfdb_coa.m) as a delta on this
% repository's coa.m: the tendency line and the birth draw are the only changes.
% As in that copy, the roulette FDB ranks all coyotes on the global costs as
% committed by the packs already processed this year, and levy(D) draws D+1
% normals of which only the first D-2 magnitudes are used. Added: the roulette
% falls back to the last index when rounding or a NaN score leaves the cumulative
% sum short of the draw (the find() empty-index crash fixed in sdo). Inherited
% from coa: the reported fitness and solution come from one population snapshot,
% and the last partial pack of a run is discarded as in the reference.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lrfdb_coa(problem)
    
    % Extract problem parameters
    D = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;
    
    % Algorithm parameters
    n_coy = 5;                    % Number of coyotes per pack
    n_packs = 20;                 % Number of packs
    p_leave = 0.005 * n_coy^2;    % Probability of leaving a pack
    Ps = 1 / D;                   % Probability for pup generation
    
    pop_total = n_packs * n_coy;  % Total population size
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Initialize coyotes population
    coyotes = repmat(VarMin, pop_total, 1) + rand(pop_total, D) .* ...
              (repmat(VarMax, pop_total, 1) - repmat(VarMin, pop_total, 1));
    
    ages = zeros(pop_total, 1);       % Ages of coyotes
    packs = reshape(randperm(pop_total), n_packs, []);  % Pack organization
    coypack = repmat(n_coy, n_packs, 1);  % Number of coyotes per pack
    
    % Evaluate initial population
    [costs, FE] = calculate_fitness(coyotes', problem, FE);
    costs = costs(:);  % Force column vector
    
    % Find initial best solution
    [GlobalMin, ibest] = min(costs);
    GlobalParams = coyotes(ibest, :);
    
    % Record best fitness for each initial evaluation and store history
    for eval_count = 1:pop_total
        curve(eval_count) = GlobalMin;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, coyotes, costs, population_history, fitness_history, ...
            history_index, maxFE);
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
            
            % Social tendency: roulette-FDB pick over all coyotes replaces the pack median (Eq. 6)
            tendency = coyotes(rouletteFitnessDistanceBalance(coyotes, costs), :);
            
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
                
                % Social condition updated from the alpha and the pack tendency, Eq. (12)
                new_c = coyotes_aux(c, :) + rand * (c_alpha - coyotes_aux(rc1, :)) + ...
                                            rand * (tendency - coyotes_aux(rc2, :));
                
                % Keep the coyotes in the search space (boundary control)
                new_coyotes(c, :) = min(max(new_c, VarMin), VarMax);
                
                % Evaluate the new social condition (Eq. 13)
                [new_cost, FE] = calculate_fitness(new_coyotes(c, :)', problem, FE);
                new_cost = new_cost(1);  % Extract scalar from potential array
                
                % Record convergence curve and history
                if FE <= maxFE
                    [GlobalMin, ibest] = min(costs);
                    GlobalParams = coyotes(ibest, :);
                    curve(FE) = GlobalMin;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, coyotes, costs, population_history, fitness_history, ...
                        history_index, maxFE);
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
            
            % Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = randperm(n_coy_aux, 2);
            prob1 = (1 - Ps) / 2;
            prob2 = prob1;
            pdr = randperm(D);
            p1 = zeros(1, D);
            p2 = zeros(1, D);
            p1(pdr(1)) = 1; % Guarantee 1 characteristic per individual
            p2(pdr(2)) = 1; % Guarantee 1 characteristic per individual
            % Levy magnitudes replace the uniform gene-selection draw of Alg. 1
            L = levy(D);
            r = L(1:D-2);
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
                [GlobalMin, ibest] = min(costs);
                GlobalParams = coyotes(ibest, :);
                curve(FE) = GlobalMin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, coyotes, costs, population_history, fitness_history, ...
                    history_index, maxFE);
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
            
            % Update the pack information
            coyotes(pack_indices, :) = coyotes_aux;
            costs(pack_indices, :) = costs_aux;
            ages(pack_indices, 1) = ages_aux;
        end
        
        if FE >= maxFE
            break;
        end
        
        % A coyote can leave a pack and enter another pack (Eq. 4)
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
        
        % Update coyotes ages
        ages = ages + 1;
        
        % Update global best (best alpha coyote among all alphas)
        [GlobalMin, ibest] = min(costs);
        GlobalParams = coyotes(ibest, :);
    end
    
    % Return best solution
    best_fitness = GlobalMin;
    best_solution = GlobalParams;
    
end

% Roulette-wheel FDB selection: score = normalised fitness + normalised distance to the best
function index = rouletteFitnessDistanceBalance(population, fitness)
    fitness = fitness(:)';
    populationSize = numel(fitness);
    [~, bestIndex] = min(fitness);
    best = population(bestIndex, :);
    if (min(fitness) == max(fitness)) || (sum(fitness) >= Inf) || ~(sum(best) < Inf)
        index = randi(populationSize);
        return;
    end
    distances = sum(abs(best - population), 2)';
    minFitness = min(fitness); maxMinFitness = max(fitness) - minFitness;
    minDistance = min(distances); maxMinDistance = max(distances) - minDistance;
    normFitness = 1 - ((fitness - minFitness) / maxMinFitness);
    normDistances = (distances - minDistance) / maxMinDistance;
    divDistances = normFitness + normDistances;
    r = rand * sum(divDistances);
    index = find(r <= cumsum(divDistances), 1, 'first');
    % Rounding (or a NaN score) can leave the cumulative sum short of r
    if isempty(index), index = populationSize; end
end

% Mantegna Levy magnitudes |u|/|v|^(1/beta), capped at 1
function L = levy(D)
    beta = 3/2;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1, D) * sigma;
    v = randn(1);
    step = u ./ abs(v).^(1/beta);
    L = 100 * abs(0.01 * step);
    L(L > 1) = 1;
end

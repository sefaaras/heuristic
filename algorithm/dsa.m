% ----------------------------------------------------------------------- %
% Differential Search Algorithm (DSA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   size_of_superorganism = 30  % Population size (superorganism size)
%   method = [1, 2]             % Direction generation methods
%   p1, p2 = 0.3*rand          % Morphogenesis probabilities
%   R = 1./gamrnd(1,0.5)       % Scale factor (pseudo-stable walk)
%
% Algorithm Concept:
%   - Direction generation: B-DSA (bijective), S-DSA (surjective) or elitist
%   - Map generation selects the active and passive individuals of each clan
%   - Bio-interaction: stopover = pop + (R.*map).*(direction - pop)
%   - Greedy selection between each stopover and its parent
%
% Reference:
% P. Civicioglu,
% Transforming geocentric cartesian coordinates to geodetic coordinates
% by using differential search algorithm,
% Computers & Geosciences, Volume 46, 2012, Pages 229-247
% https://doi.org/10.1016/j.cageo.2011.12.011
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dsa(problem)

    % Extract problem parameters
    dimension = problem.dimension;
    low_habitat_limit = problem.lb;
    up_habitat_limit = problem.ub;
    maxIteration = problem.maxFe;
    
    method = [1, 2];
    size_of_superorganism = 30;
    size_of_one_clan = dimension;
    
    %Initialization
    
    % generate initial individuals, clans and superorganism.
    superorganism=genpop(size_of_superorganism,size_of_one_clan,low_habitat_limit,up_habitat_limit);
    
    fe = 0;                             % Function Evaluation Counter
    curve = zeros(1, maxIteration);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
    % Calculate initial fitness
    [fit_superorganism, fe] = calculate_fitness(superorganism', problem, fe);
    
    % Record initial best fitness and store history
    for eval_count = 1:size_of_superorganism
        [current_best, ~] = min(fit_superorganism);
        if eval_count <= maxIteration
            curve(eval_count) = current_best;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, superorganism, fit_superorganism, population_history, fitness_history, ...
                history_index, maxIteration);
        end
    end
    
    while fe < maxIteration
        
        % 'one-or-more morphogenesis', the reference default trial-pattern strategy
        p1=0.3*rand;
        p2=0.3*rand;


       [direction,~]=generate_direction(method(randi(numel(method))),superorganism,size_of_superorganism,fit_superorganism);
        
       map=generate_map_of_active_individuals(size_of_superorganism,size_of_one_clan,p1,p2);
              
        R=1./gamrnd(1,0.5);   % pseudo-stable walk scale factor of the reference


        % bio-interaction (morphogenesis) 
        stopover=superorganism+(R.*map).*(direction-superorganism);
    
       % Boundary Control
        stopover=update(stopover,low_habitat_limit,up_habitat_limit); 
        
        % Selection-II
        
        [fit_stopover, fe] = calculate_fitness(stopover', problem, fe);
    
        ind=fit_stopover<fit_superorganism; 
        fit_superorganism(ind)=fit_stopover(ind); 
        superorganism(ind,:)=stopover(ind,:);
        
        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:size_of_superorganism
            eval_count = fe - size_of_superorganism + eval_idx;
            if eval_count <= maxIteration
                [current_best, ~] = min(fit_superorganism);
                curve(eval_count) = current_best;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, superorganism, fit_superorganism, population_history, fitness_history, ...
                    history_index, maxIteration);
            end
        end
            
    end
    
    % Final best solution
    [best_fitness, indexbest] = min(fit_superorganism);    
    best_solution = superorganism(indexbest,:);

end

function pop=genpop(a,b,low,up)
    pop=ones(a,b);
    for i=1:a
        for j=1:b 
            pop(i,j)=rand*(up(j)-low(j))+low(j);
        end
    end
end

function p=update(p,low,up)
    [popsize,dim]=size(p);
    for i=1:popsize
        for j=1:dim
            % Standard method of the reference: re-draw or clamp, decided per component
            if p(i,j)<low(j), if rand<rand, p(i,j)=rand*(up(j)-low(j))+low(j); else, p(i,j)=low(j); end, end
            if p(i,j)>up(j),  if rand<rand, p(i,j)=rand*(up(j)-low(j))+low(j); else, p(i,j)=up(j); end, end
        end
    end
end

function [direction,msg]=generate_direction(method,superorganism,size_of_superorganism,fit_superorganism)
     switch method
            case 1           
                % B-DSA (bijective): move towards a permutation of the superorganism
                direction=superorganism(randperm(size_of_superorganism),:); msg=' B-DSA';
            case 2   
                % S-DSA (surjective): move towards random members of the top-best set
                ind=ones(size_of_superorganism,1); 
                [~,B]=sort(fit_superorganism); 
                for i=1:size_of_superorganism, ind(i)=B(randi(ceil(rand*size_of_superorganism),1)); end 
                direction=superorganism(ind,:);  msg=' S-DSA';   
            case 3
                % E1-DSA (elitist): move towards one randomly chosen top-best solution
                [~,jind]=sort(fit_superorganism); ibest=jind(ceil(rand*size_of_superorganism)); msg='E1-DSA'; 
                direction=repmat(superorganism(ibest,:),[size_of_superorganism 1]); 
            case 4
                % E2-DSA (elitist): move towards the single best solution
                [~,ibest]=min(fit_superorganism); msg='E2-DSA';
                direction=repmat(superorganism(ibest,:),[size_of_superorganism 1]);             
     end
end

function map=generate_map_of_active_individuals(size_of_superorganism,size_of_one_clan,p1,p2)
        % strategy-selection of active/passive individuals
        map=zeros(size_of_superorganism,size_of_one_clan);
            if rand<rand
                if rand<p1
                    % Random-mutation #1 strategy
                    for i=1:size_of_superorganism
                        map(i,:)=rand(1,size_of_one_clan) < rand;              
                    end
                else
                    % Differential-mutation strategy
                    for i=1:size_of_superorganism 
                        map(i,randi(size_of_one_clan))=1;
                    end
                end
            else
                 % Random-mutation #2 strategy
                for i=1:size_of_superorganism                
                    map(i,randi(size_of_one_clan,1,ceil(p2*size_of_one_clan)))=1;                
                end
            end
end
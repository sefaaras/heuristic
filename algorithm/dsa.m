% ----------------------------------------------------------------------- %
% Differential Search Algorithm (DSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   size_of_superorganism = 30  % Population size (superorganism size)
%   method = [1, 2]             % Direction generation methods
%   p1, p2 = 0.3*rand          % Morphogenesis probabilities
%   R = 1./gamrnd(1,0.5)       % Scale factor (pseudo-stable walk)
%
% Algorithm Phases:
%   1. Direction Generation     % B-DSA (Bijective) or S-DSA (Surjective)
%   2. Map Generation          % Active/passive individuals selection
%   3. Bio-interaction         % Morphogenesis process
%   4. Selection               % Greedy selection of better solutions
%
% Reference:
% P. Civicioglu,
% Transforming geocentric cartesian coordinates to geodetic coordinates 
% by using differential search algorithm,
% Computers & Geosciences, Volume 46, 2012, Pages 229-247
% https://doi.org/10.1016/j.cageo.2011.12.011
% ----------------------------------------------------------------------- %
% Modified to match signature: [best_fitness, best_solution, curve, population_history, fitness_history]
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds  
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dsa(problem)

    % Extract problem parameters
    dimension = problem.dimension;      % Problem dimension
    low_habitat_limit = problem.lb;     % Lower bounds
    up_habitat_limit = problem.ub;      % Upper bounds
    maxIteration = problem.maxFe;       % Maximum function evaluations
    
    method = [1, 2];
    size_of_superorganism = 30;
    size_of_one_clan = dimension;
    
    %Initialization
    
    % generate initial individuals, clans and superorganism.
    superorganism=genpop(size_of_superorganism,size_of_one_clan,low_habitat_limit,up_habitat_limit);
    
    fe = 0;                             % Function Evaluation Counter
    curve = zeros(1, maxIteration);     % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;               % Fixed history size
    sampling_interval = max(1, floor(maxIteration / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, size_of_superorganism, size_of_one_clan);     % Store population at sampled FEs
    fitness_history = zeros(history_size, size_of_superorganism);                          % Store fitness values at sampled FEs
    history_index = 1;                  % Current index in history arrays
    
    % Calculate initial fitness
    [fit_superorganism, fe] = calculate_fitness(superorganism', problem, fe);
    
    % Record initial best fitness and store history
    for eval_count = 1:size_of_superorganism
        [current_best, ~] = min(fit_superorganism);
        if eval_count <= maxIteration
            curve(eval_count) = current_best;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, superorganism, fit_superorganism, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    while fe < maxIteration
        
        % SETTING OF ALGORITHMIC CONTROL PARAMETERS
        % Trial-pattern generation strategy for morphogenesis; 'one-by-one morphogenesis'. 
        % p1=0.0*rand;  % i.e.,  0.0 <= p1 <= 0.0
        % p2=0.0*rand;  % i.e.,  0.0 <= p2 <= 0.0
        
        % Trial-pattern generation strategy for morphogenesis; 'one-or-more morphogenesis'. (DEFAULT)
        p1=0.3*rand;  % i.e.,  0.0 <= p1 <= 0.3
        p2=0.3*rand;  % i.e.,  0.0 <= p2 <= 0.3
        
        %-------------------------------------------------------------------
        
       [direction,~]=generate_direction(method(randi(numel(method))),superorganism,size_of_superorganism,fit_superorganism);
        
       map=generate_map_of_active_individuals(size_of_superorganism,size_of_one_clan,p1,p2);
              
      %-------------------------------------------------------------------
        % Recommended Methods for generation of Scale-Factor; R 
        % R=4*randn;  % brownian walk
        % R=4*randg;  % brownian walk
        % R=lognrnd(rand,5*rand);  % brownian walk
         R=1./gamrnd(1,0.5);   % pseudo-stable walk
        % R=1/normrnd(0,5);    % pseudo-stable walk
    
        %-------------------------------------------------------------------
        
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
                    history_index, sampling_interval, history_size);
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
            % first (standard)-method
            if p(i,j)<low(j), if rand<rand, p(i,j)=rand*(up(j)-low(j))+low(j); else, p(i,j)=low(j); end, end
            if p(i,j)>up(j),  if rand<rand, p(i,j)=rand*(up(j)-low(j))+low(j); else, p(i,j)=up(j); end, end
            
            %{
           %  second-method
            if rand<rand,                    
                if p(i,j)<low(j) || p(i,j)>up(j), p(i,j)=rand*(up(j)-low(j))+low(j); end
            else
                if p(i,j)<low(j), p(i,j)=low(j); end
                if p(i,j)>up(j),  p(i,j)=up(j); end            
            end
            %}
        end        
    end
end

function [direction,msg]=generate_direction(method,superorganism,size_of_superorganism,fit_superorganism)
     switch method
            case 1           
                % BIJECTIVE DSA  (B-DSA) (i.e., go-to-rnd DSA);             
                % philosophy: evolve the superorganism (i.e.,population) towards to "permuted-superorganism (i.e., random directions)" 
                direction=superorganism(randperm(size_of_superorganism),:); msg=' B-DSA';
            case 2   
                % SURJECTIVE DSA (S-DSA) (i.e., go-to-good DSA)
                % philosophy: evolve the superorganism (i.e.,population) towards to "some of the random top-best" solutions
                ind=ones(size_of_superorganism,1); 
                [~,B]=sort(fit_superorganism); 
                for i=1:size_of_superorganism, ind(i)=B(randi(ceil(rand*size_of_superorganism),1)); end 
                direction=superorganism(ind,:);  msg=' S-DSA';   
            case 3
                % ELITIST DSA #1 (E1-DSA) (i.e., go-to-best DSA)
                % philosophy: evolve the superorganism (i.e.,population) towards to "one of the random top-best" solution
                [~,jind]=sort(fit_superorganism); ibest=jind(ceil(rand*size_of_superorganism)); msg='E1-DSA'; 
                direction=repmat(superorganism(ibest,:),[size_of_superorganism 1]); 
            case 4
                % ELITIST DSA #2 (E2-DSA) (i.e., go-to-best DSA)
                % philosophy: evolve the superorganism (i.e.,population) towards to "the best" solution
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
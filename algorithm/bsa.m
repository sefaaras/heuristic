% ----------------------------------------------------------------------- %
% Backtracking Search Optimization Algorithm (BSA) for unconstrained 
% benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 30        % Population size
%   DIM_RATE = 1        % Dimension rate for crossover map
%   F = 3*randn         % Scale factor (mutation strength)
%
% Reference:
% P. Civicioglu, 
% Backtracking Search Optimization Algorithm for numerical optimization 
% problems, Applied Mathematics and Computation 219 (2013) 8121-8144
% https://doi.org/10.1016/j.amc.2013.02.017
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = bsa(problem)

    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    low = problem.lb;              % Lower bounds
    up = problem.ub;               % Upper bounds
    maxIteration = problem.maxFe;  % Maximum function evaluations
    
    popsize = 30; DIM_RATE = 1;
    %INITIALIZATION
    pop=GeneratePopulation(popsize,dim,low,up); % see Eq.1 in [1]
    
    FE = 0;                         % Function Evaluation Counter
    curve = zeros(1, maxIteration); % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;           % Fixed history size
    sampling_interval = max(1, floor(maxIteration / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, popsize, dim);     % Store population at sampled FEs
    fitness_history = zeros(history_size, popsize);            % Store fitness values at sampled FEs
    history_index = 1;              % Current index in history arrays
    
    [fitnesspop, FE] = calculate_fitness(pop', problem, FE);
    
    % Record initial best fitness and store history
    for eval_count = 1:popsize
        [current_best, ~] = min(fitnesspop);
        if eval_count <= maxIteration
            curve(eval_count) = current_best;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, pop, fitnesspop, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    historical_pop=GeneratePopulation(popsize,dim,low,up); % see Eq.2 in [1]
    
    % historical_pop  is swarm-memory of BSA as mentioned in [1].
    
    % ------------------------------------------------------------------------------------------
        while FE<maxIteration
    
            %SELECTION-I
            if rand<rand, historical_pop=pop; end  % see Eq.3 in [1]
            historical_pop=historical_pop(randperm(popsize),:); % see Eq.4 in [1]
            F=get_scale_factor; % see Eq.5 in [1], you can other F generation strategies 
            map=zeros(popsize,dim); % see Algorithm-2 in [1]         
            if rand<rand
                for i=1:popsize,  u=randperm(dim); map(i,u(1:ceil(DIM_RATE*rand*dim)))=1; end
            else
                for i=1:popsize,  map(i,randi(dim))=1; end
            end
            % RECOMBINATION (MUTATION+CROSSOVER)   
            offsprings=pop+(map.*F).*(historical_pop-pop);   % see Eq.5 in [1]    
            offsprings=BoundaryControl(offsprings,low,up); % see Algorithm-3 in [1]
            % SELECTON-II
            
            [fitnessoffsprings, FE] = calculate_fitness(offsprings', problem, FE);
            ind=fitnessoffsprings<fitnesspop;
            fitnesspop(ind)=fitnessoffsprings(ind);
            pop(ind,:)=offsprings(ind,:);
            
            % Record convergence curve for each evaluation and store history
            for eval_idx = 1:popsize
                eval_count = FE - popsize + eval_idx;
                if eval_count <= maxIteration
                    [current_best, ~] = min(fitnesspop);
                    curve(eval_count) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        eval_count, pop, fitnesspop, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
        end
        
        % Final best solution
        [best_fitness, idx] = min(fitnesspop);    
        best_solution = pop(idx,:);
    
    end
    
    function pop=GeneratePopulation(popsize,dim,low,up)
    pop=ones(popsize,dim);
    for i=1:popsize
        for j=1:dim
            pop(i,j)=rand*(up(j)-low(j))+low(j);
        end
    end
    end
    
    function pop=BoundaryControl(pop,low,up)
    [popsize,dim]=size(pop);
    for i=1:popsize
        for j=1:dim                
            k=rand<rand; % you can change boundary-control strategy
            if pop(i,j)<low(j)
                if k
                    pop(i,j)=low(j); 
                else 
                    pop(i,j)=rand*(up(j)-low(j))+low(j); 
                end
            end        
            if pop(i,j)>up(j)
                if k
                    pop(i,j)=up(j); 
                else 
                    pop(i,j)=rand*(up(j)-low(j))+low(j); 
                end
            end        
        end
    end
    end
    
    function F=get_scale_factor % you can change generation strategy of scale-factor,F    
         F=3*randn; % STANDARD brownian-walk
        % F=4*randg;  % brownian-walk    
        % F=lognrnd(rand,5*rand);  % brownian-walk              
        % F=1/normrnd(0,5);        % pseudo-stable walk (levy-like)
        % F=1./gamrnd(1,0.5);      % pseudo-stable walk (levy-like, simulates inverse gamma distribution; levy-distiribution)   
    end
% ----------------------------------------------------------------------- %
% Symbiotic Organisms Search (SOS) for unconstrained benchmark problems
% a simplified version, last revised: 2014.08.27
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   ecosize = 50        % Ecosystem size (population size)
%   BF1, BF2 = 1 or 2   % Beneficial factors in mutualism phase
%   
% Algorithm Phases:
%   1. Mutualism Phase   % Both organisms benefit from interaction
%   2. Commensalism Phase % One organism benefits, other unaffected
%   3. Parasitism Phase  % One organism benefits, other harmed
%
% Reference:
% Min-Yuan Cheng, Doddy Prayogo,         
% Symbiotic Organisms Search: A new metaheuristic optimization algorithm, 
% Computers & Structures 139 (2014), 98-112   
% http://dx.doi.org/10.1016/j.compstruc.2014.03.007                          
% ----------------------------------------------------------------------- %
% Written by Doddy Prayogo at National Taiwan University of Science and 
% Technology (NTUST)
% Email: doddyprayogo@ymail.com
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = sos(problem)
    
    % Extract problem parameters
    n = problem.dimension;       % Problem dimension
    lb = problem.lb;             % Lower bounds
    ub = problem.ub;             % Upper bounds
    maxFE = problem.maxFe;       % Maximum function evaluations
    
    ecosize = 50;
    
    FE=0;                           % Function of Evaluation Counter
    curve = zeros(1, maxFE);        % Convergence curve - preallocate for maxFe elements
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;           % Fixed history size
    sampling_interval = max(1, floor(maxFE / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, ecosize, n);     % Store population at sampled FEs
    fitness_history = zeros(history_size, ecosize);           % Store fitness values at sampled FEs
    history_index = 1;              % Current index in history arrays
    
    eco=zeros(ecosize,n);
    for i=1:ecosize
        eco(i,:)=rand(1,n).*(ub-lb)+lb;
    end
    
    [fitness, FE] = calculate_fitness(eco', problem, FE);
    
    % Record best fitness for each initial evaluation and store population/fitness history
    for eval_count = 1:ecosize
        [current_best, ~] = min(fitness);
        curve(eval_count) = current_best;
        % Store history with sampling
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, eco, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    % --- Main Looping
    while FE<maxFE 
        
        for i=1:ecosize % Organisms' Looping
            
            % Update the best Organism
            [~, idx]=min(fitness); bestOrganism=eco(idx,:);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mutualism Phase
                % Choose organism j randomly other than organism i           
                j=i;
                while i==j
                    seed=randperm(ecosize); 
                    j=seed(1);                  
                end
                
                % Determine Mutual Vector & Beneficial Factor
                mutualVector=mean([eco(i,:);eco(j,:)]);
                BF1=round(1+rand); BF2=round(1+rand);
                
                % Calculate new solution after Mutualism Phase
                ecoNew1=eco(i,:)+rand(1,n).*(bestOrganism-BF1.*mutualVector); 
                ecoNew2=eco(j,:)+rand(1,n).*(bestOrganism-BF2.*mutualVector);
                ecoNew1=bound(ecoNew1,ub,lb); 
                ecoNew2=bound(ecoNew2,ub,lb);
                    
                % Evaluate the fitness of the new solution
                [fitnessNew1, FE] = calculate_fitness(ecoNew1', problem, FE);
                
                % Accept the new solution if the fitness is better
                if fitnessNew1<fitness(i)
                    fitness(i)=fitnessNew1;
                    eco(i,:)=ecoNew1;
                end
                
                % Record best fitness in curve after potential update and store history
                if FE <= maxFE
                    [current_best, ~] = min(fitness);
                    curve(FE) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, eco, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                if FE >= maxFE, break; end
                
                [fitnessNew2, FE] = calculate_fitness(ecoNew2', problem, FE);
                
                % Accept the new solution if the fitness is better
                if fitnessNew2<fitness(j)
                   fitness(j)=fitnessNew2;
                   eco(j,:)=ecoNew2;
                end
                
                % Record best fitness in curve after potential update and store history
                if FE <= maxFE
                    [current_best, ~] = min(fitness);
                    curve(FE) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, eco, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                
                if FE >= maxFE, break; end
                
            % End of Mutualism Phase 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            % Commensialism Phase
                
                % Choose organism j randomly other than organism i
                j=i;
                while i==j
                    seed=randperm(ecosize); 
                    j=seed(1);                  
                end
                
                % Calculate new solution after Commensalism Phase    
                ecoNew1=eco(i,:)+(rand(1,n)*2-1).*(bestOrganism-eco(j,:));
                ecoNew1=bound(ecoNew1,ub,lb);
    
                % Evaluate the fitness of the new solution
                [fitnessNew1, FE] = calculate_fitness(ecoNew1', problem, FE);
                
                % Accept the new solution if the fitness is better
                if fitnessNew1<fitness(i)
                    fitness(i)=fitnessNew1;
                    eco(i,:)=ecoNew1;
                end
                
                % Record best fitness in curve and store history
                if FE <= maxFE
                    [current_best, ~] = min(fitness);
                    curve(FE) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, eco, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                if FE >= maxFE, break; end
                
            % End of Commensalism Phase
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Parasitism Phase
    
                % Choose organism j randomly other than organism i 
                j=i;
                while i==j
                    seed=randperm(ecosize);
                    j=seed(1);
                end
                
                % Determine Parasite Vector & Calculate the fitness
                parasiteVector=eco(i,:);
                seed=randperm(n);           
                pick=seed(1:ceil(rand*n));  % select random dimension
                parasiteVector(:,pick)=rand(1,length(pick)).*(ub(pick)-lb(pick))+lb(pick);
                
                [fitnessParasite, FE] = calculate_fitness(parasiteVector', problem, FE);
                
                % Kill organism j and replace it with the parasite 
                % if the fitness is lower than the parasite
                if fitnessParasite < fitness(j)
                    fitness(j)=fitnessParasite;
                    eco(j,:)=parasiteVector;
                end
                
                % Record best fitness in curve and store history
                if FE <= maxFE
                    [current_best, ~] = min(fitness);
                    curve(FE) = current_best;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, eco, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                if FE >= maxFE, break; end
            
            % End of Parasitism Phase
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                 
        end % End of Organisms' Looping
        
       
    end % End of Main Looping
    
    % --- Update the best Organism
    [best_fitness,idx]=min(fitness); 
    best_solution = eco(idx,:);
    
    end
    
    %% --- Boundary Handling --- 
    function a=bound(a,ub,lb)
        a(a>ub)=ub(a>ub); a(a<lb)=lb(a<lb);
    end
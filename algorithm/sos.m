% ----------------------------------------------------------------------- %
% Symbiotic Organisms Search (SOS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   ecosize = 50        % Ecosystem size (population size)
%   BF1, BF2 = 1 or 2   % Beneficial factors in mutualism phase
%
% Algorithm Concept:
%   - Mutualism phase: both organisms benefit from the interaction
%   - Commensalism phase: one organism benefits, the other is unaffected
%   - Parasitism phase: one organism benefits, the other is harmed
%
% Reference:
% Min-Yuan Cheng, Doddy Prayogo,
% Symbiotic Organisms Search: A new metaheuristic optimization algorithm,
% Computers & Structures 139 (2014), 98-112
% http://dx.doi.org/10.1016/j.compstruc.2014.03.007
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the simplified reference implementation released by the paper's
% second author (Doddy Prayogo, NTUST), revision 2014.08.27.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sos(problem)
    
    % Extract problem parameters
    n = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    
    ecosize = 50;
    
    FE=0;                           % Function of Evaluation Counter
    curve = zeros(1, maxFE);
    
    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    
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
            history_index, maxFE);
    end
    
    % --- Main Looping
    while FE<maxFE 
        
        for i=1:ecosize % Organisms' Looping
            
            % Update the best Organism
            [~, idx]=min(fitness); bestOrganism=eco(idx,:);
            
            % Mutualism phase: organism j drawn at random, other than organism i
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
                        history_index, maxFE);
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
                        history_index, maxFE);
                end
                
                if FE >= maxFE, break; end
                
            % Commensalism phase
                
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
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
                
            % Parasitism phase
    
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
                
                % Organism j is killed and replaced when the parasite scores better
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
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
            
            % End of Parasitism Phase
                 
        end % End of Organisms' Looping
        
       
    end % End of Main Looping
    
    % --- Update the best Organism
    [best_fitness,idx]=min(fitness); 
    best_solution = eco(idx,:);
    
    end
    
    % Boundary Handling
    function a=bound(a,ub,lb)
        a(a>ub)=ub(a>ub); a(a<lb)=lb(a<lb);
    end
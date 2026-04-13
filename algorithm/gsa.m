% ----------------------------------------------------------------------- %
% Gravitational Search Algorithm (GSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                  % Population size (number of agents)
%   ElitistCheck = 1        % Use elitist strategy (1=yes, 0=no)
%   Rpower = 1              % Power of distance in force calculation
%   Rnorm = 2               % Norm type for distance calculation
%   final_per = 2           % Final percentage of agents applying force
%
% Algorithm Concept:
%   - Based on law of gravity and mass interactions
%   - Agents have masses proportional to their fitness
%   - Better agents attract others with stronger gravitational force
%   - Gravitational constant decreases over time
%   - Agents move according to gravitational forces
%
% Reference:
% Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi,
% GSA: A Gravitational Search Algorithm,
% Information Sciences 179 (2009) 2232–2248
% https://doi.org/10.1016/j.ins.2009.03.004
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = gsa(problem)

    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    low = problem.lb;              % Lower bounds
    up = problem.ub;               % Upper bounds
    maxIteration = problem.maxFe;  % Maximum function evaluations
    
    N = 50;                        % Population size
    ElitistCheck = 1;              % Use elitist strategy
    Rpower = 1;                    % Power of distance in force calculation
    Rnorm = 2;                     % Norm type for distance calculation
    
    FE = 0;                        % Function Evaluation Counter
    curve = zeros(1, maxIteration); % Convergence curve
    
    % Initialize storage for population and fitness history with 1/10000 sampling
    history_size = 10000;           % Fixed history size
    sampling_interval = max(1, floor(maxIteration / history_size));  % Calculate sampling interval
    population_history = zeros(history_size, N, dim);     % Store population at sampled FEs
    fitness_history = zeros(history_size, N);             % Store fitness values at sampled FEs
    history_index = 1;              % Current index in history arrays
    
    % Random initialization for agents
    X = initialization(dim, N, up, low); 
    
    V = zeros(N, dim);             % Velocity initialization
    max_it = ceil(maxIteration/N) + 1;
    
    for iter = 1:max_it
        % Checking allowable range
        X = space_bound(X, up, low); 

        % Calculate fitness
        [fitness, FE] = calculate_fitness(X', problem, FE);
        
        [best, best_X] = min(fitness); % minimization
        
        if iter == 1
            best_fitness = best;
            best_solution = X(best_X, :);
        end

        if best < best_fitness % minimization
            best_fitness = best;
            best_solution = X(best_X, :);
        end    

        % Record convergence curve for each evaluation and store history
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count <= maxIteration
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        
        if FE >= maxIteration
            break;
        end

        % Calculation of M. eq.14-20
        [M] = massCalculation(fitness); 

        % Calculation of Gravitational constant. eq.13.
        G = Gconstant(iter, max_it); 

        % Calculation of acceleration in gravitational field. eq.7-10,21.
        a = Gfield(M, X, G, Rnorm, Rpower, ElitistCheck, iter, max_it);

        % Agent movement. eq.11-12
        [X, V] = move(X, a, V);

    end % iteration

end
    
    function [X]=initialization(dim,N,up,down)
        if size(up,2)==1
            X=rand(N,dim).*(up-down)+down;
        end
        if size(up,2)>1
            for i=1:dim
            high=up(i);low=down(i);
            X(:,i)=rand(N,1).*(high-low)+low;
            end
        end
    end
    
    function  X=space_bound(X,up,low)
        [N,dim]=size(X);
        for i=1:N 
        %     %%Agents that go out of the search space, are reinitialized randomly .
            Tp=X(i,:)>up;Tm=X(i,:)<low;X(i,:)=(X(i,:).*(~(Tp+Tm)))+((rand(1,dim).*(up-low)+low).*(Tp+Tm));
        end
    end
    
    function [M]=massCalculation(fit)
        %%%%here, make your own function of 'mass calculation'
        % Ensure fit is a column vector
        fit = fit(:);
        
        Fmax=max(fit); Fmin=min(fit); 
        N = length(fit);
        if Fmax==Fmin
           M=ones(N,1);
        else
            best=Fmin;worst=Fmax; %eq.17-18
           M=(fit-worst)./(best-worst); %eq.15,
        end
        M=M./sum(M); %eq. 16.
    end
    
    function G=Gconstant(iteration,max_it)
    %%%here, make your own function of 'G'
      alfa=20;G0=100;
      G=G0*exp(-alfa*iteration/max_it); %eq. 28.
    end
    
    function a=Gfield(M,X,G,Rnorm,Rpower,ElitistCheck,iteration,max_it)
    
        [N,dim]=size(X);
         final_per=2; %In the last iteration, only 2 percent of agents apply force to the others.
    
        %%%%total force calculation
         if ElitistCheck==1
             kbest=final_per+(1-iteration/max_it)*(100-final_per); %kbest in eq. 21.
             kbest=round(N*kbest/100);
         else
             kbest=N; %eq.9.
         end
         % Ensure kbest is valid and within bounds
         kbest = max(1, min(kbest, N));
         
        [~, ds]=sort(M,'descend');
        E=zeros(N,dim);
         for i=1:N
             E(i,:)=zeros(1,dim);
             for ii=1:kbest
                 j=ds(ii);
                 if j~=i
                    R=norm(X(i,:)-X(j,:),Rnorm); %Euclidian distanse.
                 for k=1:dim 
                     E(i,k)=E(i,k)+rand*(M(j))*((X(j,k)-X(i,k))/(R^Rpower+eps));
                      %note that Mp(i)/Mi(i)=1
                 end
                 end
             end
         end
        %%acceleration
        a=E.*G; %note that Mp(i)/Mi(i)=1
    end
    
    function [X,V]=move(X,a,V)
        %movement.
        [N,dim]=size(X);
        V=rand(N,dim).*V+a; %eq. 11.
        X=X+V; %eq. 12.
    end
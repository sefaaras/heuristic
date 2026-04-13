% ----------------------------------------------------------------------- %
% Yin-Yang-Pair Optimization (YYPO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Imin = 2, Imax = 4   % Min and max archive size
%   alpha = 10           % Radius reduction factor
%   del(1) = 0.5, del(2) = 0.5  % Initial search radii
%   
% Algorithm Concept:
%   - Binary splitting strategy in normalized space
%   - Two points (Yin-Yang pair) guide search
%   - Archive-based selection of best solutions
%   - Adaptive radius adjustment
%
% Reference:
% Saurabh Punnathanam, Pradeep Kotecha,
% Yin-Yang-pair Optimization: A novel lightweight optimization algorithm,
% Engineering Applications of Artificial Intelligence 54 (2016) 62-79
% http://dx.doi.org/10.1016/j.engappai.2016.04.004
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = yypo(problem)
    
    % Extract problem parameters
    D = problem.dimension;         % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    % Algorithm parameters
    Imin = 2; 
    Imax = 4; 
    alpha = 10;
    del(1) = 0.5; 
    del(2) = 0.5;
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize storage for population and fitness history
    % For YYPO, population size varies, so we store archive snapshots
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, 2, D);  % Store two main points
    fitness_history = zeros(history_size, 2);        % Store their fitness values
    history_index = 1;
    
    if D <= 52
        BMatrixFn = @UseMatlabInbuilt;
    else
        BMatrixFn = @UseTailorMade;
    end
    
    % Generate two random points between 0 and 1 (normalized space)
    P = rand(2, D);
    
    % Scale the points and evaluate the objective function
    ScaleP = repmat(lb, 2, 1) + P .* repmat((ub - lb), 2, 1);
    [fitness_temp, FE] = calculate_fitness(ScaleP', problem, FE);
    P(:, D + 1) = fitness_temp;
    
    % Record initial evaluations
    for eval_count = 1:2
        curve(eval_count) = min(P(:, D + 1));
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, P(:, 1:D), P(:, D + 1), population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end
    
    Acount = 0;  % Archive Counter
    SizeArch = randi([Imin, Imax], 1, 1);  % Selection of I between min and max
    
    % Initialization of a matrix to store the points in archive
    Arch = NaN(SizeArch * 2, D + 1);
    
    % Main optimization loop
    while FE < maxFE
        
        % If the point P2 is better than P1, then swap P1 and P2 with its del
        if P(2, D + 1) < P(1, D + 1)
            [P(2, :), P(1, :), del(2), del(1)] = deal(P(1, :), P(2, :), del(1), del(2));
        end
        
        % Stack the points in the archive
        Acount = Acount + 1;
        Arch(2 * Acount - 1 : 2 * Acount, :) = P;
        
        % Execute the splitting function for both the points
        [P, FE_new] = SplitFn(problem, BMatrixFn, D, P, del, lb, ub, FE);
        
        % Record curve for all new evaluations
        for i = (FE + 1):FE_new
            if i <= maxFE
                curve(i) = min(P(:, D + 1));
                [population_history, fitness_history, history_index] = record_history(...
                    i, P(:, 1:D), P(:, D + 1), population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
        FE = FE_new;
        
        % Check if the Archiving Stage is reached
        if Acount == SizeArch
            [P, del] = Archivefn(Arch, P, del, alpha, D);
            SizeArch = randi([Imin, Imax], 1, 1);
            % Matrix to store the points in archive
            Arch = NaN(SizeArch * 2, D + 1);
            Acount = 0;
        end
        
    end
    
    % Stack the latest set of points in the Archive
    Arch = [Arch; P];
    
    % Extracting the best solution
    [best_fitness, ind] = min(Arch(:, D + 1));
    best_solution = lb + Arch(ind, 1:D) .* (ub - lb);
    
end

%% --- Helper Functions ---

function B = UseMatlabInbuilt(D)
    B = de2bi(randperm(2^D, 2*D) - ones(1, 2*D));
    B = [zeros(2*D, D - size(B, 2)) B];
end

function B = UseTailorMade(D)
    p = 1;
    while p < 200
        B = randi([0 1], 2*D, D);
        B = unique(B, 'rows');
        [r, ~] = size(B);
        % Check if the number of unique rows satisfies the requirement
        if r == 2*D
            break
        end
        p = p + 1;
    end
end

function [Points, FE] = SplitFn(problem, BMatrixFn, D, Points, radius, lb, ub, FE)
    for i = 1:2
        
        P = Points(i, 1:D); 
        del = radius(i);
        
        if rand < 0.5
            % Equation (1) in paper
            NP1 = repmat(P, D, 1) + diag(rand(D, 1)) .* del;
            NP2 = repmat(P, D, 1) - diag(rand(D, 1)) .* del;
            NP = [NP1; NP2];
        else
            % Generate the Binary Matrix
            B = BMatrixFn(D);
            % Equation (2) in paper
            B(B == 0) = -1;  % To aid in Equation (2)
            NP = B .* rand(2*D, D) .* del / sqrt(2) + repmat(P, 2*D, 1);
        end
        
        % Bounding the variables which are not in the bounds
        NP(NP < 0) = rand(length(find(NP < 0)), 1);
        NP(NP > 1) = rand(length(find(NP > 1)), 1);
        
        % Scaling the variables to its original domains
        ScaledNP = repmat(lb, 2*D, 1) + NP .* repmat((ub - lb), 2*D, 1);
        [Obj, FE] = calculate_fitness(ScaledNP', problem, FE);
        
        % Determining the best point from the 2D solutions created using the splitting stage
        [BestObj, ind] = min(Obj);
        P = NP(ind, :);
        Points(i, :) = [P BestObj];
    end
end

function [P, del] = Archivefn(Arch, P, del, alpha, D)
    Arch = [Arch; P];
    
    [~, ind] = min(Arch(:, D + 1));
    P(1, :) = Arch(ind, :);
    
    Arch(ind, D + 1) = NaN;
    
    [~, ind] = min(Arch(:, D + 1));
    P(2, :) = Arch(ind, :);
    
    % Equation (3) - Radius adjustment
    del(1) = del(1) - del(1) / alpha;
    del(2) = del(2) + del(2) / alpha;
    
    % del2 is to be capped at 0.75
    del(2) = min(del(2), 0.75);
end


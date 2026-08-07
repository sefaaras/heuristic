% ----------------------------------------------------------------------- %
% Crystal Structure Algorithm (CryStAl)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Cr_Number = 10   % Number of crystals (population)
%
% Algorithm Concept:
%   - Inspired by crystallography: crystals on Bravais lattice points
%   - Four "cubicle" update rules per crystal, driven by a main crystal, the
%     best crystal (Crb) and the mean of randomly selected crystals (Fc)
%
% Reference:
% Siamak Talatahari, Mahdi Azizi, Mohamad Tolouei, Babak Talatahari,
% Pooya Sareh,
% Crystal Structure Algorithm (CryStAl): A Metaheuristic Optimization Method,
% IEEE Access 9 (2021) 71244-71261.
% https://doi.org/10.1109/ACCESS.2021.3079161
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = crystal(problem)

    % Extract problem parameters
    Var_Number = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    Cr_Number = 10;                              % Maximum number of initial crystals
    MaxIteation = ceil(maxFE / (Cr_Number * 4)); % four evaluations per crystal per iteration

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    if length(LB) == 1
        LB = repmat(LB, 1, Var_Number);
    end
    if length(UB) == 1
        UB = repmat(UB, 1, Var_Number);
    end

    % Initialization
    Crystal = zeros(Cr_Number, Var_Number);
    Fun_eval = zeros(1, Cr_Number);
    for i = 1:Cr_Number
        Crystal(i, :) = unifrnd(LB, UB);
        [Fun_eval(i), FE] = calculate_fitness(Crystal(i, :)', problem, FE);
    end

    [BestFitness, idbest] = min(Fun_eval);
    Crb = Crystal(idbest, :);
    bsf = BestFitness;
    best_solution = Crb;

    for eval_count = 1:Cr_Number
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Crystal, Fun_eval, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Search process
    Iter = 1;
    while Iter <= MaxIteation && FE < maxFE
        for i = 1:Cr_Number
            % Main crystal
            Crmain = Crystal(randperm(Cr_Number, 1), :);
            % Random-selected crystals
            RandNumber = randperm(Cr_Number, 1);
            RandSelectCrystal = randperm(Cr_Number, RandNumber);
            % Mean of randomly-selected crystals
            Fc = mean(Crystal(RandSelectCrystal, :)) .* (length(RandSelectCrystal) ~= 1) ...
                + Crystal(RandSelectCrystal(1, 1), :) * (length(RandSelectCrystal) == 1);
            r = 2 * rand - 1; r1 = 2 * rand - 1;
            r2 = 2 * rand - 1; r3 = 2 * rand - 1;
            % New crystals
            NewCrystal(1, :) = Crystal(i, :) + r * Crmain;
            NewCrystal(2, :) = Crystal(i, :) + r1 * Crmain + r2 * Crb;
            NewCrystal(3, :) = Crystal(i, :) + r1 * Crmain + r2 * Fc;
            NewCrystal(4, :) = Crystal(i, :) + r1 * Crmain + r2 * Crb + r3 * Fc;

            for i2 = 1:4
                NewCrystal(i2, :) = bound(NewCrystal(i2, :), UB, LB);
                [Fun_evalNew_i2, FE] = calculate_fitness(NewCrystal(i2, :)', problem, FE);
                if Fun_evalNew_i2 < Fun_eval(i)
                    Fun_eval(i) = Fun_evalNew_i2;
                    Crystal(i, :) = NewCrystal(i2, :);
                end
                if Fun_evalNew_i2 < bsf
                    bsf = Fun_evalNew_i2;
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Crystal, Fun_eval, population_history, fitness_history, ...
                        history_index, maxFE);
                end
                if FE >= maxFE, break; end
            end
            if FE >= maxFE, break; end
        end
        Iter = Iter + 1;
        [BestFitness, idbest] = min(Fun_eval);
        Crb = Crystal(idbest, :);
        best_solution = Crb;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness = BestFitness;
end

% Boundary handling
function x = bound(x, UB, LB)
    x(x > UB) = UB(x > UB);
    x(x < LB) = LB(x < LB);
end

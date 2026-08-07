% ----------------------------------------------------------------------- %
% Sunflower Optimization (SFO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n = 100               % Population size (number of plants)
%   p = 0.6               % Pollination rate
%   m = 0.05              % Mortality rate
%
% Algorithm Concept:
%   - Inspired by the motion of sunflowers following the sun
%   - A fraction p of plants are recombined by pollination between neighbours
%   - Intermediate plants take an oriented step towards the best plant
%   - A fraction m of the worst plants die and are randomly regenerated
%
% Reference:
% Gabriel Filipe Costa Gomes, Sebastiao Simoes da Cunha, Antonio Carlos Ancelotti,
% A sunflower optimization (SFO) algorithm applied to damage identification on
% laminated composite plates,
% Engineering with Computers 35 (2019) 619-626
% https://doi.org/10.1007/s00366-018-0620-8
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sfo(problem)

    % Extract problem parameters
    d = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    n = 100;      % number of plants
    p = 0.6;      % pollination rate
    m = 0.05;     % mortality rate

    pol_end  = round(p * n);        % last index of the pollination band
    step_end = round(n * (1 - m));  % last index of the step band

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize the plants
    Plants = zeros(n, d);
    for i = 1:n
        Plants(i, :) = LB + (UB - LB) .* rand(1, d);
    end

    % An unevaluated plant keeps Inf, so it cannot pose as the best of the population
    Fitness = inf(n, 1);
    for i = 1:n
        [Fitness(i), FE] = calculate_fitness(Plants(i, :)', problem, FE);
        if FE >= 1 && FE <= maxFE
            curve(FE) = min(Fitness(1:i));
        end
        % Fitness is still Inf past i, so the field is only worth recording once
        % every plant has been evaluated
        if i == n && FE >= 1 && FE <= maxFE
            [population_history, fitness_history, history_index] = record_history(...
                FE, Plants, Fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    [fmin, I] = min(Fitness);
    best = Plants(I, :);

    while FE < maxFE
        for i = 1:n
            if FE >= maxFE
                break;
            end

            if i <= pol_end
                % pollination
                S_i = (Plants(i, :) - Plants(i + 1, :)) * rand(1) + Plants(i + 1, :);
            elseif i <= step_end
                % oriented step towards the best plant
                S_i = Plants(i, :) + rand * ((best - Plants(i, :)) / (norm((best - Plants(i, :)))));
            else
                % mortality: random regeneration
                S_i = zeros(1, d);
                for k = 1:length(LB)
                    S_i(k) = (UB(k) - LB(k)) * rand + LB(k);
                end
            end

            S_i = bound_check(S_i, LB, UB);

            [Fnew, FE] = calculate_fitness(S_i', problem, FE);
            if (Fnew <= Fitness(i))
                Plants(i, :) = S_i;
                Fitness(i) = Fnew;
            end
            if Fnew <= fmin
                best = S_i;
                fmin = Fnew;
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = fmin;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Plants, Fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    best_solution = best;
    best_fitness = fmin;

end

% Boundary Handling
function s = bound_check(s, LB, UB)
    ns_tmp = s;
    I = ns_tmp < LB;
    ns_tmp(I) = LB(I);
    J = ns_tmp > UB;
    ns_tmp(J) = UB(J);
    s = ns_tmp;
end

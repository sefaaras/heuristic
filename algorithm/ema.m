% ----------------------------------------------------------------------- %
% Exchange Market Algorithm (EMA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50          % Number of shareholders (population size)
%   g1 = g2 = [0.2 0.05]  % Maximum/minimum risk levels
%   Group ratios: 0.2 / 0.3 / 0.5 (non-oscillation),
%                 0.2 / 0.5 / 0.3 (oscillation)
%
% Algorithm Concept:
%   - Models the behaviour of shareholders trading in a stock market
%   - Non-oscillation mode: recombination and competitive search around the
%     elite shareholders
%   - Oscillation mode: risk-based buying/selling of shares controlled by a
%     linearly decreasing risk level
%
% Reference:
% Naser Ghorbani, Ebrahim Babaei,
% Exchange market algorithm,
% Applied Soft Computing 19 (2014) 177-187
% https://doi.org/10.1016/j.asoc.2014.02.006
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ema(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    popsize = 50;
    num_pop = popsize;
    num_par = dim;

    % Adjustable parameters of the algorithm (Table 2 of the paper)
    g1 = [2*10^-1 5*10^-2];
    g2 = [2*10^-1 5*10^-2];
    num_pop11 = 0.2 * num_pop;   % elite group (preserved) in non-oscillation
    num_pop12 = 0.3 * num_pop;
    num_pop21 = 0.2 * num_pop;   % elite group (preserved) in oscillation
    num_pop22 = 0.5 * num_pop;

    % Analogue of the original iteration budget (drives the risk schedule)
    num_iter = (maxFE / (popsize * 2)) + 1;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, num_pop, dim);
    fitness_history = zeros(history_size, num_pop);
    history_index = 1;

    % ---- First iteration: initialise and sort ----
    pop = zeros(num_pop, num_par);
    for i = 1:num_pop
        pop(i, :) = lb + rand(1, num_par) .* (ub - lb);
    end
    [FC, FE] = calculate_fitness(pop', problem, FE);
    FC = FC(:);
    [FC, index] = sort(FC);
    pop = pop(index, :);

    best_fitness = FC(1);
    best_solution = pop(1, :);

    for eval_count = 1:num_pop
        curve(eval_count) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, pop, FC, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    iteration = 1;
    while FE < maxFE
        iteration = iteration + 1;

        % ---- Non-oscillation mode (Section 3.1) ----
        pop = notoscillation(pop, num_pop, num_par, num_pop11, num_pop12);
        for ii = 1:num_pop
            pop(ii, :) = min(max(pop(ii, :), lb), ub);
        end
        FE_before = FE;
        [FC, FE] = calculate_fitness(pop', problem, FE);
        FC = FC(:);
        [FC, index] = sort(FC);
        pop = pop(index, :);
        if FC(1) < best_fitness
            best_fitness = FC(1);
            best_solution = pop(1, :);
        end
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, FC, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE
            break;
        end

        % ---- Oscillation mode (Section 3.2) ----
        pop = oscillation(pop, iteration, num_pop, num_par, num_iter, num_pop21, num_pop22, g1, g2);
        for ii = 1:num_pop
            pop(ii, :) = min(max(pop(ii, :), lb), ub);
        end
        FE_before = FE;
        [FC, FE] = calculate_fitness(pop', problem, FE);
        FC = FC(:);
        [FC, index] = sort(FC);
        pop = pop(index, :);
        if FC(1) < best_fitness
            best_fitness = FC(1);
            best_solution = pop(1, :);
        end
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, pop, FC, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

end

%% --- Non-oscillation trading (Eqs. 1-3) ---
function pop = notoscillation(pop, num_pop, num_par, num_pop11, num_pop12)
    comp = zeros(num_pop, num_par);
    for j = num_pop11 + 1:num_pop
        if j <= num_pop11 + num_pop12
            person(1) = ceil(num_pop11 * rand);
            person(2) = ceil(num_pop11 * rand);
            r1 = rand;
            pop(j, :) = r1 * pop(person(1), :) + (1 - r1) * pop(person(2), :);   % Eq. (1)
        else
            c1 = 2; c2 = 2;
            r1 = rand(1, num_par); r2 = rand(1, num_par);
            person(1) = ceil(num_pop11 * rand);
            person(2) = ceil(num_pop11 * rand);
            comp(j, :) = c1 * r1 .* (pop(person(1), :) - pop(j, :)) + c2 * r2 .* (pop(person(2), :) - pop(j, :));   % Eq. (2)
            pop(j, :) = pop(j, :) + 0.8 * comp(j, :);   % Eq. (3)
        end
    end
end

%% --- Oscillation trading (Eqs. 4-11) ---
function pop = oscillation(pop, iteration, num_pop, num_par, num_iter, num_pop21, num_pop22, g1, g2)
    for j = num_pop21 + 1:num_pop
        risk  = g1(1, 1) - (g1(1, 1) - g1(1, 2)) * (iteration / num_iter);   % Eq. (8)
        risk2 = g2(1, 1) - (g2(1, 1) - g2(1, 2)) * (iteration / num_iter);
        u = j / num_pop;                 % Eq. (5)
        nt1 = sum(abs(pop(j, :)));        % Eq. (6)
        ntt = sum(pop(j, :));
        pg = ntt;                         % market information (Section 3.2.2.1)

        if j <= num_pop22 + num_pop21     % trading stocks in group 2
            % buying new stocks
            dnt1 = abs(ntt - pg + (2 * rand * u * risk * nt1));   % Eq. (4)
            tt = ceil(rand * rand * num_par);
            xx1 = rand * rand; x2n0 = rand(1, tt - 1); sumx2n0 = sum(x2n0);
            xx2 = [xx1, (1 - xx1) * x2n0 / sumx2n0]; xx3 = xx2 * dnt1;
            for h = 1:tt
                g = ceil(rand * num_par);
                pop(j, g) = pop(j, g) + xx3(1, h);
            end
            % selling new stocks
            nt2 = sum(pop(j, :)); dnt2 = abs(nt2 - pg);   % Eq. (9)
            an = nt2 - pg;
            tt = ceil(rand * rand * num_par);
            xx1 = rand * rand; x2n0 = rand(1, tt - 1); sumx2n0 = sum(x2n0);
            xx2 = [xx1, (1 - xx1) * x2n0 / sumx2n0]; xx3 = xx2 * dnt2;
            for h = 1:tt
                g = ceil(rand * num_par);
                if an > 0
                    d11 = pop(j, g) - xx3(1, h);
                else
                    d11 = pop(j, g) + xx3(1, h);
                end
                pop(j, g) = d11;
            end
        else                              % trading stock in group 3
            rs = 0.5 - rand;                          % Eq. (11)
            dnt3 = (4 * rs * u * risk2 * nt1);        % Eq. (10)
            tt = ceil(rand * rand * num_par);
            xx1 = rand * rand; x2n0 = rand(1, tt - 1); sumx2n0 = sum(x2n0);
            xx2 = [xx1, (1 - xx1) * x2n0 / sumx2n0]; xx3 = xx2 * dnt3;
            for h = 1:tt
                g = ceil(rand * num_par);
                pop(j, g) = pop(j, g) + xx3(1, h);
            end
        end
    end
end

% ----------------------------------------------------------------------- %
% Spider Wasp Optimizer (SWO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 100  % Initial population size (spider wasps)
%   TR    = 0.3   % Trade-off probability (hunting vs mating)
%   Cr    = 0.2   % Crossover probability
%   N_min = 20    % Minimum population size (population reduction, Eq.25)
%
% Algorithm Concept:
%   - Hunting/nesting behavior: searching, following, escaping stages
%   - Mating behavior: crossover of a high-quality "male" spider wasp
%   - Linear population reduction from 100 down to N_min
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Mohammed Jameel,
% Mohamed Abouhawwash,
% Spider wasp optimizer: a novel meta-heuristic optimization algorithm,
% Artificial Intelligence Review 56 (2023) 11675-11738.
% https://doi.org/10.1007/s10462-023-10446-y
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = swo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 100;
    Tmax = maxFE;
    TR = 0.3;
    Cr = 0.2;
    N_min = 20;

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    Positions = initialization(SearchAgents_no, dim, ub, lb);
    [SW_Fit, FE] = calculate_fitness(Positions', problem, FE);
    SW_Fit = SW_Fit(:)';

    [Best_score, bidx] = min(SW_Fit);
    Best_SW = Positions(bidx, :);

    for e = 1:SearchAgents_no
        if e <= maxFE
            curve(e) = Best_score;
            [population_history, fitness_history, history_index] = record_history(...
                e, Positions, SW_Fit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    N = SearchAgents_no;
    while FE < maxFE
        frac = FE / Tmax;                     % progress ratio (analogous to t/Tmax)
        a  = 2 - 2 * frac;                    % 2 -> 0
        a2 = -1 + (-1 * frac);                % -1 -> -2
        k  = 1 - frac;                        % 1 -> 0
        JK = randperm(N);

        if rand < TR
            %% Hunting and nesting behavior
            for i = 1:N
                r1 = rand(); r2 = rand(); r3 = rand(); p = rand();
                C = a * (2 * r1 - 1);
                l = (a2 - 1) * rand + 1;
                L = Levy(1);
                vc = unifrnd(-k, k, 1, dim);
                rn1 = randn;
                O_P = Positions(i, :);
                for j = 1:dim
                    if i < k * N
                        if p < (1 - frac)      % Searching stage (exploration)
                            if r1 < r2
                                m1 = abs(rn1) * r1;
                                Positions(i, j) = Positions(i, j) + m1 * (Positions(JK(1), j) - Positions(JK(2), j));
                            else
                                B = 1 / (1 + exp(l));
                                m2 = B * cos(l * 2 * pi);
                                Positions(i, j) = Positions(JK(i), j) + m2 * (lb(j) + rand * (ub(j) - lb(j)));
                            end
                        else                    % Following and escaping stage
                            if r1 < r2
                                Positions(i, j) = Positions(i, j) + C * abs(2 * rand * Positions(JK(3), j) - Positions(i, j));
                            else
                                Positions(i, j) = Positions(i, j) .* vc(j);
                            end
                        end
                    else
                        if r1 < r2
                            Positions(i, j) = Best_SW(j) + cos(2 * l * pi) * (Best_SW(j) - Positions(i, j));
                        else
                            Positions(i, j) = Positions(JK(1), j) + r3 * abs(L) * (Positions(JK(1), j) - Positions(i, j)) + (1 - r3) * (rand > rand) * (Positions(JK(3), j) - Positions(JK(2), j));
                        end
                    end
                end
                Positions(i, :) = reflect_bounds(Positions(i, :), ub, lb);
                [SW_Fit1, FE] = calculate_fitness(Positions(i, :)', problem, FE);
                if SW_Fit1 < SW_Fit(i)
                    SW_Fit(i) = SW_Fit1;
                    if SW_Fit(i) < Best_score
                        Best_score = SW_Fit(i);
                        Best_SW = Positions(i, :);
                    end
                else
                    Positions(i, :) = O_P;
                end
                if FE <= maxFE
                    curve(FE) = Best_score;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Positions, SW_Fit, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                if FE >= maxFE, break; end
            end
        else
            %% Mating behavior
            for i = 1:N
                l = (a2 - 1) * rand + 1;
                SW_m = zeros(1, dim);
                O_P = Positions(i, :);
                if SW_Fit(JK(1)) < SW_Fit(i)
                    v1 = Positions(JK(1), :) - Positions(i, :);
                else
                    v1 = Positions(i, :) - Positions(JK(1), :);
                end
                if SW_Fit(JK(2)) < SW_Fit(JK(3))
                    v2 = Positions(JK(2), :) - Positions(JK(3), :);
                else
                    v2 = Positions(JK(3), :) - Positions(JK(2), :);
                end
                rn1 = randn; rn2 = randn;
                for j = 1:dim
                    SW_m(j) = Positions(i, j) + (exp(l)) * abs(rn1) * v1(j) + (1 - exp(l)) * abs(rn2) * v2(j);
                    if rand < Cr
                        Positions(i, j) = SW_m(j);
                    end
                end
                Positions(i, :) = reflect_bounds(Positions(i, :), ub, lb);
                [SW_Fit1, FE] = calculate_fitness(Positions(i, :)', problem, FE);
                if SW_Fit1 < SW_Fit(i)
                    SW_Fit(i) = SW_Fit1;
                    if SW_Fit(i) < Best_score
                        Best_score = SW_Fit(i);
                        Best_SW = Positions(i, :);
                    end
                else
                    Positions(i, :) = O_P;
                end
                if FE <= maxFE
                    curve(FE) = Best_score;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, Positions, SW_Fit, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
                if FE >= maxFE, break; end
            end
        end

        %% Population reduction (Eq.25)
        N = fix(N_min + (N - N_min) * ((Tmax - FE) / Tmax));
        N = max(N, N_min);
    end

    curve(min(FE, maxFE):end) = Best_score;
    best_fitness = Best_score;
    best_solution = Best_SW;
end

%% --- Levy sample (1 x d) ---
function L = Levy(d)
    beta = 3 / 2;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    L = 0.05 * step;
end

%% --- Boundary relocation (random re-init of out-of-range dims) ---
function x = reflect_bounds(x, ub, lb)
    for j = 1:numel(x)
        if x(j) > ub(j) || x(j) < lb(j)
            x(j) = lb(j) + rand * (ub(j) - lb(j));
        end
    end
end

%% --- Initialization ---
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = length(ub);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

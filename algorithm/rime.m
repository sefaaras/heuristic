% ----------------------------------------------------------------------- %
% RIME Optimization Algorithm (RIME) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30   % Population size (rime-agents)
%   W = 5    % Soft-rime parameter (subsection 4.3.1 of the paper)
%
% Algorithm Concept:
%   - Soft-rime search strategy (exploration) driven by RimeFactor
%   - Hard-rime puncture mechanism (exploitation) using normalized rates
%   - Positive greedy selection between old and new rime agents
%
% Reference:
% Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang,
% Majdi Mafarja, Huiling Chen,
% RIME: A physics-based optimization,
% Neurocomputing 532 (2023) 183-214.
% https://doi.org/10.1016/j.neucom.2023.02.010
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rime(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;                        % Number of search agents
    W = 5;                         % Soft-rime parameter
    Max_iter = ceil(maxFE / N);    % Iteration budget (for RimeFactor / E)

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, dim);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    % Initialize the set of random solutions
    Rimepop = initialization(N, dim, ub, lb);

    [Rime_rates, FE] = calculate_fitness(Rimepop', problem, FE);
    Rime_rates = Rime_rates(:)';   % 1 x N row

    [bsf, bidx] = min(Rime_rates);
    best_pos = Rimepop(bidx, :);

    for e = 1:N
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, Rimepop, Rime_rates, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    t = 0;
    while FE < maxFE
        t = t + 1;
        RimeFactor = (rand - 0.5) * 2 * cos((pi * t / (Max_iter / 10))) * (1 - round(t * W / Max_iter) / W); % Eq.(3),(4),(5)
        E = sqrt(t / Max_iter);    % Eq.(6)
        newRimepop = Rimepop;
        normalized_rime_rates = normr_local(Rime_rates); % Eq.(7)
        for i = 1:N
            for j = 1:dim
                % Soft-rime search strategy
                if rand() < E
                    newRimepop(i, j) = best_pos(1, j) + RimeFactor * ((ub(j) - lb(j)) * rand + lb(j)); % Eq.(3)
                end
                % Hard-rime puncture mechanism
                if rand() < normalized_rime_rates(i)
                    newRimepop(i, j) = best_pos(1, j); % Eq.(7)
                end
            end
        end

        % Boundary absorption
        for i = 1:N
            newRimepop(i, :) = bound(newRimepop(i, :), ub, lb);
        end

        [newRime_rates, FE] = calculate_fitness(newRimepop', problem, FE);
        newRime_rates = newRime_rates(:)';

        for i = 1:N
            % Positive greedy selection mechanism
            if newRime_rates(i) < Rime_rates(i)
                Rime_rates(i) = newRime_rates(i);
                Rimepop(i, :) = newRimepop(i, :);
                if Rime_rates(i) < bsf
                    bsf = Rime_rates(i);
                    best_pos = Rimepop(i, :);
                end
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Rimepop, Rime_rates, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end

        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

%% --- Row normalization (avoids Neural Network Toolbox dependency on normr) ---
function y = normr_local(x)
    % Normalize the row vector x to unit L2 norm (identical to normr for 1 row)
    nrm = sqrt(sum(x.^2, 2));
    if nrm == 0
        y = x;
    else
        y = x ./ nrm;
    end
end

%% --- Initialization ---
function Positions = initialization(N, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(N, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(N, dim);
        for i = 1:dim
            Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

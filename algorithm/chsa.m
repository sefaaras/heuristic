% ----------------------------------------------------------------------- %
% Chameleon Swarm Algorithm (CSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% NOTE: The original acronym is "CSA"; it is stored here as `chsa` because
% the CSA name is already used by another algorithm in this suite.
%
% Algorithm Parameters:
%   searchAgents = 30
%   rho = 1.0, gamma = 2.0, alpha = 4.0, beta = 3.0  (rotation/velocity)
%
% Algorithm Concept:
%   - Models chameleon foraging: tracking prey, ~360 deg eye rotation for
%     search, and a high-speed sticky-tongue prey capture (velocity update)
%
% Reference:
% Malik Shehadeh Braik,
% Chameleon Swarm Algorithm: A bio-inspired optimizer for solving
% engineering design problems,
% Expert Systems with Applications 174 (2021) 114685.
% https://doi.org/10.1016/j.eswa.2021.114685
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = chsa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    searchAgents = 30;
    iteMax = ceil(maxFE / searchAgents);

    if size(ub, 2) == 1
        ub = ones(1, dim) * ub;
        lb = ones(1, dim) * lb;
    end

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, searchAgents, dim);
    fitness_history = zeros(history_size, searchAgents);
    history_index = 1;

    %% Initial population
    chameleonPositions = initialization(searchAgents, dim, ub, lb);

    fit = zeros(searchAgents, 1);
    [fit(:), FE] = calculate_fitness(chameleonPositions', problem, FE);
    fit = fit(:);

    fitness = fit;
    [fmin0, index] = min(fit);

    chameleonBestPosition = chameleonPositions;
    gPosition = chameleonPositions(index, :);

    v = 0.1 * chameleonBestPosition;
    v0 = 0.0 * v;

    bsf = fmin0;
    for eval_count = 1:searchAgents
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, chameleonPositions, fit', population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    %% CSA main parameters
    rho = 1.0;
    p1 = 2.0;
    p2 = 2.0;
    c1 = 2.0;
    c2 = 1.80; %#ok<NASGU>
    gamma = 2.0;
    alpha = 4.0;
    beta = 3.0;

    %% Start CSA
    for t = 1:iteMax
        if FE >= maxFE, break; end

        a = 2590 * (1 - exp(-log(t)));
        omega = (1 - (t / iteMax))^(rho * sqrt(t / iteMax));
        p1 = 2 * exp(-2 * (t / iteMax)^2);
        p2 = 2 / (1 + exp((-t + iteMax / 2) / 100));
        mu = gamma * exp(-(alpha * t / iteMax)^beta);

        ch = ceil(searchAgents * rand(1, searchAgents));

        % Update the position of CSA (exploration)
        for i = 1:searchAgents
            if rand >= 0.1
                chameleonPositions(i, :) = chameleonPositions(i, :) + p1 * (chameleonBestPosition(ch(i), :) - chameleonPositions(i, :)) * rand() + ...
                    + p2 * (gPosition - chameleonPositions(i, :)) * rand();
            else
                for j = 1:dim
                    chameleonPositions(i, j) = gPosition(j) + mu * ((ub(j) - lb(j)) * rand + lb(j)) * sign(rand - 0.50);
                end
            end
        end

        % Chameleon velocity updates and find a food source
        for i = 1:searchAgents
            v(i, :) = omega * v(i, :) + p1 * (chameleonBestPosition(i, :) - chameleonPositions(i, :)) * rand + ...
                + p2 * (gPosition - chameleonPositions(i, :)) * rand;
            chameleonPositions(i, :) = chameleonPositions(i, :) + (v(i, :).^2 - v0(i, :).^2) / (2 * a);
        end
        v0 = v;

        % Handling boundary violations
        for i = 1:searchAgents
            if chameleonPositions(i, :) < lb
                chameleonPositions(i, :) = lb;
            elseif chameleonPositions(i, :) > ub
                chameleonPositions(i, :) = ub;
            end
        end

        % Relocation of chameleon positions (randomization) and evaluation
        for i = 1:searchAgents
            ub_ = sign(chameleonPositions(i, :) - ub) > 0;
            lb_ = sign(chameleonPositions(i, :) - lb) < 0;
            chameleonPositions(i, :) = (chameleonPositions(i, :) .* (~xor(lb_, ub_))) + ub .* ub_ + lb .* lb_;

            [fit(i, 1), FE] = calculate_fitness(chameleonPositions(i, :)', problem, FE);

            if fit(i) < fitness(i)
                chameleonBestPosition(i, :) = chameleonPositions(i, :);
                fitness(i) = fit(i);
            end

            if fit(i) < bsf
                bsf = fit(i);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, chameleonPositions, fit', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end

        [fmin, index] = min(fitness);
        if fmin < fmin0
            gPosition = chameleonBestPosition(index, :);
            fmin0 = fmin;
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    ngPosition = find(fitness == min(fitness));
    best_solution = chameleonBestPosition(ngPosition(1), :);
    best_fitness = fmin0;
end

%% --- Initialization ---
function pos = initialization(searchAgents, dim, u, l)
    Boundary_no = size(u, 2);
    if Boundary_no == 1
        u_new = ones(1, dim) * u;
        l_new = ones(1, dim) * l;
    else
        u_new = u;
        l_new = l;
    end
    for i = 1:dim
        u_i = u_new(i);
        l_i = l_new(i);
        pos(:, i) = rand(searchAgents, 1) .* (u_i - l_i) + l_i;
    end
end

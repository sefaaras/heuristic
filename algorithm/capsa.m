% ----------------------------------------------------------------------- %
% Capuchin Search Algorithm (CapSA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   noP = 100       % Population size (capuchins)
%   bf  = 0.70      % Balance factor
%   cr  = 11.0      % Modulus of elasticity
%   g   = 9.81      % Gravity
%   a1 = 1.25, a2 = 1.5  % Velocity coefficients
%
% Algorithm Concept:
%   - Physics-based model of capuchin locomotion: leaping (projection),
%     jumping (land), ground movement, swinging, and climbing
%   - Population split into leaders (first half) and followers (second half)
%
% Reference:
% Malik Braik, Alaa Sheta, Heba Al-Hiary,
% A novel meta-heuristic search algorithm for solving optimization problems:
% capuchin search algorithm,
% Neural Computing and Applications 33 (7) (2021) 2515-2547.
% https://doi.org/10.1007/s00521-020-05145-6
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = capsa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    LB = problem.lb;
    UB = problem.ub;
    maxFE = problem.maxFe;

    noP = 100;
    maxite = ceil(maxFE / noP);

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, noP, dim);
    fitness_history = zeros(history_size, noP);
    history_index = 1;

    %% CapSA initialization
    CapPos = initialization(noP, dim, UB, LB);

    v = 0.1 * CapPos;          % initial velocity
    v0 = zeros(noP, dim);
    CapFit = zeros(noP, 1);

    [CapFit(:), FE] = calculate_fitness(CapPos', problem, FE);
    CapFit = CapFit(:);

    Fit = CapFit;
    [fitCapSA, index] = min(CapFit);

    CapBestPos = CapPos;        % best position initialization
    Pos = CapPos;
    gFoodPos = CapPos(index, :); % initial global position

    bsf = fitCapSA;
    for eval_count = 1:noP
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, CapPos, CapFit', population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    %% CapSA parameters
    bf = 0.70;   % Balance factor
    cr = 11.0;   % Modulus of elasticity
    g = 9.81;
    a1 = 1.250; a2 = 1.5;
    beta = [2 11 2];
    wmax = 0.8;
    wmin = 0.1;

    %% CapSA main loop
    for t = 1:maxite
        if FE >= maxFE, break; end

        tau = beta(1) * exp(-beta(2) * t / maxite)^beta(3);   % life-time convergence
        w = wmax - (wmax - wmin) * (t / maxite);
        fol = ceil((noP - 1) .* rand(noP, 1))';

        % CapSA velocity update
        for i = 1:noP
            for j = 1:dim
                v(i, j) = w * v(i, j) + ...
                    a1 * (CapBestPos(i, j) - CapPos(i, j)) * rand + ...
                    a2 * (gFoodPos(j) - CapPos(i, j)) * rand;
            end
        end

        % CapSA position update
        for i = 1:noP
            if i < noP / 2
                if (rand() >= 0.1)
                    r = rand;
                    if r <= 0.15
                        CapPos(i, :) = gFoodPos + bf * ((v(i, :).^2) * sin(2 * rand() * 1.5)) / g;        % Jumping (Projection)
                    elseif r > 0.15 && r <= 0.30
                        CapPos(i, :) = gFoodPos + cr * bf * ((v(i, :).^2) * sin(2 * rand() * 1.5)) / g;   % Jumping (Land)
                    elseif r > 0.30 && r <= 0.9
                        CapPos(i, :) = CapPos(i, :) + v(i, :);                                            % Movement on the ground
                    elseif r > 0.9 && r <= 0.95
                        CapPos(i, :) = gFoodPos + bf * sin(rand() * 1.5);                                 % Swing (local search)
                    elseif r > 0.95
                        CapPos(i, :) = gFoodPos + bf * (v(i, :) - v0(i, :));                              % Climbing (local search)
                    end
                else
                    CapPos(i, :) = tau * (LB + rand * (UB - LB));
                end
            elseif i >= noP / 2 && i <= noP
                eps_ = ((rand() + 2 * rand()) - (3 * rand())) / (1 + rand());
                Pos(i, :) = gFoodPos + 2 * (CapBestPos(fol(i), :) - CapPos(i, :)) * eps_ + ...
                    2 * (CapPos(i, :) - CapBestPos(i, :)) * eps_;
                CapPos(i, :) = (Pos(i, :) + CapPos(i - 1, :)) / (2);
            end
        end
        v0 = v;

        % relocation (update, exploration) and evaluation
        for i = 1:noP
            u = UB - CapPos(i, :) < 0;
            l = LB - CapPos(i, :) > 0;
            CapPos(i, :) = LB .* l + UB .* u + CapPos(i, :) .* ~xor(u, l);

            [CapFit(i, 1), FE] = calculate_fitness(CapPos(i, :)', problem, FE);

            if CapFit(i, 1) < Fit(i)
                CapBestPos(i, :) = CapPos(i, :);
                Fit(i) = CapFit(i, 1);
            end

            if CapFit(i, 1) < bsf
                bsf = CapFit(i, 1);
            end
            if FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, CapPos, CapFit', population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE, break; end
        end

        % finding out the best position; update global position and best fitness
        [fmin, index] = min(Fit);
        if fmin < fitCapSA
            gFoodPos = CapBestPos(index, :);
            fitCapSA = fmin;
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness = fitCapSA;
    best_solution = gFoodPos;
end

%% --- Initialization ---
function pos = initialization(searchAgents, dim, u, l)
    Boundary_no = size(u, 2);
    if Boundary_no == 1
        u = ones(1, dim) * u;
        l = ones(1, dim) * l;
    end
    for i = 1:dim
        u_i = u(i);
        l_i = l(i);
        pos(:, i) = rand(searchAgents, 1) .* (u_i - l_i) + l_i;
    end
end

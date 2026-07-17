% ----------------------------------------------------------------------- %
% Kepler Optimization Algorithm (KOA) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 25   % Population size (planets)
%   Tc = 3, M0 = 0.1, lambda = 15   % Controlling parameters
%
% Algorithm Concept:
%   - Physics-based: each planet orbits the best-so-far solution (the Sun)
%   - Operators derived from Kepler's laws (gravitational force, mass,
%     orbital velocity, semimajor axis) drive exploration/exploitation
%   - Elitism keeps each planet only if its new position is not worse
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Shaimaa A. Abdel Azeem,
% Mohammed Jameel, Mohamed Abouhawwash,
% Kepler optimization algorithm: A new metaheuristic algorithm inspired by
% Kepler's laws of planetary motion,
% Knowledge-Based Systems 268 (2023) 110454.
% https://doi.org/10.1016/j.knosys.2023.110454
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = koa(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    SearchAgents_no = 25;   % Number of search agents (Planets)
    Tmax = maxFE;           % Maximum number of function evaluations

    FE = 0;
    curve = zeros(1, maxFE);
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, SearchAgents_no, dim);
    fitness_history = zeros(history_size, SearchAgents_no);
    history_index = 1;

    Sun_Pos = zeros(1, dim);   % best-so-far (the Sun)
    Sun_Score = inf;

    %% Controlling parameters
    Tc = 3;
    M0 = 0.1;
    lambda = 15;

    %% Initialization
    orbital = rand(1, SearchAgents_no);        % Orbital Eccentricity (Eq. 4)
    T = abs(randn(1, SearchAgents_no));         % Orbital Period (Eq. 5)
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    t = 0;                                       % Function evaluation counter

    [PL_Fit, FE] = calculate_fitness(Positions', problem, FE);
    PL_Fit = PL_Fit(:)';
    for i = 1:SearchAgents_no
        if PL_Fit(i) < Sun_Score
            Sun_Score = PL_Fit(i);
            Sun_Pos = Positions(i, :);
        end
    end

    for eval_count = 1:SearchAgents_no
        if eval_count <= maxFE
            curve(eval_count) = Sun_Score;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Positions, PL_Fit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    while FE < maxFE
        [Order] = sort(PL_Fit);
        worstFitness = Order(SearchAgents_no);        % Eq. (11)
        M = M0 * (exp(-lambda * (t / Tmax)));          % Eq. (12)

        % Euclidean distance between the Sun and the ith solution
        R = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            R(i) = 0;
            for j = 1:dim
                R(i) = R(i) + (Sun_Pos(j) - Positions(i, j))^2;   % Eq. (7)
            end
            R(i) = sqrt(R(i));
        end

        % Mass of the Sun and object i
        MS = zeros(1, SearchAgents_no);
        m = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            sum_ = 0;
            for k = 1:SearchAgents_no
                sum_ = sum_ + (PL_Fit(k) - worstFitness);
            end
            MS(i) = rand * (Sun_Score - worstFitness) / (sum_);   % Eq. (8)
            m(i) = (PL_Fit(i) - worstFitness) / (sum_);           % Eq. (9)
        end

        % Gravitational force F
        Rnorm = zeros(1, SearchAgents_no);
        MSnorm = zeros(1, SearchAgents_no);
        Mnorm = zeros(1, SearchAgents_no);
        Fg = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            Rnorm(i) = (R(i) - min(R)) / (max(R) - min(R));       % Eq. (24)
            MSnorm(i) = (MS(i) - min(MS)) / (max(MS) - min(MS));
            Mnorm(i) = (m(i) - min(m)) / (max(m) - min(m));
            Fg(i) = orbital(i) * M * ((MSnorm(i) * Mnorm(i)) / (Rnorm(i) * Rnorm(i) + eps)) + (rand);   % Eq. (6)
        end

        % Semimajor axis
        a1 = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            a1(i) = rand * (T(i)^2 * (M * (MS(i) + m(i)) / (4 * pi * pi)))^(1 / 3);   % Eq. (23)
        end

        V = zeros(SearchAgents_no, dim);
        for i = 1:SearchAgents_no
            a2 = -1 + -1 * (rem(t, Tmax / Tc) / (Tmax / Tc));   % Eq. (29)
            n = (a2 - 1) * rand + 1;                             % Eq. (28)
            a = randi(SearchAgents_no);
            b = randi(SearchAgents_no);
            rd = rand(1, dim);
            r = rand;
            U1 = rd < r;                                         % Eq. (21)
            O_P = Positions(i, :);

            if rand < rand
                h = (1 / (exp(n .* randn)));                     % Eq. (27)
                Xm = (Positions(b, :) + Sun_Pos + Positions(i, :)) / 3.0;
                Positions(i, :) = Positions(i, :) .* U1 + (Xm + h .* (Xm - Positions(a, :))) .* (1 - U1);   % Eq. (26)
            else
                if rand < 0.5                                    % Eq. (18)
                    f = 1;
                else
                    f = -1;
                end
                L = (M * (MS(i) + m(i)) * abs((2 / (R(i) + eps)) - (1 / (a1(i) + eps))))^(0.5);   % Eq. (15)
                U = rd > rand(1, dim);
                if Rnorm(i) < 0.5                                % Eq. (13)
                    M = (rand .* (1 - r) + r);                   % Eq. (16)
                    l = L * M * U;                               % Eq. (14)
                    Mv = (rand * (1 - rd) + rd);                 % Eq. (20)
                    l1 = L .* Mv .* (1 - U);                     % Eq. (19)
                    V(i, :) = l .* (2 * rand * Positions(i, :) - Positions(a, :)) + l1 .* (Positions(b, :) - Positions(a, :)) + (1 - Rnorm(i)) * f * U1 .* rand(1, dim) .* (ub - lb);   % Eq. (13a)
                else
                    U2 = rand > rand;                            % Eq. (22)
                    V(i, :) = rand .* L .* (Positions(a, :) - Positions(i, :)) + (1 - Rnorm(i)) * f * U2 * rand(1, dim) .* (rand * ub - lb);   % Eq. (13b)
                end

                if rand < 0.5                                    % Eq. (18)
                    f = 1;
                else
                    f = -1;
                end
                Positions(i, :) = ((Positions(i, :) + V(i, :) .* f) + (Fg(i) + abs(randn)) * U .* (Sun_Pos - Positions(i, :)));   % Eq. (25)
            end

            % Return search agents that exceed the bounds
            if rand < rand
                for j = 1:size(Positions, 2)
                    if Positions(i, j) > ub(j)
                        Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
                    elseif Positions(i, j) < lb(j)
                        Positions(i, j) = lb(j) + rand * (ub(j) - lb(j));
                    end
                end
            else
                Positions(i, :) = min(max(Positions(i, :), lb), ub);
            end

            [PL_Fit1, FE] = calculate_fitness(Positions(i, :)', problem, FE);
            % Elitism (Eq. 30)
            if PL_Fit1 < PL_Fit(i)
                PL_Fit(i) = PL_Fit1;
                if PL_Fit(i) < Sun_Score
                    Sun_Score = PL_Fit(i);
                    Sun_Pos = Positions(i, :);
                end
            else
                Positions(i, :) = O_P;
            end

            t = t + 1;

            if FE <= maxFE
                curve(FE) = Sun_Score;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Positions, PL_Fit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(FE, maxFE):end) = Sun_Score;

    best_fitness = Sun_Score;
    best_solution = Sun_Pos;
end

%% --- Initialization ---
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = length(ub);
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

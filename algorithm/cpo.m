% ----------------------------------------------------------------------- %
% Crested Porcupine Optimizer (CPO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N0    = 130   % Initial population size N'
%   N_min = 120   % Floor of the cyclic population reduction
%   T     = 2     % Number of reduction cycles over the budget
%   alpha = 0.2   % Convergence rate of the physical attack
%   Tf    = 0.8   % Probability of odor (else physical attack) when exploiting
%
% Algorithm Concept:
%   - Porcupines update one at a time; tau8 < tau9 explores, otherwise exploits
%   - Sight: x + randn*|2*rand*best - y|, y the midpoint of x and a random peer
%   - Sound: masked dimensions move to y plus a scaled difference of two peers
%   - Odor: masked dimensions jump to a peer plus S*(peer difference), with the
%     odor factor S = exp(f_i / sum f)
%   - Physical attack: best + (alpha*(1-r2) + r2)*(delta*best - x) - force term
%   - Greedy replacement per porcupine; out-of-box coordinates are re-drawn uniformly
%   - Cyclic population reduction from N' towards N_min, restarted T times
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Mohamed Abouhawwash,
% Crested Porcupine Optimizer: A new nature-inspired metaheuristic,
% Knowledge-Based Systems 284 (2024) 111257.
% https://doi.org/10.1016/j.knosys.2023.111257
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (CPO.m, main.m, "source code demo 1.0") through its
% verbatim mirror github.com/NourElDmerdash/CPO (commit ac600d9), since the paper's
% MATLAB Drive link needs a login; seeded runs reproduce the release exactly (9/9).
% Parameters are the release's; later studies quote N_min = 80, alpha = 0.1, Tf = 0.5.
% Kept as released: rand(1,dim) < 0.5*2-1 is never true, so delta = 0 (no S or force
% term; the attack is best - c*x), and rem(t,Tmax/T)/Tmax/T divides by Tmax*T, so the
% population only falls a quarter of the way to N_min (130 -> 127). Fixing both is
% no clear gain (median error over 5 seeds, cec14 D=10 at 1e5 FE, RW F5 at 2e4 FE):
%      kept   F1 252   F4 0.073   F6 0.0037   RW F5 violation 33.2 (infeasible)
%      fixed  F1 352   F4 0.042   F6 0.0095   RW F5 violation 24.4 (infeasible)
% Tmax = maxFE - N0 - 1 ends the release's FE schedule on the budget, its stop at the
% known optimum is dropped, and NaN coordinates are re-drawn like out-of-box ones.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cpo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N0    = 130;
    N_min = 120;
    T     = 2;
    alpha = 0.2;
    Tf    = 0.8;
    Tmax  = max(1, maxFE - N0 - 1);   % the release counts Tmax + 1 evaluations after initialisation

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = initialization(N0, dim, ub, lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';
    [Gb_Fit, index] = min(fit);
    Gb_Sol = X(index, :);

    for e = 1:N0
        if e <= maxFE
            curve(e) = Gb_Fit;
            [population_history, fitness_history, history_index] = record_history(...
                e, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % X always holds the personal bests (the release's Xp): a rejected move is undone
    Pop_size = N0;
    t = 0;
    while FE < maxFE
        r2 = rand;
        for i = 1:Pop_size
            U1 = rand(1, dim) > rand;
            if rand < rand
                if rand < rand
                    % First defense mechanism (sight)
                    y = (X(i, :) + X(randi(Pop_size), :)) / 2;
                    xn = X(i, :) + randn .* abs(2 * rand * Gb_Sol - y);
                else
                    % Second defense mechanism (sound)
                    y = (X(i, :) + X(randi(Pop_size), :)) / 2;
                    xn = U1 .* X(i, :) + (1 - U1) .* (y + rand * (X(randi(Pop_size), :) - X(randi(Pop_size), :)));
                end
            else
                Yt = 2 * rand * (1 - t / Tmax) ^ (t / Tmax);
                U2 = rand(1, dim) < 0;   % delta: the release's rand < 0.5*2-1, never true (kept)
                S = rand * U2;
                if rand < Tf
                    % Third defense mechanism (odor); sum over all N0, dormant porcupines included
                    St = exp(fit(i) / (sum(fit) + eps));
                    S = S .* Yt .* St;
                    xn = (1 - U1) .* X(i, :) + U1 .* (X(randi(Pop_size), :) + St * (X(randi(Pop_size), :) - X(randi(Pop_size), :)) - S);
                else
                    % Fourth defense mechanism (physical attack)
                    Mt = exp(fit(i) / (sum(fit) + eps));
                    vt = X(i, :);
                    Vtp = X(randi(Pop_size), :);
                    Ft = rand(1, dim) .* (Mt * (-vt + Vtp));
                    S = S .* Yt .* Ft;
                    xn = (Gb_Sol + (alpha * (1 - r2) + r2) * (U2 .* Gb_Sol - X(i, :))) - S;
                end
            end
            % The negated test also catches NaN, which the release's > ub / < lb tests miss
            out = ~(xn >= lb & xn <= ub);
            xn(out) = lb(out) + rand(1, nnz(out)) .* (ub(out) - lb(out));

            [nF, FE] = calculate_fitness(xn', problem, FE);
            if ~(fit(i) < nF)
                X(i, :) = xn;
                fit(i) = nF;
                if fit(i) <= Gb_Fit
                    Gb_Sol = xn;
                    Gb_Fit = nF;
                end
            end
            t = t + 1;
            if FE <= maxFE
                curve(FE) = Gb_Fit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X(1:Pop_size, :), fit(1:Pop_size), population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        % Cyclic population reduction, with the release's /Tmax/T kept (see the note)
        Pop_size = fix(N_min + (N0 - N_min) * (1 - (rem(t, Tmax / T) / Tmax / T)));
    end

    curve(min(FE, maxFE):end) = Gb_Fit;
    best_fitness  = Gb_Fit;
    best_solution = Gb_Sol;
end

% Initialization, column by column as in the release
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

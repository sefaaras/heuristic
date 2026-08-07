% ----------------------------------------------------------------------- %
% Adaptive Gold Rush Optimizer (AGRO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N             = 30      % Population size (gold prospectors)
%   sigma_initial = 2       % Initial scale of the l1/l2 decay
%   sigma_final   = 1/Max_iter
%
% Algorithm Concept:
%   - Three search strategies of the Gold Rush Optimizer: collaboration,
%     mining and migration
%   - AGRO replaces GRO's fixed strategy probabilities with an adaptive
%     mechanism: a decayed success record (goodSel) rewards the strategies
%     that improved solution quality, and the resulting p1/p2/p3 steer the
%     per-agent strategy choice
%   - Mining picks its partner by fitness-weighted roulette (random_ks_two)
%
% Reference:
% Costas Panagiotakis,
% AGRO: An Adaptive Gold Rush Optimizer with Dynamic Strategy Selection,
% Algorithms 2026, 19(3), 192.
% https://doi.org/10.3390/a19030192
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = agro(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N        = 30;
    Max_iter = max(2, ceil(maxFE / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    sigma_initial = 2;
    sigma_final   = 1 / Max_iter;

    best_pos   = zeros(1, dim);
    best_score = inf;

    Positions = initialization(N, dim, lb, ub);
    Fit       = inf(1, N);
    X_NEW     = Positions;
    Fit_NEW   = Fit;

    goodSel   = zeros(1, 3);
    Strategy  = randi(3, 1, N);
    p1 = zeros(1, Max_iter);
    p2 = zeros(1, Max_iter);
    p3 = zeros(1, Max_iter);
    LS = zeros(1, 3);
    Init_iter = 1;

    bsf  = inf;
    iter = 1;

    while iter <= Max_iter && FE < maxFE

        newgoodSel = zeros(1, 3);

        for i = 1:3
            LS(i) = length(find(Strategy == i));
        end

        % Evaluate the candidate positions
        [Fit_NEW, FE] = calculate_fitness(X_NEW', problem, FE);
        Fit_NEW = Fit_NEW(:)';

        for i = 1:N
            if Fit_NEW(i) < Fit(i)
                Fit(i) = Fit_NEW(i);
                Positions(i, :) = X_NEW(i, :);
                newgoodSel(Strategy(i)) = newgoodSel(Strategy(i)) + (3/4) * (1 / LS(Strategy(i)));
            end
        end

        [m, i] = min(Fit);
        if m < best_score
            best_score = Fit(i);
            best_pos   = Positions(i, :);
            newgoodSel(Strategy(i)) = newgoodSel(Strategy(i)) + (1/4);
        end
        if best_score < bsf
            bsf = best_score;
        end

        for k = 1:N
            ec = FE - N + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Positions, Fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        if iter == 1
            goodSel = newgoodSel;
        else
            goodSel = 0.75 * goodSel + 0.25 * newgoodSel;
        end

        l2 = ((Max_iter - iter) / (Max_iter - Init_iter)) ^ 2 * (sigma_initial - sigma_final) + sigma_final;
        l1 = ((Max_iter - iter) / (Max_iter - Init_iter)) ^ 1 * (sigma_initial - sigma_final) + sigma_final;

        M = max(Fit);
        m = min(Fit);
        if M ~= m
            Scores = (M - Fit) / (M - m);
            Scores = 0.75 * Scores + 0.0001 * rand(1, N) + 0.25;
        else
            Scores = ones(1, N);
        end

        p1(iter) = goodSel(1) / sum(goodSel);
        p2(iter) = goodSel(2) / sum(goodSel);
        p3(iter) = goodSel(3) / sum(goodSel);

        % Generate the next candidate positions
        for i = 1:size(Positions, 1)
            m = rand;

            if m < 1/12 + 0.75 * p1(iter)
                % collaboration
                Strategy(i) = 1;
                coworkers = randperm(N - 1, 2);
                diggers = 1:N;
                diggers(i) = [];
                coworkers = diggers(coworkers);
                digger1 = coworkers(1);
                digger2 = coworkers(2);

                for d = 1:dim
                    r1 = rand;
                    D3 = Positions(digger2, d) - Positions(digger1, d);
                    X_NEW(i, d) = Positions(i, d) + r1 * D3;
                end

            elseif m < 1/6 + 0.75 * (p1(iter) + p2(iter))
                % mining
                Strategy(i) = 2;
                k_vec = random_ks_two(N, i, Scores);
                digger1 = k_vec(1);
                for d = 1:dim
                    r1 = rand;
                    A2 = 2 * l2 * r1 - l2;
                    D2 = Positions(i, d) - Positions(digger1, d);
                    X_NEW(i, d) = Positions(digger1, d) + A2 * D2;
                end

            else
                % migration
                Strategy(i) = 3;
                for d = 1:dim
                    r1 = rand;
                    r2 = rand;
                    C1 = 1 + l1 * (r2 - 1/2);
                    A1 = 1 + l1 * (r1 - 1/2);
                    D1 = C1 * best_pos(d) - Positions(i, d);
                    X_NEW(i, d) = Positions(i, d) + A1 * D1;
                end
            end

            X_NEW(i, :) = boundConstraint(X_NEW(i, :), Positions(i, :), lb, ub);
        end

        iter = iter + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = best_score;
    best_solution = best_pos;
end

% Fitness-weighted partner selection (excluding i)
function k_vec = random_ks_two(N, i, Scores)
    values  = 1:N;
    weights = Scores;
    weights(i) = 0;
    weights = weights / sum(weights);
    k1 = randsample(values, 1, true, weights);
    k_vec = k1;
end

% Initialization
function Positions = initialization(N, dim, lb, ub)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

% Domain control: revert violating components to the old position
function newPos = boundConstraint(newPos, oldPos, lb, ub)
    pos = newPos < lb;
    newPos(pos) = oldPos(pos);
    pos = newPos > ub;
    newPos(pos) = oldPos(pos);
end

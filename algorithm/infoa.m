% ----------------------------------------------------------------------- %
% weIghted meaN oF vectOrs (INFO)
% Stored as infoa; info is a MATLAB builtin function
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nP    = 30                  % Population size
%   alpha = 2*exp(-4*t)         % Range of del and sigm, t the spent budget ratio
%   r     = 0.1 + 0.4*rand      % Weight of WM1 against WM2 in the mean rule
%   epsi  = 1e-25*rand          % Offset that keeps a weighted mean off exact zero
%   mu    = 0.05*randn          % Per-coordinate spread of the combining stage
%   pr    = 0.5                 % Chance of the local search stage
%
% Algorithm Concept:
%   - Mean rule: a weighted mean of the three differences among three random
%     members, blended with the same mean over the best, better and worst
%   - Weights w = cos(df + pi)*exp(-df/omega), df a fitness difference and omega
%     the largest of the three fitnesses involved
%   - Updating rule: two candidates z1, z2 around the member and the best (or a
%     random member and the better), each with a scaled mean-rule step
%   - Vector combining: each coordinate takes z1 or z2 plus mu*|z1 - z2| with
%     probability 0.5, otherwise keeps the member's value
%   - Local search, with probability pr: a step around the best, or around a
%     random blend of three members' mean, the better and the best
%   - Greedy replacement; best refreshed at once, better (a random one of ranks
%     2-5) and worst once per iteration; out-of-box coordinates are clamped
%
% Reference:
% Iman Ahmadianfar, Ali Asghar Heidari, Saeed Noshadian, Huiling Chen,
% Amir H. Gandomi,
% INFO: An efficient optimization algorithm based on weighted mean of vectors,
% Expert Systems with Applications 195 (2022) 116516.
% https://doi.org/10.1016/j.eswa.2022.116516
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (INFO.m from aliasgharheidari.com/INFO.html,
% nP = 30 from its Main.m). The demo schedules alpha on it/MaxIt with MaxIt = 500;
% here it runs on the spent budget FE/maxFe and FE >= maxFe is the only
% terminator. As released, the second best/better/worst weight (the code's
% Eq. 4.8) repeats the best-better difference. omega can be zero or negative and
% the (f_a - f_b + 1) denominators can vanish, so a step can overflow, and the
% reference's clamp turns Inf into NaN (Inf*0); here a non-finite coordinate is
% redrawn uniformly in its own box before the clamp, as in avoa, gto and nrbo.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = infoa(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    nP = 30;
    pr = 0.5;
    e  = 1e-25;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = repmat(lb, nP, 1) + rand(nP, dim) .* repmat(ub - lb, nP, 1);
    [Cost, FE] = calculate_fitness(X', problem, FE);
    Cost = Cost(:);
    M = Cost;

    bsf          = inf;
    bsf_solution = X(1, :);
    for i = 1:nP
        if Cost(i) < bsf
            bsf          = Cost(i);
            bsf_solution = X(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    [~, ind] = sort(Cost);
    Best_X     = X(ind(1), :);
    Best_Cost  = Cost(ind(1));
    Worst_Cost = Cost(ind(end));
    Worst_X    = X(ind(end), :);
    I = randi([2 5]);
    Better_X    = X(ind(I), :);
    Better_Cost = Cost(ind(I));

    W = zeros(1, 3);
    while FE < maxFE
        alpha = 2 * exp(-4 * (FE / maxFE));                          % Eqs. (5.1) and (9.1)

        M_Best   = Best_Cost;
        M_Better = Better_Cost;
        M_Worst  = Worst_Cost;

        for i = 1:nP
            if FE >= maxFE
                break;
            end

            del  = 2 * rand * alpha - alpha;                          % Eq. (5)
            sigm = 2 * rand * alpha - alpha;                          % Eq. (9)

            A1 = randperm(nP);
            A1(A1 == i) = [];
            a = A1(1); b = A1(2); c = A1(3);

            epsi = e * rand;

            omg = max([M(a) M(b) M(c)]);
            MM = [(M(a) - M(b)) (M(a) - M(c)) (M(b) - M(c))];
            W(1) = cos(MM(1) + pi) * exp(-(MM(1)) / omg);             % Eq. (4.2)
            W(2) = cos(MM(2) + pi) * exp(-(MM(2)) / omg);             % Eq. (4.3)
            W(3) = cos(MM(3) + pi) * exp(-(MM(3)) / omg);             % Eq. (4.4)
            Wt = sum(W);
            WM1 = del .* (W(1) .* (X(a, :) - X(b, :)) + W(2) .* (X(a, :) - X(c, :)) + ...
                  W(3) .* (X(b, :) - X(c, :))) / (Wt + 1) + epsi;      % Eq. (4.1)

            omg = max([M_Best M_Better M_Worst]);
            MM = [(M_Best - M_Better) (M_Best - M_Better) (M_Better - M_Worst)];
            W(1) = cos(MM(1) + pi) * exp(-MM(1) / omg);               % Eq. (4.7)
            W(2) = cos(MM(2) + pi) * exp(-MM(2) / omg);               % Eq. (4.8)
            W(3) = cos(MM(3) + pi) * exp(-MM(3) / omg);               % Eq. (4.9)
            Wt = sum(W);
            WM2 = del .* (W(1) .* (Best_X - Better_X) + W(2) .* (Best_X - Worst_X) + ...
                  W(3) .* (Better_X - Worst_X)) / (Wt + 1) + epsi;     % Eq. (4.6)

            r = 0.1 + 0.4 * rand;
            MeanRule = r .* WM1 + (1 - r) .* WM2;                     % Eq. (4)

            if rand < 0.5
                z1 = X(i, :) + sigm .* (rand .* MeanRule) + randn .* (Best_X - X(a, :)) / (M_Best - M(a) + 1);
                z2 = Best_X + sigm .* (rand .* MeanRule) + randn .* (X(a, :) - X(b, :)) / (M(a) - M(b) + 1);
            else                                                      % Eq. (8)
                z1 = X(a, :) + sigm .* (rand .* MeanRule) + randn .* (X(b, :) - X(c, :)) / (M(b) - M(c) + 1);
                z2 = Better_X + sigm .* (rand .* MeanRule) + randn .* (X(a, :) - X(b, :)) / (M(a) - M(b) + 1);
            end

            u = zeros(1, dim);
            for j = 1:dim
                mu = 0.05 * randn;
                if rand < 0.5
                    if rand < 0.5
                        u(j) = z1(j) + mu * abs(z1(j) - z2(j));      % Eq. (10.1)
                    else
                        u(j) = z2(j) + mu * abs(z1(j) - z2(j));      % Eq. (10.2)
                    end
                else
                    u(j) = X(i, j);                                   % Eq. (10.3)
                end
            end

            if rand < pr
                L = rand < 0.5;
                v1 = (1 - L) * 2 * (rand) + L;                        % Eqs. (11.5) and (11.6)
                v2 = rand .* L + (1 - L);
                Xavg = (X(a, :) + X(b, :) + X(c, :)) / 3;             % Eq. (11.4)
                phi = rand;
                Xrnd = phi .* (Xavg) + (1 - phi) * (phi .* Better_X + (1 - phi) .* Best_X);   % Eq. (11.3)
                Randn = L .* randn(1, dim) + (1 - L) .* randn;
                if rand < 0.5
                    u = Best_X + Randn .* (MeanRule + randn .* (Best_X - X(a, :)));          % Eq. (11.1)
                else
                    u = Xrnd + Randn .* (MeanRule + randn .* (v1 * Best_X - v2 * Xrnd));      % Eq. (11.2)
                end
            end

            NF = ~isfinite(u);
            u(NF) = lb(NF) + rand(1, sum(NF)) .* (ub(NF) - lb(NF));
            New_X = min(max(u, lb), ub);

            [fv, FE] = calculate_fitness(New_X', problem, FE);
            New_Cost = fv(1);

            if New_Cost < bsf
                bsf          = New_Cost;
                bsf_solution = New_X;
            end
            if New_Cost < Cost(i)
                X(i, :) = New_X;
                Cost(i) = New_Cost;
                M(i)    = Cost(i);
                if Cost(i) < Best_Cost
                    Best_X    = X(i, :);
                    Best_Cost = Cost(i);
                end
            end

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, Cost, population_history, fitness_history, ...
                history_index, maxFE);
        end

        [~, ind] = sort(Cost);
        Worst_X    = X(ind(end), :);
        Worst_Cost = Cost(ind(end));
        I = randi([2 5]);
        Better_X    = X(ind(I), :);
        Better_Cost = Cost(ind(I));
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

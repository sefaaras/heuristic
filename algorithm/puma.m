% ----------------------------------------------------------------------- %
% Puma Optimizer (PO)
% Stored as puma; the acronym PO collides with po (Political Optimizer)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nSol = 30              % Population size (pumas)
%   PF   = [0.5 0.5 0.3]   % Weights of the f1, f2 and f3 phase-selection terms
%   Mega = 0.99 -> 0.01    % Score weight alpha, -0.01 per iteration the phase loses
%   pCR  = 0.20            % Exploration crossover rate, + (1 - pCR)/nSol per failure
%   Q    = 0.67            % Exploitation: share of the reflected-projection move
%   Beta = 2               % Exploitation: divisor 1 + Beta*rand of the mean-based move
%
% Algorithm Concept:
%   - Unexperienced phase (iterations 1-3): exploration and exploitation both
%     start from the same population; the best nSol of the three survive
%   - Experienced phase: one phase per iteration, the one with the higher score
%     (f1 last gain per time, f2 gain over three changes, f3 time since chosen)
%   - Exploration: DE-like six-vector difference or a uniform random point,
%     binomial crossover whose rate rises after every failed puma
%   - Exploitation: an exp(randn)-scaled difference step from the best, a
%     reflection of the best plus a scalar projection, or a mean-based move
%   - Greedy replacement of each puma by its candidate
%
% Reference:
% Benyamin Abdollahzadeh, Nima Khodadadi, Saeid Barshandeh, Pavel Trojovsky,
% Farhad Soleimanian Gharehchopogh, El-Sayed M. El-kenawy, Laith Abualigah,
% Seyedali Mirjalili,
% Puma optimizer (PO): a novel metaheuristic optimization algorithm and its
% application in machine learning,
% Cluster Computing 27(4) (2024) 5235-5283.
% https://doi.org/10.1007/s10586-023-04221-5
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's release (github.com/Benyamin-abdollahzadeh/
% Puma-Optimizer-PO-, commit 004599e, Puma.m + Exploration.m + Exploitation.m);
% with unifrnd initialisation a seeded run matches it bit for bit (cec14).
% Two release quirks are kept: VEC = S2/S1 is a matrix right division of two
% row vectors, i.e. one least-squares scalar, not an element-wise ratio, and
% mbest = mean([Sol.X])/nSol averages every coordinate of every puma into one
% scalar (Eq. (32) writes it as mean(Sol_total)/Npop).
% MaxIter = ceil(maxFE/nSol) - 4: initialisation and the three unexperienced
% iterations cost 7*nSol, every later iteration nSol, so Iter/MaxIter in the
% exploitation step reaches 1 at the end of the budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = puma(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    nSol = 30;
    MaxIter = max(4, ceil(maxFE / nSol) - 4);
    PF = [0.5 0.5 0.3];   % 1 and 2 weight intensification (f1, f2), 3 diversification (f3)

    UnSelected = ones(1, 2);
    F3_Explore = 0;
    F3_Exploit = 0;
    Seq_Time_Explore = ones(1, 3);
    Seq_Time_Exploit = ones(1, 3);
    Seq_Cost_Explore = ones(1, 3);
    Seq_Cost_Exploit = ones(1, 3);
    Costs_Explor = zeros(1, 3);
    Costs_Exploit = zeros(1, 3);
    minPF3 = [];   % running min of the release's PF_F3 list of nonzero gains; empty until the first
    Mega_Explor = 0.99;
    Mega_Exploit = 0.99;
    Score_Explore = 0;
    Score_Exploit = 0;
    Flag_Change = 1;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = zeros(nSol, dim);
    for i = 1:nSol
        X(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [bsf, ib] = min(fit);
    bsx = X(ib, :);
    for e = 1:min(nSol, maxFE)
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % The release's Puma male: refreshed only between phases, unlike the reported bsf
    [BestC, ind] = min(fit);
    BestX = X(ind, :);
    InitialBestC = BestC;

    Iter = 0;
    while FE < maxFE
        Iter = Iter + 1;
        if Iter <= 3
            modes = [1 2];
        elseif Score_Explore > Score_Exploit
            modes = 1;
        else
            modes = 2;
        end

        PX = {X, X};
        PFit = {fit, fit};
        for m = modes
            P = X;
            Pf = fit;
            if m == 1
                [Pf, s] = sort(Pf);   % exploration works on the population sorted by cost
                P = P(s, :);
                pCR = 0.20;
                p = (1 - pCR) / nSol;   % Eqs. (28)-(29)
            end
            for i = 1:nSol
                if m == 1
                    z = explore(P, i, pCR, lb, ub, nSol, dim);
                else
                    z = exploit(P, i, BestX, Iter, MaxIter, lb, ub, nSol, dim);
                end
                [fz, FE] = calculate_fitness(z', problem, FE);
                if fz < Pf(i)
                    P(i, :) = z;
                    Pf(i) = fz;
                elseif m == 1
                    pCR = pCR + p;   % Eq. (30)
                end
                if fz < bsf
                    bsf = fz;
                    bsx = z;
                end
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, P, Pf, population_history, fitness_history, ...
                    history_index, maxFE);
                if FE >= maxFE, break; end
            end
            PX{m} = P;
            PFit{m} = Pf;
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        if Iter <= 3
            Costs_Explor(Iter) = min(PFit{1});
            Costs_Exploit(Iter) = min(PFit{2});
            allX = [X; PX{1}; PX{2}];
            allF = [fit, PFit{1}, PFit{2}];
            [~, s] = sort(allF);
            X = allX(s(1:nSol), :);
            fit = allF(s(1:nSol));
            BestX = X(1, :);
            BestC = fit(1);

            if Iter == 3
                Seq_Cost_Explore = abs([InitialBestC - Costs_Explor(1), ...
                    Costs_Explor(2) - Costs_Explor(1), Costs_Explor(3) - Costs_Explor(2)]);   % Eqs. (5)-(7)
                Seq_Cost_Exploit = abs([InitialBestC - Costs_Exploit(1), ...
                    Costs_Exploit(2) - Costs_Exploit(1), Costs_Exploit(3) - Costs_Exploit(2)]);   % Eqs. (8)-(10)
                for k = 1:3
                    minPF3 = add_gain(minPF3, Seq_Cost_Explore(k));
                    minPF3 = add_gain(minPF3, Seq_Cost_Exploit(k));
                end
                F1_Explor = PF(1) * (Seq_Cost_Explore(1) / Seq_Time_Explore(1));   % Eq. (1)
                F1_Exploit = PF(1) * (Seq_Cost_Exploit(1) / Seq_Time_Exploit(1));   % Eq. (2)
                F2_Explor = PF(2) * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore));   % Eq. (3)
                F2_Exploit = PF(2) * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit));   % Eq. (4)
                Score_Explore = (PF(1) * F1_Explor) + (PF(2) * F2_Explor);   % Eq. (11)
                Score_Exploit = (PF(1) * F1_Exploit) + (PF(2) * F2_Exploit);   % Eq. (12)
            end
            continue;
        end

        SelectFlag = modes;
        X = PX{SelectFlag};
        fit = PFit{SelectFlag};
        [TBC, tb] = min(fit);
        TBX = X(tb, :);
        Count_select = UnSelected;
        if SelectFlag == 1
            UnSelected(2) = UnSelected(2) + 1;
            UnSelected(1) = 1;
            F3_Explore = PF(3);
            F3_Exploit = F3_Exploit + PF(3);
            Seq_Cost_Explore = [abs(BestC - TBC), Seq_Cost_Explore(1:2)];
            minPF3 = add_gain(minPF3, Seq_Cost_Explore(1));
        else
            UnSelected(1) = UnSelected(1) + 1;
            UnSelected(2) = 1;
            F3_Explore = F3_Explore + PF(3);
            F3_Exploit = PF(3);
            Seq_Cost_Exploit = [abs(BestC - TBC), Seq_Cost_Exploit(1:2)];
            minPF3 = add_gain(minPF3, Seq_Cost_Exploit(1));
        end
        if TBC < BestC
            BestC = TBC;
            BestX = TBX;
        end

        if Flag_Change ~= SelectFlag
            Flag_Change = SelectFlag;
            Seq_Time_Explore = [Count_select(1), Seq_Time_Explore(1:2)];
            Seq_Time_Exploit = [Count_select(2), Seq_Time_Exploit(1:2)];
        end

        F1_Explor = PF(1) * (Seq_Cost_Explore(1) / Seq_Time_Explore(1));   % Eq. (14)
        F1_Exploit = PF(1) * (Seq_Cost_Exploit(1) / Seq_Time_Exploit(1));   % Eq. (13)
        F2_Explor = PF(2) * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore));   % Eq. (16)
        F2_Exploit = PF(2) * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit));   % Eq. (15)

        % Eq. (21) compares the scores of the previous iteration
        if Score_Explore < Score_Exploit
            Mega_Explor = max((Mega_Explor - 0.01), 0.01);
            Mega_Exploit = 0.99;
        elseif Score_Explore > Score_Exploit
            Mega_Explor = 0.99;
            Mega_Exploit = max((Mega_Exploit - 0.01), 0.01);
        end
        lmn_Explore = 1 - Mega_Explor;   % Eq. (23)
        lmn_Exploit = 1 - Mega_Exploit;   % Eq. (22)

        Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (minPF3 * F3_Explore));   % Eq. (19)
        Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (minPF3 * F3_Exploit));   % Eq. (18)
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsx;
end

% Running min over the nonzero gains, NaN-ignoring like min() over the release's list
function m = add_gain(m, g)
    if g ~= 0
        if isempty(m)
            m = g;
        else
            m = min(m, g);
        end
    end
end

% Exploration candidate for puma i of the cost-sorted population P, Eqs. (25)-(27)
function z = explore(P, i, pCR, lb, ub, nSol, dim)
    A = randperm(nSol);
    A(A == i) = [];
    G = 2 * rand - 1;   % Eq. (26)
    if rand < 0.5
        y = rand(1, dim) .* (ub - lb) + lb;
    else
        y = P(A(1), :) + G .* (P(A(1), :) - P(A(2), :)) + G .* (((P(A(1), :) - P(A(2), :)) - ...
            (P(A(3), :) - P(A(4), :))) + ((P(A(3), :) - P(A(4), :)) - (P(A(5), :) - P(A(6), :))));
    end
    y = max(y, lb);
    y = min(y, ub);
    x = P(i, :);
    z = x;
    j0 = randi([1, dim]);
    for j = 1:dim
        if j == j0 || rand <= pCR   % short-circuit: no draw at j0, as in the release
            z(j) = y(j);
        end
    end
end

% Exploitation candidate for puma i of P, Eqs. (32)-(38)
function z = exploit(P, i, BestX, Iter, MaxIter, lb, ub, nSol, dim)
    Q = 0.67;
    Beta = 2;
    beta1 = 2 * rand;
    beta2 = randn(1, dim);
    w = randn(1, dim);   % Eq. (37)
    v = randn(1, dim);   % Eq. (38)
    F1 = randn(1, dim) .* exp(2 - Iter * (2 / MaxIter));   % Eq. (35)
    F2 = w .* v .^ 2 .* cos((2 * rand) * w);   % Eq. (36)
    mbest = mean(reshape(P', 1, [])) / nSol;   % one scalar, as the release's mean([Sol.X])/nSol
    R_1 = 2 * rand - 1;   % Eq. (34)
    S1 = (2 * rand - 1 + randn(1, dim));
    S2 = (F1 * R_1 .* P(i, :) + F2 * (1 - R_1) .* BestX);
    VEC = S2 / S1;   % mrdivide of two row vectors: a least-squares scalar, kept from the release
    if rand() <= 0.5
        if rand() > Q
            z = BestX + beta1 .* (exp(beta2)) .* (P(randi(nSol, 1), :) - P(i, :));   % Eq. (32)
        else
            z = beta1 .* VEC - BestX;
        end
    else
        r1 = round(1 + (nSol - 1) * rand());   % Eq. (33)
        z = (mbest .* P(r1, :) - ((-1) .^ (randi([0 1], 1, 1))) .* P(i, :)) ./ (1 + (Beta * rand));
    end
    z = max(z, lb);
    z = min(z, ub);
end

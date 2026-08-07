% ----------------------------------------------------------------------- %
% Wave Optics Optimizer (WOO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop   = 50        % Population size (light waves)
%   Lambda = 5e-4      % Wave length (mm)
%   d      = 1e-2      % Slit width (mm)
%   I0     = 1         % Initial relative light intensity
%
% Algorithm Concept:
%   - Single-slit diffraction builds the initial population: each wave is
%     placed between an upper and a lower fringe boundary derived from the
%     diffraction angle sin(theta) = m*Lambda/(2d)
%   - Random (diffraction) strategy: the intensity I = I0*(sin(u)/u)^2 and
%     the relative intensity Ir drive a mixture of the interference vectors
%     U (mid-rank difference) and U2 (rank-partitioned difference), gated by
%     the per-dimension mask Z1
%   - Reinforcement (interference) strategy: rotation/oscillation operators
%     around the central bright fringe (the best solution)
%   - Adjustment strategy: the per-wave parameters Ua_s / Ua_w are updated
%     from the successful moves recorded in the matrix Ma_b
%
% Reference:
% Yong Peng, Shaowei Gu, Yunbin Liang, Kaichen Ouyang, Yingli Li, Kui Wang,
% Guohua Wu, Chaojie Fan,
% Wave Optics Optimizer: A novel meta-heuristic algorithm for engineering
% optimization,
% Communications in Nonlinear Science and Numerical Simulation 152 (2026) 109337.
% https://doi.org/10.1016/j.cnsns.2025.109337
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = woo(problem)

    Dim   = problem.dimension;
    Low   = problem.lb;
    Up    = problem.ub;
    maxFE = problem.maxFe;

    nPop  = 50;
    MaxIt = max(1, ceil((maxFE - nPop) / nPop));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial parameters
    Lambda = 5 * 10 ^ -4;   % wave length (mm)
    d      = 1 * 10 ^ -2;   % slit width (mm)
    I0     = 1;             % initial relative light intensity

    k = 0; Time_0 = 0; Time_p = 0;
    m_l = 0; m_d = 0;

    % Single-slit diffraction (initialisation)
    S = zeros(nPop, Dim); US = S; LS = S; PopPos = S;
    deltaS = zeros(1, nPop);
    for i = 1:nPop
        S(i, :) = rand(1, Dim) .* (Up - Low) + Low;
        us = max(S(i, :));
        ls = min(S(i, :));
        if i == 1
            sintheta = 0;
            US(i, :) = S(i, :) + d * (rand * (us - ls) + ls);
            LS(i, :) = S(i, :) - d * (rand * (us - ls) + ls);
        else
            if mod(i, 2) == 0
                m_d = m_d + 1;
                sintheta = (2 * m_d) * Lambda / (2 * d);
            else
                m_l = m_l + 1;
                sintheta = (2 * m_l + 1) * Lambda / (2 * d);
            end
            US(i, :) = S(i, :) + d * (rand * (us - ls) + ls) / sintheta;
            LS(i, :) = S(i, :) - d * (rand * (us - ls) + ls) / sintheta;
        end
        deltaS(i) = d * sintheta;

        PopPos(i, :) = (US(i, :) + LS(i, :)) / 2 + sign(rand - 0.5) * rand * deltaS(i);
        PopPos(i, :) = SpaceBound(PopPos(i, :), Up, Low);
    end

    [PopFit, FE] = calculate_fitness(PopPos', problem, FE);
    PopFit = PopFit(:)';

    [BestF, bi] = min(PopFit);
    BestX = PopPos(bi, :);

    Ua_deltaF = ones(1, nPop);
    Ua_deltaS = ones(1, nPop);
    Ua_deltaX = zeros(1, nPop);
    for i = 1:nPop
        Ua_deltaX(i) = sqrt(mean((PopPos(i, :) - BestX) .^ 2));
    end

    [~, index] = sort(PopFit);
    PopPos    = PopPos(index, :);
    PopFit    = PopFit(index);
    Ua_deltaX = Ua_deltaX(index);

    bsf = BestF;
    for eval_count = 1:min(nPop, maxFE)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, PopPos, PopFit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Individual parameters
    Ua     = (Lambda / (4 * pi)) * (3 * nPop / d);
    MaDim0 = round(3 / Ua * Dim / log(Dim));
    Ua_s   = ones(1, nPop) * Ua;
    Ua_w   = ones(1, nPop) * cos(pi / 2 * Ua);

    t0 = round(1/2 * Ua * nPop);  t = t0;
    T  = abs(round((1 - sin(pi / 2 * Ua)) * nPop));
    Mid = zeros(1, 2);
    Mid(1) = T + 1; Mid(1) = SpaceMid(Mid(1), nPop);
    fes = zeros(nPop); sim = round(nPop / 2);

    newPopPos = zeros(nPop, Dim);
    newPopFit = zeros(1, nPop);
    flag_e1   = zeros(1, nPop);
    Leader_Pop = 1;
    s = Ua; w = 1; pb_m = 1;

    % Main loop
    for It = 1:MaxIt
        if FE >= maxFE, break; end

        C    = (1 - It / MaxIt);
        tau1 = I0 * (Ua / 3 * (C - 1) + 1);
        tau2 = exp(tau1) / 3;
        bc   = round(nPop / 3 * (1 + C * 2));

        MaDim = round((1 + C) * MaDim0 / 2);
        Ma_b  = zeros(4, max(1, MaDim));

        m_l = 0; m_d = 0; Time_1 = 0;

        for i = 1:nPop
            pw = sin((-exp(-It) + 1) * pi / 2);
            Mid(2) = round(nPop - 1 / (2 * Ua ^ 2) * C * T);
            Mid(2) = SpaceMid(Mid(2), nPop);
            U = (PopPos(randi([min(Mid), max(Mid)]), :) - PopPos(i, :)) * pw;
            if rand > C
                if i < 1/2 * nPop
                    U2 = PopPos(randi([1, sim]), :) - PopPos(randi(nPop), :);
                else
                    U2 = PopPos(randi([1, sim]), :) - PopPos(randi([sim + 1, nPop]), :);
                end
            else
                U2 = PopPos(randi(nPop), :) - PopPos(randi(nPop), :);
            end

            % Random (diffraction) strategy
            if tau2 > rand
                if i == 1
                    I = I0 * sin(pi / 2 * C ^ (2 * (1 + C)));
                    m_r = 1;
                else
                    u = pi * deltaS(i) / Lambda;
                    I = I0 * (sin(u) / u) ^ 2;
                    if mod(i, 2) == 0
                        m_d = m_d + 1;
                        m_r = m_d;
                    else
                        m_l = m_l + 1;
                        m_r = m_l;
                    end
                end
                Ir = exp(I * abs(BestF - PopFit(i)) / (BestF + eps));
                if Ir == inf || isnan(Ir)
                    Ir = 1;
                end

                if It < 1 / t * MaxIt
                    flag_e1(i) = 0;
                    r1 = rand;
                    s = exp(-2 * r1 * (1 - r1));
                    Z1 = rand(1, Dim) < s;
                    om = (rand(1, Dim) .* abs(randn(1, Dim))) .^ (C + 3) + randn / m_r;
                    if bc > i
                        w = Ir * exp(-rand(1, Dim) .^ (C + 3));
                        DU1 = (1 - w) .* U + w .* U2;
                    else
                        w = Ir * exp(-rand(1, Dim) .^ (C + 3)) / m_r;
                        DU1 = ((1 - w) .* U + w .* (U2 + BestX / m_r)) .^ 2;
                    end
                else
                    flag_e1(i) = 1;
                    om = 1;
                    if tau2 > rand
                        w = max(Ua_w(i), 0);
                        Po = round(3 * (1 - m_r / sim));
                        r_Pop1 = randi(nPop, 1);
                        s = Ua_s(r_Pop1) ^ (Po * C) + Ua / 4 * randn * rand;
                        s = max(min(s, 1), 0);
                        Z1 = rand(1, Dim) <= s;
                        Z1(randi(Dim)) = 1;
                    else
                        w = Ir;
                        Po = 3;
                        s = (Ua) ^ (Po * C);
                        Z1 = rand(1, Dim) <= s;
                    end

                    if Time_p == 0
                        p = round(C ^ 2 * log(Dim) * T + 1 / (w) + Ua);
                        p = min(max(p, 1), nPop);
                        Leader_Pop = randi([1, p], 1);
                    else
                        Leader_Pop = pb_m;
                        Time_p = 0;
                    end
                    if tau2 > rand
                        Su = exp(Ir - 2 * T * Lambda / d);
                        if fes(i, It) / MaxIt > rand && bc > i
                            DU1 = U + Su * U2;
                        else
                            DU1 = w * (PopPos(Leader_Pop, :) + Su * (U - PopPos(randi(nPop), :)));
                        end
                    else
                        U3 = tau1 * (rand(1, Dim) .* S(i, :)) + (1 - tau1) * PopPos(Leader_Pop, :);
                        DU1 = (1 - w) * U + w * Ua * (Ir * U2 + U3);
                    end
                end
                newPopPos(i, :) = PopPos(i, :) + om .* Z1 .* DU1;

            % Reinforcement (interference) strategy
            else
                flag_e1(i) = 2;
                if rand > tau1
                    if rand > C
                        wv = sqrt(2) * sqrt(abs(sin(It .* (rand - Ua)))) + 2 * rand - 1;
                        SW = (sin(rand * pi) + cos(rand * pi));
                        om2 = SW ./ wv;
                        Vr = randn(1, Dim) .* om2;
                        newPopPos(i, :) = Vr .* BestX;
                    else
                        Rt1 = randi(360, 2) * pi / 360;
                        om2 = cos(Rt1(1)) * sin(Rt1(2)) / i;
                        Vr = randn(1, Dim) .* om2;
                        newPopPos(i, :) = Vr .* (BestX + PopPos(i, :)) / 2;
                    end
                else
                    r1 = rand;
                    Dv = ((BestX + PopPos(i, :)) / 2 - PopPos(randi(nPop), :));
                    om3 = abs(randn) * (r1 * (1 - r1)) / i;
                    Z2 = rand > rand;
                    DX = Z2 * om3 * Dv;
                    om4 = exp(-randn / i) * rand;
                    newPopPos(i, :) = PopPos(i, :) + om4 .* U2 + DX;
                end
            end

            newPopPos(i, :) = SpaceBound(newPopPos(i, :), Up, Low);
            [newPopFit(i), FE] = calculate_fitness(newPopPos(i, :)', problem, FE);

            if newPopFit(i) < PopFit(i)
                if flag_e1(i) == 0
                    Time_0 = Time_0 + 1;
                    if Time_0 > (3 * C + 3) * nPop && t > Ua * t0
                        t = t - 1;
                        Time_0 = 0;
                    end
                elseif flag_e1(i) == 1
                    Time_1 = Time_1 + 1;
                    Ua_deltaF(i) = abs((PopFit(i) - newPopFit(i)) / (PopFit(i) - BestF + eps));
                    Ua_deltaS(i) = sqrt(mean((PopPos(i, :) - newPopPos(i, :)) .^ 2));
                    Ua_deltaX(i) = sqrt(mean((PopPos(i, :) - BestX) .^ 2));
                    if Time_1 <= MaDim
                        Ma_b(1, Time_1) = i;
                        Ma_b(2, Time_1) = Leader_Pop;
                        Ma_b(3, Time_1) = s;
                        Ma_b(4, Time_1) = w(1);
                    elseif newPopFit(i) < BestF && Time_1 <= 2 * MaDim
                        Ma_b(1, Time_1 - MaDim) = i;
                        Ma_b(2, Time_1 - MaDim) = Leader_Pop;
                        Ma_b(3, Time_1 - MaDim) = s;
                        Ma_b(4, Time_1 - MaDim) = w(1);
                    end
                end

                S(i, :)      = tau1 * PopPos(i, :) + (1 - tau1) * S(i, :);
                S(i, :)      = SpaceBound(S(i, :), Up, Low);
                PopFit(i)    = newPopFit(i);
                PopPos(i, :) = newPopPos(i, :);
            end

            if PopFit(i) < BestF
                BestF = PopFit(i);
                BestX = PopPos(i, :);
                fes(i, It + 1) = 0;
            else
                fes(i, It + 1) = fes(i, It) + 1;
            end

            if BestF < bsf, bsf = BestF; end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, PopPos, PopFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end

        % Adjustment strategy
        if Time_1 > 0
            Ma_b(:, all(Ma_b == 0, 1)) = [];
            if isempty(Ma_b), continue; end
            if Time_1 >= MaDim
                Sa = MaDim;
            else
                Sa = Time_1;
            end

            deltaF_Better = Ua_deltaF(Ma_b(1, :));  SumF_Better = sum(deltaF_Better);
            deltaS_Better = Ua_deltaS(Ma_b(1, :));  SumS_Better = sum(deltaS_Better);
            deltaX_Better = Ua_deltaX(Ma_b(1, :));  SumX_Better = sum(deltaX_Better);

            om5 = Sa * (deltaF_Better / SumF_Better + deltaX_Better / (SumX_Better + SumS_Better));
            om6 = log(mean(Ma_b(1, :))) * pw;
            L = om5 / om6;
            [~, Lminr] = sort(L);
            [~, Xminr] = sort(deltaX_Better);
            Pool = zeros(2, 3);
            Pool(1, 1) = mean(Ma_b(3, :)); Pool(2, 1) = mean(Ma_b(4, :));
            Pool(1, 2) = Ma_b(3, Lminr(1)); Pool(2, 2) = Ma_b(4, Lminr(1));
            Pool(1, 3) = Ma_b(3, Xminr(1)); Pool(2, 3) = Ma_b(4, Xminr(1));
            if Sa <= sin(pi / 2 * Ua) * MaDim
                rpool = 1;
                k = k(1) + 1;
                if k > nPop
                    k = 1;
                end
            else
                rpool = randi(3);
                rL = randi(min(Sa, size(Ma_b, 2)));
                k = Ma_b(1, Lminr(1:min(rL, numel(Lminr))));
            end
            L1 = mean(L) * Pool(1, rpool);
            L2 = mean(L) * Pool(2, rpool);
            f = sign(rand ^ (Dim / 10) - Ua);
            if rand > C
                if tau2 > rand
                    Ua_s(k) = (tau1 * L1 + (1 - tau1) * Ua_s(k));
                else
                    Ua_s(k) = exp(-sqrt((tau1 * L1 + (1 - tau1) * Ua_s(k)) .^ 2)) * (It ^ f);
                    kpbest = Ma_b(2, randi(size(Ma_b, 2)));
                    Time_p = 1;
                    pb_m = kpbest;
                end
                Ua_w(k) = sin(L1 * pi / 2);
            else
                Ua_s(k) = (L1 ^ 2) * (It ^ f);
                Ua_w(k) = tau1 * exp(-sqrt(L2)) + (1 - tau1) * Ua_w(k);
            end
        end

        [~, index] = sort(PopFit);
        PopPos = PopPos(index, :);
        PopFit = PopFit(index);
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = BestF;
    best_solution = BestX;
end

% Bound handling: re-draw violating dimensions
function X = SpaceBound(X, Up, Low)
    Dim = length(X);
    S = (X > Up) + (X < Low);
    X = (rand(1, Dim) .* (Up - Low) + Low) .* S + X .* (~S);
end

% Clamp an index into [1, nPop]
function Mid = SpaceMid(Mid, nPop)
    if Mid <= 0
        Mid = 1;
    elseif Mid > nPop
        Mid = nPop;
    end
end

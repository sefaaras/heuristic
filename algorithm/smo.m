% ----------------------------------------------------------------------- %
% Starling Murmuration Optimizer (SMO) for benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize  = 100   % Population size (starlings)
%   FlockNum = 10    % Number of dynamic flocks
%
% Algorithm Concept:
%   - Separating strategy (diversity) via a quantum operator (Eq.2)
%   - Dynamic multi-flock construction, then per-flock:
%       Whirling search (exploitation) and Quantum random Diving (exploration)
%
% Reference:
% Hoda Zamani, Mohammad H. Nadimi-Shahraki, Amir H. Gandomi,
% Starling murmuration optimizer: A novel bio-inspired algorithm for global
% and engineering optimization,
% Computer Methods in Applied Mechanics and Engineering 392 (2022) 114616.
% https://doi.org/10.1016/j.cma.2022.114616
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension, lb, ub, maxFe, fhd, number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = smo(problem)

    ProblemSize = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    PopSize = 100;
    FlockNum = 10;
    lu = [lb; ub];

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, PopSize, ProblemSize);
    fitness_history = zeros(history_size, PopSize);
    history_index = 1;

    Pop = repmat(lu(1, :), PopSize, 1) + rand(PopSize, ProblemSize) .* (repmat(lu(2, :) - lu(1, :), PopSize, 1));
    [Val, FE] = calculate_fitness(Pop', problem, FE);
    Val = Val(:)';

    [~, sorted_index] = sort(Val, 'ascend');
    Pop = Pop(sorted_index, :);
    Val = Val(sorted_index);

    BestVal = Val(1);
    BestPos = Pop(1, :);
    bsf = BestVal;
    best_pos = BestPos;
    PosConvergance = Pop(1:20, :);

    for e = 1:PopSize
        if e <= maxFE
            curve(e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                e, Pop, Val, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    MaxIt = max(2, ceil(maxFE / PopSize));
    t = 0;
    while FE < maxFE
        t = t + 1;

        %% A) Separating strategy
        Sep_rate = log(t + ProblemSize) / (log(MaxIt) * 2);
        SepSize = max(round(Sep_rate * size(Pop, 1)), 2);
        % Keep at least FlockNum starlings outside the separated set so the
        % dynamic multi-flock construction always has enough representatives
        % (only binds for a tiny budget relative to dimension).
        SepSize = min(SepSize, PopSize - FlockNum);
        SepInd = (randperm(size(Pop, 1), SepSize))';
        PopSep = Pop(SepInd, :);
        U = rand(size(PopSep, 1), size(PopSep, 2));
        Fsep = QHO(U);
        Fsep(Fsep == -1) = 0;
        Fsep = min(Fsep, 1); Fsep = max(Fsep, 0);
        Param2Change = randperm(ProblemSize, randi(ProblemSize));
        r1 = randperm(size(PopSep, 1), size(PopSep, 1))';
        Xr_hat = [PosConvergance; PopSep];
        r11 = randperm(size(Xr_hat, 1), size(PopSep, 1))';
        PopSep(:, Param2Change) = BestPos(1, Param2Change) + Fsep(:, Param2Change) .* (Xr_hat(r11, Param2Change) - Pop(r1, Param2Change));
        PopSep = BoundCorrection(PopSep, Pop(SepInd, :), lu);
        [ValSep, FE] = calculate_fitness(PopSep', problem, FE);
        ValSep = ValSep(:)';
        [bsf, best_pos, curve, population_history, fitness_history, history_index] = recordBatch(...
            ValSep, PopSep, bsf, best_pos, curve, FE, maxFE, Pop, Val, ...
            population_history, fitness_history, history_index, sampling_interval, history_size);
        [Pop(SepInd, :), Val(SepInd)] = Update(PopSep, ValSep, Pop(SepInd, :), Val(SepInd), ProblemSize);
        if FE >= maxFE, break; end

        %% B) Dynamic multi-flock construction
        Ind = (setdiff(1:PopSize, SepInd))';
        [EVal, sorted_index] = sort(Val(Ind), 'ascend');
        EPop = Pop(sorted_index, :);
        Rep_Set = (1:FlockNum)';
        Rep_Pos = EPop(Rep_Set, :);
        SubPop_Num = (FlockNum + 1:size(Ind, 1))';
        FlockPopSize = floor((size(Ind, 1) - mod(size(Ind, 1), FlockNum)) / FlockNum);
        Result = cell(1, FlockNum);
        F_Mean = zeros(1, FlockNum);
        for k = 1:FlockNum
            if k == FlockNum
                lo = (k - 1) * (FlockPopSize - 1) + 1;
                lo = min(lo, numel(SubPop_Num) + 1);
                Result{k} = [FlockNum; SubPop_Num(lo:end)];
            else
                lo = (k - 1) * (FlockPopSize - 1) + 1;
                hi = min(k * (FlockPopSize - 1), numel(SubPop_Num));
                lo = min(lo, hi + 1);
                Result{k} = [Rep_Set(k); SubPop_Num(lo:hi)];
            end
            F_Mean(k) = mean(EVal(Result{k}));
        end
        F_Quality = (sum(F_Mean)) ./ repmat(F_Mean, FlockNum, 1);
        F_Quality = F_Quality(1, :);

        %% C) Whirling search strategy (exploitation)
        [~, FW] = find(F_Quality > mean(F_Quality));
        for i = 1:size(FW, 2)
            WhirIdx = Result{FW(i)};
            WhirlingPop = Pop(Ind(WhirIdx), :);
            I = randperm(size(Rep_Set, 1), 1)';
            X_RW = Rep_Pos(Rep_Set(I, :), :);
            idx = randperm(size(WhirlingPop, 1), size(WhirlingPop, 1))';
            XN = WhirlingPop(idx, :);
            WhirlingPop = WhirlingPop + cos(rand) .* (X_RW - XN);
            WhirlingPop = BoundCorrection(WhirlingPop, Pop(Ind(WhirIdx), :), lu);
            [WhirlingVal, FE] = calculate_fitness(WhirlingPop', problem, FE);
            WhirlingVal = WhirlingVal(:)';
            [bsf, best_pos, curve, population_history, fitness_history, history_index] = recordBatch(...
                WhirlingVal, WhirlingPop, bsf, best_pos, curve, FE, maxFE, Pop, Val, ...
                population_history, fitness_history, history_index, sampling_interval, history_size);
            [Pop(Ind(WhirIdx), :), Val(Ind(WhirIdx))] = Update(Pop(Ind(WhirIdx), :), Val(Ind(WhirIdx)), WhirlingPop, WhirlingVal, ProblemSize);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        %% D) Diving search strategy (exploration)
        [~, FD] = find(F_Quality <= mean(F_Quality));
        for q = 1:size(FD, 2)
            DivIdx = Result{FD(q)};
            DivingPop = Pop(Ind(DivIdx), :);
            R_D = repmat(Rep_Pos(FD(q), :), size(DivingPop, 1), 1);
            mu = 1; lambda = 20;
            IGD_1 = random('InverseGaussian', mu, lambda, size(DivingPop, 1), 1);
            pos = find(IGD_1 > 1);
            while ~isempty(pos)
                IGD_1(pos) = random('InverseGaussian', mu, lambda, size(pos, 1), 1);
                pos = find(IGD_1 > 1);
            end
            IGD_2 = random('InverseGaussian', mu, lambda, size(DivingPop, 1), 1);
            pos = find(IGD_2 > 1);
            while ~isempty(pos)
                IGD_2(pos) = random('InverseGaussian', mu, lambda, size(pos, 1), 1);
                pos = find(IGD_2 > 1);
            end
            C = [cos(rand) .* (angle(exp(1i .* rand ./ 2))), sin(.5) .* (angle(exp(1i .* 1.8 ./ 2))); ...
                -sin(.5) .* (angle(exp(-1i .* 1.8 ./ 2))), cos(-.5 * rand) .* (angle(exp(-1i .* (-.5 * rand) ./ 2)))];
            UP = C(1, 1) .* IGD_1 + C(1, 2) .* IGD_1;
            Down = C(2, 1) .* IGD_2 + C(2, 2) .* IGD_2;
            Downward = find(UP <= Down);
            Upward = find(UP > Down);
            a1 = randperm(size(DivingPop, 1));
            UnionSet = [Pop; PosConvergance];
            X_j = UnionSet(a1, :);
            VminPop = min(UnionSet, [], 1);
            VmaxPop = max(UnionSet, [], 1);
            Si_delta = repmat(VminPop(1, :), size(DivingPop, 1), 1) + rand(size(DivingPop, 1), ProblemSize) .* (repmat(VmaxPop(1, :) - VminPop(1, :), size(DivingPop, 1), 1));
            r1d = randperm(size(DivingPop, 1), size(DivingPop, 1))';
            if ~isempty(Downward)
                DivingPop(Downward, :) = R_D(Downward, :) - Down(Downward) .* (DivingPop(Downward, :) - DivingPop(r1d(Downward), :));
            end
            if ~isempty(Upward)
                DivingPop(Upward, :) = R_D(Upward, :) + UP(Upward) .* (DivingPop(Upward, :) - X_j(Upward, :) + Si_delta(Upward, :));
            end
            DivingPop = BoundCorrection(DivingPop, Pop(Ind(DivIdx), :), lu);
            [DivingVal, FE] = calculate_fitness(DivingPop', problem, FE);
            DivingVal = DivingVal(:)';
            [bsf, best_pos, curve, population_history, fitness_history, history_index] = recordBatch(...
                DivingVal, DivingPop, bsf, best_pos, curve, FE, maxFE, Pop, Val, ...
                population_history, fitness_history, history_index, sampling_interval, history_size);
            [Pop(Ind(DivIdx), :), Val(Ind(DivIdx))] = Update(Pop(Ind(DivIdx), :), Val(Ind(DivIdx)), DivingPop, DivingVal, ProblemSize);
            if FE >= maxFE, break; end
        end

        [BestVal, IdBst] = min(Val);
        BestPos = Pop(IdBst, :);
        if t <= size(PosConvergance, 1)
            PosConvergance(t, :) = BestPos;
        else
            PosConvergance(t, :) = BestPos;
        end
        if FE >= maxFE, break; end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = best_pos;
end

%% --- Update best-so-far and convergence curve for an evaluated batch ---
function [bsf, best_pos, curve, ph, fh, hi] = recordBatch(bval, bpos, bsf, best_pos, curve, FE, maxFE, Pop, Val, ph, fh, hi, si, hs)
    nb = numel(bval);
    for kk = 1:nb
        if bval(kk) < bsf
            bsf = bval(kk);
            best_pos = bpos(kk, :);
        end
        ec = FE - nb + kk;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, Pop, Val, ph, fh, hi, si, hs);
        end
    end
end

%% --- Bound correction (reflect toward midpoint with the parent) ---
function Xi = BoundCorrection(Xi, Pop, lu)
    Lower = repmat(lu(1, :), size(Pop, 1), 1);
    Xi(Xi < Lower) = (Pop(Xi < Lower) + Lower(Xi < Lower)) / 2;
    Upper = repmat(lu(2, :), size(Pop, 1), 1);
    Xi(Xi > Upper) = (Pop(Xi > Upper) + Upper(Xi > Upper)) / 2;
end

%% --- Greedy update (keep the better of two candidate sets) ---
function [OutPop, OutVal] = Update(Pop, Val, TPop, TVal, D)
    tmp = (Val <= TVal);
    tmp1 = repmat(tmp', 1, D);
    OutPop = tmp1 .* Pop + (1 - tmp1) .* TPop;
    OutVal = tmp .* Val + (1 - tmp) .* TVal;
end

%% --- Quantum harmonic oscillator operator ---
function Out = QHO(y)
    n = 1;
    h_bar = 1.05457168e-34; m = 9.1093826e-31; k = 2 * pi * 1e6;
    alpha = (m * k / h_bar)^(1 / 4);
    H = zeros(n + 1, n + 1);
    H(1, 1) = 1; H(2, 2) = 2;
    for i = 3:n + 1
        H(i, :) = 2 * [0, H(i - 1, 1:end - 1)] - 2 * (i - 2) * H(i - 2, :);
    end
    Tmp = sqrt(alpha ./ ((2.^n) .* factorial(n) .* sqrt(pi)));
    Phi = cell(size(n));
    for i = 1:length(n)
        Phi{i} = zeros(size(y));
        for j = 1:max(n) + 1
            Phi{i} = Phi{i} + H(n(i) + 1, j) * ((alpha * y).^(j - 1));
        end
        Phi{i} = [y, exp(-0.5 .* alpha^2 .* y.^2) .* Tmp(i) .* Phi{i}];
    end
    Out = cell2mat(Phi);
    Out = Out(1:size(y, 1), 1:size(y, 2));
end

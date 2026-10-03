% ----------------------------------------------------------------------- %
% L-SHADE with Eigenvector Crossover and Successful-Parent Selection (SPS-L-SHADE-EIG)
% CEC 2015 learning-based track -- 1st place (organiser's draft ranking)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 600 -> 4               % Linear reduction over the budget
%   H = 6, p = 0.11             % Memory size and pbest rate
%   arc_rate = 2.6              % Archive size, a multiple of NP
%   Q = 64                      % Consecutive failures before SPS takes over
%   cw = 0.3 -> 0 (linear)      % Learning rate of the accumulated covariance
%   CR in [0.05, 0.3]           % Crossover rate clamp
%   fw = 0.1, crw = 0.1, erw = 0.2  % Cauchy/Gaussian widths for F, CR and ER
%
% Algorithm Concept:
%   - current-to-pbest/1 with an external archive, as in L-SHADE
%   - Every individual carries a consecutive-failure counter; past Q failures its
%     mutation and crossover read the successful-parent archive SP, not the
%     current population, so a stagnated individual restarts from what worked
%   - SP holds the last NP successful trials, written cyclically
%   - With probability ER the crossover runs in the eigenvector basis of an
%     accumulated covariance matrix, which makes it rotation invariant;
%     otherwise it is binomial
%   - A coordinate outside the box is moved to the midpoint between the bound
%     and its parent
%   - F, CR and ER memories take fitness-weighted means, Lehmer for F
%
% Reference:
% Shu-Mei Guo, Jason Sheng-Hong Tsai, Chin-Chang Yang, Pang-Han Hsu,
% A self-optimization approach for L-SHADE incorporated with eigenvector-based
% crossover and successful-parent-selecting framework on CEC 2015 benchmark set,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 1003-1010.
% https://doi.org/10.1109/CEC.2015.7256999
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released CEC2015 code (SPS_L_SHADE_EIG.m), whose
% framework around the search -- recording, plotting, epsilon constraint
% handling, noise handling -- is dropped; only 'Interpolation' bound handling is
% kept, which is what its defaults select. Two of its quirks are reproduced
% deliberately: the archive stores the successful CHILD, where L-SHADE stores the
% replaced parent, and after the population shrinks the SP rank index keeps its
% pre-shrink values, which stay in range and only scramble which SP member counts
% as pbest. The reference stops NP evaluations short of the budget; here the run
% ends on FE >= maxFe like every other algorithm, and NP is 4 by then, so it
% overruns by at most three evaluations.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sps_lshade_eig(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    NP_init = 600;
    NP_min = 4;
    H = 6;
    p_best_rate = 0.11;
    arc_rate = 2.6;
    Q = 64;
    cw_init = 0.3;
    fw = 0.1;
    crw = 0.1;
    erw = 0.2;
    CR_min = 0.05;
    CR_max = 0.3;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    NP = NP_init;
    X = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(ub - lb, NP, 1);
    [fx, FE] = calculate_fitness(X', problem, FE);
    fx = fx(:);

    bsf_fit_var = inf;
    bsf_solution = zeros(1, dim);
    for i = 1:NP
        if fx(i) < bsf_fit_var
            bsf_fit_var = fx(i);
            bsf_solution = X(i, :);
        end
        if i <= maxFE
            curve(i) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, fx', population_history, fitness_history, history_index, maxFE);
        end
    end

    [fx, sort_idx] = sort(fx);
    X = X(sort_idx, :);

    MF = 0.5 * ones(H, 1);
    MCR = 0.5 * ones(H, 1);
    MER = ones(H, 1);
    iM = 1;
    FC = zeros(NP, 1);        % consecutive failures, per individual
    C = cov(X);               % accumulated covariance, the basis of the eigen crossover
    SP = X;                   % successful-parent archive, written cyclically
    fSP = fx;
    iSP = 1;
    [~, sortidx_fSP] = sort(fSP);
    A = zeros(0, dim);        % external archive
    cw = cw_init;

    while FE < maxFE
        r = floor(1 + H * rand(NP, 1));

        ER = MER(r) + erw * randn(NP, 1);
        ER = min(max(ER, 0), 1);

        F = MF(r) + fw * tan(pi * (rand(NP, 1) - 0.5));
        pos = find(F <= 0);
        while ~isempty(pos)
            F(pos) = MF(r(pos)) + fw * tan(pi * (rand(numel(pos), 1) - 0.5));
            pos = find(F <= 0);
        end
        F = min(F, 1);

        CR = MCR(r) + crw * randn(NP, 1);
        CR = min(max(CR, CR_min), CR_max);

        pbest = 1 + floor(max(2, round(p_best_rate * NP)) * rand(NP, 1));

        nA = size(A, 1);
        XA = [X; A];
        SPA = [SP; A];

        r1 = zeros(NP, 1);
        r2 = zeros(NP, 1);
        for i = 1:NP
            r1(i) = floor(1 + NP * rand);
            while i == r1(i)
                r1(i) = floor(1 + NP * rand);
            end
            % The reference rejects r2 == r1 but never r2 == i
            r2(i) = floor(1 + (NP + nA) * rand);
            while i == r1(i) || r1(i) == r2(i)
                r2(i) = floor(1 + (NP + nA) * rand);
            end
        end

        use_sp = FC > Q;          % stagnated individuals mutate out of SP instead
        base = X;
        base(use_sp, :) = SP(use_sp, :);

        V = zeros(NP, dim);
        n = ~use_sp;
        V(n, :) = X(n, :) + F(n) .* (X(pbest(n), :) - X(n, :)) ...
                          + F(n) .* (X(r1(n), :) - XA(r2(n), :));
        s = use_sp;
        V(s, :) = SP(s, :) + F(s) .* (SP(sortidx_fSP(pbest(s)), :) - SP(s, :)) ...
                           + F(s) .* (SP(r1(s), :) - SPA(r2(s), :));

        C = (1 - cw) * C + cw * cov(X);
        [B, ~] = eig(C);

        mask = rand(NP, dim) < CR(:, ones(1, dim));   % true takes the mutant coordinate
        jrand = floor(rand(NP, 1) * dim) + 1;
        mask(sub2ind([NP dim], (1:NP)', jrand)) = true;

        use_eig = rand(NP, 1) < ER;
        U = base;
        b = ~use_eig;
        Ub = base(b, :); Vb = V(b, :); mb = mask(b, :);
        Ub(mb) = Vb(mb);
        U(b, :) = Ub;
        e = use_eig;
        if any(e)
            % Rows here are the reference's columns, so B' * x becomes x * B
            XT = base(e, :) * B;
            VT = V(e, :) * B;
            me = mask(e, :);
            XT(me) = VT(me);
            U(e, :) = XT * B';
        end

        LB = repmat(lb, NP, 1);
        UB = repmat(ub, NP, 1);
        below = U < LB;
        above = U > UB;
        U(below) = 0.5 * (LB(below) + base(below));
        U(above) = 0.5 * (UB(above) + base(above));

        [fu, FE] = calculate_fitness(U', problem, FE);
        fu = fu(:);

        for i = 1:NP
            if fu(i) < bsf_fit_var
                bsf_fit_var = fu(i);
                bsf_solution = U(i, :);
            end
        end

        for eval_idx = 1:NP
            eval_count = FE - NP + eval_idx;
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = bsf_fit_var;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, X, fx', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        S_ER = []; S_F = []; S_CR = []; S_df = [];
        A_size = round(arc_rate * NP);
        for i = 1:NP
            if fu(i) < fx(i)
                S_ER(end+1, 1) = ER(i); %#ok<AGROW>
                S_F(end+1, 1) = F(i);   %#ok<AGROW>
                S_CR(end+1, 1) = CR(i); %#ok<AGROW>
                S_df(end+1, 1) = abs(fu(i) - fx(i)); %#ok<AGROW>
                X(i, :) = U(i, :);
                fx(i) = fu(i);
                % The reference archives the accepted child, not the parent it replaced
                if size(A, 1) < A_size
                    A(end+1, :) = X(i, :); %#ok<AGROW>
                elseif A_size > 0
                    A(floor(1 + A_size * rand), :) = X(i, :);
                end
                FC(i) = 0;
                SP(iSP, :) = U(i, :);
                fSP(iSP) = fu(i);
                iSP = mod(iSP, NP) + 1;
            else
                FC(i) = FC(i) + 1;
            end
        end

        if ~isempty(S_df)
            w = S_df ./ sum(S_df);
            MER(iM) = sum(w .* S_ER);
            MCR(iM) = sum(w .* S_CR);
            MF(iM) = sum(w .* S_F .* S_F) / sum(w .* S_F);
            iM = mod(iM, H) + 1;
        end

        cw = (1 - FE / maxFE) * cw_init;

        [fx, sort_idx] = sort(fx);
        X = X(sort_idx, :);
        FC = FC(sort_idx);

        NP_new = round(NP_init - (NP_init - NP_min) * FE / maxFE);
        NP_new = max(NP_new, NP_min);
        if NP_new < NP
            fx = fx(1:NP_new);
            X = X(1:NP_new, :);
            FC = FC(1:NP_new);
            A_size = round(arc_rate * NP_new);
            if size(A, 1) > A_size
                A = A(1:A_size, :);
            end
            % Keeps the SP members whose pre-shrink rank fits the new size
            [~, sortidx_fSP] = sort(fSP);
            keep = sortidx_fSP <= NP_new;
            SP = SP(keep, :);
            fSP = fSP(keep);
            sortidx_fSP = sortidx_fSP(keep);
            iSP = mod(iSP - 1, NP_new) + 1;
            NP = NP_new;
        else
            [~, sortidx_fSP] = sort(fSP);
        end
    end

    curve(min(FE, maxFE):end) = bsf_fit_var;

    best_fitness = bsf_fit_var;
    best_solution = bsf_solution;
end

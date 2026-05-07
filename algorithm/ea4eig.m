% ----------------------------------------------------------------------- %
% EA4eig Algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N_init = 100              % Initial population size
%   Nmin = 10                 % Minimum population size
%   h = 4                     % Number of cooperating heuristics
%   peig = 0.4                % Eigenvector crossover probability
%   CBps = 0.5                % Population ratio for eigenvector basis
%
% Algorithm Concept:
%   - Cooperative ensemble of four evolutionary heuristics:
%     CoBiDE, IDE, CMA-ES, and jSO
%   - Roulette selection chooses the active heuristic according to
%     accumulated success counts
%   - Eigenvector crossover is used to improve coordinate-system
%     adaptation in the cooperating heuristics
%   - Linear population size reduction is applied from N_init to Nmin
%
% Reference:
% Petr Bujok and Patrik Kolenovsky,
% Eigen Crossover in Cooperative Model of Evolutionary Algorithms Applied
% to CEC 2022 Single Objective Numerical Optimisation,
% 2022 IEEE Congress on Evolutionary Computation (CEC), 2022, pp. 1-8.
% https://doi.org/10.1109/CEC55065.2022.9870433
% ----------------------------------------------------------------------- %
% Implementation Note:
% Integrated single-file version adapted to the repository algorithm API.
%
% The original EA4eig runner was written as a standalone CEC script with
% external helper files and text output. This version keeps the algorithmic
% operators in local functions and uses the shared problem/calculate_fitness
% interface used by the rest of the system.
%
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ea4eig(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = max(1, round(problem.maxFe));

    if isscalar(lb)
        lb = repmat(lb, 1, dim);
    end
    if isscalar(ub)
        ub = repmat(ub, 1, dim);
    end

    % Original EA4eig_N100_10 population schedule.
    N_init = min(100, maxFE);
    N = N_init;
    Nmin = min(10, N_init);
    h = 4;
    n0 = 2;

    FE = 0;
    curve = zeros(1, maxFE);
    last_curve_fe = 0;

    % EA4eig reduces population size, so history stores the current top
    % solutions in a fixed-width array and pads unused rows with NaN.
    history_pop_size = 100;
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = NaN(history_size, history_pop_size, dim);
    fitness_history = NaN(history_size, history_pop_size);
    history_index = 1;

    ni = zeros(1, h) + n0;

    % Initialize the main population.
    P = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    [fitness, FE] = calculate_fitness(P', problem, FE);
    P(:, dim + 1) = fitness(:);

    [bsf_fit_var, best_idx] = min(P(:, dim + 1));
    best_solution_current = P(best_idx, 1:dim);

    [curve, last_curve_fe] = fill_curve(curve, last_curve_fe, FE, bsf_fit_var);
    [population_history, fitness_history, history_index] = record_ea4eig_history( ...
        min(FE, maxFE), P, dim, history_pop_size, population_history, ...
        fitness_history, history_index, sampling_interval, history_size);

    if FE >= maxFE
        [best_fitness, best_solution, curve] = finalize_output( ...
            bsf_fit_var, best_solution_current, curve, last_curve_fe);
        return;
    end

    delta = 1 / (5 * h);

    % IDE parameters.
    gmax = max(1, round(maxFE / N));
    SRg = zeros(gmax, 1) + 2; %#ok<NASGU>
    T = gmax / 10;
    GT = fix(gmax / 2);
    gt = GT;
    g = 0;
    Tcurr = 0;

    % CoBiDE / eigenvector crossover parameters.
    CBps = 0.5;
    peig = 0.4;
    CBF = zeros(N, 1);
    CBCR = zeros(N, 1);
    for i = 1:N
        CBF(i) = sample_cobide_f();
        CBCR(i) = sample_cobide_cr();
    end

    % CMA-ES parameters.
    sigma = max((ub(1) - lb(1)) / 2, eps);
    oldPop = P(:, 1:dim)';
    myeps = 1e-15;
    [mu, weights, mueff, cc, cs, c1, cmu, damps] = cma_parameters(N, dim);

    pc = zeros(dim, 1);
    ps = zeros(dim, 1);
    B = eye(dim, dim);
    D = ones(dim, 1);
    CC = B * diag(D .^ 2) * B';
    invsqrtC = B * diag(D .^ -1) * B';
    eigeneval = 0;
    chiN = dim ^ 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ^ 2));

    % jSO parameters.
    Asize_max = round(N * 2.6);
    H = 5;
    MF = 0.3 * ones(1, H);
    MCR = 0.8 * ones(1, H);
    MF(1, H) = 0.9;
    MCR(1, H) = 0.9;
    k = 1;
    Asize = 0;
    A = [];
    pmax = 0.25;
    pmin = pmax / 2;

    terminate = false;

    while FE < maxFE && ~terminate
        [hh, p_min] = roulette_select(ni);
        if p_min < delta
            ni = zeros(1, h) + n0;
        end

        switch hh
            case 1
                % One CoBiDE generation.
                Q = zeros(N, dim + 1);
                if rand < peig
                    Popeig = sortrows(P, dim + 1);
                    keep_count = max(2, min(N, round(N * CBps)));
                    Popeig(keep_count + 1:N, :) = [];
                    EigVect = safe_eig_vectors(Popeig(:, 1:dim), dim);

                    for i = 1:N
                        vyb = random_except(N, 3, i);
                        r1 = P(vyb(1), 1:dim);
                        r2 = P(vyb(2), 1:dim);
                        r3 = P(vyb(3), 1:dim);
                        v = r1 + CBF(i) * (r2 - r3);
                        v = mirror_bounds(v, lb, ub);

                        y = P(i, 1:dim);
                        yeig = EigVect' * y';
                        veig = EigVect' * v';
                        change = find(rand(1, dim) < CBCR(i));
                        if isempty(change)
                            change = 1 + fix(dim * rand(1));
                        end
                        yeig(change) = veig(change);
                        y = (EigVect * yeig)';
                        y = mirror_bounds(y, lb, ub);

                        [fy, FE] = calculate_fitness(y', problem, FE);
                        Q(i, :) = [y, fy(1)];
                    end
                else
                    for i = 1:N
                        vyb = random_except(N, 3, i);
                        r1 = P(vyb(1), 1:dim);
                        r2 = P(vyb(2), 1:dim);
                        r3 = P(vyb(3), 1:dim);
                        v = r1 + CBF(i) * (r2 - r3);
                        v = mirror_bounds(v, lb, ub);

                        y = P(i, 1:dim);
                        change = find(rand(1, dim) < CBCR(i));
                        if isempty(change)
                            change = 1 + fix(dim * rand(1));
                        end
                        y(change) = v(change);
                        y = mirror_bounds(y, lb, ub);

                        [fy, FE] = calculate_fitness(y', problem, FE);
                        Q(i, :) = [y, fy(1)];
                    end
                end

                for i = 1:N
                    if Q(i, dim + 1) <= P(i, dim + 1)
                        P(i, :) = Q(i, :);
                        ni(hh) = ni(hh) + 1;
                    else
                        CBF(i) = sample_cobide_f();
                        CBCR(i) = sample_cobide_cr();
                    end
                end

            case 2
                % One IDE generation.
                [P, I] = sortrows(P, dim + 1);
                CBF = CBF(I);
                CBCR = CBCR(I);
                Q = P;
                IDEps = 0.1 + 0.9 * 10^(5 * (g / gmax - 1));
                pd = 0.1 * IDEps;
                if g < gt
                    SRT = 0;
                else
                    SRT = 0.1;
                end

                for i = 1:N
                    vyb = random_except(N, 4, i);
                    o = vyb(1);
                    if g <= gt
                        o = i;
                    end
                    r1 = vyb(2);
                    r2 = vyb(3);
                    r3 = vyb(4);
                    xo = P(o, 1:dim);
                    xr1 = P(r1, 1:dim);
                    xr2 = P(r2, 1:dim);
                    xr3 = P(r3, 1:dim);

                    indperturb = find(rand(1, dim) < pd);
                    random_point = lb + rand(1, dim) .* (ub - lb);
                    xr3(indperturb) = random_point(indperturb);

                    Fo = o / N + 0.1 * randn(1);
                    while Fo <= 0 || Fo > 1
                        Fo = o / N + 0.1 * randn(1);
                    end

                    high_ind_S = max(1, fix(IDEps * N));
                    if o > high_ind_S && r1 > high_ind_S
                        candidates = setdiff(1:high_ind_S, unique([vyb, i]));
                        if ~isempty(candidates)
                            r1 = candidates(1 + fix(rand(1) * length(candidates)));
                            xr1 = P(r1, 1:dim);
                        end
                    end

                    if (g > gt) && (rand < 0.5)
                        Q(i, 1:dim) = P(i, 1:dim) + Fo * (xr1 - xo) + Fo * (xr2 - xr3);
                    else
                        Q(i, 1:dim) = xo + Fo * (xr1 - xo) + Fo * (xr2 - xr3);
                    end
                end

                if rand < peig
                    Popeig = sortrows(P, dim + 1);
                    keep_count = max(2, min(N, round(N * CBps)));
                    Popeig(keep_count + 1:N, :) = [];
                    EigVect = safe_eig_vectors(Popeig(:, 1:dim), dim);

                    for i = 1:N
                        y = P(i, 1:dim);
                        v = Q(i, 1:dim);
                        yeig = EigVect' * y';
                        veig = EigVect' * v';
                        change = find(rand(1, dim) < CBCR(i));
                        if isempty(change)
                            change = 1 + fix(dim * rand(1));
                        end
                        yeig(change) = veig(change);
                        y = (EigVect * yeig)';
                        Q(i, 1:dim) = mirror_bounds(y, lb, ub);
                    end
                else
                    for i = 1:N
                        CR = i / N + 0.1 * randn(1);
                        while CR < 0 || CR > 1
                            CR = i / N + 0.1 * randn(1);
                        end
                        jrand = 1 + fix(rand(1) * dim);
                        for j = 1:dim
                            if ~(rand(1) <= CR || j == jrand)
                                Q(i, j) = P(i, j);
                            end
                            if Q(i, j) < lb(j) || Q(i, j) > ub(j)
                                Q(i, j) = lb(j) + rand(1) * (ub(j) - lb(j));
                            end
                        end
                    end
                end

                [fitQ, FE] = calculate_fitness(Q(:, 1:dim)', problem, FE);
                Q(:, dim + 1) = fitQ(:);
                indsucc = find(Q(:, dim + 1) <= P(:, dim + 1));
                ni(hh) = ni(hh) + length(indsucc);
                SR = length(indsucc) / N;
                if g < gt
                    if SR <= SRT
                        Tcurr = Tcurr + 1;
                    else
                        Tcurr = 0;
                    end
                    if Tcurr >= T
                        gt = g;
                    end
                end
                P(indsucc, :) = Q(indsucc, :);
                [P, I] = sortrows(P, dim + 1);
                CBF = CBF(I);
                CBCR = CBCR(I);
                g = g + 1;

            case 3
                % One CMA-ES generation.
                [P, I] = sortrows(P, dim + 1);
                CBF = CBF(I);
                CBCR = CBCR(I);

                xmean = P(1:mu, 1:dim)' * weights;
                Pop = zeros(dim, N);
                PopFit = zeros(1, N);

                for kk = 1:N
                    Pop(:, kk) = xmean + sigma * B * (D .* randn(dim, 1));

                    zrca = find(Pop(:, kk) < lb');
                    if ~isempty(zrca)
                        Pop(zrca, kk) = (oldPop(zrca, kk) + lb(zrca)') / 2;
                    end
                    zrcb = find(Pop(:, kk) > ub');
                    if ~isempty(zrcb)
                        Pop(zrcb, kk) = (oldPop(zrcb, kk) + ub(zrcb)') / 2;
                    end

                    [fit_kk, FE] = calculate_fitness(Pop(:, kk), problem, FE);
                    PopFit(kk) = fit_kk(1);

                    [maxf, maxind] = max(P(:, dim + 1));
                    if PopFit(kk) < maxf
                        P(maxind, :) = [Pop(:, kk)', PopFit(kk)];
                        ni(hh) = ni(hh) + 1;
                    end
                end

                [PopFit, FitInd] = sort(PopFit);
                xold = xmean;
                xmean = Pop(:, FitInd(1:mu)) * weights;

                oldPop = Pop;
                ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
                hsig = sum(ps .^ 2) / (1 - (1 - cs) ^ (2 * FE / N)) / dim < 2 + 4 / (dim + 1);
                pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

                artmp = (1 / sigma) * (Pop(:, FitInd(1:mu)) - repmat(xold, 1, mu));
                CC = (1 - c1 - cmu) * CC + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * CC) + cmu * artmp * diag(weights) * artmp';
                sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));
                if (sigma > 1e+300) || (sigma < 1e-300) || isnan(sigma)
                    sigma = max((ub(1) - lb(1)) / 2, eps);
                end

                if FE - eigeneval > N / (c1 + cmu) / dim / 10
                    eigeneval = FE;
                    CC = triu(CC) + triu(CC, 1)';
                    [B, eigD] = eig(CC);
                    D = sqrt(max(real(diag(eigD)), realmin));
                    invsqrtC = B * diag(D .^ -1) * B';
                end

                if (PopFit(1) <= myeps) || (max(D) > 1e7 * max(min(D), realmin))
                    terminate = true;
                end

            case 4
                % One jSO generation.
                Fpole = -1 * ones(1, N);
                CRpole = -1 * ones(1, N);
                SCR = [];
                SF = [];
                suc = 0;
                Q = zeros(N, dim + 1);
                deltafce = -1 * ones(1, N);
                pp = pmax - ((pmax - pmin) * (FE / maxFE));

                if rand < peig
                    Popeig = sortrows(P, dim + 1);
                    keep_count = max(2, min(N, round(N * CBps)));
                    Popeig(keep_count + 1:N, :) = [];
                    EigVect = safe_eig_vectors(Popeig(:, 1:dim), dim);

                    for i = 1:N
                        [F, CR] = sample_jso_params(MF, MCR, H, FE, maxFE);
                        Fpole(1, i) = F;
                        CRpole(1, i) = CR;

                        p = max(2, ceil(pp * N));
                        pom = sortrows(P, dim + 1);
                        pbest = pom(1:p, 1:dim);
                        xpbest = pbest(1 + fix(p * rand(1)), :);

                        xi = P(i, 1:dim);
                        vyb = random_except(N, 1, i);
                        r1 = P(vyb, 1:dim);
                        expt = [i, vyb];

                        vyb = random_except(N + Asize, 1, expt);
                        sjed = [P(:, 1:dim); A];
                        r2 = sjed(vyb, :);

                        Fw = weighted_jso_f(F, FE, maxFE);
                        v = xi + Fw * (xpbest - xi) + F * (r1 - r2);
                        y = xi;

                        yeig = EigVect' * y';
                        veig = EigVect' * v';
                        change = find(rand(1, dim) < CBCR(i));
                        if isempty(change)
                            change = 1 + fix(dim * rand(1));
                        end
                        yeig(change) = veig(change);
                        y = (EigVect * yeig)';
                        Q(i, 1:dim) = mirror_bounds(y, lb, ub);
                    end
                else
                    for i = 1:N
                        [F, CR] = sample_jso_params(MF, MCR, H, FE, maxFE);
                        Fpole(1, i) = F;
                        CRpole(1, i) = CR;

                        p = max(2, ceil(pp * N));
                        pom = sortrows(P, dim + 1);
                        pbest = pom(1:p, 1:dim);
                        xpbest = pbest(1 + fix(p * rand(1)), :);

                        xi = P(i, 1:dim);
                        vyb = random_except(N, 1, i);
                        r1 = P(vyb, 1:dim);
                        expt = [i, vyb];

                        vyb = random_except(N + Asize, 1, expt);
                        sjed = [P(:, 1:dim); A];
                        r2 = sjed(vyb, :);

                        Fw = weighted_jso_f(F, FE, maxFE);
                        v = xi + Fw * (xpbest - xi) + F * (r1 - r2);
                        y = xi;

                        change = find(rand(1, dim) < CR);
                        if isempty(change)
                            change = 1 + fix(dim * rand(1));
                        end
                        y(change) = v(change);
                        Q(i, 1:dim) = mirror_bounds(y, lb, ub);
                    end
                end

                [fitQ, FE] = calculate_fitness(Q(:, 1:dim)', problem, FE);
                Q(:, dim + 1) = fitQ(:);

                for i = 1:N
                    if Q(i, dim + 1) < P(i, dim + 1)
                        deltafce(1, i) = P(i, dim + 1) - Q(i, dim + 1);
                        suc = suc + 1;
                        if Asize < Asize_max
                            A = [A; P(i, 1:dim)]; %#ok<AGROW>
                            Asize = Asize + 1;
                        else
                            archive_idx = random_sample(Asize, 1);
                            A(archive_idx, :) = P(i, 1:dim); %#ok<AGROW>
                        end
                        SCR = [SCR, CRpole(1, i)]; %#ok<AGROW>
                        SF = [SF, Fpole(1, i)]; %#ok<AGROW>
                    end
                    if Q(i, dim + 1) <= P(i, dim + 1)
                        P(i, :) = Q(i, :);
                    end
                end

                if suc > 0
                    MCR_old = MCR(1, k);
                    MF_old = MF(1, k);
                    delty = deltafce(deltafce ~= -1);
                    sum_delta = sum(delty);
                    vahyw = (1 / sum_delta) * delty;
                    mSCR = max(SCR);
                    if (MCR(1, k) == -1) || (mSCR == 0)
                        MCR(1, k) = -1;
                    else
                        meanSCRpomjm = vahyw .* SCR;
                        meanSCRpomci = meanSCRpomjm .* SCR;
                        MCR(1, k) = sum(meanSCRpomci) / sum(meanSCRpomjm);
                    end
                    meanSFpomjm = vahyw .* SF;
                    meanSFpomci = meanSFpomjm .* SF;
                    MF(1, k) = sum(meanSFpomci) / sum(meanSFpomjm);
                    MCR(1, k) = (MCR(1, k) + MCR_old) / 2;
                    MF(1, k) = (MF(1, k) + MF_old) / 2;
                    k = k + 1;
                    if k >= H
                        k = 1;
                    end
                end

                ni(hh) = ni(hh) + suc;
        end

        [current_best, best_idx] = min(P(:, dim + 1));
        if current_best < bsf_fit_var
            bsf_fit_var = current_best;
            best_solution_current = P(best_idx, 1:dim);
        end

        [curve, last_curve_fe] = fill_curve(curve, last_curve_fe, FE, bsf_fit_var);
        [population_history, fitness_history, history_index] = record_ea4eig_history( ...
            min(FE, maxFE), P, dim, history_pop_size, population_history, ...
            fitness_history, history_index, sampling_interval, history_size);

        optN = round((((Nmin - N_init) / maxFE) * FE) + N_init);
        optN = max(Nmin, optN);

        if N > optN
            diffPop = N - optN;
            if N - diffPop < Nmin
                diffPop = N - Nmin;
            end

            N = N - diffPop;
            [P, I] = sortrows(P, dim + 1);
            CBF = CBF(I);
            CBCR = CBCR(I);

            P(N + 1:end, :) = [];
            CBF(N + 1:end) = [];
            CBCR(N + 1:end) = [];
            Asize_max = round(N * 2.6);
            while Asize > Asize_max
                index_v_arch = random_sample(Asize, 1);
                A(index_v_arch, :) = []; %#ok<AGROW>
                Asize = Asize - 1;
            end

            [mu, weights, mueff, cc, cs, c1, cmu, damps] = cma_parameters(N, dim);
        end
    end

    [best_fitness, best_solution, curve] = finalize_output( ...
        bsf_fit_var, best_solution_current, curve, last_curve_fe);
end

%% --- Progress helpers ---
function [curve, last_curve_fe] = fill_curve(curve, last_curve_fe, FE, best_fit)
    maxFE = length(curve);
    capped_fe = min(FE, maxFE);
    if capped_fe > last_curve_fe
        curve(last_curve_fe + 1:capped_fe) = best_fit;
        last_curve_fe = capped_fe;
    end
end

function [best_fitness, best_solution, curve] = finalize_output(best_fit, best_sol, curve, last_curve_fe)
    if last_curve_fe < length(curve)
        curve(last_curve_fe + 1:end) = best_fit;
    end
    best_fitness = best_fit;
    best_solution = best_sol;
end

function [pop_hist, fit_hist, hist_idx] = record_ea4eig_history( ...
    current_fe, P, dim, history_pop_size, pop_hist, fit_hist, ...
    hist_idx, sampling_interval, history_size)

    if current_fe < 1
        return;
    end

    if mod(current_fe, sampling_interval) == 0 || hist_idx <= history_size
        if hist_idx <= history_size
            [sorted_fit, sorted_idx] = sort(P(:, dim + 1));
            current_size = size(P, 1);
            top_k = min(history_pop_size, current_size);

            rec_pop = NaN(history_pop_size, dim);
            rec_fit = NaN(1, history_pop_size);
            rec_pop(1:top_k, :) = P(sorted_idx(1:top_k), 1:dim);
            rec_fit(1:top_k) = sorted_fit(1:top_k)';

            pop_hist(hist_idx, :, :) = rec_pop;
            fit_hist(hist_idx, :) = rec_fit;
            hist_idx = hist_idx + 1;
        end
    end
end

%% --- Parameter helpers ---
function F = sample_cobide_f()
    if rand < 0.5
        F = cauchy_rnd(0.65, 0.1);
    else
        F = cauchy_rnd(1, 0.1);
    end
    while F < 0
        if rand < 0.5
            F = cauchy_rnd(0.65, 0.1);
        else
            F = cauchy_rnd(1, 0.1);
        end
    end
    F = min(F, 1);
end

function CR = sample_cobide_cr()
    if rand < 0.5
        CR = cauchy_rnd(0.1, 0.1);
    else
        CR = cauchy_rnd(0.95, 0.1);
    end
    CR = min(max(CR, 0), 1);
end

function [F, CR] = sample_jso_params(MF, MCR, H, FE, maxFE)
    rr = random_sample(H, 1);
    CR = MCR(1, rr) + sqrt(0.1) * randn(1);
    CR = min(max(CR, 0), 1);

    if FE < (0.25 * maxFE)
        CR = max([CR, 0.7]);
    elseif FE < (0.5 * maxFE)
        CR = max([CR, 0.6]);
    end

    F = -1;
    while F <= 0
        F = rand * pi - pi / 2;
        F = 0.1 * tan(F) + MF(1, rr);
    end
    F = min(F, 1);

    if (FE < (0.6 * maxFE)) && (F > 0.7)
        F = 0.7;
    end
end

function Fw = weighted_jso_f(F, FE, maxFE)
    if FE < 0.2 * maxFE
        Fw = 0.7 * F;
    elseif FE < 0.4 * maxFE
        Fw = 0.8 * F;
    else
        Fw = 1.2 * F;
    end
end

function [mu, weights, mueff, cc, cs, c1, cmu, damps] = cma_parameters(N, dim)
    mu = max(1, floor(N / 2));
    weights = log(mu + 1 / 2) - log((1:mu)');
    weights = weights / sum(weights);
    mueff = sum(weights) ^ 2 / sum(weights .^ 2);
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim);
    cs = (mueff + 2) / (dim + mueff + 5);
    c1 = 2 / ((dim + 1.3) ^ 2 + mueff);
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ^ 2 + mueff));
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs;
end

%% --- Selection and boundary helpers ---
function y = cauchy_rnd(x0, gamma)
    y = x0 + gamma * tan(pi * (rand(1) - 1 / 2));
end

function [res, p_min] = roulette_select(cutpoints)
    h = length(cutpoints);
    ss = sum(cutpoints);
    p_min = min(cutpoints) / ss;
    cp = zeros(1, h);
    cp(1) = cutpoints(1);
    for i = 2:h
        cp(i) = cp(i - 1) + cutpoints(i);
    end
    cp = cp / ss;
    res = 1 + fix(sum(cp < rand(1)));
end

function sample = random_sample(N, k)
    support = 1:N;
    sample = zeros(1, k);
    for i = 1:k
        index = 1 + fix(rand(1) * length(support));
        sample(i) = support(index);
        support(index) = [];
    end
end

function sample = random_except(N, k, excluded)
    support = 1:N;
    excluded = excluded(excluded >= 1 & excluded <= N);
    support(unique(excluded)) = [];
    sample = zeros(1, k);
    for i = 1:k
        index = 1 + fix(rand(1) * length(support));
        sample(i) = support(index);
        support(index) = [];
    end
end

function result = mirror_bounds(y, lb, ub)
    result = y;
    for i = find(result < lb | result > ub)
        if ub(i) <= lb(i)
            result(i) = lb(i);
            continue;
        end
        while result(i) < lb(i) || result(i) > ub(i)
            if result(i) > ub(i)
                result(i) = 2 * ub(i) - result(i);
            elseif result(i) < lb(i)
                result(i) = 2 * lb(i) - result(i);
            end
        end
    end
end

function EigVect = safe_eig_vectors(population, dim)
    if size(population, 1) < 2
        EigVect = eye(dim);
        return;
    end

    C = cov(population);
    if any(~isfinite(C(:)))
        EigVect = eye(dim);
        return;
    end

    [EigVect, ~] = eig(C);
    if any(~isfinite(EigVect(:)))
        EigVect = eye(dim);
    end
end

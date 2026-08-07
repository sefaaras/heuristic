% ----------------------------------------------------------------------- %
% Hybrid Sampling Evolution Strategy (HS-ES)
% CEC 2018 competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   total = 200, mu = 100        % Stage 1, over 100 sampling generations
%   lambda = floor(3*log(D))+80  % Stage 2 CMA-ES population
%   sigma0 = 0.2                 % Stage 2 initial step size
%   Times = 2 (D <= 30) else 1   % Stage 2 restarts
%   stopeval = maxFE/4 (D <= 30) else maxFE/2   % Per-restart cap
%   total/mu = 200/160, 450/360, 600/480        % Stage 3, by D <= 30, 50, > 50
%   +200/+160 while FE <= 0.3*maxFE             % Stage 3 growth for D >= 50
%
% Algorithm Concept:
%   - Three stages, each seeding the next:
%     1. Univariate Gaussian sampling (a simple EDA) from N(mean, std) estimated
%        per dimension from the weighted best mu
%     2. CMA-ES restarts seeded from the stage-1 best, supplying the covariance
%        information the univariate model cannot represent
%     3. Univariate sampling again with the dimensions the restarts agreed on
%        FROZEN, so the remaining budget goes only to those that still matter
%   - Which dimensions freeze is decided by size: for D <= 30 from the spread of
%     the two restarts' best vectors, for larger D from a sensitivity probe
%   - Both sampling stages shrink their step to 0.96*randn once the best value
%     has stagnated for 20 generations
%
% Reference:
% Guohua Zhang, Yuhui Shi,
% Hybrid Sampling Evolution Strategy for Solving Single Objective Bound
% Constrained Problems,
% 2018 IEEE Congress on Evolutionary Computation (CEC), 2018, pp. 1-7.
% https://doi.org/10.1109/CEC.2018.8477908
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (HSES.m, "Codes for best 3" in
% Suganthan's CEC2018 repository), a competition script written for a fixed
% [-100,100] box and a 10000*D budget, so five adaptations were needed.
% (1) BOX: the mod(x,+/-100) repair becomes c + mod(x-c,+/-h) about the box
% centre and the position-space constants scale by mean(ub-lb)/200 -- both exact
% on a symmetric box, required for CEC2020RW. (2) BUDGET: every stage stops when
% the budget runs out and a phantom FE increment is dropped. (3) Both freeze
% branches test D > 30; the reference errors for 30 < D < 50. (4) One
% best-so-far pair is tracked across stages. (5) Negative eigenvalues clamp at 0.
% (6) The evaluation helper clamps its block, so nothing escapes the box.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hses(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';      % 1 x D
    ub    = problem.ub(:)';      % 1 x D
    maxFE = problem.maxFe;

    ctr  = (lb + ub) / 2;        % box centre
    hw   = (ub - lb) / 2;        % box half-width
    sc   = mean(ub - lb) / 200;  % position-space scale (1 on the standard CEC box)

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    ph = [];  % record_history allocates the metric buffers on its first sample
    fh = [];
    hi = 1;

    bsf  = inf;
    bsfx = lb + (ub - lb) .* rand(1, D);

    % Stage 1: univariate Gaussian sampling
    total = 200;
    mu    = 100;

    pos = repmat(lb, total, 1) + rand(total, D) .* repmat(ub - lb, total, 1);
    [e, FE, bsf, bsfx, curve, ph, fh, hi] = evalPop(pos, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

    weights = log(mu + 1/2) - log(1:mu)';
    weights = weights / sum(weights);

    [a1, a2] = sort(e);
    top      = pos(a2(1:mu), :);
    meanval  = mean(top);
    stdval   = std(top);

    pos = repmat(meanval, total, 1) + repmat(stdval, total, 1) .* randn(total, D);
    pos = resampleOut(pos, meanval, stdval, lb, ub);

    cc1 = 0;
    FV  = zeros(1, 100);
    for kk = 1:100
        if FE >= maxFE, break; end

        [e, FE, bsf, bsfx, curve, ph, fh, hi] = evalPop(pos, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

        [a1, a2] = sort(e);
        newpos   = pos(a2(1:mu), :);
        meanval  = (newpos' * weights)';
        stdval   = std(newpos);
        FV(kk)   = a1(1);

        if kk > 30 && mod(kk, 20) == 0
            [~, aa2] = min(FV(1:kk));
            if aa2 < kk - 20
                cc1 = 1;
            end
        end

        if cc1 == 1
            step = 0.96 * randn(total, D);
        else
            step = randn(total, D);
        end
        pos = repmat(meanval, total, 1) + repmat(stdval, total, 1) .* step;
        pos = wrapBox(pos, ctr, hw, lb, ub);
    end

    previousbest = a1(1);
    bestvec      = bsfx;

    % Stage 2: CMA-ES restarts
    if D <= 30
        Times    = 2;
        stopeval = maxFE / 4;
    else
        Times    = 1;
        stopeval = maxFE / 2;
    end

    arfitnessbest = bsf .* ones(1, Times);
    xvalbest      = repmat(bestvec', 1, Times);

    N      = D;
    lambda = floor(3 * log(N)) + 80;
    mucma  = floor(lambda / 2);
    wcma   = log(lambda / 2 + 1/2) - log(1:mucma)';
    wcma   = wcma / sum(wcma);
    mueff  = sum(wcma) ^ 2 / sum(wcma .^ 2);
    cc     = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
    cs     = (mueff + 2) / (N + mueff + 5);
    c1     = 2 / ((N + 1.3) ^ 2 + mueff);
    cmu    = 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ^ 2 + 2 * mueff / 2);
    damps  = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
    chiN   = N ^ 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));

    for kkk = 1:Times
        if FE >= maxFE, break; end

        sigma     = 0.2 * sc;
        pc        = zeros(N, 1);
        ps        = zeros(N, 1);
        B         = eye(N);
        DD        = eye(N);
        C         = eye(N);
        eigenval  = 0;
        counteval = 0;
        xmean     = bestvec';

        while counteval < stopeval && FE < maxFE
            arz  = randn(N, lambda);
            arxx = repmat(xmean, 1, lambda) + sigma * (B * DD * arz);
            arxx = wrapBox(arxx', ctr, hw, lb, ub)';

            [arfitness, FE, bsf, bsfx, curve, ph, fh, hi] = ...
                evalPop(arxx', problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);
            counteval = counteval + lambda;

            [arfitness, arindex] = sort(arfitness);

            if abs(arfitness(1) - previousbest) < 1e-11
                break;
            else
                previousbest = arfitness(1);
            end

            if arfitnessbest(kkk) > arfitness(1)
                arfitnessbest(kkk) = arfitness(1);
                xvalbest(:, kkk)   = arxx(:, arindex(1));
            end

            xmean = arxx(:, arindex(1:mucma)) * wcma;
            zmean = arz(:,  arindex(1:mucma)) * wcma;

            ps   = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (B * zmean);
            hsig = norm(ps) / sqrt(1 - (1 - cs) ^ (2 * counteval / lambda)) / chiN < 1.4 + 2 / (N + 1);
            pc   = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (B * DD * zmean);

            BDz = B * DD * arz(:, arindex(1:mucma));
            C   = (1 - c1 - cmu) * C + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
                  + cmu * BDz * diag(wcma) * BDz';

            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));

            if counteval - eigenval > lambda / cmu / N / 10
                eigenval = counteval;
                C = triu(C) + triu(C, 1)';
                [B, DD] = eig(C);
                B  = real(B);
                % Clamp negative eigenvalues instead of going complex
                DD = diag(sqrt(max(real(diag(DD)), 0)));
            end

            if arfitness(1) == arfitness(ceil(0.7 * lambda))
                sigma = sigma * exp(0.2 + cs / damps);
            end
        end
    end

    % Stage 3: univariate sampling with frozen dims
    if D <= 30
        total = 200; mu = 160;
    elseif D == 50
        total = 450; mu = 360;
    else
        total = 600; mu = 480;
    end
    if D >= 50 && FE <= 0.3 * maxFE
        total = total + 200;
        mu    = mu + 160;
    end

    weights = log(mu + 1/2) - log(1:mu)';
    weights = weights / sum(weights);

    ppp1        = [];
    dividevalue = 0;
    bbpbb       = ones(1, D);

    if D <= 30
        % Freeze the dimensions the two CMA-ES restarts agreed on
        ppp1 = std(xvalbest');
        ppp2 = sort(ppp1);
        if ppp2(1) > 0.2 * sc
            dividevalue = 0;
        elseif max(ppp2) < 0.01 * sc
            dividevalue = 1 * sc;
        else
            indicatorppp = zeros(1, D);
            for dd = 2:D
                if ppp2(dd - 1) ~= 0
                    indicatorppp(dd) = (ppp2(dd) - ppp2(dd - 1)) / ppp2(dd - 1);
                else
                    indicatorppp(dd) = inf;
                end
            end
            finite_ind = indicatorppp(isfinite(indicatorppp));
            if isempty(finite_ind), finite_ind = 0; end
            indicatorppp(1) = min(finite_ind) - 0.001;
            [~, value2] = sort(indicatorppp, 'descend');
            for dd = 1:D
                v = value2(dd);
                if ppp2(v) < 10 * sc
                    if ppp2(v) > 0.1 * sc
                        dividevalue = ppp2(v) - 0.001 * sc;
                        break;
                    end
                elseif ppp2(max(v - 1, 1)) < 0.01 * sc
                    dividevalue = ppp2(v) - 0.001 * sc;
                    break;
                end
                if dd == D
                    dividevalue = ppp2(v) - 0.001 * sc;
                end
            end
        end
    elseif FE < maxFE
        % Coordinate-wise sensitivity probe around the best vector
        nprobe = round(total / 5);
        bbpbbp = zeros(1, D);
        spos   = repmat(xvalbest(:, 1)', nprobe, 1);
        denom  = max(abs(arfitnessbest(1)), eps);
        for d = 1:D
            if FE >= maxFE, break; end
            spos(:, d) = xvalbest(d, 1) + ((1:nprobe)' - 0.1 * total) * (ub(d) - lb(d)) / 200;
            [ep, FE, bsf, bsfx, curve, ph, fh, hi] = ...
                evalPop(spos, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);
            bbpbbp(d)  = abs(max(ep) / denom);
            spos(:, d) = xvalbest(d, 1);
        end

        if max(bbpbbp) < 3.1
            bbpbb = ones(1, D);
        else
            aaa1  = sort(bbpbbp);
            diaaa1 = zeros(1, D - 1);
            for d = 1:D - 1
                if aaa1(d) ~= 0
                    diaaa1(d) = aaa1(d + 1) / aaa1(d);
                else
                    diaaa1(d) = inf;
                end
            end
            [~, aab2] = sort(diaaa1, 'descend');
            division = 0;
            if aaa1(max(round(D / 2), 1)) <= 2
                for d = 1:D - 1
                    if aaa1(aab2(d)) < 1.8
                        division = aaa1(aab2(d)) + 0.01;
                        break;
                    end
                end
            else
                for d = 1:D - 1
                    if aaa1(aab2(d)) < 4
                        division = aaa1(aab2(d)) + 0.01;
                        break;
                    else
                        division = 0;
                    end
                end
            end
            bbpbb = double(bbpbbp <= division);
        end
    end

    [bestCMA, seq] = min(arfitnessbest);
    xfrozen        = xvalbest(:, seq(1))';

    pos = repmat(lb, total, 1) + rand(total, D) .* repmat(ub - lb, total, 1);

    kk    = 1;
    cc2   = 0;
    xmin  = zeros(1, 1);
    while FE < maxFE
        [e1, FE, bsf, bsfx, curve, ph, fh, hi] = evalPop(pos, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi);

        [a1, a2] = sort(e1);
        xmin(kk) = a1(1);

        newpos  = pos(a2(1:min(mu, total)), :);
        meanval = (newpos' * weights(1:size(newpos, 1)))';
        stdval  = std(newpos);

        if kk == 1
            if D > 30
                frozen = (bbpbb == 0);
            else
                frozen = (ppp1 < dividevalue);
            end
            stdval(frozen)  = 0.001 * sc;
            meanval(frozen) = xfrozen(frozen);
        end

        kk = kk + 1;
        if kk > 30 && mod(kk, 20) == 0
            [~, bbb] = min(xmin);
            cc2 = double(bbb < kk - 20);
        end

        if cc2 == 1
            step = 0.96 * randn(total, D);
        else
            step = randn(total, D);
        end
        pos = repmat(meanval, total, 1) + repmat(stdval, total, 1) .* step;
        pos = resampleOut(pos, meanval, stdval, lb, ub);
    end

    % bestCMA already reaches bsf via evalPop; the guard keeps a budget ending inside stage 2 correct
    if bestCMA < bsf
        bsf  = bestCMA;
        bsfx = xfrozen;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness       = bsf;
    best_solution      = bsfx;
    population_history = ph;
    fitness_history    = fh;
end

% Helper Functions

function [fit, FE, bsf, bsfx, curve, ph, fh, hi] = evalPop(X, problem, FE, maxFE, bsf, bsfx, curve, ph, fh, hi)
% Evaluates an n-by-D block; LOCAL not nested, so in-place optimisation applies to curve/history
    X = min(max(X, problem.lb), problem.ub);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:);
    n = size(X, 1);
    for i = 1:n
        if fit(i) < bsf
            bsf  = fit(i);
            bsfx = X(i, :);
        end
        ec = FE - n + i;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hi] = record_history(ec, X, fit, ph, fh, hi, maxFE);
        end
    end
end

function X = wrapBox(X, ctr, hw, lb, ub)
% Generalised modulo bound rule; on a symmetric box this is the reference's mod(x,100)
    n = size(X, 1);
    C = repmat(ctr, n, 1);
    H = repmat(hw,  n, 1);

    hi_v = X > repmat(ub, n, 1);
    if any(hi_v(:))
        X(hi_v) = C(hi_v) + mod(X(hi_v) - C(hi_v), H(hi_v));
    end
    lo_v = X < repmat(lb, n, 1);
    if any(lo_v(:))
        X(lo_v) = C(lo_v) + mod(X(lo_v) - C(lo_v), -H(lo_v));
    end
    % Degenerate boxes (h == 0) leave NaN; pin those to the centre
    bad = ~isfinite(X);
    if any(bad(:))
        X(bad) = C(bad);
    end
end

function X = resampleOut(X, meanval, stdval, lb, ub)
% Violating component re-sampled from N(mean, std), bounded retries then clamped
    n = size(X, 1);
    M = repmat(meanval, n, 1);
    S = repmat(stdval,  n, 1);
    for it = 1:10
        bad = X < repmat(lb, n, 1) | X > repmat(ub, n, 1);
        if ~any(bad(:)), return; end
        X(bad) = M(bad) + S(bad) .* randn(sum(bad(:)), 1);
    end
    X = min(max(X, repmat(lb, n, 1)), repmat(ub, n, 1));
end

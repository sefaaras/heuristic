% ----------------------------------------------------------------------- %
% Self-adaptive Search Equation-based Artificial Bee Colony (SSEABC)
% CEC 2016 competition -- 4th place by Friedman rank (3rd by summed error)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 13 -> 25               % Food sources; one is added near the best every 13 cycles
%   limit = floor(1.7396*13*D)  % Failed trials before a scout redraws a food source
%   archive = 500               % Random search equations at the start
%   tr = 0.3, lsItr = 84        % Share of maxFe before the local-search contest; later LS length
%   lambda0 = 4 + floor(8.1864*ln D), mu = floor(lambda/1.1334)  % CMA-ES sizes
%   sigma0 = 0.1026*(ub - lb), IPOP factor 3.6703, lambda <= 200  % CMA-ES restarts
%   TolFun, TolHistFun, TolX = 10^-18.2734, 10^-12.3001, 10^-14.2054  % CMA-ES stops
%
% Algorithm Concept:
%   - Employed and onlooker bees use one archived search equation per cycle: a first
%     term (x_i, x_best or x_r1) plus up to three of five scaled difference terms
%   - Every pass through the archive drops its least successful equations, adds one
%     random equation and discounts all success counts by FE/maxFe
%   - One difference term uses a neighbour drawn with probability 1/distance
%   - After 30 % of maxFe, MTS-LS1 and IPOP-CMA-ES (up to 60 % of maxFe) run in
%     turn from the best food; whichever improved last is kept
%   - The kept method then runs 84 evaluations per cycle on a random food, or on the
%     best once that has failed; a failure reopens the contest
%
% Reference:
% Gurcan Yavuz, Dogan Aydin, Thomas Stuetzle,
% Self-adaptive search equation-based artificial bee colony algorithm on the CEC 2014
% benchmark functions,
% 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 1173-1180.
% https://doi.org/10.1109/CEC.2016.7743920
% Components: ABC (Karaboga), MTS-LS1 (Tseng and Chen), IPOP-CMA-ES (Auger and Hansen).
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' C++ (github.com/gurcanyavuz/SSEABC, identical to SSEABC.tar.gz
% in the CEC 2016 "codes for best results" archive) with its README command line; the
% CMA-ES follows the bundled cmaes.c 3.10 (its rates and stops, samples clamped in place).
% Kept: onlookers never visit the last food; Powell's method is unreachable with lsID 4
% and is not ported. Changed: the stop at error 1e-20 is removed; MTS-LS1 clamps instead
% of a growing penalty; f < 0 gets fitness 1 + |f| (the release scores error values,
% 1/(1 + f)); a step the release takes as the largest coordinate gap is taken as the
% largest gap relative to the box width, times each width (same on a uniform box); a
% grown food starts with 0 failures (uninitialised there); an onlooker pass that finds
% no food in 1e6 draws ends.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sseabc(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    nfoods      = 13;
    maxfoodsize = 25;
    growth      = 13;
    limitF      = 1.7396;
    asize       = 500;
    lsItr       = 84;
    tr          = 0.3;
    tunea = 8.1864;  tuneb = 1.1334;  tunec = 0.1026;  tuned = 3.6703;
    tunee = -18.2734; tunef = -12.3001; tuneg = -14.2054;
    max_miss    = 1e6;

    limit = floor(limitF * nfoods * D);     % the release keeps the initial food count here

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = lb + 0.5 * span;
    done = false;
    ctxX = zeros(0, D);
    ctxF = zeros(0, 1);

    X  = zeros(nfoods, D);
    Fv = zeros(nfoods, 1);
    for i0 = 1:nfoods
        ctxX = X(1:i0 - 1, :);
        ctxF = Fv(1:i0 - 1);
        X(i0, :) = lb + rand(1, D) .* span;
        Fv(i0) = evaluate(X(i0, :));
        if done
            finish();
            return;
        end
    end
    trail  = zeros(nfoods, 1);
    failed = zeros(nfoods, 1);

    % Archive: [first second third fourth] term codes, modification rate, success count
    SEt = zeros(asize, 4);
    SEmr = zeros(asize, 1);
    for k0 = 1:asize
        [SEt(k0, :), SEmr(k0)] = create_se();
    end
    SEs = zeros(asize, 1);
    c   = 0;                                % equation in use (0-based, as the release's counter)

    nols        = false;
    ls_kind     = 0;                        % 0 none, 1 MTS-LS1, 2 IPOP-CMA-ES
    mts_improve = true;
    mts_best    = inf;
    mts_bestx   = lb;
    cma_lambda  = 0;
    iterations  = 0;

    while ~done
        employed_bees();
        if done, break; end
        prob = fitness_of(Fv) / sum(fitness_of(Fv));
        onlooker_bees(prob);
        if done, break; end
        scout_bees();
        if done, break; end

        [~, bi] = min(Fv);
        if FE > maxFE * tr && ~nols
            adaptively_select(bi);
            if done, break; end
        end
        if ls_kind > 0
            solbefore = bsf;
            apply_ils(bi);
            if done, break; end
            nols = solbefore > bsf;
        end

        NP = size(X, 1);
        if iterations > 0 && mod(iterations, growth) == 0 && NP < maxfoodsize
            [~, gbi] = min(Fv);
            r  = rand;
            rp = lb + rand(1, D) .* span;
            xn = min(max(rp + r * (X(gbi, :) - rp), lb), ub);
            ctxX = X;
            ctxF = Fv;
            fn = evaluate(xn);
            if done, break; end
            X(end + 1, :) = xn;
            Fv(end + 1) = fn;
            trail(end + 1) = 0;
            failed(end + 1) = 0;
        end

        iterations = iterations + 1;
        c = c + 1;
        if c == numel(SEs)
            if c >= 2
                eliminate_worst_se();
            end
            c = 0;
            SEs = SEs * (FE / maxFE);
        end
    end

    finish();

    function finish()
        curve(min(max(FE, 1), maxFE):end) = bsf;
        best_fitness  = bsf;
        best_solution = bsfx;
    end

    % Evaluates rows of Xr within the budget, recording the context plus the evaluated prefix
    function f = evaluate(Xr)
        k = size(Xr, 1);
        f = inf(k, 1);
        k_ok = min(k, maxFE - FE);
        if k_ok > 0
            [fv, FE_new] = calculate_fitness(Xr(1:k_ok, :)', problem, FE);
            f(1:k_ok) = fv(:);
            for q = 1:k_ok
                if f(q) < bsf
                    bsf  = f(q);
                    bsfx = Xr(q, :);
                end
                ec = FE + q;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, [ctxX; Xr(1:q, :)], [ctxF; f(1:q)], population_history, ...
                    fitness_history, history_index, maxFE);
            end
            FE = FE_new;
        end
        if FE >= maxFE
            done = true;
        end
    end

    function [t, mr] = create_se()
        t = [floor(3 * rand), floor(6 * rand), floor(6 * rand), floor(6 * rand)];
        mr = floor(1 + 2 * rand);
        if mr == 2
            mr = floor(1 + 9 * rand);
        end
    end

    % Karaboga's fitness; the release's 1/(f + 1) for f >= 0
    function fit = fitness_of(f)
        fit = 1 ./ (1 + f);
        neg = f < 0;
        fit(neg) = 1 + abs(f(neg));
    end

    % Neighbour drawn with probability proportional to 1/distance (getSuitableReferencefor)
    function k = suitable_reference(i)
        NPc = size(X, 1);
        dist = sqrt(sum((X - X(i, :)) .^ 2, 2));
        others = [1:i - 1, i + 1:NPc];
        zero = others(dist(others) == 0);
        if ~isempty(zero)
            k = zero(1);                    % 1/0 = inf takes the whole roulette, as in C++
            return;
        end
        p = 1 ./ dist(others);
        cs = cumsum(p);
        k = others(find(rand * cs(end) <= cs, 1));
        if isempty(k)
            k = others(end);
        end
    end

    function v = term_value(code, xc, gb, gbd, r1, r2)
        switch code
            case 0
                v = (2 * rand - 1) * (gb - xc);
            case 1
                v = (2 * rand - 1) * (gb - r1);
            case 2
                v = (2 * rand - 1) * (xc - r1);
            case 3
                v = (2 * rand - 1) * (r1 - r2);
            case 4
                v = (2 * rand - 1) * (xc - gbd);
            otherwise
                v = 0;
        end
    end

    function [nb, nb2] = two_neighbours(i)
        NPc = size(X, 1);
        nb = i;
        while nb == i
            nb = randi(NPc);
        end
        nb2 = i;
        while nb2 == i || nb2 == nb
            nb2 = randi(NPc);
        end
    end

    % New coordinate value from the current search equation (first term + three others)
    function v = apply_equation(t, cur, j, gbi, gbd, nb, nb2)
        xc = cur(j);
        gb = X(gbi, j);
        gd = X(gbd, j);
        r1 = X(nb, j);
        r2 = X(nb2, j);
        switch t(1)
            case 0
                v = xc;
            case 1
                v = gb;
            otherwise
                v = r1;
        end
        v = v + term_value(t(2), xc, gb, gd, r1, r2) + term_value(t(3), xc, gb, gd, r1, r2) + ...
            term_value(t(4), xc, gb, gd, r1, r2);
        v = min(max(v, lb(j)), ub(j));
    end

    function greedy(i, temp, f)
        if f < Fv(i)
            SEs(c + 1) = SEs(c + 1) + 1;
            trail(i) = 0;
            X(i, :) = temp;
            Fv(i) = f;
        else
            trail(i) = trail(i) + 1;
        end
    end

    % Employed bees change one coordinate (the equation's MR is not used here, as released)
    function employed_bees()
        t = SEt(c + 1, :);
        for i = 1:size(X, 1)
            [nb, nb2] = two_neighbours(i);
            [~, gbi] = min(Fv);
            gbd = suitable_reference(i);
            j = randi(D);
            temp = X(i, :);
            temp(j) = apply_equation(t, X(i, :), j, gbi, gbd, nb, nb2);
            ctxX = X;
            ctxF = Fv;
            f = evaluate(temp);
            if done, return; end
            greedy(i, temp, f);
        end
    end

    % Onlookers cycle over foods 1..NP-1 (the release resets at NP-1) until NP have flown
    function onlooker_bees(prob)
        NPc = size(X, 1);
        t = SEt(c + 1, :);
        mr = SEmr(c + 1);
        i = 1;
        flown = 0;
        miss = 0;
        while flown < NPc
            if rand < prob(i)
                miss = 0;
                [nb, nb2] = two_neighbours(i);
                [~, gbi] = min(Fv);
                gbd = suitable_reference(i);
                flown = flown + 1;
                cur = X(i, :);
                temp = cur;
                for mm = 1:mr
                    j = randi(D);
                    temp(j) = apply_equation(t, cur, j, gbi, gbd, nb, nb2);
                end
                ctxX = X;
                ctxF = Fv;
                f = evaluate(temp);
                if done, return; end
                greedy(i, temp, f);
            else
                miss = miss + 1;
                if miss > max_miss
                    return;
                end
            end
            i = i + 1;
            if i == NPc
                i = 1;
            end
        end
    end

    function scout_bees()
        [mt, mi] = max(trail);
        if mt >= limit
            ctxX = X;
            ctxF = Fv;
            xn = lb + rand(1, D) .* span;
            fn = evaluate(xn);
            if done, return; end
            X(mi, :) = xn;
            Fv(mi) = fn;
            trail(mi) = 0;
            failed(mi) = 0;
        end
    end

    % Sort by success, drop the tail, add one random equation (eliminateWorstCandidateSE)
    function eliminate_worst_se()
        [~, o] = sort(SEs, 'descend');
        SEt = SEt(o, :);
        SEmr = SEmr(o);
        SEs = SEs(o);
        val = asize / floor((maxFE / (2 * maxfoodsize)) / asize);
        if val >= c - 2
            val = c - 2;
        end
        if c > 2
            keepn = numel(SEs) - floor(val);
            SEt = SEt(1:keepn, :);
            SEmr = SEmr(1:keepn);
            SEs = SEs(1:keepn);
        end
        [tn, mrn] = create_se();
        SEt(end + 1, :) = tn;
        SEmr(end + 1) = mrn;
        SEs(end + 1) = 0;
    end

    % Contest between fresh MTS-LS1 and IPOP-CMA-ES objects, run one after the other
    function adaptively_select(bi)
        mts_improve = true;
        mts_best = inf;
        cma_lambda = 0;
        currsol = bsf;
        f1x = X(bi, :);
        f1f = Fv(bi);
        [f1x, f1f] = mts_apply(f1x, f1f, span / 4, floor(maxFE * tr));
        if done, return; end
        sol_mts = bsf;
        cma_apply(f1x, f1f, floor(maxFE * tr * 2));
        if done, return; end
        sol_cma = bsf;
        if currsol ~= sol_cma || currsol ~= sol_mts
            if sol_cma < sol_mts
                ls_kind = 2;
            else
                ls_kind = 1;
            end
        else
            ls_kind = 0;
            nols = true;
        end
        Fv(bi) = bsf;
        X(bi, :) = bsfx;
    end

    % ILS invocation: a random food while the best has not failed, else the best itself
    function apply_ils(bi)
        NPc = size(X, 1);
        food = bi;
        foodID = 0;
        if failed(bi) == 0
            [~, fid] = min(Fv);
            while true
                foodID = randi(NPc);
                if ~(foodID == fid && failed(foodID) > 0)
                    break;
                end
            end
            food = foodID;
        end
        ref = foodID;
        while ref == foodID
            ref = randi(NPc);
        end
        step = max(abs(X(ref, :) - X(food, :)) ./ span) * span;
        before = Fv(food);
        if ls_kind == 1
            [xn, fn, upd] = mts_apply(X(food, :), Fv(food), step, lsItr);
        else
            [xn, fn, upd] = cma_apply(X(food, :), Fv(food), lsItr);
        end
        if done, return; end
        X(food, :) = xn;
        Fv(food) = fn;
        if upd
            trail(food) = 0;
        end
        if Fv(food) < before
            failed(food) = 0;
        else
            failed(food) = failed(food) + 1;
        end
    end

    % MTS-LS1 object: its best (over all its runs) replaces the food if better
    function [fx, ff, upd] = mts_apply(fx, ff, s, maxitr)
        xk = fx;
        inits = s;
        iter = 0;
        collapses = 0;
        converged = false;
        countmax = maxitr + FE;
        ctxX = X;
        ctxF = Fv;
        while true
            if ~mts_improve
                s = s / (1.5 + rand);
                if max(s) < 1e-20
                    % The release compares the last stored best with itself: a second collapse stops
                    collapses = collapses + 1;
                    if collapses >= 2
                        converged = true;
                    end
                    s = (0.7 + 0.2 * rand) * inits;
                end
            end
            mts_improve = false;
            for i = 1:D
                before1 = ls_eval(xk);
                if done, break; end
                xi = xk(i);
                xk(i) = max(min(xi - s(i), ub(i)), lb(i));
                after1 = ls_eval(xk);
                if done, break; end
                if abs(after1 - before1) <= 1e-20
                    xk(i) = xi;
                elseif after1 - before1 > 1e-20
                    xk(i) = max(min(xi + 0.5 * s(i), ub(i)), lb(i));
                    after2 = ls_eval(xk);
                    if done, break; end
                    if after2 >= before1
                        xk(i) = xi;
                    else
                        mts_improve = true;
                    end
                else
                    mts_improve = true;
                end
            end
            iter = iter + 1;
            if done || FE >= countmax || iter >= lsItr || converged
                break;
            end
        end
        upd = false;
        if ff > mts_best
            ff = mts_best;
            fx = mts_bestx;
            upd = true;
        end
    end

    function f = ls_eval(x)
        f = evaluate(x);
        if f < mts_best
            mts_best = f;
            mts_bestx = x;
        end
    end

    % ICMAESLS::apply: the food takes the global best if that is better
    function [fx, ff, upd] = cma_apply(fx, ff, allow)
        icmaes(allow, fx);
        upd = false;
        if ff > bsf
            ff = bsf;
            fx = bsfx;
            upd = true;
        end
    end

    % The release's icmaes(): restarts within the allowance; lambda persists in the object
    function icmaes(allow, xfeed)
        countmax = allow + FE;
        irun = 0;
        while FE < countmax && ~done
            if irun == 0
                xinit = xfeed;
            else
                xinit = lb + rand(1, D) .* span;
            end
            irun = irun + 1;
            ctxX = X;
            ctxF = Fv;
            evaluate(xinit);                % evaluated and discarded, as released
            if done, return; end
            lam = cma_lambda;
            if lam < 2
                lam = 4 + floor(tunea * log(D));
            end
            cma_run(xinit(:), lam, countmax);
            cma_lambda = min(200, floor(tuned * lam));
        end
    end

    % One run of cmaes.c 3.10 with initials.par: until a stopping test or the allowance
    function cma_run(xmean, lambda, countmax)
        N = D;
        mu = floor(lambda / tuneb);
        w = log(mu + 1) - log(1:mu)';
        mueff = sum(w) ^ 2 / sum(w .^ 2);
        w = w / sum(w);
        cs = (mueff + 2) / (N + mueff + 3);
        ccum = 4 / (N + 4);
        mucov = mueff;
        t1 = 2 / ((N + 1.4142) * (N + 1.4142));
        t2 = min(1, (2 * mueff - 1) / ((N + 2) * (N + 2) + mueff));
        ccov = (1 / mucov) * t1 + (1 - 1 / mucov) * t2;
        ccov1 = min(ccov / mucov, 1);
        ccovmu = min(ccov * (1 - 1 / mucov), 1 - ccov1);
        damps = (1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1)) * ...
                max(0.3, 1 - N / (1e-6 + min(1e299, 1e299 / lambda))) + cs;
        modulo = 1 / ccov / N / 10;
        stds = tunec * span(:);
        sigma = sqrt(sum(stds .^ 2) / N);
        Dv = stds / sigma;
        C = diag(Dv .^ 2);
        B = eye(N);
        minEW = min(Dv) ^ 2;
        maxEW = max(Dv) ^ 2;
        chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2));
        kond = 2 ^ 53 / 1000;               % dMaxSignifKond
        pc = zeros(N, 1);
        ps = zeros(N, 1);
        gen = 0;
        uptodate = true;
        genEig = 0;
        histLen = 10 + ceil(30 * N / lambda);
        fhist = zeros(histLen, 1);
        fvals = zeros(lambda, 1);
        tolfun = 10 ^ tunee;
        tolhist = 10 ^ tunef;
        tolx = 10 ^ tuneg;
        lbc = lb(:);
        ubc = ub(:);

        while ~stop_test() && FE < countmax && ~done
            if ~uptodate && gen >= genEig + modulo
                Cs = triu(C) + triu(C, 1)';
                [V, E] = eig(Cs);
                ev = diag(E);
                minEW = min(ev);
                maxEW = max(ev);
                if ~all(isfinite(ev)) || minEW <= 0
                    return;                 % ConditionNumber would stop it; sqrt would turn NaN
                end
                B = V;
                Dv = sqrt(ev);
                uptodate = true;
                genEig = gen;
            end
            arx = xmean + sigma * (B * (Dv .* randn(N, lambda)));
            gen = gen + 1;
            arx = min(max(arx, lbc), ubc);  % inbound(): the clamped samples drive the update
            nev = min([lambda, countmax - FE, maxFE - FE]);
            ctxX = zeros(0, D);
            ctxF = zeros(0, 1);
            f = evaluate(arx(:, 1:nev)');
            if nev < lambda || done
                return;                     % the release updates on stale values, then quits
            end

            [~, idx] = sort(f);
            fvals = f;
            if f(idx(1)) == f(idx(floor(lambda / 2) + 1))
                sigma = sigma * exp(0.2 + cs / damps);
            end
            fhist = [f(idx(1)); fhist(1:end - 1)];

            xold = xmean;
            xmean = arx(:, idx(1:mu)) * w;
            BDz = sqrt(mueff) * (xmean - xold) / sigma;
            z = (B' * BDz) ./ Dv;
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs)) * (B * z);
            psxps = sum(ps .^ 2);
            hsig = sqrt(psxps) / sqrt(1 - (1 - cs) ^ (2 * gen)) / chiN < 1.4 + 2 / (N + 1);
            pc = (1 - ccum) * pc + hsig * sqrt(ccum * (2 - ccum)) * BDz;
            Y = (arx(:, idx(1:mu)) - xold) / sigma;
            C = (1 - ccov1 - ccovmu) * C + ccov1 * (pc * pc' + (1 - hsig) * ccum * (2 - ccum) * C) + ...
                ccovmu * (Y * (w .* Y'));
            uptodate = false;
            sigma = sigma * exp((sqrt(psxps) / chiN - 1) * cs / damps);
        end

        % cmaes_TestForTermination with MaxFunEvals and MaxIter at 1e299
        function stop = stop_test()
            stop = false;
            dC = diag(C);
            if gen > 0
                nh = min(gen, histLen);
                rg = max(max(fhist(1:nh)), max(fvals)) - min(min(fhist(1:nh)), min(fvals));
                if rg <= tolfun
                    stop = true;
                    return;
                end
            end
            if gen > histLen && max(fhist) - min(fhist) <= tolhist
                stop = true;
                return;
            end
            if all(sigma * sqrt(dC) < tolx) && all(sigma * pc < tolx)
                stop = true;
                return;
            end
            if any(sigma * sqrt(dC) > 1e3 * stds) || maxEW >= minEW * kond
                stop = true;
                return;
            end
            for ia = 1:N
                if all(xmean == xmean + 0.1 * sigma * Dv(ia) * B(:, ia))
                    stop = true;
                    return;
                end
            end
            if any(xmean == xmean + 0.2 * sigma * sqrt(dC))
                stop = true;
            end
        end
    end
end

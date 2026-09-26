% ----------------------------------------------------------------------- %
% United Multi-Operator Evolutionary Algorithms (UMOEAs)
% CEC 2014 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PS = 100                    % Population of each of the three MOEAs
%   CS = 50 (D = 10), else 100  % Cycle length in generations
%   limit = 50 % of maxFe       % After it the chosen MOEA runs alone
%   F, CR ~ N(0.5, 0.1) in [0.1, 0.95]  % Initial per-individual DE parameters
%   sigma0 = 1.5, lambda = PS   % CMA-ES initial step size and sample size
%   beta = N(0.7, 0.1), p = 0.1 % Multi-parent crossover step, archive gene swap rate
%   eta = 3, b = 5, pm = 0.1    % SBX index, non-uniform mutation shape and rate
%
% Algorithm Concept:
%   - Three MOEAs (a two-operator DE, CMA-ES and a two-operator GA) evolve their
%     own populations, each operator's share following its success
%   - For CS generations all three run; an exponential curve fitted to each one's
%     best value then predicts who leads after 2*CS, and only that one runs next
%   - Every 2*CS generations the best solution is copied into all three and one
%     sample drawn around the leader's two best joins the other two
%   - Past half the budget the leader keeps the search to itself
%   - DE mixes DE/phi-rand/1 and DE/current-to-phi-best/1, F and CR self-adapted
%     from three random members; GA mixes multi-parent crossover and SBX
%
% Reference:
% Saber M. Elsayed, Ruhul A. Sarker, Daryl L. Essam, Noha M. Hamza,
% Testing united multi-operator evolutionary algorithms on the CEC2014
% real-parameter numerical optimization,
% 2014 IEEE Congress on Evolutionary Computation (CEC), 2014, pp. 1650-1657.
% https://doi.org/10.1109/CEC.2014.6900308
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from UMOEAs_Final.rar in Suganthan's CEC2014 Top-Methods-Part-A, the
% authors' corrected release that repairs CMA-ES samples (the published numbers
% came from an unrepaired run). With no f* in the harness, the choice fits
% a*exp(b*t) to the raw best values by least squares (the release fits |f - f*|
% with the Curve Fitting Toolbox), and the stop at a 1e-8 error is removed. The
% release pairs its best value with a parent's position in DE and GA and returns
% CMA-ES samples unsorted beside sorted fitness; here both pairs are coherent,
% which changes what the sharing step copies. Kept as written: GA survivors are
% credited with operator labels shifted by the elite count, and the CMA-ES bound
% penalty initialises only the first weight. The last batch is budget-truncated.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = umoeas(problem)

    n     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    PopSize = 100;
    if n == 10
        CS = 50;
    else
        CS = 100;
    end
    T_gen     = maxFE / PopSize;
    limit_all = maxFE / 2;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf          = inf;
    bsf_solution = lb + rand(1, n) .* span;

    x0 = repmat(lb, PopSize, 1) + rand(PopSize, n) .* repmat(span, PopSize, 1);
    fit0 = eval_rows(x0);
    k0 = numel(fit0);
    for e0 = 1:k0
        [population_history, fitness_history, history_index] = record_history(...
            e0, x0(1:e0, :), fit0(1:e0), population_history, fitness_history, history_index, maxFE);
    end
    x0 = x0(1:k0, :);
    fit0 = fit0(:).';
    [fit0, order0] = sort(fit0);
    x0 = x0(order0, :);

    EA_1 = x0;  EA_obj1 = fit0;
    EA_2 = x0;  EA_obj2 = fit0;
    EA_3 = x0;  EA_obj3 = fit0;

    [setting, es_hist, bnd] = init_cmaes_par(n, PopSize, lb, ub, EA_obj2(1));

    mu_F  = min(0.95, max(0.1, 0.5 + 0.1 * randn(1, PopSize)));
    cr_DE = min(0.95, max(0.1, 0.5 + 0.1 * randn(1, PopSize)));
    prob_de = 0.5;
    prob_ga = 0.5;

    success    = zeros(CS, 3);
    impro      = [EA_obj1(1), EA_obj2(1), EA_obj3(1)];
    count_iter = 0;
    iter       = 1;
    indx       = 0;

    while FE < maxFE
        count_iter = count_iter + 1;

        if count_iter == CS + 1
            indx = fit_exp(success, CS);
            success(:) = 0;
            prob_de = 0.5;
            prob_ga = 0.5;
        elseif count_iter == 2 * CS && FE <= limit_all
            success(:) = 0;
            count_iter = 1;
            inf_sharing();
            [EA_obj1, o] = sort(EA_obj1);  EA_1 = EA_1(o, :);
            [EA_obj2, o] = sort(EA_obj2);  EA_2 = EA_2(o, :);
            [EA_obj3, o] = sort(EA_obj3);  EA_3 = EA_3(o, :);
            setting.xmean = EA_2(1:setting.mu, :)' * setting.weights;
            prob_de = 0.5;
            prob_ga = 0.5;
        end

        if FE < maxFE && (count_iter < CS || indx == 1)
            samo_de();
            if count_iter < CS
                impro(1) = EA_obj1(1);
            end
        end
        if FE < maxFE && (count_iter < CS || indx == 2)
            samo_es();
            if count_iter < CS
                impro(2) = EA_obj2(1);
            end
        end
        if FE < maxFE && (count_iter < CS || indx == 3)
            samo_ga();
            if count_iter < CS
                impro(3) = EA_obj3(1);
            end
        end

        if count_iter <= CS
            success(count_iter, :) = impro;
        end
        iter = iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;

    % SAMO-DE: DE/phi-rand/1 or DE/current-to-phi-best/1 per individual, F and CR self-adapted
    function samo_de()
        x = EA_1;
        fitx = EA_obj1;
        NPd = size(x, 1);
        prob_all = rand(1, NPd);
        phi = ones(1, NPd);
        for i = 1:NPd
            if prob_all(i) <= prob_de
                phi(i) = max(1, ceil(rand * 0.5 * NPd));
            end
        end

        trial = x;
        F_try = mu_F;
        Cr_try = cr_DE;
        contribution = zeros(1, NPd);
        for i = 1:NPd
            randnum = zeros(1, 3);
            for t = 1:3
                randnum(t) = ceil(NPd * rand);
                while randnum(t) == i || randnum(t) == phi(i) || any(randnum(1:t-1) == randnum(t))
                    randnum(t) = ceil(NPd * rand);
                end
            end

            if rand < 0.75
                Fi = mu_F(randnum(3)) + rand * (mu_F(randnum(1)) - mu_F(randnum(2)));
                if Fi < 0
                    Fi = abs(Fi);
                elseif Fi > 1
                    Fi = 2 - Fi;
                end
                Ci = cr_DE(randnum(3)) + rand * (cr_DE(randnum(1)) - cr_DE(randnum(2)));
                if Ci < 0
                    Ci = abs(Ci);
                elseif Ci > 1
                    Ci = 2 - Ci;
                end
            else
                Fi = rand;
                Ci = rand;
            end
            F_try(i) = Fi;
            Cr_try(i) = Ci;

            take = rand(1, n) < Ci;
            take(ceil(n * rand)) = true;
            if prob_all(i) <= prob_de
                v = x(phi(i), :) + Fi * (x(randnum(1), :) - x(randnum(2), :));
                contribution(i) = 1;
            else
                v = x(i, :) + Fi * (x(randnum(1), :) - x(randnum(2), :)) + Fi * (x(phi(i), :) - x(i, :));
                contribution(i) = 2;
            end
            u = x(i, :);
            u(take) = v(take);
            trial(i, :) = reflect_rows(u, lb, ub);
        end
        trial = redraw_nonfinite(trial, lb, span);

        fe0 = FE;
        f = eval_rows(trial);
        k = numel(f);
        x_new = x;
        fitx_new = fitx;
        F_i = mu_F;
        Cr_i = cr_DE;
        count1 = 0;
        count2 = 0;
        for i = 1:k
            % Ties are accepted; a worse trial keeps the parent and the parent's F and CR
            if f(i) <= fitx(i)
                x_new(i, :) = trial(i, :);
                fitx_new(i) = f(i);
                F_i(i) = F_try(i);
                Cr_i(i) = Cr_try(i);
                if contribution(i) == 1
                    count1 = count1 + 1;
                else
                    count2 = count2 + 1;
                end
            end
            [population_history, fitness_history, history_index] = record_history(...
                fe0 + i, x_new, fitx_new, population_history, fitness_history, history_index, maxFE);
        end

        count1 = count1 / sum(prob_all <= prob_de);
        count2 = count2 / sum(prob_all > prob_de);
        prob_de = max(0.05, min(0.95, count1 / (count1 + count2)));

        [EA_obj1, o] = sort(fitx_new);
        EA_1  = x_new(o, :);
        mu_F  = F_i(o);
        cr_DE = Cr_i(o);
    end

    % SAMO-ES: one CMA-ES generation of PS samples with Hansen's bound penalty
    function samo_es()
        lam = PopSize;
        arz = randn(n, lam);
        arx = repmat(setting.xmean, 1, lam) + setting.sigma * (setting.BD * arz);
        arxvalid = xintobounds(arx, lb', ub');
        bad = ~isfinite(arxvalid);
        if any(bad(:))
            arxvalid = redraw_nonfinite(arxvalid', lb, span)';
            arx(bad) = arxvalid(bad);
        end

        fe0 = FE;
        raw = eval_rows(arxvalid')';
        k = numel(raw);
        if k < lam
            [raw, o] = sort(raw);
            EA_2 = arxvalid(:, o)';
            EA_obj2 = raw;
            record_batch(fe0 + 1, FE, EA_2, EA_obj2);
            return;
        end

        sel = raw;
        val = myprctile(raw, [25 75]);
        val = (val(2) - val(1)) / n / mean(setting.diagC) / setting.sigma^2;
        if ~isfinite(val)
            val = max(bnd.dfithist);
        elseif val == 0
            val = min(bnd.dfithist(bnd.dfithist > 0));
        elseif bnd.validfitval == 0
            bnd.dfithist = [];
            bnd.validfitval = 1;
        end
        if numel(bnd.dfithist) < 20 + (3 * n) / lam
            bnd.dfithist = [bnd.dfithist val];
        else
            bnd.dfithist = [bnd.dfithist(2:end) val];
        end

        [tx, ti] = xintobounds(setting.xmean, lb', ub');
        if bnd.iniphase && any(ti)
            % The release indexes with a 0/1 double vector, which only reaches weights(1)
            bnd.weights(1) = 2.0002 * median(bnd.dfithist);
            dd = setting.diagC / mean(setting.diagC);
            bnd.weights = bnd.weights ./ dd;
            if bnd.validfitval && iter > 2
                bnd.iniphase = 0;
            end
        end
        if any(ti)
            tx = setting.xmean - tx;
            idx = (ti ~= 0 & abs(tx) > 3 * max(1, sqrt(n) / setting.mueff) * setting.sigma * sqrt(setting.diagC));
            idx = idx & (sign(tx) == sign(setting.xmean - setting.xold));
            bnd.weights(idx) = 1.2^(min(1, setting.mueff / 10 / n)) * bnd.weights(idx);
        end
        arpenalty = bnd.weights' * (arxvalid - arx).^2;
        sel = sel + arpenalty;

        [raw_sorted, idx_raw] = sort(raw);
        [sel_sorted, idxsel] = sort(sel);
        es_hist(2:end) = es_hist(1:end-1);
        es_hist(1) = raw_sorted(1);

        s = setting;
        s.xold = s.xmean;
        s.xmean = arx(:, idxsel(1:s.mu)) * s.weights;
        zmean = arz(:, idxsel(1:s.mu)) * s.weights;
        s.ps = (1 - s.cs) * s.ps + sqrt(s.cs * (2 - s.cs) * s.mueff) * (s.B * zmean);
        hsig = norm(s.ps) / sqrt(1 - (1 - s.cs)^(2 * iter)) / s.chiN < 1.4 + 2 / (n + 1);
        s.pc = (1 - s.cc) * s.pc + hsig * (sqrt(s.cc * (2 - s.cc) * s.mueff) / s.sigma) * (s.xmean - s.xold);
        arpos = (arx(:, idxsel(1:s.mu)) - repmat(s.xold, 1, s.mu)) / s.sigma;
        s.C = (1 - s.ccov1 - s.ccovmu + (1 - hsig) * s.ccov1 * s.cc * (2 - s.cc)) * s.C ...
            + s.ccov1 * (s.pc * s.pc') ...
            + s.ccovmu * arpos * (repmat(s.weights, 1, n) .* arpos');
        s.diagC = diag(s.C);
        s.sigma = s.sigma * exp(min(1, (sqrt(sum(s.ps.^2)) / s.chiN - 1) * s.cs / s.damps));

        if mod(iter, 1 / (s.ccov1 + s.ccovmu) / n / 10) < 1
            s.C = triu(s.C) + triu(s.C, 1)';
            % The release stops with an error here; a broken matrix restarts from the identity
            if any(~isfinite(s.C(:)))
                s.C = eye(n);
            end
            [s.B, tmp] = eig(s.C);
            s.diagD = diag(tmp);
            if min(s.diagD) <= 0
                s.diagD(s.diagD < 0) = 0;
                tmp = max(s.diagD) / 1e14;
                s.C = s.C + tmp * eye(n);
                s.diagD = s.diagD + tmp * ones(n, 1);
            end
            if max(s.diagD) > 1e14 * min(s.diagD)
                tmp = max(s.diagD) / 1e14 - min(s.diagD);
                s.C = s.C + tmp * eye(n);
                s.diagD = s.diagD + tmp * ones(n, 1);
            end
            s.diagC = diag(s.C);
            s.diagD = sqrt(s.diagD);
            s.BD = s.B .* repmat(s.diagD', n, 1);
        end

        if s.sigma > 1e10 * max(s.diagD) && s.sigma > 8e14 * max(s.insigma)
            fac = s.sigma;
            s.sigma = s.sigma / fac;
            s.pc = fac * s.pc;
            s.diagD = fac * s.diagD;
            s.C = fac^2 * s.C;
            s.BD = s.B .* repmat(s.diagD', n, 1);
            s.diagC = fac^2 * s.diagC;
        end
        if any(s.sigma * sqrt(s.diagC) > s.maxdx)
            s.sigma = min(s.maxdx ./ sqrt(s.diagC));
        end
        frozen = s.xmean == s.xmean + 0.2 * s.sigma * sqrt(s.diagC);
        if any(frozen)
            s.C = s.C + (s.ccov1 + s.ccovmu) * diag(s.diagC .* frozen);
            s.sigma = s.sigma * exp(0.05 + s.cs / s.damps);
        end
        tmp = 0.1 * s.sigma * s.BD(:, 1 + floor(mod(iter, n)));
        if all(s.xmean == s.xmean + tmp)
            s.sigma = s.sigma * exp(0.2 + s.cs / s.damps);
        end
        span_hist = [es_hist sel_sorted(1)];
        if iter > 2 && max(span_hist) - min(span_hist) == 0
            s.sigma = s.sigma * exp(0.2 + s.cs / s.damps);
        end
        setting = s;

        EA_2 = arxvalid(:, idx_raw)';
        EA_obj2 = raw_sorted;
        record_batch(fe0 + 1, FE, EA_2, EA_obj2);
    end

    % SAMO-GA: multi-parent crossover with the archive swap, or SBX with non-uniform mutation
    function samo_ga()
        x = EA_3;
        fitx = EA_obj3;
        NPg = size(x, 1);
        arc_size = ceil(NPg / 2);
        archive = x(1:arc_size, :);
        x_new = zeros(NPg + 2, n);
        contribution = zeros(NPg, 1);
        % The horizon T is maxFe/PS generations; iter can pass it by the few idle generations
        lambda_nu = max(0, 1 - iter / T_gen)^5;

        i = 1;
        while i <= NPg
            if rand < prob_ga
                beta = 0.7 + 0.1 * randn;
                TcSize = randi([2, 3]);
                best = zeros(3, 1);
                for j = 1:3
                    r = randperm(NPg);
                    best(j) = min(r(1:TcSize));
                end
                a = sort(best);
                if a(1) == a(2)
                    while a(2) == a(1) || a(2) == a(3)
                        a(2) = randi(NPg);
                    end
                    a = sort(a);
                end
                if a(1) == a(3)
                    while a(3) == a(1) || a(3) == a(2)
                        a(3) = randi(NPg);
                    end
                    a = sort(a);
                end
                if a(2) == a(3)
                    while a(3) == a(1) || a(3) == a(2)
                        a(3) = randi(NPg);
                    end
                    a = sort(a);
                end
                x_new(i, :)   = x(a(1), :) + beta * (x(a(2), :) - x(a(3), :));
                x_new(i+1, :) = x(a(2), :) + beta * (x(a(3), :) - x(a(1), :));
                x_new(i+2, :) = x(a(3), :) + beta * (x(a(1), :) - x(a(2), :));
                x_new(i:i+2, :) = reflect_rows(x_new(i:i+2, :), lb, ub);
                for ip = i:min(NPg, i + 2)
                    swap = rand(1, n) < 0.1;
                    donors = ceil(arc_size * rand(1, n));
                    genes = archive(sub2ind([arc_size, n], donors, 1:n));
                    x_new(ip, swap) = genes(swap);
                    contribution(ip) = 1;
                end
                i = i + 3;
            else
                TcSize = randi([2, 3]);
                best = zeros(2, 1);
                for j = 1:2
                    r = randperm(NPg);
                    best(j) = min(r(1:TcSize));
                end
                a = best;
                while a(2) == a(1)
                    a(2) = randi(NPg);
                end
                u = rand;
                if u <= 0.5
                    beta2 = (2 * u)^(1 / (1 + 3));
                else
                    beta2 = (1 / (2 * (1 - u)))^(1 / (1 + 3));
                end
                x_new(i, :)   = 0.5 * ((1 + beta2) * x(a(1), :) + (1 - beta2) * x(a(2), :));
                x_new(i+1, :) = 0.5 * ((1 - beta2) * x(a(1), :) + (1 + beta2) * x(a(2), :));
                for ip = i:min(NPg, i + 1)
                    y = reflect_rows(x_new(ip, :), lb, ub);
                    for j = 1:n
                        if rand < 0.1
                            if rand <= 0.5
                                delta = (ub(j) - y(j)) * (1 - rand^lambda_nu);
                            else
                                delta = (lb(j) - y(j)) * (1 - rand^lambda_nu);
                            end
                            y(j) = y(j) + delta;
                        end
                    end
                    x_new(ip, :) = reflect_rows(y, lb, ub);
                    contribution(ip) = 2;
                end
                i = i + 2;
            end
        end
        x_new = redraw_nonfinite(x_new(1:NPg, :), lb, span);

        fe0 = FE;
        fitx_new = eval_rows(x_new)';
        k = numel(fitx_new);

        count_mpc = sum(contribution == 1);
        FSIZE = ceil(count_mpc / 2);
        xx  = [x(1:FSIZE, :); x_new(1:k, :)];
        fff = [fitx(1:FSIZE), fitx_new];
        % Labels listed offspring-first against elite-first rows, as in the release
        fcc = [contribution(1:k)', zeros(1, FSIZE)];
        [fff, o] = sort(fff);
        keep = min(NPg, numel(fff));
        EA_3 = xx(o(1:keep), :);
        EA_obj3 = fff(1:keep);
        survivors = fcc(o(1:keep));
        count1 = sum(survivors == 1);
        count2 = sum(survivors == 2);
        prob_ga = max(0.05, min(0.95, count1 / (count1 + count2)));
        if count1 == 0 && count2 == 0
            prob_ga = 0.5;
        end

        record_batch(fe0 + 1, FE, EA_3, EA_obj3);
    end

    % The best joins every MOEA; the leader's two best seed one sample in each of the others
    function inf_sharing()
        if EA_obj1(1) ~= bsf
            EA_1(end, :) = bsf_solution;
            EA_obj1(end) = bsf;
        end
        if EA_obj2(1) ~= bsf
            EA_2(end, :) = bsf_solution;
            EA_obj2(end) = bsf;
        end
        if EA_obj3(1) ~= bsf
            EA_3(end, :) = bsf_solution;
            EA_obj3(end) = bsf;
        end
        if indx == 1
            src = EA_1;
        elseif indx == 2
            src = EA_2;
        else
            src = EA_3;
        end
        xmean2 = mean(src(1:2, :), 1);
        sigma2 = std(src(1:2, :), 0, 1);
        for target = setdiff(1:3, indx)
            if FE >= maxFE
                break;
            end
            y = redraw_nonfinite(reflect_rows(xmean2 + sigma2 .* randn(1, n), lb, ub), lb, span);
            fe0 = FE;
            fy = eval_rows(y);
            switch target
                case 1
                    EA_1(end - 1, :) = y;  EA_obj1(end - 1) = fy;
                    record_batch(fe0 + 1, FE, EA_1, EA_obj1);
                case 2
                    EA_2(end - 1, :) = y;  EA_obj2(end - 1) = fy;
                    record_batch(fe0 + 1, FE, EA_2, EA_obj2);
                case 3
                    EA_3(end - 1, :) = y;  EA_obj3(end - 1) = fy;
                    record_batch(fe0 + 1, FE, EA_3, EA_obj3);
            end
        end
    end

    % Evaluates as many rows of X as the budget allows, with bsf and curve per FE
    function f = eval_rows(X)
        kk = min(size(X, 1), maxFE - FE);
        f = zeros(0, 1);
        if kk <= 0
            return;
        end
        fe_start = FE;
        [fv, FE] = calculate_fitness(X(1:kk, :)', problem, FE);
        f = fv(:);
        for ee = 1:kk
            if f(ee) < bsf
                bsf          = f(ee);
                bsf_solution = X(ee, :);
            end
            curve(fe_start + ee) = bsf;
        end
    end

    function record_batch(fe_from, fe_to, P, Pf)
        for fe = fe_from:fe_to
            [population_history, fitness_history, history_index] = record_history(...
                fe, P, Pf, population_history, fitness_history, history_index, maxFE);
        end
    end
end

function [s, hist, bnd] = init_cmaes_par(n, lam, lb, ub, f_first)
    s.xmean = lb' + (ub - lb)' .* rand(n, 1);
    s.insigma = 1.5 * ones(n, 1);
    s.sigma = 1.5;
    s.pc = zeros(n, 1);
    s.ps = zeros(n, 1);
    s.diagD = ones(n, 1);
    s.diagC = s.diagD .^ 2;
    s.B = eye(n);
    s.BD = s.B .* repmat(s.diagD', n, 1);
    s.C = diag(s.diagC);
    s.maxdx = ((ub - lb) / 2)';
    if any(s.sigma * sqrt(s.diagC) > s.maxdx)
        s.sigma = min(s.maxdx ./ sqrt(s.diagC));
    end
    s.chiN = n^0.5 * (1 - 1 / (4 * n) + 1 / (21 * n^2));
    s.mu = ceil(lam / 2);
    s.weights = log(max(s.mu, n / 2) + 1 / 2) - log(1:s.mu)';
    s.mueff = sum(s.weights)^2 / sum(s.weights.^2);
    s.weights = s.weights / sum(s.weights);
    s.cc = (4 + s.mueff / n) / (n + 4 + 2 * s.mueff / n);
    s.cs = (s.mueff + 2) / (n + s.mueff + 3);
    s.ccov1 = 2 / ((n + 1.3)^2 + s.mueff);
    s.ccovmu = 2 * (s.mueff - 2 + 1 / s.mueff) / ((n + 2)^2 + s.mueff);
    s.damps = 0.5 + 0.5 * min(1, (0.27 * lam / s.mueff - 1)^2) + ...
              2 * max(0, sqrt((s.mueff - 1) / (n + 1)) - 1) + s.cs;
    s.xold = s.xmean;

    hist = NaN(1, 10 + ceil(3 * 10 * n / lam));
    hist(1) = f_first;

    bnd.weights = zeros(n, 1);
    bnd.dfithist = 1;
    bnd.validfitval = 0;
    bnd.iniphase = 1;
end

% The release's scalar-bound reflection, per dimension; idx is +1 above ub, -1 below lb
function [x, idx] = xintobounds(x, lbc, ubc)
    L = repmat(lbc, 1, size(x, 2));
    U = repmat(ubc, 1, size(x, 2));
    below = x < L;
    x(below) = min(U(below), max(L(below), 2 * L(below) - x(below)));
    above = x > U;
    x(above) = max(L(above), min(U(above), 2 * U(above) - x(above)));
    idx = above - below;
end

function X = reflect_rows(X, lb, ub)
    X = xintobounds(X', lb', ub')';
end

function X = redraw_nonfinite(X, lb, span)
    bad = ~isfinite(X);
    if any(bad(:))
        R = repmat(lb, size(X, 1), 1) + rand(size(X)) .* repmat(span, size(X, 1), 1);
        X(bad) = R(bad);
    end
end

function res = myprctile(inar, perc)
    N = numel(inar);
    sar = sort(inar);
    res = zeros(1, numel(perc));
    for q = 1:numel(perc)
        p = perc(q);
        if p <= 100 * (0.5 / N)
            res(q) = sar(1);
        elseif p >= 100 * ((N - 0.5) / N)
            res(q) = sar(N);
        else
            avail = 100 * ((1:N) - 0.5) / N;
            i = find(p > avail, 1, 'last');
            res(q) = sar(i) + (sar(i+1) - sar(i)) * (p - avail(i)) / (avail(i+1) - avail(i));
        end
    end
end

% Picks the MOEA whose fitted a*exp(b*t) is lowest at t = 2*CS
function indx = fit_exp(success, CS)
    si = (ceil(CS / 2):CS)';
    expected = zeros(1, 3);
    for oi = 1:3
        expected(oi) = exp1_extrapolate(si, success(si, oi), 2 * CS);
    end
    [~, indx] = min(expected);
end

% Least-squares a*exp(b*t) through (t, y), profiled over b, evaluated at t_new
function yhat = exp1_extrapolate(t, y, t_new)
    if any(~isfinite(y))
        yhat = y(end);
        if isnan(yhat)
            yhat = inf;
        end
        return;
    end
    tc = t - t(end);
    bgrid = [-fliplr(logspace(-6, 1, 60)), 0, logspace(-6, 1, 60)];
    sse = zeros(size(bgrid));
    for q = 1:numel(bgrid)
        sse(q) = profile_sse(bgrid(q), tc, y);
    end
    [~, q] = min(sse);
    lo = bgrid(max(1, q - 1));
    hi = bgrid(min(numel(bgrid), q + 1));
    gr = (sqrt(5) - 1) / 2;
    for it = 1:60
        b1 = hi - gr * (hi - lo);
        b2 = lo + gr * (hi - lo);
        if profile_sse(b1, tc, y) <= profile_sse(b2, tc, y)
            hi = b2;
        else
            lo = b1;
        end
    end
    b = (lo + hi) / 2;
    if profile_sse(b, tc, y) > sse(q)
        b = bgrid(q);
    end
    e = exp(b * tc);
    a = (e' * y) / (e' * e);
    yhat = a * exp(b * (t_new - t(end)));
end

function s = profile_sse(b, tc, y)
    e = exp(b * tc);
    a = (e' * y) / (e' * e);
    s = sum((a * e - y) .^ 2);
end

% ----------------------------------------------------------------------- %
% United Multi-Operator Evolutionary Algorithms II (UMOEAs-II)
% CEC 2016 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PS1 = 18*D -> 4             % DE subpopulation, reduced linearly
%   PS2 = 4 + floor(3*ln D)     % CMA-ES subpopulation
%   CS = 50/100/150             % Cycle length at D = 10 / 30 / other
%   memory = 6, arc_rate = 2.6  % F and CR memory slots, archive size
%   prob_ls = 0.1 or 0.01       % Local search probability, set by its last result
%   sigma = 0.3, LS_FE = 2 % of maxFe   % CMA-ES step size and local search budget
%
% Algorithm Concept:
%   - Two subpopulations evolve side by side, one by multi-operator DE and one
%     by CMA-ES, and each cycle a probability decides which of them runs
%   - That probability comes from both quality and diversity, so a converged
%     subpopulation loses its share even while its best is good
%   - Every second cycle the stronger one seeds the weaker: either the DE
%     population re-initialises CMA-ES, or the CMA-ES best joins the DE population
%   - The DE phase picks per individual between three operators, whose
%     probabilities follow their measured relative improvement
%   - Past three quarters of the budget an interior-point local search polishes
%     the best solution, and its success sets how often it runs again
%
% Reference:
% Saber M. Elsayed, Noha Hamza, Ruhul A. Sarker,
% Testing United Multi-Operator Evolutionary Algorithms-II on Single Objective
% Optimization Problems,
% 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 2966-2973.
% https://doi.org/10.1109/CEC.2016.7744164
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released MATLAB (UMOEAsII.rar in Suganthan's CEC2016
% archive). Its CMA-ES phase shares its lineage with this repository's ebocmar,
% so the common parts follow that file, including the eigen guards ebocmar needed
% and its unscaled sigma = 0.3. Two things from the release are kept as written:
% CMA-ES samples are left unrepaired until half the budget is spent, and the
% feasibility test on a new best compares against a hard -100/100 rather than the
% problem's own box, which on a suite with another box simply never fires.
% That test is coordinate-wise, so a NaN fails it: the release's min/max skip NaN,
% and on CEC2020RW F12 the clamped flat plateau inflates sigma to Inf, whose NaN
% samples reached best_solution in 4 campaign runs.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = umoea2(problem)

    n = problem.dimension;
    xmin = problem.lb;
    xmax = problem.ub;
    maxFE = problem.maxFe;

    Par = introd_par(n, xmin, xmax, maxFE);
    PS1 = Par.PopSize;
    PS2 = 4 + floor(3 * log(n));

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    pop_total = PS1 + PS2;
    x = repmat(xmin, pop_total, 1) + rand(pop_total, n) .* repmat(xmax - xmin, pop_total, 1);
    [fitx, FE] = calculate_fitness(x', problem, FE);
    fitx = fitx(:)';

    [bestold, bes_l] = min(fitx);
    bestx = x(bes_l, :);
    for i = 1:min(pop_total, maxFE)
        curve(i) = min(fitx(1:i));
        [population_history, fitness_history, history_index] = record_history(...
            i, x, fitx, population_history, fitness_history, history_index, maxFE);
    end

    EA_1 = x(1:PS1, :);          EA_obj1 = fitx(1:PS1);
    EA_2 = x(PS1+1:end, :);      EA_obj2 = fitx(PS1+1:end);
    [EA_obj1, ord] = sort(EA_obj1);  EA_1 = EA_1(ord, :);
    [EA_obj2, ord] = sort(EA_obj2);  EA_2 = EA_2(ord, :);

    setting = init_cma_par(struct(), EA_2, n, PS2);
    fitness = struct('hist', NaN(1, 10 + ceil(3 * 10 * n / PS2)));
    fitness.hist(1) = EA_obj2(1);

    probDE1 = ones(1, Par.n_opr) / Par.n_opr;
    arch_rate = 2.6;
    archive.NP = round(arch_rate * PS1);
    archive.pop = zeros(0, n);
    archive.funvalues = zeros(0, 1);
    memory_size = 6;
    archive_f = 0.5 * ones(1, memory_size);
    archive_Cr = 0.5 * ones(1, memory_size);
    hist_pos = 1;

    InitPop = PS1;
    iter = 0;
    cy = 0;
    indx = 0;
    Probs = ones(1, 2);

    while FE < maxFE
        iter = iter + 1;
        cy = cy + 1;

        if cy == ceil(Par.CS + 1)
            % Share of each subpopulation, from quality and diversity together
            qual = [EA_obj1(1), EA_obj2(1)];
            norm_qual = 1 - qual ./ sum(qual);
            D = [mean(pdist2_local(EA_1(2:PS1, :), EA_1(1, :))), ...
                 mean(pdist2_local(EA_2(2:PS2, :), EA_2(1, :)))];
            norm_div = D ./ sum(D);
            Probs = norm_qual + norm_div;
            Probs = max(0.1, min(0.9, Probs ./ sum(Probs)));
            [~, indx] = max(Probs);
            if Probs(1) == Probs(2)
                indx = 0;
            end
        elseif cy == 2 * ceil(Par.CS)
            if indx == 1
                list_ind = randperm(PS1);
                list_ind = list_ind(1:min(PS2, PS1));
                EA_2(1:numel(list_ind), :) = EA_1(list_ind, :);
                EA_obj2(1:numel(list_ind)) = EA_obj1(list_ind);
                setting = init_cma_par(setting, EA_2, n, PS2);
                setting.sigma = setting.sigma * (1 - FE / maxFE);
            elseif all(EA_2(1, :) > -100 & EA_2(1, :) < 100)
                EA_1(PS1, :) = EA_2(1, :);
                EA_obj1(PS1) = EA_obj2(1);
                [EA_obj1, ord] = sort(EA_obj1);
                EA_1 = EA_1(ord, :);
            end
            cy = 1;
            Probs = ones(1, 2);
        end

        if FE < maxFE && rand < Probs(1)
            UpdPopSize = round(((Par.MinPopSize - InitPop) / maxFE) * FE + InitPop);
            if PS1 > UpdPopSize
                reduction = min(PS1 - UpdPopSize, PS1 - Par.MinPopSize);
                EA_1(end-reduction+1:end, :) = [];
                EA_obj1(end-reduction+1:end) = [];
                PS1 = PS1 - reduction;
                archive.NP = round(arch_rate * PS1);
                if size(archive.pop, 1) > archive.NP
                    keep = randperm(size(archive.pop, 1), archive.NP);
                    archive.pop = archive.pop(keep, :);
                    archive.funvalues = archive.funvalues(keep);
                end
            end

            [EA_1, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, archive_f, archive_Cr, ...
             FE, curve, population_history, fitness_history, history_index] = ...
                samo_de(EA_1, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, ...
                        archive_f, archive_Cr, xmin, xmax, n, PS1, FE, problem, maxFE, curve, ...
                        population_history, fitness_history, history_index);
        end

        if FE < maxFE && rand < Probs(2)
            [EA_2, EA_obj2, setting, bestold, bestx, fitness, FE, curve, ...
             population_history, fitness_history, history_index] = ...
                samo_es(EA_2, setting, iter, bestold, bestx, fitness, xmin, xmax, n, PS2, ...
                        FE, problem, maxFE, curve, population_history, fitness_history, history_index);
        end

        % The local search only starts while budget is left, as the phases above do
        if FE > 0.75 * maxFE && FE < maxFE && rand < Par.prob_ls
            old_FE = FE;
            [ls_x, ls_f, FE, succ] = local_search(bestx, bestold, Par, FE, problem, xmin, xmax);
            for e = (old_FE + 1):min(FE, maxFE)
                curve(e) = min(bestold, ls_f);
                [population_history, fitness_history, history_index] = record_history(...
                    e, EA_1, EA_obj1, population_history, fitness_history, history_index, maxFE);
            end
            if succ == 1
                bestx = ls_x;
                bestold = ls_f;
                EA_1(PS1, :) = bestx;
                EA_obj1(PS1) = bestold;
                [EA_obj1, ord] = sort(EA_obj1);
                EA_1 = EA_1(ord, :);
                EA_2 = repmat(EA_1(1, :), PS2, 1);
                setting = init_cma_par(setting, EA_2, n, PS2);
                setting.sigma = 1e-05;
                EA_obj2(1:PS2) = EA_obj1(1);
                Par.prob_ls = 0.1;
            else
                Par.prob_ls = 0.01;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bestold;
    best_fitness = bestold;
    best_solution = bestx;
end

function [x, fitx, prob, bestold, bestx, archive, hist_pos, archive_f, archive_Cr, ...
          FE, curve, ph, fh, hi] = ...
    samo_de(x, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, archive_f, ...
            archive_Cr, xmin, xmax, n, PopSize, FE, problem, maxFE, curve, ph, fh, hi)

    mem_rand_index = ceil(memory_size * rand(PopSize, 1));
    mu_sf = archive_f(mem_rand_index);
    mu_cr = archive_Cr(mem_rand_index);

    cr = normrnd_local(mu_cr, 0.1)';
    cr(mu_cr == -1) = 0;
    cr = min(cr, 1);

    F = mu_sf + 0.1 * tan(pi * (rand(1, PopSize) - 0.5));
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1, numel(pos)) - 0.5));
        pos = find(F <= 0);
    end
    F = min(F, 1)';

    popAll = [x; archive.pop];
    [r1, r2, r3] = gnR1R2(PopSize, size(popAll, 1), 1:PopSize);

    % Each individual takes one of the three operators, by their current shares
    bb = rand(PopSize, 1);
    l2 = sum(prob(1:2));
    op_1 = bb <= prob(1);
    op_2 = bb > prob(1) & bb <= l2;
    op_3 = bb > l2;

    vi = zeros(PopSize, n);
    pNP = max(round(0.1 * PopSize), 2);
    phix = x(max(1, ceil(rand(1, PopSize) * pNP)), :);
    vi(op_1, :) = x(op_1, :) + F(op_1, ones(1, n)) .* ...
        (phix(op_1, :) - x(op_1, :) + x(r1(op_1), :) - popAll(r2(op_1), :));
    vi(op_2, :) = x(op_2, :) + F(op_2, ones(1, n)) .* ...
        (phix(op_2, :) - x(op_2, :) + x(r1(op_2), :) - x(r3(op_2), :));

    pNP = max(round(0.5 * PopSize), 2);
    phix = x(max(1, ceil(rand(1, PopSize) * pNP)), :);
    vi(op_3, :) = F(op_3, ones(1, n)) .* x(r1(op_3), :) + phix(op_3, :) - x(r3(op_3), :);

    vi = han_boun(vi, xmax, xmin, x, PopSize, 1);

    mask = rand(PopSize, n) > cr(:, ones(1, n));
    jrand = sub2ind([PopSize n], (1:PopSize)', floor(rand(PopSize, 1) * n) + 1);
    mask(jrand) = false;
    ui = vi;
    ui(mask) = x(mask);

    [fitx_new, FE] = calculate_fitness(ui', problem, FE);
    fitx_new = fitx_new(:)';

    for e = 1:PopSize
        eval_count = FE - PopSize + e;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = min(bestold, min(fitx_new(1:e)));
            [ph, fh, hi] = record_history(eval_count, x, fitx, ph, fh, hi, maxFE);
        end
    end

    diff = abs(fitx - fitx_new);
    I = fitx_new < fitx;
    goodCR = cr(I)';
    goodF = F(I)';

    archive = update_archive(archive, x(I, :), fitx(I)');

    % Operator shares follow their relative improvement
    diff2 = max(0, (fitx - fitx_new)) ./ abs(fitx);
    count_S = [max(0, mean_or_zero(diff2(op_1))), max(0, mean_or_zero(diff2(op_2))), ...
               max(0, mean_or_zero(diff2(op_3)))];
    if all(count_S ~= 0)
        prob = max(0.1, min(0.9, count_S ./ sum(count_S)));
    else
        prob = ones(1, 3) / 3;
    end

    fitx(I) = fitx_new(I);
    x(I, :) = ui(I, :);

    if numel(goodCR) > 0
        weightsDE = diff(I) ./ sum(diff(I));
        archive_f(hist_pos) = (weightsDE * (goodF .^ 2)') / (weightsDE * goodF');
        if max(goodCR) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (weightsDE * (goodCR .^ 2)') / (weightsDE * goodCR');
        end
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size
            hist_pos = 1;
        end
    end

    [fitx, ord] = sort(fitx);
    x = x(ord, :);
    if fitx(1) < bestold
        bestold = fitx(1);
        bestx = x(1, :);
    end
end

function [x, fitx, setting, bestold, bestx, fitness, FE, curve, ph, fh, hi] = ...
    samo_es(x, setting, iter, bestold, bestx, fitness, xmin, xmax, n, PopSize, FE, problem, ...
            maxFE, curve, ph, fh, hi)

    arz = randn(n, PopSize);
    arx = repmat(setting.xmean, 1, PopSize) + setting.sigma * (setting.BD * arz);

    % The release leaves samples unrepaired for the first half of the budget
    handle_limit = 0.5;
    if FE >= handle_limit * maxFE
        arxvalid = han_boun(arx', xmax, xmin, x, PopSize, 2)';
    else
        arxvalid = arx;
    end

    [raw, FE] = calculate_fitness(arxvalid, problem, FE);
    raw = raw(:)';

    % Only the batch best can replace bestold, and only inside the hard-coded box
    inbox = all(arxvalid >= -100 & arxvalid <= 100, 1);
    [~, ib] = min(raw);
    accept = raw(ib) < bestold && inbox(ib);
    run_best = bestold;
    for e = 1:PopSize
        if accept && inbox(e) && raw(e) < run_best
            run_best = raw(e);
        end
        eval_count = FE - PopSize + e;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = run_best;
            [ph, fh, hi] = record_history(eval_count, arxvalid', raw, ph, fh, hi, maxFE);
        end
    end
    if accept
        bestold = raw(ib);
        bestx = arxvalid(:, ib)';
    end

    [sel, idxsel] = sort(raw);
    raw = raw(idxsel);
    arxvalid = arxvalid(:, idxsel);
    arx = arx(:, idxsel);
    arz = arz(:, idxsel);

    setting.xold = setting.xmean;
    setting.xmean = arx(:, 1:setting.mu) * setting.weights;
    if FE >= handle_limit * maxFE
        setting.xmean = han_boun(setting.xmean', xmax, xmin, x(1, :), 1, 2)';
    end
    zmean = arz(:, 1:setting.mu) * setting.weights;

    setting.ps = (1 - setting.cs) * setting.ps + ...
        sqrt(setting.cs * (2 - setting.cs) * setting.mueff) * (setting.B * zmean);
    hsig = norm(setting.ps) / sqrt(1 - (1 - setting.cs)^(2 * iter)) / setting.chiN < 1.4 + 2 / (n + 1);
    setting.pc = (1 - setting.cc) * setting.pc + ...
        hsig * (sqrt(setting.cc * (2 - setting.cc) * setting.mueff) / setting.sigma) * ...
        (setting.xmean - setting.xold);

    if setting.ccov1 + setting.ccovmu > 0
        arpos = (arx(:, 1:setting.mu) - repmat(setting.xold, 1, setting.mu)) / setting.sigma;
        setting.C = (1 - setting.ccov1 - setting.ccovmu + (1 - hsig) * setting.ccov1 * setting.cc * (2 - setting.cc)) * setting.C ...
            + setting.ccov1 * (setting.pc * setting.pc') ...
            + setting.ccovmu * arpos * (repmat(setting.weights, 1, n) .* arpos');
        setting.diagC = diag(setting.C);
    end

    setting.sigma = setting.sigma * exp(min(1, (sqrt(sum(setting.ps.^2)) / setting.chiN - 1) * setting.cs / setting.damps));

    if (setting.ccov1 + setting.ccovmu) > 0 && mod(iter, 1 / (setting.ccov1 + setting.ccovmu) / n / 10) < 1
        setting.C = triu(setting.C) + triu(setting.C, 1)';
        setting.C = real((setting.C + setting.C') / 2);
        if any(~isfinite(setting.C(:)))
            setting.C = eye(n, n);
        end
        [setting.B, tmp] = eig(setting.C);
        setting.diagD = real(diag(tmp));
        if min(setting.diagD) <= 0
            setting.diagD(setting.diagD < 0) = 0;
            if max(setting.diagD) <= 0
                setting.C = eye(n, n); setting.B = eye(n, n); setting.diagD = ones(n, 1);
            else
                tmp = max(setting.diagD) / 1e14;
                setting.C = setting.C + tmp * eye(n, n);
                setting.diagD = setting.diagD + tmp * ones(n, 1);
            end
        end
        if max(setting.diagD) > 1e14 * min(setting.diagD)
            tmp = max(setting.diagD) / 1e14 - min(setting.diagD);
            setting.C = setting.C + tmp * eye(n, n);
            setting.diagD = setting.diagD + tmp * ones(n, 1);
        end
        setting.diagC = diag(setting.C);
        setting.diagD = sqrt(setting.diagD);
        setting.BD = setting.B .* repmat(setting.diagD', n, 1);
    end

    % Flat fitness inflates the step size, as in Hansen's code
    if sel(1) == sel(1 + ceil(0.1 + PopSize / 4))
        setting.sigma = setting.sigma * exp(0.2 + setting.cs / setting.damps);
    end
    fitness.hist = [sel(1), fitness.hist(1:end-1)];
    if iter > 2 && (max(fitness.hist(~isnan(fitness.hist))) - min(fitness.hist(~isnan(fitness.hist)))) == 0
        setting.sigma = setting.sigma * exp(0.2 + setting.cs / setting.damps);
    end

    x = arxvalid';
    fitx = raw;
end

function [x, f, FE, succ] = local_search(bestx, f, Par, FE, problem, xmin, xmax)
    LS_FE = min(ceil(0.02 * Par.Max_FES), Par.Max_FES - FE);
    options = optimset('Display', 'off', 'algorithm', 'interior-point', ...
                       'UseParallel', 'never', 'MaxFunEvals', LS_FE);
    x = bestx;
    succ = 0;
    f0 = evaluate_for_ls(bestx(:), problem);
    FE = FE + 1;
    if ~isfinite(f0)
        return;
    end
    try
        [Xls, FUN, ~, details] = fmincon(@(xx) evaluate_for_ls(xx, problem), bestx(:), ...
                                         [], [], [], [], xmin, xmax, [], options);
    catch
        return;
    end
    if (f - FUN) > 0
        succ = 1;
        f = FUN;
        x = Xls(:)';
    end
    FE = FE + details.funcCount;
end

function f = evaluate_for_ls(x, problem)
    f = calculate_fitness(x(:), problem, 0);
    f = f(1);
end

function [Par] = introd_par(n, xmin, xmax, maxFE)
    Par.n_opr = 3;
    Par.n = n;
    if n == 10
        Par.CS = 50;
    elseif n == 30
        Par.CS = 100;
    else
        Par.CS = 150;
    end
    Par.xmin = xmin;
    Par.xmax = xmax;
    Par.Max_FES = maxFE;
    Par.PopSize = 18 * n;
    Par.MinPopSize = 4;
    Par.prob_ls = 0.1;
end

function [setting] = init_cma_par(setting, EA_2, n, n2)
    setting.xmean = mean(EA_2)';
    setting.insigma = 0.3;
    setting.sigma = setting.insigma;
    setting.pc = zeros(n, 1);
    setting.ps = zeros(n, 1);
    setting.diagD = ones(n, 1);
    setting.diagC = setting.diagD .^ 2;
    setting.B = eye(n, n);
    setting.BD = setting.B .* repmat(setting.diagD', n, 1);
    setting.C = diag(setting.diagC);
    setting.chiN = n^0.5 * (1 - 1/(4*n) + 1/(21*n^2));
    setting.mu = ceil(n2 / 2);
    setting.weights = log(max(setting.mu, n/2) + 1/2) - log(1:setting.mu)';
    setting.mueff = sum(setting.weights)^2 / sum(setting.weights.^2);
    setting.weights = setting.weights / sum(setting.weights);
    setting.cc = (4 + setting.mueff/n) / (n + 4 + 2*setting.mueff/n);
    setting.cs = (setting.mueff + 2) / (n + setting.mueff + 3);
    setting.ccov1 = 2 / ((n + 1.3)^2 + setting.mueff);
    setting.ccovmu = 2 * (setting.mueff - 2 + 1/setting.mueff) / ((n + 2)^2 + setting.mueff);
    setting.damps = 0.5 + 0.5*min(1, (0.27*n2/setting.mueff - 1)^2) + ...
                    2*max(0, sqrt((setting.mueff - 1)/(n + 1)) - 1) + setting.cs;
    setting.xold = setting.xmean;
end

function x = han_boun(x, xmax, xmin, x2, PopSize, hb)
    x_L = repmat(xmin, PopSize, 1);
    x_U = repmat(xmax, PopSize, 1);
    switch hb
        case 1
            pos = x < x_L;
            x(pos) = (x2(pos) + x_L(pos)) / 2;
            pos = x > x_U;
            x(pos) = (x2(pos) + x_U(pos)) / 2;
        case 2
            pos = x < x_L;
            x(pos) = min(x_U(pos), max(x_L(pos), 2*x_L(pos) - x2(pos)));
            pos = x > x_U;
            x(pos) = max(x_L(pos), min(x_U(pos), 2*x_L(pos) - x2(pos)));
    end
end

function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
    NP0 = numel(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0
            break;
        end
        r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    r1 = r1(:); r2 = r2(:); r3 = r3(:);
end

function archive = update_archive(archive, pop, funvalue)
    if archive.NP == 0 || isempty(pop)
        return;
    end
    popAll = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, keep] = unique(popAll, 'rows', 'stable');
    popAll = popAll(keep, :);
    funvalues = funvalues(keep);
    if size(popAll, 1) <= archive.NP
        archive.pop = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1), archive.NP);
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos);
    end
end

function d = pdist2_local(A, b)
    d = sqrt(sum((A - repmat(b, size(A, 1), 1)) .^ 2, 2));
end

function v = normrnd_local(mu, sigma)
    v = mu + sigma * randn(size(mu));
end

function m = mean_or_zero(v)
    if isempty(v)
        m = 0;
    else
        m = mean(v);
    end
end

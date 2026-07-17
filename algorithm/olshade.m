% ----------------------------------------------------------------------- %
% OLSHADE-CS (Orthogonal-Learning LSHADE with Conservative Selection)
% for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize   = 6 * n^2        % LSHADE phase population size
%   nich_size = 10             % Neighbourhood/niche size
%   n_opr     = 4              % Number of mutation operators
%
% Algorithm Concept:
%   - Phase 1: orthogonal-array based initialization + orthogonal-learning DE
%     over the first 20% of the budget
%   - Phase 2: LSHADE ensemble (4 operators, linear population reduction,
%     archive) with a conservative greedy selection in the late budget
%
% Reference:
% Abhishek Kumar, Partha P. Biswas, Ponnuthurai N. Suganthan,
% Differential evolution with orthogonal array-based initialization and a
% novel selection strategy,
% Swarm and Evolutionary Computation 68 (2022) 101010.
% https://doi.org/10.1016/j.swevo.2021.101010
%
% Note: the reference implementation hardcodes the [-100,100] CEC box in the
% orthogonal-array scaling, the mutation clamp, and the feasibility check.
% Those are generalized here to the actual problem bounds (identical on the
% [-100,100] CEC suites) so the algorithm also runs on CEC2020RW. The
% orthogonal-array size Q for non-standard dimensions uses round(dim*2.5) so
% Q stays integer (identical for the standard 5/10/15/20/30/50/100 dims).
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = olshade(problem)

    % Extract problem parameters
    Par.n = problem.dimension;
    Par.Max_FES = problem.maxFe;
    Par.xmin = problem.lb;
    Par.xmax = problem.ub;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    Par.n_opr = 4;
    Par.Printing = 0;
    Par.PopSize = 6 * Par.n .^ 2;
    Par.MinPopSize = 4;
    Par.nich_size = 10;

    FE = 0;
    curve = zeros(1, maxFE);
    history_pop_size = 100;
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, history_pop_size, Par.n);
    fitness_history = zeros(history_size, history_pop_size);
    history_index = 1;

    %% Niche / orthogonal-array based initialization
    [pop_fix, pop_near_idx_fix] = genpop(Par.n, Par.nich_size, lb, ub);
    Par.Max_FES1 = 0.2 * Par.Max_FES;

    %% Phase 1: Orthogonal learning
    pop = pop_fix; pop_near_idx = pop_near_idx_fix;
    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    [bestold, bes_l] = min(fitness); bestx = pop(bes_l, :);

    [curve, population_history, fitness_history, history_index] = rec_block(...
        pop, fitness, 1, FE, bestold, curve, population_history, fitness_history, ...
        history_index, sampling_interval, history_size, maxFE, history_pop_size);

    hist_pos = 1;
    memory_size = 20 * Par.n;
    archive_f = ones(1, memory_size) .* 0.2;
    archive_Cr = ones(1, memory_size) .* 0.2;
    res_det = repmat(bestold, 1, size(pop, 1));
    stop_con = 0;

    while stop_con == 0
        ev0 = FE;
        [pop, fitness, pop_near_idx, archive_f, archive_Cr, hist_pos, bestold, bestx, FE, res_det] = ...
            oDE(pop, fitness, pop_near_idx, archive_f, archive_Cr, hist_pos, bestold, bestx, memory_size, ...
            Par.xmin, Par.xmax, FE, Par.nich_size, res_det, Par.Printing, problem, lb, ub);
        [curve, population_history, fitness_history, history_index] = rec_block(...
            pop, fitness, ev0 + 1, FE, bestold, curve, population_history, fitness_history, ...
            history_index, sampling_interval, history_size, maxFE, history_pop_size);
        if FE > Par.Max_FES1 || FE >= maxFE
            stop_con = 1;
        end
    end

    [fitness, idx] = sort(fitness);
    pop = pop(idx, :);
    PS1 = Par.PopSize;
    EA = pop(1:PS1, :); EA_obj = fitness(1:PS1);

    probDE1 = 1 ./ Par.n_opr .* ones(1, Par.n_opr);
    arch_rate = 2.6;
    archive.NP = arch_rate * PS1;
    archive.pop = zeros(0, Par.n);
    archive.funvalues = zeros(0, 1);

    hist_pos = 1;
    memory_size = 20 * Par.n;
    archive_f = ones(1, memory_size) .* 0.2;
    archive_Cr = ones(1, memory_size) .* 0.2;

    stop_con = 0; InitPop = PS1;

    %% Phase 2: LSHADE ensemble
    while stop_con == 0
        UpdPopSize = round((((Par.MinPopSize - InitPop) / Par.Max_FES) * (FE)) + InitPop);
        if PS1 > UpdPopSize
            reduction_ind_num = PS1 - UpdPopSize;
            if PS1 - reduction_ind_num < Par.MinPopSize
                reduction_ind_num = PS1 - Par.MinPopSize;
            end
            for r = 1:reduction_ind_num
                vv = PS1;
                EA(vv, :) = [];
                EA_obj(vv) = [];
                PS1 = PS1 - 1;
            end
            archive.NP = round(arch_rate * PS1);
            if size(archive.pop, 1) > archive.NP
                rndpos = randperm(size(archive.pop, 1));
                rndpos = rndpos(1:archive.NP);
                archive.pop = archive.pop(rndpos, :);
            end
        end

        ev0 = FE;
        [EA, EA_obj, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, FE, res_det] = ...
            LSHADE_MODE_greedy(EA, EA_obj, probDE1, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, ...
            Par.xmin, Par.xmax, Par.n, PS1, FE, res_det, Par.Printing, Par.Max_FES, problem, lb, ub);
        [curve, population_history, fitness_history, history_index] = rec_block(...
            EA, EA_obj, ev0 + 1, FE, bestold, curve, population_history, fitness_history, ...
            history_index, sampling_interval, history_size, maxFE, history_pop_size);

        if (FE >= Par.Max_FES - 4 * UpdPopSize) || FE >= maxFE
            stop_con = 1;
        end
    end

    curve(min(FE, maxFE):end) = bestold;

    best_solution = bestx;
    best_fitness = bestold;
end

%% --- Record best-so-far curve + top-k history over an FE block ---
function [curve, ph, fh, hidx] = rec_block(pop, costs, fe_from, fe_to, bestval, curve, ph, fh, hidx, si, hs, maxFE, hps)
    fe_to = min(fe_to, maxFE);
    if fe_from < 1, fe_from = 1; end
    if fe_to < fe_from, return; end
    dim = size(pop, 2);
    costs = costs(:)';
    [sf, sidx] = sort(costs);
    tk = min(hps, numel(costs));
    rp = NaN(hps, dim); rf = NaN(1, hps);
    rp(1:tk, :) = pop(sidx(1:tk), :);
    rf(1:tk) = sf(1:tk);
    for ec = fe_from:fe_to
        curve(ec) = bestval;
        [ph, fh, hidx] = record_history(ec, rp, rf, ph, fh, hidx, si, hs);
    end
end

%% --- Orthogonal-array based population ---
function [pop, nghbr_idx] = genpop(dim, niche_size, lb, ub)
    if dim == 5
        Q = 15;
    elseif dim == 10
        Q = 25;
    elseif dim == 15
        Q = 40;
    elseif dim == 20
        Q = 50;
    else
        Q = round(dim * 2.5);   % keep Q integer for non-standard dims
    end
    J = 2;
    N = (Q^J - 1) / (Q - 1);

    pop_init = oa_permut(Q, N, J);
    pop_init(:, (dim + 1):N) = [];

    gmin = min(pop_init(:)); gmax = max(pop_init(:));
    pop = lb + (pop_init - gmin) .* (ub - lb) / (gmax - gmin);

    leng = Q^J;
    nghbr_idx = zeros(leng, niche_size);
    % Per-row nearest-neighbour search (avoids allocating a leng-by-leng
    % distance matrix; identical neighbours to the reference implementation).
    for i = 1:leng
        d = sqrt(sum((pop - pop(i, :)).^2, 2));
        [~, sindex] = sort(d);
        nghbr_idx(i, :) = sindex(1:niche_size);
    end
end

function A = oa_permut(q, n, j)
    if n ~= (q^j - 1) / (q - 1)
        A = [];
        return
    end
    row = q^j;
    A = zeros(row, n);
    for k = 1:j
        J = ((q^(k - 1) - 1) / (q - 1)) + 1;
        for i = 1:q^j
            A(i, J) = floor(((i - 1) / (q^(j - k))));
        end
    end
    for k = 2:j
        J = ((q^(k - 1) - 1) / (q - 1)) + 1;
        for s = 1:J - 1
            for t = 1:q - 1
                x = J + (s - 1) * (q - 1) + t;
                A(:, x) = mod(A(:, s) * t + A(:, J), q);
            end
        end
    end
    A = mod(A, q);
end

%% --- Orthogonal-learning DE stage ---
function [pop, fitness, pop_near_idx, archive_f, archive_Cr, hist_pos, bestold, bestx, current_eval, res_det] = ...
        oDE(pop, fitness, pop_near_idx, archive_f, archive_Cr, hist_pos, bestold, bestx, memory_size, ...
        ~, ~, current_eval, nich_size, res_det, Printing, problem, lb, ub)

    [pop_size, problem_size] = size(pop);
    mem_rand_index1 = ceil(memory_size * rand(pop_size, 1));
    mu_sf1 = archive_f(mem_rand_index1)';
    mu_cr1 = archive_Cr(mem_rand_index1)';

    [cr1, sf1] = gencrsf(mu_cr1, mu_sf1, pop_size);

    children_fitness = zeros(pop_size, 1);
    for jj = 1:pop_size
        pop_nghbr = pop(pop_near_idx(jj, :), :);
        fitness_nghbr = [pop_near_idx(jj, :)', fitness(pop_near_idx(jj, :), :)];
        [~, best_mem_idx] = sort(fitness_nghbr(:, 2));
        nn = floor(rand * min(nich_size, 0.01 * pop_size)) + 1;
        bestmem = pop(fitness_nghbr(best_mem_idx(nn), 1), :); %#ok<NASGU>
        X1 = DE(pop_nghbr, bestmem, 3, sf1(jj), cr1(jj), problem_size, nich_size, lb, ub);
        [children_fitness(jj, :), current_eval] = calculate_fitness(X1', problem, current_eval);

        if children_fitness(jj, :) < fitness(jj, :)
            pop(jj, :) = X1;
            fitness(jj, :) = children_fitness(jj, :);
            if fitness(jj, :) <= bestold, bestold = fitness(jj, :); bestx = pop(jj, :); end
        end
    end

    dif = abs(fitness - children_fitness);
    I = (fitness > children_fitness);
    goodCR = cr1(I == 1);
    goodF = sf1(I == 1);
    dif_val = dif(I == 1);
    num_success_params = numel(goodCR);

    if num_success_params > 0
        sum_dif = sum(dif_val);
        dif_val = dif_val / sum_dif;
        archive_f(hist_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
        if max(goodCR) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
        end
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size, hist_pos = 1; end
    end

    if Printing == 1
        res_det = [res_det repmat(bestold, 1, pop_size)];
    end
end

function [cr, sf] = gencrsf(mu_cr, mu_sf, pop_size)
    cr = normrnd(mu_cr, 0.1);
    term_pos = mu_cr == -1;
    cr(term_pos) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);
    sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
    pos = find(sf <= 0);
    while ~isempty(pos)
        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end
    sf = min(sf, 1);
end

function ui = DE(pop, bm, st, F, CR, n, NP, lb, ub)
    jj = 1;
    r1 = round(rand * NP); r2 = round(rand * NP); r3 = round(rand * NP);
    while (r1 == jj || r1 == 0), r1 = ceil(rand * NP); end
    while (r2 == jj || r2 == r1 || r2 == 0), r2 = ceil(rand * NP); end
    while (r3 == jj || r3 == r1 || r3 == r2 || r3 == 0), r3 = ceil(rand * NP); end
    pm1 = pop(r1, 1:n);
    pm2 = pop(r2, 1:n);
    pm3 = pop(r3, 1:n);
    popold = pop(jj, :);

    mui = rand(1, n) < CR;
    if mui == zeros(1, n), nn = randperm(n); mui(nn(1)) = 1; end
    mpo = mui < 0.5;

    if (st == 1)
        ui = pm3 + F * (pm1 - pm2);
        ui = popold .* mpo + ui .* mui;
    elseif (st == 2)
        ui = bm + F * (pm1 - pm2);
        ui = popold .* mpo + ui .* mui;
    elseif (st == 3)
        ui = popold + F * (bm - popold) + F * (pm1 - pm2);
        ui = popold .* mpo + ui .* mui;
    end

    ui = max(min(ui, ub), lb);
end

%% --- LSHADE multi-operator (greedy/conservative) stage ---
function [x, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, current_eval, res_det] = ...
    LSHADE_MODE_greedy(x, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, archive_f, archive_Cr, xmin, xmax, n, ...
    PopSize, current_eval, res_det, Printing, Max_FES, problem, lb, ub)

    vi = zeros(PopSize, n);

    mem_rand_index = ceil(memory_size * rand(PopSize, 1));
    mu_sf = archive_f(mem_rand_index);
    mu_cr = archive_Cr(mem_rand_index);

    cr = normrnd(mu_cr, 0.1);
    term_pos = find(mu_cr == -1);
    cr(term_pos) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);

    F = mu_sf + 0.1 * tan(pi * (rand(1, PopSize) - 0.5));
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1, length(pos)) - 0.5));
        pos = find(F <= 0);
    end
    F = min(F, 1);
    F = F';
    [fitx, inddd] = sort(fitx);
    x = x(inddd, :);
    [cr, ~] = sort(cr);

    popAll = [x; archive.pop];
    r0 = 1:PopSize;
    [r1, r2, r3] = gnR1R2(PopSize, size(popAll, 1), r0);
    bb = rand(PopSize, 1);
    probiter = prob(1, :);
    l2 = sum(prob(1:2));
    l3 = sum(prob(2:3));
    op_1 = bb <= probiter(1) * ones(PopSize, 1);
    op_2 = bb > probiter(1) * ones(PopSize, 1) & bb <= (l2 * ones(PopSize, 1));
    op_3 = bb > l2 * ones(PopSize, 1) & bb <= (l3 * ones(PopSize, 1));
    op_4 = bb > l3 * ones(PopSize, 1) & bb <= (1 * ones(PopSize, 1));

    pNP = max(round(0.25 * PopSize), 1);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_1 == 1, :) = x(op_1 == 1, :) + F(op_1 == 1, ones(1, n)) .* (phix(op_1 == 1, :) - x(op_1 == 1, :) + x(r1(op_1 == 1), :) - popAll(r2(op_1 == 1), :));
    vi(op_2 == 1, :) = x(op_2 == 1, :) + F(op_2 == 1, ones(1, n)) .* (phix(op_2 == 1, :) - x(op_2 == 1, :) + x(r1(op_2 == 1), :) - x(r3(op_2 == 1), :));
    vi(op_4 == 1, :) = phix(op_4 == 1, :) + F(op_4 == 1, ones(1, n)) .* (x(r3(op_4 == 1), :) - popAll(r2(op_4 == 1), :));
    pNP = max(round(0.5 * PopSize), 2);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_3 == 1, :) = F(op_3 == 1, ones(1, n)) .* x(r1(op_3 == 1), :) + (phix(op_3 == 1, :) - x(r3(op_3 == 1), :));

    vi = han_boun(vi, xmax, xmin, x, PopSize, 2);

    if rand < 0.4
        mask = false(PopSize, n);
        for iii = 1:PopSize
            mask(iii, :) = rand(1, n) > cr(:, ones(1, n));
        end
        rows = (1:PopSize)'; cols = floor(rand(PopSize, 1) * n) + 1;
        jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
        ui = vi;
        ui(mask) = x(mask);
    else
        ui = x;
        startLoc = randi(n, PopSize, 1);
        for i = 1:PopSize
            l = startLoc(i);
            while (rand < cr(i) && l < n)
                l = l + 1;
            end
            for j = startLoc(i):l
                ui(i, j) = vi(i, j);
            end
        end
    end

    [fitx_new, current_eval] = calculate_fitness(ui', problem, current_eval);
    fitx_new = fitx_new(:);

    diff = abs(fitx - fitx_new);
    if current_eval / Max_FES > 0.6
        I = (fitx_new < fitx);
    else
        I = SelectionGreedy2(x, ui, fitx(:), fitx_new(:));
    end

    goodCR = cr(I == 1);
    goodF = F(I == 1);

    archive = updateArchive(archive, x(I == 1, :), fitx(I == 1));

    diff2 = max(0, (fitx - fitx_new)) ./ abs(fitx);
    count_S(1) = max(0, mean(diff2(op_1 == 1)));
    count_S(2) = max(0, mean(diff2(op_2 == 1)));
    count_S(3) = max(0, mean(diff2(op_3 == 1)));
    count_S(4) = max(0, mean(diff2(op_4 == 1)));
    if count_S ~= 0
        prob = max(0.1, min(0.9, count_S ./ (sum(count_S))));
    else
        prob = 1 / 4 * ones(1, 4);
    end

    fitx(I == 1) = fitx_new(I == 1);
    x(I == 1, :) = ui(I == 1, :);

    if size(goodF, 1) == 1, goodF = goodF'; end
    if size(goodCR, 1) == 1, goodCR = goodCR'; end
    num_success_params = numel(goodCR);
    if num_success_params > 0
        weightsDE = diff(I == 1) ./ sum(diff(I == 1));
        archive_f(hist_pos) = (weightsDE' * (goodF .^ 2)) ./ (weightsDE' * goodF);
        if max(goodCR) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (weightsDE' * (goodCR .^ 2)) / (weightsDE' * goodCR);
        end
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size, hist_pos = 1; end
    else
        archive_Cr(hist_pos) = 0.5;
        archive_f(hist_pos) = 0.5;
    end

    [fitx, ind] = sort(fitx);
    x = x(ind, :);
    if fitx(1) < bestold && all(x(ind(1), :) >= lb) && all(x(ind(1), :) <= ub)
        bestold = fitx(1);
        bestx = x(1, :);
    end

    if Printing == 1
        res_det = [res_det repmat(bestold, 1, PopSize)];
    end
end

function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0, break; else, r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1; end
        if i > 1000, error('Cannot generate r1 in 1000 iterations'); end
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; else, r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1; end
        if i > 1000, error('Cannot generate r2 in 1000 iterations'); end
    end
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0, break; else, r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1; end
        if i > 1000, error('Cannot generate r3 in 1000 iterations'); end
    end
end

function x = han_boun(x, xmax, xmin, x2, PopSize, ~)
    hb = randi(3);
    switch hb
        case 1
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x(pos) = (x2(pos) + x_L(pos)) / 2;
            x_U = repmat(xmax, PopSize, 1);
            pos = x > x_U;
            x(pos) = (x2(pos) + x_U(pos)) / 2;
        case 2
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x_U = repmat(xmax, PopSize, 1);
            x(pos) = min(x_U(pos), max(x_L(pos), 2 * x_L(pos) - x2(pos)));
            pos = x > x_U;
            x(pos) = max(x_L(pos), min(x_U(pos), 2 * x_L(pos) - x2(pos)));
        case 3
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x_U = repmat(xmax, PopSize, 1);
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
            pos = x > x_U;
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
    end
end

function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0, return; end
    if size(pop, 1) ~= size(funvalue, 1), error('check it'); end
    popAll = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end
    if size(popAll, 1) <= archive.NP
        archive.pop = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:ceil(archive.NP));
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

function I = SelectionGreedy2(x, ui, fitx, fitx_new)
    N = size(x, 1);
    n = max(ceil(N * 0.15), 1);
    fit = [fitx; fitx_new];
    I1 = false(N, 1);
    for i = 1:N
        ind = randperm(2 * N, n);
        neigh = fit(ind(1:n)); %#ok<NASGU>
        imp = sum(fit(ind(1:n)) > fitx_new(i)) / n;
        if imp > 0.25
            I1(i) = true;
        end
    end
    I2 = (fitx_new < fitx);
    I = I1 & I2;
end

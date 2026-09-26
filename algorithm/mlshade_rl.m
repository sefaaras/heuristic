% ----------------------------------------------------------------------- %
% Multi-operator Ensemble LSHADE with Restart and Local Search (mLSHADE-RL)
% CEC 2024 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18*D -> 4 (linear)     % Linear population size reduction
%   memory_size = 5, arc_rate = 1.4   % Memory slots (F, CR, freq start at 0.5); archive factor
%   p_best_rate = 0.11                % pbest share, also for the ordered operator
%   pb = 0.4, ps = 0.5                % Eigen-crossover chance; neighbour share for its covariance
%   freq = 0.5, LP = 20               % Sinusoidal F: initial frequency, learning period
%   P_MS = 1/3 each -> [0.1, 0.9]     % Operator probabilities, adapted by improvement
%   prob_ls = 0.01 (0.1 after success) % Local-search chance per generation after 85 % of FEs
%   LS_FE = ceil(0.02*maxFE)          % Evaluations per local-search call
%
% Algorithm Concept:
%   - LSHADE-cnEpSin frame: in the first half F follows one of two sinusoidal
%     schedules chosen by recent success, afterwards a Cauchy memory draw
%   - Three mutations: current-to-pbest-w/1 with archive, current-to-pbest/1
%     without it, and current-to-ordpbest-w/1 (pbest, r1, r2 sorted by fitness)
%   - Each individual picks an operator by probability; the probabilities follow
%     the mean relative improvement of each operator, kept inside [0.1, 0.9]
%   - With probability 0.4 the crossover runs in the eigenbasis of the covariance
%     of the half of the population nearest the best
%   - Restart: once the population volume ratio drops below 0.001, long-stagnant
%     individuals are rebuilt by horizontal or vertical crossover
%   - In the last 15 % of the budget a gradient-based local search polishes the best
%
% Reference:
% Dikshit Chauhan, Anupam Trivedi, Shivani,
% A Multi-operator Ensemble LSHADE with Restart and Local Search Mechanisms for
% Single-objective Optimization,
% arXiv preprint arXiv:2409.15994 (2024).
% https://doi.org/10.48550/arXiv.2409.15994
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from mLSHADE_RL.m and helpers (organisers' CEC 2024 BC-SOP package, same
% code as the authors' repository). Its LS2 calls fmincon (sqp, Optimization
% Toolbox); here a local projected BFGS with forward-difference gradients runs on
% the same budget from the population best. The reference charges neither LS2
% nor restart evaluations; here every evaluation counts. Restarted individuals
% are re-evaluated (the reference comments that call out and keeps the stale
% fitness) and bound-repaired, as are Eigen-crossover trials (evaluated outside
% the box in the reference). G_Max is counted from the LPSR plan, not the fixed
% 2745 of D = 30; the volume ratio is taken in logs so high D cannot overflow it;
% normrnd, pdist2 and rands are replaced by base MATLAB.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mlshade_rl(problem)

    problem_size = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    freq_inti   = 0.5;
    pb          = 0.4;
    ps          = 0.5;
    prob        = ones(1, 3) / 3;
    p_best_rate = 0.11;
    arc_rate    = 1.4;
    memory_size = 5;
    pop_size    = 18 * problem_size;
    SEL         = round(ps * pop_size);
    prob_ls     = 0.01;
    max_pop_size = pop_size;
    min_pop_size = 4;
    LP          = 20;

    G_Max = lpsr_generations(maxFE, max_pop_size, min_pop_size);

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf_fit_var  = inf;
    popold = repmat(lb, pop_size, 1) + rand(pop_size, problem_size) .* repmat(ub - lb, pop_size, 1);
    bsf_solution = popold(1, :);
    n0 = min(pop_size, maxFE);
    if n0 < pop_size
        popold = popold(1:n0, :);
        pop_size = n0;
    end
    [fitness, FE] = calculate_fitness(popold', problem, FE);
    fitness = fitness(:);
    track(popold, fitness, popold, fitness);

    memory_sf   = 0.5 .* ones(memory_size, 1);
    memory_cr   = 0.5 .* ones(memory_size, 1);
    memory_freq = freq_inti * ones(memory_size, 1);
    memory_pos  = 1;

    archive.NP  = arc_rate * pop_size;
    archive.pop = zeros(0, problem_size);

    gg = 0;
    goodF1all = [];
    goodF2all = [];
    badF1all  = [];
    badF2all  = [];

    counter = zeros(pop_size, 1);              % stagnation counters, never shrunk with the population
    log_vlim = 0.5 * sum(log(abs(lu(1, :) - lu(2, :))));

    while FE < maxFE
        gg = gg + 1;

        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf   = memory_sf(mem_rand_index);
        mu_cr   = memory_cr(mem_rand_index);
        mu_freq = memory_freq(mem_rand_index);

        cr = mu_cr + 0.1 * randn(pop_size, 1);
        cr(mu_cr == -1) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end

        freq = mu_freq + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos_f = find(freq <= 0);
        while ~isempty(pos_f)
            freq(pos_f) = mu_freq(pos_f) + 0.1 * tan(pi * (rand(length(pos_f), 1) - 0.5));
            pos_f = find(freq <= 0);
        end

        sf   = min(sf, 1);
        freq = min(freq, 1);

        % The reference widens sf to pop_size x D with equal columns; one column is the same thing
        flag1 = false;
        flag2 = false;
        if FE <= maxFE / 2
            if gg <= LP
                use_first = rand < 0.5;
            else
                ns1_sum = sum(goodF1all(gg-LP:gg-1));
                nf1_sum = sum(badF1all(gg-LP:gg-1));
                sumS1 = (ns1_sum / (ns1_sum + nf1_sum)) + 0.01;
                ns2_sum = sum(goodF2all(gg-LP:gg-1));
                nf2_sum = sum(badF2all(gg-LP:gg-1));
                sumS2 = (ns2_sum / (ns2_sum + nf2_sum)) + 0.01;
                use_first = sumS1 / (sumS1 + sumS2) > sumS2 / (sumS2 + sumS1);
            end
            if use_first
                sf = 0.5 .* (sin(2 .* pi .* freq_inti .* gg + pi) .* ((G_Max - gg) / G_Max) + 1) .* ones(pop_size, 1);
                flag1 = true;
            else
                sf = 0.5 * (sin(2 * pi .* freq .* gg) .* (gg / G_Max) + 1);
                flag2 = true;
            end
        end

        bb = rand(pop_size, 1);
        l2 = sum(prob(1:2));
        op_1 = bb <= prob(1);
        op_2 = bb > prob(1) & bb <= l2;
        op_3 = bb > l2 & bb <= 1;

        popAll = [pop; archive.pop];
        [r1, r2, r3] = gnR1R2R3(pop_size, size(popAll, 1), 1:pop_size);

        pNP = max(round(p_best_rate * pop_size), 2);
        randindex = max(1, ceil(rand(1, pop_size) .* pNP));
        pbest = pop(sorted_index(randindex), :);
        if FE <= 0.2 * maxFE
            FW = 0.7 * sf;
        elseif FE <= 0.4 * maxFE
            FW = 0.8 * sf;
        else
            FW = 1.2 * sf;
        end

        vi = zeros(pop_size, problem_size);
        vi(op_1, :) = pop(op_1, :) + FW(op_1, ones(1, problem_size)) .* ...
            (pbest(op_1, :) - pop(op_1, :) + pop(r1(op_1), :) - popAll(r2(op_1), :));
        vi(op_2, :) = pop(op_2, :) + sf(op_2, ones(1, problem_size)) .* ...
            (pbest(op_2, :) - pop(op_2, :) + pop(r1(op_2), :) - pop(r3(op_2), :));

        % current-to-ordpbest-w/1: two random donors and a pbest, re-sorted by fitness
        EDErandindex = max(1, ceil(rand(1, pop_size) .* pNP));
        EDEpestind = sorted_index(EDErandindex);
        Rg = Gen_R(pop_size, 2);
        R1 = [Rg(:, 2:3), EDEpestind(:)];
        [~, I1] = sort(fitness(R1), 2);
        R_S = zeros(pop_size, 3);
        for i = 1:pop_size
            R_S(i, :) = R1(i, I1(i, :));
        end
        rb = R_S(:, 1);
        rm = R_S(:, 2);
        rw = R_S(:, 3);
        vi(op_3, :) = pop(op_3, :) + FW(op_3, ones(1, problem_size)) .* ...
            (pop(rb(op_3), :) - pop(op_3, :) + pop(rm(op_3), :) - popAll(rw(op_3), :));
        vi = boundConstraint(vi, pop, lu);

        J_ = mod(floor(rand(pop_size, 1) * problem_size), problem_size) + 1;
        J = (J_ - 1) * pop_size + (1:pop_size)';
        crs = rand(pop_size, problem_size) < cr(:, ones(1, problem_size));
        if rand < pb
            best = pop(sorted_index(1), :);
            Dis = sqrt(sum((pop - best) .^ 2, 2));
            [~, idx_ordered] = sort(Dis, 'ascend');
            Xsel = pop(idx_ordered(1:SEL), :);
            xmean = mean(Xsel, 1);
            C = 1 / (SEL - 1) * (Xsel - xmean(ones(SEL, 1), :))' * (Xsel - xmean(ones(SEL, 1), :));
            C = triu(C) + transpose(triu(C, 1));
            [R, Dg] = eig(C);
            if max(diag(Dg)) > 1e20 * min(diag(Dg))
                tmp = max(diag(Dg)) / 1e20 - min(diag(Dg));
                C = C + tmp * eye(problem_size);
                [R, ~] = eig(C);
            end
            Xr = pop * R;
            vr = vi * R;
            Ur = Xr;
            Ur(J) = vr(J);
            Ur(crs) = vr(crs);
            ui = Ur * R';
        else
            ui = pop;
            ui(J) = vi(J);
            ui(crs) = vi(crs);
        end
        ui = sanitise(ui, lb, ub);
        ui = boundConstraint(ui, pop, lu);

        n_eval = min(pop_size, maxFE - FE);
        children_fitness = inf(pop_size, 1);
        [cf, FE] = calculate_fitness(ui(1:n_eval, :)', problem, FE);
        children_fitness(1:n_eval) = cf(:);

        dif = abs(fitness - children_fitness);
        I = (fitness > children_fitness);
        goodCR   = cr(I);
        goodF    = sf(I);
        goodFreq = freq(I);
        dif_val  = dif(I);
        badF     = sf(~I);
        if flag1
            goodF1all = [goodF1all numel(goodF)]; %#ok<AGROW>
            badF1all  = [badF1all numel(badF)]; %#ok<AGROW>
            goodF2all = [goodF2all 1]; %#ok<AGROW>
            badF2all  = [badF2all 1]; %#ok<AGROW>
        end
        if flag2
            goodF2all = [goodF2all numel(goodF)]; %#ok<AGROW>
            badF2all  = [badF2all numel(badF)]; %#ok<AGROW>
            goodF1all = [goodF1all 1]; %#ok<AGROW>
            badF1all  = [badF1all 1]; %#ok<AGROW>
        end

        % Operator probabilities from the mean relative improvement, Eq. (22)
        diff2 = max(0, (fitness - children_fitness)) ./ abs(children_fitness);
        count_S = [max(0, mean(diff2(op_1))), max(0, mean(diff2(op_2))), max(0, mean(diff2(op_3)))];
        if all(count_S ~= 0)
            prob = max(0.1, min(0.9, count_S ./ sum(count_S)));
        else
            prob = ones(1, 3) / 3;
        end

        archive = updateArchive(archive, popold(I, :));

        [fitness, Isel] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(Isel == 2, :) = ui(Isel == 2, :);
        track(ui(1:n_eval, :), children_fitness(1:n_eval), popold, fitness);

        if ~isempty(goodCR)
            dif_val = dif_val / sum(dif_val);
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
            end
            if max(goodFreq) == 0 || memory_freq(memory_pos) == -1
                memory_freq(memory_pos) = -1;
            else
                memory_freq(memory_pos) = (dif_val' * (goodFreq .^ 2)) / (dif_val' * goodFreq);
            end
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size
                memory_pos = 1;
            end
        end

        if FE < maxFE
            [~, bes_l] = min(fitness);
            [rebuilt, counter, redo] = restart_mechanism(popold, counter, Isel, bes_l, log_vlim);
            if any(redo)
                rows = find(redo);
                rows = rows(1:min(numel(rows), maxFE - FE));
                new_pos = boundConstraint(sanitise(rebuilt(rows, :), lb, ub), popold(rows, :), lu);
                [rf, FE] = calculate_fitness(new_pos', problem, FE);
                popold(rows, :) = new_pos;
                fitness(rows) = rf(:);
                track(new_pos, rf(:), popold, fitness);
            end
        end

        plan_pop_size = round((((min_pop_size - max_pop_size) / maxFE) * FE) + max_pop_size);
        if pop_size > plan_pop_size
            reduction_ind_num = pop_size - plan_pop_size;
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = pop_size - min_pop_size;
            end
            pop_size = pop_size - reduction_ind_num;
            SEL = round(ps * pop_size);
            for r = 1:reduction_ind_num
                [~, indBest] = sort(fitness, 'ascend');
                worst_ind = indBest(end);
                popold(worst_ind, :) = [];
                fitness(worst_ind, :) = [];
            end
            archive.NP = round(arc_rate * pop_size);
            if size(archive.pop, 1) > archive.NP
                rndpos = randperm(size(archive.pop, 1));
                archive.pop = archive.pop(rndpos(1:archive.NP), :);
            end
        end

        if FE > 0.85 * maxFE && FE < maxFE && rand < prob_ls
            [f_ls0, indx] = min(fitness);
            [x_ls, f_ls] = local_search(popold(indx, :), f_ls0, ceil(0.02 * maxFE));
            if f_ls0 - f_ls > 0
                popold(pop_size, :) = x_ls;
                fitness(pop_size) = f_ls;
                [fitness, sort_indx] = sort(fitness);
                popold = popold(sort_indx, :);
                prob_ls = 0.1;
            else
                prob_ls = 0.01;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf_fit_var;

    best_fitness  = bsf_fit_var;
    best_solution = bsf_solution;

    % Best-so-far per evaluation, then one history sample per evaluation with the settled population
    function track(X, f, P, fP)
        n = size(X, 1);
        for q = 1:n
            if f(q) < bsf_fit_var
                bsf_fit_var  = f(q);
                bsf_solution = X(q, :);
            end
            ec = FE - n + q;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf_fit_var;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, P, fP, population_history, fitness_history, history_index, maxFE);
            end
        end
    end

    % Projected BFGS with forward differences, in place of the reference's fmincon sqp call
    function [x, f] = local_search(x0, f0, budget)
        x = x0;
        f = f0;
        n = numel(x);
        used = 0;
        Hinv = eye(n);
        [g, ok] = fd_grad(x, f);
        while ok && used < budget && FE < maxFE
            act = (x <= lb & g > 0) | (x >= ub & g < 0);
            free = ~act;
            if ~any(free) || max(abs(g(free))) <= 1e-6 * max(1, abs(f))
                break;
            end
            d = zeros(1, n);
            d(free) = -(Hinv(free, free) * g(free)')';
            if g * d' >= 0
                Hinv = eye(n);
                d = zeros(1, n);
                d(free) = -g(free);
            end
            alpha = 1;
            accepted = false;
            while used < budget && FE < maxFE && alpha > 1e-10
                xt = min(max(x + alpha * d, lb), ub);
                if all(xt == x)
                    break;
                end
                ft = probe(xt);
                if ft <= f + 1e-4 * (g * (xt - x)')
                    accepted = true;
                    break;
                end
                alpha = alpha / 2;
            end
            if ~accepted
                break;
            end
            s = xt - x;
            [gt, ok] = fd_grad(xt, ft);
            x = xt;
            f = ft;
            if ~ok
                break;
            end
            y = gt - g;
            g = gt;
            sy = s * y';
            if sy > 1e-12 * norm(s) * norm(y)
                rho = 1 / sy;
                V = eye(n) - rho * (s' * y);
                Hinv = V * Hinv * V' + rho * (s' * s);
            end
            if norm(s) <= 1e-10 * (1 + norm(x))
                break;
            end
        end

        function fv1 = probe(z)
            [fv0, FE] = calculate_fitness(z', problem, FE);
            fv1 = fv0(1);
            used = used + 1;
            if fv1 < bsf_fit_var
                bsf_fit_var  = fv1;
                bsf_solution = z;
            end
            curve(FE) = bsf_fit_var;
            [population_history, fitness_history, history_index] = record_history(...
                FE, popold, fitness, population_history, fitness_history, history_index, maxFE);
        end

        % Forward step sqrt(eps)*max(|x|, 1), taken backwards where it would leave the box
        function [gr, done] = fd_grad(xc, fc)
            gr = zeros(1, n);
            done = false;
            for jj = 1:n
                if used >= budget || FE >= maxFE
                    return;
                end
                h = sqrt(eps) * max(abs(xc(jj)), 1);
                if xc(jj) + h > ub(jj)
                    h = -h;
                end
                xh = xc;
                xh(jj) = xc(jj) + h;
                gr(jj) = (probe(xh) - fc) / h;
            end
            done = all(isfinite(gr));
        end
    end
end

function g = lpsr_generations(maxFE, max_pop_size, min_pop_size)
% Generations the linear population size reduction plan allows on this budget
    g = 0;
    fe = 0;
    while fe < maxFE
        fe = fe + max(min_pop_size, ...
                      round(((min_pop_size - max_pop_size) / maxFE) * fe + max_pop_size));
        g = g + 1;
    end
    g = max(1, g - 1);
end

function X = sanitise(X, lb, ub)
% Non-finite coordinates are redrawn uniformly inside the box
    bad = ~isfinite(X);
    if any(bad(:))
        [~, cols] = find(bad);
        X(bad) = lb(cols)' + rand(numel(cols), 1) .* (ub(cols) - lb(cols))';
    end
end

function [x, counter, redo] = restart_mechanism(x, counter, I, gbestid, log_vlim)
% Restart_mechanism.m; returns which rows it rebuilt so the caller can evaluate them
    [PopSize, n] = size(x);
    redo = false(PopSize, 1);
    Co = 0;
    for i = 1:PopSize
        if I(i) == 1
            counter(i) = counter(i) + 1;
            Co = Co + counter(i);
        else
            counter(i) = 0;
        end
    end
    % sqrt(V_pop / V_lim) with V_pop = sqrt(prod(range/2)) and V_lim = sqrt(prod(ub - lb))
    nVOL = exp(0.5 * (0.5 * sum(log((max(x, [], 1) - min(x, [], 1)) ./ 2)) - log_vlim));
    if nVOL < 0.001
        for i = 1:PopSize
            if Co > 2 * PopSize && counter(i) > 2 * n && i ~= gbestid
                nDim_j = randi(n);
                nSeq_j = randperm(n);
                j = nSeq_j(1:nDim_j);
                nDim_i = randi(PopSize);
                nSeq_i1 = randperm(PopSize);
                j_i1 = nSeq_i1(1:nDim_i);
                nSeq_i2 = randperm(PopSize);
                j_i2 = nSeq_i2(1:nDim_i);
                if rand > 0.5
                    for i1 = 1:nDim_i
                        x(i, :) = rand * x(j_i1(i1), :) + (1 - rand) * x(j_i2(i1), :) ...
                                  + (2 * rand - 1) * (x(j_i1(i1), :) - x(j_i2(i1), :));
                    end
                else
                    for j1 = 1:nDim_j
                        j_num = j(j1);
                        pop_num_1 = x(randi(PopSize), j_num);
                        x(i, j_num) = rand * x(i, j_num) + (1 - rand) * pop_num_1;
                    end
                end
                counter(i) = 0;
                redo(i) = true;
            end
        end
    end
end

function R = Gen_R(NP_Size, N)
% Row i holds i followed by N distinct indices different from i (Hadi's Gen_R)
    R = zeros(N + 1, NP_Size);
    R(1, :) = 1:NP_Size;
    for i = 2:N + 1
        R(i, :) = ceil(rand(NP_Size, 1) * NP_Size);
        flag = 0;
        while flag ~= 1
            pos = (R(i, :) == R(1, :));
            for w = 2:i - 1
                pos = or(pos, (R(i, :) == R(w, :)));
            end
            if sum(pos) == 0
                flag = 1;
            else
                R(i, pos) = floor(rand(sum(pos), 1) * NP_Size) + 1;
            end
        end
    end
    R = R';
end

function [r1, r2, r3] = gnR1R2R3(NP1, NP2, r0)
% r1 ~= r0 from the population, r2 ~= r0, r1 from population + archive, r3 ~= r0, r1, r2
    NP0 = length(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1001
        pos = (r1 == r0);
        if ~any(pos)
            break;
        end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        if i > 1000
            error('mlshade_rl:r1', 'Cannot generate r1 in 1000 iterations');
        end
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:1001
        pos = ((r2 == r1) | (r2 == r0));
        if ~any(pos)
            break;
        end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        if i > 1000
            error('mlshade_rl:r2', 'Cannot generate r2 in 1000 iterations');
        end
    end
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1001
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if ~any(pos)
            break;
        end
        r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        if i > 1000
            error('mlshade_rl:r3', 'Cannot generate r3 in 1000 iterations');
        end
    end
end

function archive = updateArchive(archive, pop)
% Add the replaced parents, drop duplicates, then trim at random to the archive size
    if archive.NP == 0
        return;
    end
    popAll = [archive.pop; pop];
    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll = popAll(IX, :);
    end
    if size(popAll, 1) <= archive.NP
        archive.pop = popAll;
    else
        rndpos = randperm(size(popAll, 1));
        archive.pop = popAll(rndpos(1:floor(archive.NP)), :);
    end
end

function vi = boundConstraint(vi, pop, lu)
% L-SHADE bound handling: a violating coordinate moves to the parent/bound midpoint
    NP = size(pop, 1);
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

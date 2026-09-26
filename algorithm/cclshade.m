% ----------------------------------------------------------------------- %
% Cooperative Co-evolution L-SHADE with Restarts (CCLSHADE)
% CEC 2016 (CEC 2015 benchmark track) -- 2nd place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 18*|G| -> 4           % L-SHADE population of each group, reduced linearly
%   p = 0.11, arc_rate = 1.4    % pbest rate and archive size of every group
%   H = 5                       % Memory slots for F and CR, both initialised to 0.5
%   epsilon = 1e-10             % GDG interaction threshold, relative to min |f|
%   stall = 1000*|G| FE         % Restart window, applied only once the f spread < 1e-8
%   K = D, D/2, D/5             % Random group sizes over 50 / 30 / 20 % of the budget
%
% Algorithm Concept:
%   - Global differential grouping spends 1 + 2D + D(D-1)/2 + 10 evaluations to find
%     interacting variables; a few separable ones join the smallest group, many are cut into tens
%   - Each group runs its own L-SHADE inside a shared context vector, which takes
%     a group's best whenever it beats the whole vector
%   - A group that has not improved the context for 1000*|G| evaluations and whose
%     fitness has collapsed restarts, seeded with the context vector
%   - When grouping finds one group, the variables are split at random into 1, 2
%     and 5 groups over the budget, re-drawn while the search stagnates
%
% Reference:
% Mohammed El-Abd,
% Cooperative co-evolution using LSHADE with restarts for the CEC15 benchmarks,
% 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 4810-4814.
% https://doi.org/10.1109/CEC.2016.7744406
% Components: global differential grouping (Mei, Omidvar, Li, Yao, ACM TOMS 2016);
% L-SHADE (Tanabe and Fukunaga, CEC 2014).
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's CCLSHADE_Code.rar in Suganthan's CEC2016 package
% (CEC-2015-Learning-Based); its missing boundConstraintCC and updateArchive are
% L-SHADE's, as in lshade.m. The release reads a GDG grouping computed offline and
% charges its evaluations to every run; here GDG runs inside the run at the same
% FE cost, with its threshold on min |f| of the samples (the release subtracts the
% CEC bias, unknown here) and its mid-point at the box centre instead of 0. Kept
% as written: every group's schedule uses the last group's budget share, and a
% regroup does not reset the stall counter. Random groups differ by at most one
% variable when D is not divisible and share the larger group's population size;
% below D = 5 the last phase keeps D groups.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cclshade(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    ipr         = 18;
    p_best_rate = 0.11;
    arc_rate    = 1.4;
    memory_size = 5;
    epsilon     = 1e-10;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf          = inf;
    bsf_solution = lb + rand(1, dim) .* span;

    Groups = gdg_groups();

    NG = numel(Groups);
    problem_size   = zeros(1, NG);
    pop_size       = zeros(1, NG);
    max_pop_size   = zeros(1, NG);
    min_pop_size   = 4 * ones(1, NG);
    nfes           = zeros(1, NG);
    max_nfes_group = zeros(1, NG);
    memory_sf      = 0.5 * ones(memory_size, NG);
    memory_cr      = 0.5 * ones(memory_size, NG);
    memory_pos     = ones(1, NG);
    arch_pop = cell(1, NG);
    arch_fit = cell(1, NG);
    arch_NP  = zeros(1, NG);
    popold   = cell(1, NG);
    fitness  = cell(1, NG);
    popfull  = cell(1, NG);
    context  = zeros(1, dim);
    cf       = inf;

    if FE < maxFE && NG > 1
        run_gdg_groups();
    elseif FE < maxFE
        run_random_groups();
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;

    % CC_LSHADE_PopRestart: fixed GDG groups, each restarted on its own
    function run_gdg_groups()
        for g = 1:NG
            problem_size(g) = numel(Groups{g});
        end
        pop_size     = ipr * problem_size;
        max_pop_size = pop_size;
        min_pop_size = 4 * ones(1, NG);
        max_nfes     = maxFE - FE;
        max_nfes_group = (max_nfes * pop_size(NG) / sum(pop_size)) * ones(1, NG);
        for g = 1:NG
            reset_archive(g);
            popold{g} = random_part(g, pop_size(g));
            context(Groups{g}) = popold{g}(ceil(rand * pop_size(g)), :);
        end
        f0 = eval_rows(context);
        cf = f0(1);
        record_batch(FE, FE, context, cf);
        for g = 1:NG
            evaluate_group(g, popold{g});
        end

        stall = zeros(1, NG);
        while FE < maxFE
            for g = 1:NG
                if FE >= maxFE
                    break;
                end
                [improved, np_used] = evolve_group(g);
                if improved
                    stall(g) = 0;
                else
                    stall(g) = stall(g) + np_used;
                end

                for rg = 1:NG
                    if FE < maxFE && stall(rg) >= 1000 * problem_size(rg) && ...
                            max(fitness{rg}) - min(fitness{rg}) < 1e-8
                        stall(rg) = 0;
                        pop_size(rg)     = ipr * problem_size(rg);
                        max_pop_size(rg) = pop_size(rg);
                        min_pop_size(rg) = 4;
                        memory_sf(:, rg) = 0.5;
                        memory_cr(:, rg) = 0.5;
                        memory_pos(rg)   = 1;
                        max_nfes_group(rg) = max_nfes_group(rg) - nfes(rg);
                        nfes(rg) = 0;
                        reset_archive(rg);
                        P = random_part(rg, pop_size(rg));
                        P(1, :) = context(Groups{rg});
                        evaluate_group(rg, P);
                    end
                end
            end
        end
    end

    % CC_LSHADE_PopRestart_OneGroup: random groups of D, D/2 and D/5 variables in turn
    function run_random_groups()
        mfes      = [0.5, 0.3, 0.2];
        NumGroups = [1, 2, 5];
        Kindex    = 1;
        new_random_grouping(NumGroups(Kindex));
        max_nfes0 = maxFE - FE;
        max_nfes_group = (mfes(Kindex) * max_nfes0 * pop_size(end) / sum(pop_size)) * ones(1, NG);
        for g = 1:NG
            reset_archive(g);
            popold{g} = random_part(g, pop_size(g));
            context(Groups{g}) = popold{g}(ceil(rand * pop_size(g)), :);
        end
        f0 = eval_rows(context);
        cf = f0(1);
        record_batch(FE, FE, context, cf);
        % The release counts its phases from here, after the context evaluation
        FE_ctx   = FE;
        max_nfes = maxFE - FE_ctx;
        for g = 1:NG
            evaluate_group(g, popold{g});
        end

        stall = 0;
        while FE < maxFE
            if Kindex < 3 && (FE - FE_ctx) > sum(mfes(1:Kindex)) * max_nfes
                Kindex = Kindex + 1;
                new_random_grouping(NumGroups(Kindex));
                max_nfes_group = (mfes(Kindex) * max_nfes * pop_size(end) / sum(pop_size)) * ones(1, NG);
                for g = 1:NG
                    reset_archive(g);
                    P = random_part(g, pop_size(g));
                    P(1, :) = context(Groups{g});
                    popold{g} = P;
                end
                for g = 1:NG
                    evaluate_group(g, popold{g});
                end
                stall = 0;
            end

            for g = 1:NG
                if FE >= maxFE
                    break;
                end
                [improved, np_used] = evolve_group(g);
                if improved
                    stall = 0;
                else
                    stall = stall + np_used;
                end
            end

            if FE < maxFE && stall > 1000 * dim
                if NG == 1
                    if max(fitness{1}) - min(fitness{1}) < 1e-8
                        stall = 0;
                        pop_size(1)     = ipr * problem_size(1);
                        max_pop_size(1) = pop_size(1);
                        min_pop_size(1) = 4;
                        memory_sf(:, 1) = 0.5;
                        memory_cr(:, 1) = 0.5;
                        memory_pos(1)   = 1;
                        % Recomputed from the phase share, not from what the last restart had left
                        max_nfes_group(1) = mfes(Kindex) * max_nfes - nfes(1);
                        nfes(1) = 0;
                        reset_archive(1);
                        P = random_part(1, pop_size(1));
                        P(1, :) = context(Groups{1});
                        evaluate_group(1, P);
                    end
                elseif all(cellfun(@(A) size(A, 1), popold) == size(popold{1}, 1))
                    % Rows of the groups are glued into full vectors, then cut along a new grouping
                    OldPopulation = zeros(size(popold{1}, 1), dim);
                    for g = 1:NG
                        OldPopulation(:, Groups{g}) = popold{g};
                    end
                    Groups = split_groups(problem_size);
                    for g = 1:NG
                        reset_archive(g);
                        popold{g} = OldPopulation(:, Groups{g});
                    end
                    for g = 1:NG
                        evaluate_group(g, popold{g});
                    end
                end
            end
        end

        % A new phase: ng groups of near-equal size, every group's L-SHADE state fresh
        function new_random_grouping(ng)
            ng = min(ng, dim);
            sizes = floor(dim / ng) * ones(1, ng);
            sizes(1:mod(dim, ng)) = sizes(1:mod(dim, ng)) + 1;
            Groups = split_groups(sizes);
            NG = ng;
            problem_size   = sizes;
            pop_size       = ipr * max(sizes) * ones(1, ng);
            max_pop_size   = pop_size;
            min_pop_size   = 4 * ones(1, ng);
            nfes           = zeros(1, ng);
            memory_sf      = 0.5 * ones(memory_size, ng);
            memory_cr      = 0.5 * ones(memory_size, ng);
            memory_pos     = ones(1, ng);
            arch_pop = cell(1, ng);
            arch_fit = cell(1, ng);
            arch_NP  = zeros(1, ng);
            popold   = cell(1, ng);
            fitness  = cell(1, ng);
            popfull  = cell(1, ng);
        end
    end

    % One L-SHADE generation of group g inside the context; NP is its size before the reduction
    function [improved, NP] = evolve_group(g)
        G  = Groups{g};
        NP = pop_size(g);
        ps = problem_size(g);
        pop = popold{g};
        fit = fitness{g};
        [~, sorted_index] = sort(fit, 'ascend');

        mem_rand_index = ceil(memory_size * rand(NP, 1));
        mu_sf = memory_sf(mem_rand_index, g);
        mu_cr = memory_cr(mem_rand_index, g);

        cr = mu_cr + 0.1 * randn(NP, 1);
        cr(mu_cr == -1) = 0;
        cr = max(min(cr, 1), 0);

        sf = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(numel(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end
        sf = min(sf, 1);

        popAll = [pop; arch_pop{g}];
        [r1, r2] = gnR1R2(NP, size(popAll, 1), 1:NP);
        pNP = max(round(p_best_rate * NP), 2);
        randindex = max(1, ceil(rand(1, NP) .* pNP));
        pbest = pop(sorted_index(randindex), :);

        vi = pop + sf(:, ones(1, ps)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
        vi = bound_midpoint(vi, pop, lb(G), ub(G));

        mask = rand(NP, ps) > cr(:, ones(1, ps));
        jrand = sub2ind([NP ps], (1:NP)', floor(rand(NP, 1) * ps) + 1);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);
        bad = ~isfinite(ui);
        if any(bad(:))
            redraw = random_part(g, NP);
            ui(bad) = redraw(bad);
        end

        children = repmat(context, NP, 1);
        children(:, G) = ui;
        fe0 = FE;
        cfit = eval_rows(children);
        k = numel(cfit);
        nfes(g) = nfes(g) + k;

        % Only the evaluated trials take part in the selection
        dif = abs(fit(1:k) - cfit);
        I = fit(1:k) > cfit;
        goodCR  = cr(I);
        goodF   = sf(I);
        dif_val = dif(I);
        update_archive(g, pop(I, :), fit(I));

        [newfit, win] = min([fit(1:k), cfit], [], 2);
        won = find(win == 2);
        fit(1:k) = newfit;
        pop(won, :) = ui(won, :);
        fullpop = popfull{g};
        fullpop(won, :) = children(won, :);

        popold{g}  = pop;
        fitness{g} = fit;
        popfull{g} = fullpop;

        [minfit, minidx] = min(fit);
        improved = minfit < cf;
        if improved
            cf = minfit;
            context(G) = pop(minidx, :);
        end

        if ~isempty(goodCR)
            w = dif_val / sum(dif_val);
            memory_sf(memory_pos(g), g) = (w' * (goodF .^ 2)) / (w' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos(g), g) == -1
                memory_cr(memory_pos(g), g) = -1;
            else
                memory_cr(memory_pos(g), g) = (w' * (goodCR .^ 2)) / (w' * goodCR);
            end
            memory_pos(g) = memory_pos(g) + 1;
            if memory_pos(g) > memory_size
                memory_pos(g) = 1;
            end
        end

        plan_pop_size = round(((min_pop_size(g) - max_pop_size(g)) / max_nfes_group(g)) * nfes(g) + max_pop_size(g));
        if pop_size(g) > plan_pop_size
            reduction = pop_size(g) - plan_pop_size;
            if pop_size(g) - reduction < min_pop_size(g)
                reduction = max(0, pop_size(g) - min_pop_size(g));
            end
            pop_size(g) = pop_size(g) - reduction;
            for r = 1:reduction
                [~, order] = sort(fitness{g}, 'ascend');
                worst = order(end);
                popold{g}(worst, :)  = [];
                fitness{g}(worst)    = [];
                popfull{g}(worst, :) = [];
            end
            arch_NP(g) = round(arc_rate * pop_size(g));
            if size(arch_pop{g}, 1) > arch_NP(g)
                keep = randperm(size(arch_pop{g}, 1), arch_NP(g));
                arch_pop{g} = arch_pop{g}(keep, :);
                arch_fit{g} = arch_fit{g}(keep);
            end
        end

        record_batch(fe0 + 1, FE, popfull{g}, fitness{g});
    end

    % Evaluates group g's part P inside the context and adopts its best if it beats the context
    function evaluate_group(g, P)
        population = repmat(context, size(P, 1), 1);
        population(:, Groups{g}) = P;
        fe0 = FE;
        f = eval_rows(population);
        k = numel(f);
        nfes(g) = nfes(g) + k;
        popold{g}  = P(1:k, :);
        fitness{g} = f;
        popfull{g} = population(1:k, :);
        pop_size(g) = k;
        if k > 0
            [minfit, minidx] = min(f);
            if minfit < cf
                cf = minfit;
                context = population(minidx, :);
            end
            record_batch(fe0 + 1, FE, popfull{g}, fitness{g});
        end
    end

    % Evaluates as many rows of X as the budget allows, with bsf and curve per FE
    function f = eval_rows(X)
        kk = min(size(X, 1), maxFE - FE);
        f = zeros(0, 1);
        if kk <= 0
            return;
        end
        fe0 = FE;
        [fv, FE] = calculate_fitness(X(1:kk, :)', problem, FE);
        f = fv(:);
        for ee = 1:kk
            if f(ee) < bsf
                bsf          = f(ee);
                bsf_solution = X(ee, :);
            end
            curve(fe0 + ee) = bsf;
        end
    end

    function record_batch(fe_from, fe_to, P, Pf)
        for fe = fe_from:fe_to
            [population_history, fitness_history, history_index] = record_history(...
                fe, P, Pf, population_history, fitness_history, history_index, maxFE);
        end
    end

    function P = random_part(g, n)
        G = Groups{g};
        P = repmat(lb(G), n, 1) + rand(n, numel(G)) .* repmat(span(G), n, 1);
    end

    function reset_archive(g)
        arch_NP(g)  = round(arc_rate * pop_size(g));
        arch_pop{g} = zeros(0, numel(Groups{g}));
        arch_fit{g} = zeros(0, 1);
    end

    function update_archive(g, P, Pf)
        if arch_NP(g) == 0 || isempty(P)
            return;
        end
        popAll = [arch_pop{g}; P];
        funvalues = [arch_fit{g}; Pf(:)];
        [~, IX] = unique(popAll, 'rows');
        if numel(IX) < size(popAll, 1)
            popAll = popAll(IX, :);
            funvalues = funvalues(IX);
        end
        if size(popAll, 1) > arch_NP(g)
            keep = randperm(size(popAll, 1), arch_NP(g));
            popAll = popAll(keep, :);
            funvalues = funvalues(keep);
        end
        arch_pop{g} = popAll;
        arch_fit{g} = funvalues;
    end

    % Global differential grouping, then CCLSHADE's handling of the separable variables
    function groups = gdg_groups()
        samples = repmat(lb, 10, 1) + rand(10, dim) .* repmat(span, 10, 1);
        fe0 = FE;
        fs = eval_rows(samples);
        samples = samples(1:numel(fs), :);
        record_batch(fe0 + 1, FE, samples, fs);
        threshold = min(abs(fs)) * epsilon;

        mid = (lb + ub) / 2;
        P1 = probe(lb);
        X2 = repmat(lb, dim, 1);
        X2(1:dim + 1:end) = ub;
        P2 = probe(X2);
        X3 = repmat(lb, dim, 1);
        X3(1:dim + 1:end) = mid;
        P3 = probe(X3);
        [jj, ii] = find(triu(true(dim), 1)');
        npair = numel(ii);
        X4 = repmat(lb, npair, 1);
        X4(sub2ind([npair dim], (1:npair)', ii)) = ub(ii);
        X4(sub2ind([npair dim], (1:npair)', jj)) = mid(jj);
        P4 = probe(X4);

        % A budget too small for the probes leaves every variable in one group
        if numel(P1) < 1 || numel(P2) < dim || numel(P3) < dim || numel(P4) < npair
            groups = {1:dim};
            return;
        end
        delta1 = P1 - P2(ii);
        delta2 = P3(jj) - P4;
        Delta = zeros(dim);
        Delta(sub2ind([dim dim], ii, jj)) = abs(delta1 - delta2);
        Delta = Delta + Delta';
        labels = connected_components(Delta > threshold);

        seps = [];
        nonseps = {};
        for c = 1:max(labels)
            members = find(labels == c);
            if isscalar(members)
                seps = [seps, members]; %#ok<AGROW>
            else
                nonseps{end + 1} = members; %#ok<AGROW>
            end
        end
        seps = sort(seps);

        if isempty(seps) || isempty(nonseps)
            if isempty(seps)
                groups = nonseps;
            else
                groups = {seps};
            end
        else
            groups = [{seps}, nonseps];
        end
        has_sep = ~isempty(seps);

        if has_sep && numel(seps) <= 0.1 * dim && numel(groups) > 1
            sizes = cellfun(@numel, groups(2:end));
            [~, smallest] = min(sizes);
            groups{smallest + 1} = [groups{smallest + 1}, seps];
            groups = groups(2:end);
        elseif has_sep && numel(seps) > 10
            starts = 1:10:numel(seps);
            chunks = cell(1, numel(starts));
            for c = 1:numel(starts)
                chunks{c} = seps(starts(c):min(numel(seps), starts(c) + 9));
            end
            if numel(chunks{end}) <= 0.1 * dim && numel(chunks{end}) < 10
                chunks{end - 1} = [chunks{end - 1}, chunks{end}];
                chunks = chunks(1:end - 1);
            end
            groups = [groups(2:end), chunks];
        end

        % Each probe batch is recorded with the ten samples plus the probe just evaluated
        function f = probe(X)
            fp0 = FE;
            f = eval_rows(X);
            for q = 1:numel(f)
                [population_history, fitness_history, history_index] = record_history(...
                    fp0 + q, [samples; X(q, :)], [fs; f(q)], population_history, ...
                    fitness_history, history_index, maxFE);
            end
        end
    end
end

% Cuts a fresh permutation of the variables into consecutive groups of the given sizes
function groups = split_groups(sizes)
    perm = randperm(sum(sizes));
    edges = [0, cumsum(sizes)];
    groups = cell(1, numel(sizes));
    for g = 1:numel(sizes)
        groups{g} = perm(edges(g) + 1:edges(g + 1));
    end
end

function labels = connected_components(C)
    L = size(C, 1);
    labels = zeros(1, L);
    ccc = 0;
    while true
        ind = find(labels == 0, 1);
        if isempty(ind)
            break;
        end
        ccc = ccc + 1;
        labels(ind) = ccc;
        list = ind;
        while ~isempty(list)
            reach = any(C(list, :), 1) & labels == 0;
            labels(reach) = ccc;
            list = find(reach);
        end
    end
end

function vi = bound_midpoint(vi, pop, lbg, ubg)
    NP = size(pop, 1);
    xl = repmat(lbg, NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(ubg, NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
    NP0 = numel(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    pos = (r1 == r0);
    while any(pos)
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        pos = (r1 == r0);
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    pos = (r2 == r1) | (r2 == r0);
    while any(pos)
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        pos = (r2 == r1) | (r2 == r0);
    end
    r1 = r1(:);
    r2 = r2(:);
end

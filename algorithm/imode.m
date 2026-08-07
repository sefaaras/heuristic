% ----------------------------------------------------------------------- %
% Improved Multi-operator Differential Evolution (IMODE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize     = 6*n^2 (n <= 20) / 18*n (n > 20)   % see the port note
%   MinPopSize  = 4                                 % linear reduction target
%   n_opr       = 3                                 % DE operators
%   arch_rate   = 2.6                               % archive size factor
%   memory_size = 20*n                              % CR/F memory length
%   prob_ls     = 0.1                               % SQP local-search trigger
%
% Algorithm Concept:
%   - Three DE mutation operators run in one population and are selected per
%     individual by the adaptive probabilities probDE, which are refreshed
%     from each operator's normalised fitness improvement
%       DE1: current-to-phi-best/1 with archive
%       DE2: current-to-phi-best/1 without archive
%       DE3: weighted-rand/1 around the top 50 %
%   - Randomised boundary repair (han_boun), binomial or exponential crossover
%     chosen at random, SHADE-style CR/F memories and linear population size
%     reduction
%   - In the last 15 % of the budget an SQP local search (LS2, fmincon) is
%     applied to the incumbent with probability prob_ls, which is raised to
%     0.1 on success and lowered to 0.01 on failure
%
% Reference:
% Karam M. Sallam, Saber M. Elsayed, Ripon K. Chakrabortty, Michael J. Ryan,
% Improved Multi-operator Differential Evolution Algorithm for Solving
% Unconstrained Problems,
% 2020 IEEE Congress on Evolutionary Computation (CEC), pp. 1-8.
% https://doi.org/10.1109/CEC48606.2020.9185577
% (winner of the CEC2020 single-objective bound-constrained competition)
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = imode(problem)

    n       = problem.dimension;
    xmin    = problem.lb;
    xmax    = problem.ub;
    Max_FES = problem.maxFe;

    n_opr = 3;
    if n <= 20
        PopSize = 6 * n * n;
    else
        PopSize = 18 * n;
    end
    PopSize = max(8, min(PopSize, max(8, floor(Max_FES / 4))));
    MinPopSize = 4;
    prob_ls = 0.1;
    Gmax = 3401;

    current_eval = 0;
    curve = zeros(1, Max_FES);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialise x
    x = repmat(xmin, PopSize, 1) + repmat(xmax - xmin, PopSize, 1) .* rand(PopSize, n);
    [fitx, current_eval] = calculate_fitness(x', problem, current_eval);
    fitx = fitx(:)';

    [bestold, bes_l] = min(fitx);
    bestx = x(bes_l, :);
    bsf = bestold;
    bsx = bestx;

    for eval_count = 1:min(PopSize, Max_FES)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, x, fitx, population_history, fitness_history, ...
            history_index, Max_FES);
    end

    PS1 = PopSize;
    EA_1 = x;
    EA_obj1 = fitx;
    EA_1old = x(randperm(PS1), :);

    probDE1 = 1 ./ n_opr .* ones(1, n_opr);

    arch_rate = 2.6;
    archive.NP = arch_rate * PS1;
    archive.pop = zeros(0, n);
    archive.funvalues = zeros(0, 1);

    hist_pos = 1;
    memory_size = 20 * n;
    archive_f  = ones(1, memory_size) .* 0.2;
    archive_Cr = ones(1, memory_size) .* 0.2;

    InitPop = PS1;
    iter = 0;
    UpdPopSize = PS1;
    F  = normrnd(0.5, 0.15, 1, PS1);
    cr = normrnd(0.5, 0.15, 1, PS1);

    % Main loop
    while current_eval < Max_FES
        iter = iter + 1;

        % Linear reduction of PS1
        UpdPopSize = round((((MinPopSize - InitPop) / Max_FES) * current_eval) + InitPop);
        if PS1 > UpdPopSize
            reduction_ind_num = PS1 - UpdPopSize;
            if PS1 - reduction_ind_num < MinPopSize
                reduction_ind_num = PS1 - MinPopSize;
            end
            for r = 1:reduction_ind_num
                vv = PS1;
                EA_1(vv, :) = [];
                EA_1old(vv, :) = [];
                EA_obj1(vv) = [];
                PS1 = PS1 - 1;
            end
            archive.NP = round(arch_rate * PS1);
            if size(archive.pop, 1) > archive.NP
                rndpos = randperm(size(archive.pop, 1));
                rndpos = rndpos(1:archive.NP);
                archive.pop = archive.pop(rndpos, :);
                archive.funvalues = archive.funvalues(rndpos, :);
            end
        end

        % Apply IMODE
        prev_eval = current_eval;
        [EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, ...
         archive_f, archive_Cr, current_eval] = ...
            imode_step(EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, ...
                       hist_pos, memory_size, archive_f, archive_Cr, xmin, xmax, n, ...
                       PS1, current_eval, problem, Max_FES, Gmax, iter);

        if bestold < bsf, bsf = bestold; bsx = bestx; end
        nAdd = current_eval - prev_eval;
        for k = 1:nAdd
            ec = prev_eval + k;
            if ec >= 1 && ec <= Max_FES
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, EA_1, EA_obj1, population_history, fitness_history, ...
                    history_index, Max_FES);
            end
        end

        % LS2 (SQP local search)
        if current_eval > 0.85 * Max_FES && current_eval < Max_FES
            if rand < prob_ls
                old_fit_eva = current_eval;
                [bestx_new, bestold_new, current_eval, succ, ls_curve, ls_best, ls_bestx] = ...
                    ls2_sqp(bestx, bestold, problem, current_eval, Max_FES, xmin, xmax, bsf);
                if succ == 1
                    bestx = bestx_new;
                    bestold = bestold_new;
                    EA_1(PS1, :) = bestx;
                    EA_obj1(PS1) = bestold;
                    [EA_obj1, sort_indx] = sort(EA_obj1);
                    EA_1 = EA_1(sort_indx, :);
                    prob_ls = 0.1;
                else
                    prob_ls = 0.01;
                end
                if bestold < bsf, bsf = bestold; bsx = bestx; end
                if ls_best < bsf, bsf = ls_best; bsx = ls_bestx; end
                for k = 1:numel(ls_curve)
                    ec = old_fit_eva + k;
                    if ec >= 1 && ec <= Max_FES
                        curve(ec) = ls_curve(k);
                    end
                end
                if current_eval >= 1 && current_eval <= Max_FES
                    curve(current_eval) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        current_eval, EA_1, EA_obj1, population_history, fitness_history, ...
                        history_index, Max_FES);
                end
            end
        end

        % Stopping criterion
        if current_eval >= Max_FES - 4 * UpdPopSize
            break;
        end
    end

    curve(min(current_eval, Max_FES):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% one IMODE generation
function [x, xold, fitx, prob, bestold, bestx, archive, hist_pos, ...
          archive_f, archive_Cr, current_eval] = ...
    imode_step(x, xold, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, ...
               archive_f, archive_Cr, xmin, xmax, n, PopSize, current_eval, ...
               problem, Max_FES, G_Max, gg) %#ok<INUSD>

    vi = zeros(PopSize, n);

    % CR and F
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
    cr = sort(cr);

    % Generate the new x
    popAll = [x; archive.pop];
    r0 = 1:PopSize;
    [r1, r2, r3] = gnR1R2(PopSize, size(popAll, 1), r0);

    bb = rand(PopSize, 1);
    probiter = prob(1, :);
    l2 = sum(prob(1:2));
    op_1 = bb <= probiter(1) * ones(PopSize, 1);
    op_2 = bb > probiter(1) * ones(PopSize, 1) & bb <= (l2 * ones(PopSize, 1));
    op_3 = bb > l2 * ones(PopSize, 1) & bb <= (ones(PopSize, 1));

    pNP = max(round(0.25 * PopSize), 1);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_1 == 1, :) = x(op_1 == 1, :) + F(op_1 == 1, ones(1, n)) .* ...
        (phix(op_1 == 1, :) - x(op_1 == 1, :) + x(r1(op_1 == 1), :) - popAll(r2(op_1 == 1), :));
    vi(op_2 == 1, :) = x(op_2 == 1, :) + F(op_2 == 1, ones(1, n)) .* ...
        (phix(op_2 == 1, :) - x(op_2 == 1, :) + x(r1(op_2 == 1), :) - x(r3(op_2 == 1), :));

    pNP = max(round(0.5 * PopSize), 2);
    randindex = ceil(rand(1, PopSize) .* pNP);
    randindex = max(1, randindex);
    phix = x(randindex, :);
    vi(op_3 == 1, :) = F(op_3 == 1, ones(1, n)) .* x(r1(op_3 == 1), :) + ...
        F(op_3 == 1, ones(1, n)) .* (phix(op_3 == 1, :) - x(r3(op_3 == 1), :));

    % Boundary handling
    vi = han_boun(vi, xmax, xmin, x, PopSize);

    % Crossover
    if rand < 0.4
        mask = rand(PopSize, n) > cr(:, ones(1, n));
        rows = (1:PopSize)'; cols = floor(rand(PopSize, 1) * n) + 1;
        jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
        ui = vi; ui(mask) = x(mask);
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

    % Evaluate
    [fitx_new, current_eval] = calculate_fitness(ui', problem, current_eval);
    fitx_new = fitx_new(:)';

    % CR/F improvement and operator probabilities
    diff = abs(fitx - fitx_new);
    I = (fitx_new < fitx);
    goodCR = cr(I == 1);
    goodF  = F(I == 1);

    archive = updateArchive(archive, x(I == 1, :), fitx(I == 1)');

    diff2 = max(0, (fitx - fitx_new)) ./ abs(fitx);
    count_S(1) = max(0, mean(diff2(op_1 == 1)));
    count_S(2) = max(0, mean(diff2(op_2 == 1)));
    count_S(3) = max(0, mean(diff2(op_3 == 1)));

    if count_S ~= 0
        prob = max(0.1, min(0.9, count_S ./ (sum(count_S))));
    else
        prob = 1/3 * ones(1, 3);
    end

    fitx(I == 1) = fitx_new(I == 1);
    xold(I == 1, :) = x(I == 1, :);
    x(I == 1, :) = ui(I == 1, :);

    % Update the CR and F memories
    if size(goodF, 1) == 1, goodF = goodF'; end
    if size(goodCR, 1) == 1, goodCR = goodCR'; end
    num_success_params = numel(goodCR);
    if num_success_params > 0
        weightsDE = diff(I == 1) ./ sum(diff(I == 1));
        archive_f(hist_pos) = (weightsDE * (goodF .^ 2)) ./ (weightsDE * goodF);
        if max(goodCR) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (weightsDE * (goodCR .^ 2)) / (weightsDE * goodCR);
        end
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size, hist_pos = 1; end
    else
        archive_Cr(hist_pos) = 0.5;
        archive_f(hist_pos)  = 0.5;
    end

    [fitx, ind] = sort(fitx);
    x = x(ind, :);
    xold = xold(ind, :);

    % Record the best after checking feasibility
    if fitx(1) < bestold && all(x(1, :) >= xmin) && all(x(1, :) <= xmax)
        bestold = fitx(1);
        bestx = x(1, :);
    end
end

% SQP local search on the incumbent (LS2)
function [x, f, current_eval, succ, ls_curve, run_best, run_bestx] = ...
        ls2_sqp(bestx, f, problem, current_eval, Max_FES, xmin, xmax, bsf_in)

    LS_FE = min(ceil(20.0000e-003 * Max_FES), (Max_FES - current_eval));
    x = bestx;
    succ = 0;
    ls_curve = [];
    run_best  = bsf_in;
    run_bestx = bestx;
    if LS_FE < 2
        return;
    end

    options = optimset('Display', 'off', 'algorithm', 'sqp', ...
                       'UseParallel', 'never', 'MaxFunEvals', LS_FE);

    cnt = 0;
    trace = zeros(1, LS_FE + 4 * numel(bestx) + 8);

    [Xsqp, FUN] = fmincon(@obj, bestx(1, :)', [], [], [], [], xmin(:), xmax(:), [], options);

    current_eval = current_eval + cnt;
    ls_curve = trace(1:cnt);

    if (f - FUN) > 0
        succ = 1;
        f = FUN;
        x(1, :) = Xsqp(:)';
    else
        succ = 0;
        x = bestx;
    end

    function v = obj(z)
        [vv, ~] = calculate_fitness(z(:), problem, 0);
        v = vv(1);
        cnt = cnt + 1;
        if v < run_best, run_best = v; run_bestx = z(:)'; end
        if cnt <= numel(trace)
            trace(cnt) = run_best;
        end
    end
end

% Randomised boundary handling
function x = han_boun(x, xmax, xmin, x2, PopSize)
    hb = randi(3);
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
            x(pos) = min(x_U(pos), max(x_L(pos), 2 * x_L(pos) - x2(pos)));
            pos = x > x_U;
            x(pos) = max(x_L(pos), min(x_U(pos), 2 * x_L(pos) - x2(pos)));
        case 3
            pos = x < x_L;
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
            pos = x > x_U;
            x(pos) = x_L(pos) + rand * (x_U(pos) - x_L(pos));
    end
end

% Random index generation
function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = (r1 == r0);
        if sum(pos) == 0, break; end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end

    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:1000
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; end
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end

    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0, break; end
        r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
end

% Archive update
function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0, return; end
    if size(pop, 1) ~= size(funvalue, 1), error('imode:archive', 'check it'); end

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

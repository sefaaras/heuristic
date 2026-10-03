% ----------------------------------------------------------------------- %
% Differential Evolution with Domain Transform (DTDE)
% CEC 2025 session paper (not ranked)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size = 18*D -> 4 (linear)    % Linear population size reduction
%   memory_size = 5                  % SHADE memory of F and CR, both start at 0.5
%   p_best_rate = 0.11, arc_rate = 1.4  % pbest share; archive size / pop_size
%   Thes = 0.1                       % Share of the spectrum cut at each end in the transform
%   F = CR = 0.5                     % Fixed for everyone once the transform is on
%   check every 10 generations       % Imp_S > Imp_I switches the transform on for good
%
% Algorithm Concept:
%   - L-SHADE: current-to-pbest/1 with archive, Cauchy F and normal CR from the
%     success memory (Lehmer means), linear population reduction
%   - Selective candidate (SCSS): two trials per parent; rand > rank/NP keeps the
%     one nearer the parent, otherwise the farther one
%   - Domain transform: per coordinate, sort parents + trials, FFT their fitness,
%     zero the highest frequencies, inverse FFT; mean over coordinates is the new fitness
%   - Selection, ranking, memory and reduction then use the smoothed fitness, which is
%     kept and transformed again next generation
%   - Trigger: total improvement by the better half (Imp_S) vs the worse half (Imp_I)
%     over 10 generations; once Imp_S > Imp_I the transform stays on
%
% Reference:
% Sheng Xin Zhang, Yi Nan Wen, Yu Hong Liu, Li Ming Zheng, Shao Yong Zheng,
% Differential Evolution With Domain Transform,
% IEEE Transactions on Evolutionary Computation 27 (5) (2023) 1440-1455.
% https://doi.org/10.1109/TEVC.2022.3220424
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from DTDE_main.m in DTDE.zip (zsxhomepage.github.io/code). The CEC 2025 paper
% (Hu, Luo, Zhang, doi 10.1109/CEC65147.2025.11042984) is closed access; its abstract and
% references do not settle DTDE vs DTDE_RSP.zip, so this is the TEVC paper's DTDE. Kept
% as released: the archive stores and the reduction ranks by the smoothed fitness, and
% the archive's fitness list is not cut with it (never read). The run reports and the
% history records the raw fitness. The budget-cut last generation skips the transform
% (it needs every trial) and selects on raw fitness. normrnd is replaced by
% mu + 0.1*randn (same draws). Removed: the stop at the known optimum. Same rng seed
% gives the release's run exactly up to the cut generation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dtde(problem)

    D = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;
    lu = [lb; ub];

    Thes = 0.1;
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    pop_size = 18 * D;
    max_pop_size = pop_size;
    min_pop_size = 4.0;

    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    FE = 0;
    bsf = inf;

    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, D) .* repmat(lu(2, :) - lu(1, :), pop_size, 1);
    pop = popold;
    bsf_solution = pop(1, :);
    fitness = inf(pop_size, 1);
    n_eval = min(pop_size, maxFE);
    [fv, FE] = calculate_fitness(pop(1:n_eval, :)', problem, FE);
    fitness(1:n_eval) = fv(:);
    fraw = fitness;                 % objective values of popold; fitness turns smoothed under DT
    track(pop(1:n_eval, :), fitness(1:n_eval));

    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = floor(arc_rate * pop_size);  % release: 1.4*NP, read only via 1:NP and <=, so floor is exact
    archive.pop = zeros(0, D);
    archive.funvalues = zeros(0, 1);
    IMP_S = 0;
    IMP_I = 0;
    CT = 0;
    DT_Trig = 0;

    while FE < maxFE
        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');
        for VN = 1:2
            mem_rand_index = ceil(memory_size * rand(pop_size, 1));
            mu_sf = memory_sf(mem_rand_index);
            mu_cr = memory_cr(mem_rand_index);

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
            sf = min(sf, 1);

            if DT_Trig == 1
                sf = 0.5 * ones(pop_size, 1);
                cr = 0.5 * ones(pop_size, 1);
            end

            popAll = [pop; archive.pop];
            [r1, r2] = gnR1R2(pop_size, size(popAll, 1), 1:pop_size);

            pNP = max(round(p_best_rate * pop_size), 2);
            randindex = ceil(rand(1, pop_size) .* pNP);
            randindex = max(1, randindex);
            pbest = pop(sorted_index(randindex), :);

            vi = pop + sf(:, ones(1, D)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
            vi = boundConstraint(vi, pop, lu);

            mask = rand(pop_size, D) > cr(:, ones(1, D));
            rows = (1:pop_size)';
            cols = floor(rand(pop_size, 1) * D) + 1;
            jrand = sub2ind([pop_size D], rows, cols);
            mask(jrand) = false;
            ui = vi;
            ui(mask) = pop(mask);

            dist = zeros(1, pop_size);
            for i = 1:pop_size
                dist(i) = norm(ui(i, :) - pop(i, :));
            end
            if VN == 1
                u1 = ui;
                Distance1 = dist;
                sf1 = sf;
                cr1 = cr;
            else
                u2 = ui;
                Distance2 = dist;
                sf2 = sf;
                cr2 = cr;
            end
        end

        ranki = zeros(1, pop_size);
        ranki(sorted_index) = 1:pop_size;
        for i = 1:pop_size
            if rand > ranki(i) / pop_size
                [~, Idx] = min([Distance1(i), Distance2(i)]);
            else
                [~, Idx] = max([Distance1(i), Distance2(i)]);
            end
            if Idx == 1
                ui(i, :) = u1(i, :);
                sf(i) = sf1(i);
                cr(i) = cr1(i);
            else
                ui(i, :) = u2(i, :);
                sf(i) = sf2(i);
                cr(i) = cr2(i);
            end
        end

        if DT_Trig == 0
            CT = CT + 1;
            if CT == 10
                CT = 0;
                if IMP_S > IMP_I
                    DT_Trig = 1;
                end
                IMP_S = 0;
                IMP_I = 0;
            end
        end

        n_eval = min(pop_size, maxFE - FE);
        fo = inf(pop_size, 1);
        [fv, FE] = calculate_fitness(ui(1:n_eval, :)', problem, FE);
        fo(1:n_eval) = fv(:);
        if DT_Trig == 1 && n_eval == pop_size
            f_trans = DTEO([pop; ui], [fitness; fo], Thes);
            fitness = f_trans(1:pop_size);
            children_fitness = f_trans(pop_size + 1:2 * pop_size);
        elseif DT_Trig == 1
            fitness = fraw;
            children_fitness = fo;
        else
            children_fitness = fo;
        end

        dif = abs(fitness - children_fitness);
        I = (fitness > children_fitness);
        goodCR = cr(I == 1);
        goodF = sf(I == 1);
        dif_val = dif(I == 1);

        if DT_Trig == 0
            SP = ranki <= ceil(pop_size / 2);
            IP = ranki > ceil(pop_size / 2);
            dif_val_S = dif((I == 1) & (SP' == 1));
            dif_val_I = dif((I == 1) & (IP' == 1));
            IMP_S = IMP_S + sum(dif_val_S);
            IMP_I = IMP_I + sum(dif_val_I);
        end

        archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));
        [fitness, I] = min([fitness, children_fitness], [], 2);

        popold = pop;
        popold(I == 2, :) = ui(I == 2, :);
        fraw(I == 2) = fo(I == 2);
        track(ui(1:n_eval, :), fo(1:n_eval));

        num_success_params = numel(goodCR);
        if num_success_params > 0
            sum_dif = sum(dif_val);
            dif_val = dif_val / sum_dif;
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
            end
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size
                memory_pos = 1;
            end
        end

        plan_pop_size = round((((min_pop_size - max_pop_size) / maxFE) * FE) + max_pop_size);
        if pop_size > plan_pop_size
            reduction_ind_num = pop_size - plan_pop_size;
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = pop_size - min_pop_size;
            end
            pop_size = pop_size - reduction_ind_num;
            for r = 1:reduction_ind_num
                [~, indBest] = sort(fitness, 'ascend');
                worst_ind = indBest(end);
                popold(worst_ind, :) = [];
                pop(worst_ind, :) = [];
                fitness(worst_ind, :) = [];
                fraw(worst_ind, :) = [];
            end
            archive.NP = round(arc_rate * pop_size);
            if size(archive.pop, 1) > archive.NP
                rndpos = randperm(size(archive.pop, 1));
                rndpos = rndpos(1:archive.NP);
                archive.pop = archive.pop(rndpos, :);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsf_solution;

    % Best-so-far per evaluation (raw fitness), then one history sample per evaluation
    function track(X, f)
        n = size(X, 1);
        for q = 1:n
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = X(q, :);
            end
            ec = FE - n + q;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, popold, fraw, population_history, fitness_history, history_index, maxFE);
            end
        end
    end
end

function f_trans = DTEO(x, f, Thes)
% Domain transform: low-pass the fitness series sorted along each coordinate, average over coordinates
    [Np, D] = size(x);
    f_trans = zeros(Np, D);
    for j = 1:D
        [~, x_address] = sort(x(:, j));
        fs = f(x_address);
        f_fftshift = fftshift(fft(fs, Np));
        if Thes > 0
            f_fftshift(1:ceil(Np * Thes)) = 0;
            f_fftshift(Np - ceil(Np * Thes) + 1:Np) = 0;
        end
        y = real(ifft(ifftshift(f_fftshift), Np));
        y(x_address) = y;
        f_trans(:, j) = y;
    end
    f_trans = mean(f_trans, 2);
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
% r1 from 1..NP1 with r1 ~= r0; r2 from 1..NP2 with r2 ~= r1, r0 (J. Zhang, JADE)
    NP0 = length(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000
            error('Can not genrate r1 in 1000 iterations');
        end
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else
            r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000
            error('Can not genrate r2 in 1000 iterations');
        end
    end
end

function archive = updateArchive(archive, pop, funvalue)
% Append, drop duplicate rows, then randomly trim to archive.NP (J. Zhang, JADE)
    if archive.NP == 0
        return;
    end
    if size(pop, 1) ~= size(funvalue, 1)
        error('check it');
    end
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
        rndpos = rndpos(1:archive.NP);
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

function vi = boundConstraint(vi, pop, lu)
% Out-of-box coordinate set to the midpoint of the parent and the violated bound
    [NP, ~] = size(pop);
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

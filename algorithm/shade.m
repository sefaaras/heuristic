% ----------------------------------------------------------------------- %
% Success-History Based Parameter Adaptation for Differential Evolution (SHADE)
% CEC 2013 entry; direct ancestor of L-SHADE and the CEC-winning DE lineage
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size    = 100            % Population size (constant -- no reduction)
%   memory_size = D              % Number of historical memory slots (SHADE 1.1)
%   p_best_rate = 0.1            % Greediness of the pbest term
%   arc_rate    = 2              % Archive size as a multiple of pop_size
%
% Algorithm Concept:
%   - Replaces JADE's single pair (muCR, muF), which forgets at a fixed rate,
%     with a CIRCULAR MEMORY of H (CR, F) pairs. Every individual draws its own
%     slot uniformly, so the whole history of what once worked stays available
%     and a single bad generation cannot wipe the adaptation out
%   - The memory entries are WEIGHTED Lehmer means: successes that improved the
%     fitness a lot count more than marginal ones
%   - A slot whose successful CRs are all zero is poisoned with the terminal
%     marker -1; individuals drawing it use CR = 0 forever after
%   - Mutation is JADE's DE/current-to-pbest/1 with an external archive, but p
%     is drawn per individual in the original paper; SHADE 1.1 fixes it to 0.1
%   - F is Cauchy-distributed and regenerated while non-positive; CR is normal
%
% Reference:
% Ryoji Tanabe, Alex Fukunaga,
% Success-History Based Parameter Adaptation for Differential Evolution,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 71-78.
% https://doi.org/10.1109/CEC.2013.6557555
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB/Octave release SHADE 1.1.1
% (shade11_matlab, from Ryoji Tanabe's site).
% VERSION NOTE. This is SHADE 1.1, the version the authors used as the baseline
% in the L-SHADE paper, not the CEC2013 submission: the memory size is H = D
% rather than pop_size, and p is fixed at 0.1 rather than drawn per individual
% from U(2/NP, 0.2). SHADE 1.1 is the authors' own later, better-tested code.
% Compared with lshade.m in this folder, SHADE differs only by having no linear
% population size reduction and by the constants above -- which is exactly the
% contribution of the L-SHADE paper.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = shade(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters
    p_best_rate = 0.1;
    arc_rate    = 2;
    memory_size = max(1, dim);
    pop_size    = 100;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    popold = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
    pop    = popold;

    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    bsf          = inf;
    bsf_solution = pop(1, :);
    for i = 1:pop_size
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    memory_sf  = 0.5 .* ones(memory_size, 1);
    memory_cr  = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;

    archive.NP        = arc_rate * pop_size;
    archive.pop       = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    % Main loop
    while FE < maxFE
        pop = popold;
        [~, sorted_index] = sort(fitness, 'ascend');

        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);

        % Crossover rate
        cr = normrnd(mu_cr, 0.1);
        cr(mu_cr == -1) = 0;                 % terminal memory value
        cr = min(max(cr, 0), 1);

        % Scaling factor
        sf  = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
        pos = find(sf <= 0);
        while ~isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end
        sf = min(sf, 1);

        r0     = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2_shade(pop_size, size(popAll, 1), r0);

        pNP       = max(round(p_best_rate * pop_size), 2);
        randindex = max(1, ceil(rand(1, pop_size) .* pNP));
        pbest     = pop(sorted_index(randindex), :);

        % DE/current-to-pbest/1 with archive
        vi = pop + sf(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
        vi = boundConstraint_shade(vi, pop, lu);

        % Binomial crossover
        mask  = rand(pop_size, dim) > cr(:, ones(1, dim));
        rows  = (1:pop_size)';
        cols  = floor(rand(pop_size, 1) * dim) + 1;
        jrand = sub2ind([pop_size dim], rows, cols);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        [children_fitness, FE] = calculate_fitness(ui', problem, FE);
        children_fitness = children_fitness(:);

        for i = 1:pop_size
            if children_fitness(i) < bsf
                bsf          = children_fitness(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - pop_size + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection and success bookkeeping
        dif = abs(fitness - children_fitness);

        I       = (fitness > children_fitness);
        goodCR  = cr(I == 1);
        goodF   = sf(I == 1);
        dif_val = dif(I == 1);

        archive = updateArchive_shade(archive, popold(I == 1, :), fitness(I == 1));

        [fitness, I] = min([fitness, children_fitness], [], 2);

        popold = pop;
        popold(I == 2, :) = ui(I == 2, :);

        if ~isempty(goodCR)
            dif_val = dif_val / sum(dif_val);

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
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function vi = boundConstraint_shade(vi, pop, lu)
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2_shade(NP1, NP2, r0)
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
end

function archive = updateArchive_shade(archive, pop, funvalue)
    if archive.NP == 0, return; end

    popAll    = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX]   = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll    = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end

    if size(popAll, 1) <= archive.NP
        archive.pop       = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:archive.NP);
        archive.pop       = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

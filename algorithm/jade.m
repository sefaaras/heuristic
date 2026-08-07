% ----------------------------------------------------------------------- %
% Adaptive Differential Evolution with Optional External Archive (JADE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 100                % Population size
%   p       = 0.05               % Greediness: top 100p% supply the pbest term
%   c       = 1/10               % Adaptation (forgetting) rate of muCR / muF
%   Afactor = 1                  % Archive size as a multiple of popsize
%
% Algorithm Concept:
%   - DE/current-to-pbest/1 mutation: the donor pulls the parent towards a
%     randomly chosen member of the top 100p% AND adds a difference whose
%     second term may come from the archive of recently defeated parents,
%     which restores the diversity that greedy mutation destroys
%   - Parameter adaptation without any user tuning: every individual draws
%     CR ~ N(muCR, 0.1) and F ~ Cauchy(muF, 0.1); at the end of the generation
%     muCR moves towards the ARITHMETIC mean and muF towards the LEHMER mean of
%     the values that produced a successful offspring, both with rate c
%   - The Lehmer mean is deliberate: it is biased towards large F, which
%     counteracts the drift towards small F that a plain mean would cause
%   - Violating components are set to the midpoint between the parent and the
%     violated bound, so the trial can never leave the box
%
% Reference:
% Jingqiao Zhang, Arthur C. Sanderson,
% JADE: Adaptive Differential Evolution With Optional External Archive,
% IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 945-958, 2009.
% https://doi.org/10.1109/TEVC.2009.2014613
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (JADE.m and its four helpers by
% Jingqiao Zhang, as redistributed with Y. Wang's CoDE package). One reference
% property is kept because it is harmless: the archive stores the DEFEATED
% PARENTS' positions with the OFFSPRING'S fitness values, since valParents has
% already been overwritten by the time updateArchive is called. That field is
% never read back, so it only mislabels bookkeeping.
% The reference's `if FES > 1 && ~isempty(goodCR)` guard, which relies on
% short-circuiting to skip the first generation, is written as an explicit
% first-generation test.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jade(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters
    popsize = 100;
    p       = 0.05;
    c       = 1 / 10;
    Afactor = 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    popold = repmat(lb, popsize, 1) + rand(popsize, dim) .* repmat(ub - lb, popsize, 1);

    [valParents, FE] = calculate_fitness(popold', problem, FE);
    valParents = valParents(:);

    bsf          = inf;
    bsf_solution = popold(1, :);
    for i = 1:popsize
        if valParents(i) < bsf
            bsf          = valParents(i);
            bsf_solution = popold(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, popold, valParents, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    CRm = 0.5;
    Fm  = 0.5;

    archive.NP        = Afactor * popsize;
    archive.pop       = zeros(0, dim);
    archive.funvalues = zeros(0, 1);

    goodCR = [];
    goodF  = [];
    first_generation = true;

    [~, indBest] = sort(valParents, 'ascend');

    % Main loop
    while FE < maxFE
        pop = popold;

        if ~first_generation && ~isempty(goodCR) && sum(goodF) > 0
            CRm = (1 - c) * CRm + c * mean(goodCR);
            Fm  = (1 - c) * Fm  + c * sum(goodF .^ 2) / sum(goodF);   % Lehmer mean
        end
        first_generation = false;

        [F, CR] = randFCR_jade(popsize, CRm, 0.1, Fm, 0.1);

        r0     = 1:popsize;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2_jade(popsize, size(popAll, 1), r0);

        pNP       = max(round(p * popsize), 2);
        randindex = max(1, ceil(rand(1, popsize) * pNP));
        pbest     = pop(indBest(randindex), :);

        % DE/current-to-pbest/1
        vi = pop + F(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
        vi = boundConstraint_jade(vi, pop, lu);

        % Binomial crossover
        mask  = rand(popsize, dim) > CR(:, ones(1, dim));
        rows  = (1:popsize)';
        cols  = floor(rand(popsize, 1) * dim) + 1;
        jrand = sub2ind([popsize dim], rows, cols);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        [valOffspring, FE] = calculate_fitness(ui', problem, FE);
        valOffspring = valOffspring(:);

        for i = 1:popsize
            if valOffspring(i) < bsf
                bsf          = valOffspring(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - popsize + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, valParents, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection
        [valParents, I] = min([valParents, valOffspring], [], 2);
        popold = pop;

        % Defeated parents enter the archive with the offspring fitness, as in the reference
        archive = updateArchive_jade(archive, popold(I == 2, :), valParents(I == 2));

        popold(I == 2, :) = ui(I == 2, :);

        goodCR = CR(I == 2);
        goodF  = F(I == 2);

        [~, indBest] = sort(valParents, 'ascend');
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function [F, CR] = randFCR_jade(NP, CRm, CRsigma, Fm, Fsigma)
% CR ~ N(CRm, CRsigma) truncated to [0,1]; F ~ Cauchy(Fm, Fsigma) capped at 1, redrawn if <= 0
    CR = CRm + CRsigma * randn(NP, 1);
    CR = min(1, max(0, CR));

    F = Fm + Fsigma * tan(pi * (rand(NP, 1) - 0.5));
    F = min(1, F);
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = Fm + Fsigma * tan(pi * (rand(length(pos), 1) - 0.5));
        F = min(1, F);
        pos = find(F <= 0);
    end
end

function vi = boundConstraint_jade(vi, pop, lu)
% Violating component moved to the parent/bound midpoint
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2] = gnR1R2_jade(NP1, NP2, r0)
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

function archive = updateArchive_jade(archive, pop, funvalue)
% Append, drop duplicates, then randomly thin down to archive.NP.
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

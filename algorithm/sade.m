% ----------------------------------------------------------------------- %
% Differential Evolution with Strategy Adaptation (SaDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP       = 50                % Population size
%   numst    = 4                 % Candidate mutation strategies in the pool
%   learngen = 50                % Learning period LP (generations)
%   F = N(0.5, 0.3)              % Scaling factor, drawn per individual
%   CR = N(CRm_k, 0.1)           % Per-strategy crossover rate, CRm_k adapted
%
% Algorithm Concept:
%   - Four mutation strategies compete inside one population: rand/1/bin,
%     current-to-best/2/bin, rand/2/bin, and current-to-rand/1 (no crossover,
%     rotation invariant)
%   - Strategies are assigned by STOCHASTIC UNIVERSAL SAMPLING over their recent
%     success probabilities, with a floor of 0.01 so none is ever starved
%   - Success counts live in a sliding window of LP generations, so the mix
%     tracks the phase of the search rather than the whole history
%   - CR is adapted per strategy from the MEDIAN of its successful values in
%     that window, which is what makes it robust to multimodal CR distributions
%   - F is deliberately NOT adapted but redrawn from N(0.5, 0.3) each
%     generation, supplying diversity a converged F would lose
%   - Violating components are reinitialised uniformly inside the box
%
% Reference:
% A. K. Qin, V. L. Huang, P. N. Suganthan,
% Differential Evolution Algorithm With Strategy Adaptation for Global
% Numerical Optimization,
% IEEE Transactions on Evolutionary Computation, vol. 13, no. 2, pp. 398-417, 2009.
% https://doi.org/10.1109/TEVC.2008.927706
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the MATLAB release distributed with Y. Wang's CoDE package, whose
% Readme credits the source to Dr. P. N. Suganthan.
% ONE DEFECT CORRECTED: in that release DE_gbest is assigned once, after the
% initial population is evaluated, and never again, so strategy 2
% (current-to-best/2/bin) pulls towards the best of the RANDOM INITIAL
% POPULATION for the whole run, disabling a quarter of the strategy pool.
% The paper's Eq. (4) says "the best individual vector in the current
% generation", so this is a code slip and the current-generation best is used.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sade(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    NP       = 50;
    numst    = 4;
    learngen = 50;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    pop = repmat(lb, NP, 1) + rand(NP, D) .* repmat(span, NP, 1);

    [val, FE] = calculate_fitness(pop', problem, FE);
    val = val(:);

    bsf          = inf;
    bsf_solution = pop(1, :);
    for i = 1:NP
        if val(i) < bsf
            bsf          = val(i);
            bsf_solution = pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pop, val, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Adaptation state
    aaaa = cell(1, numst);      % successful CR values per strategy: [CR, gen]
    ns   = zeros(0, numst);     % success counts, sliding window of LP rows
    nf   = zeros(0, numst);     % failure counts
    pfit = ones(1, numst);      % strategy selection probabilities
    ccm  = 0.5 * ones(1, numst);% per-strategy CR mean

    rot  = 0:(NP - 1);
    iter = 1;

    % Main loop
    while FE < maxFE
        popold = pop;

        % Shuffled donor index arrays (rotation scheme of the reference)
        ind = randperm(4);
        a1  = randperm(NP);
        a2  = a1(rem(rot + ind(1), NP) + 1);
        a3  = a2(rem(rot + ind(2), NP) + 1);
        a4  = a3(rem(rot + ind(3), NP) + 1);
        a5  = a4(rem(rot + ind(4), NP) + 1);

        pm1 = popold(a1, :);
        pm2 = popold(a2, :);
        pm3 = popold(a3, :);
        pm4 = popold(a4, :);
        pm5 = popold(a5, :);

        % Best of the CURRENT generation -- see the header note
        [~, ibest] = min(val);
        bm = repmat(popold(ibest, :), NP, 1);

        % Refresh the per-strategy CR means once the window is full
        if iter >= learngen
            for i = 1:numst
                if ~isempty(aaaa{i})
                    ccm(i) = median(aaaa{i}(:, 1));
                    aaaa{i}(aaaa{i}(:, 2) == aaaa{i}(1, 2), :) = [];  % drop the oldest generation
                else
                    ccm(i) = rand;
                end
            end
        end

        % Draw CR per individual for every strategy, truncated to [0,1]
        ccm_mat = repmat(ccm, NP, 1);
        cc      = normrnd(ccm_mat, 0.1);
        bad     = (cc > 1) | (cc < 0);
        while any(bad(:))
            redraw  = normrnd(ccm_mat, 0.1);
            cc(bad) = redraw(bad);
            bad     = (cc > 1) | (cc < 0);
        end

        % Strategy assignment by stochastic universal sampling
        rr       = rand;
        spacing  = 1 / NP;
        randnums = sort(mod(rr:spacing:(1 + rr - 0.5 * spacing), 1));

        normfit  = pfit / sum(pfit);
        partsum  = 0;
        count    = zeros(1, numst + 1);
        stpool   = [];
        for i = 1:numst
            partsum      = partsum + normfit(i);
            count(i + 1) = sum(randnums < partsum);
            stpool       = [stpool; ones(count(i + 1) - count(i), 1) * i];
        end
        % Rounding can leave the sampling one slot short; the reference would error on the randperm
        if numel(stpool) < NP
            stpool = [stpool; ones(NP - numel(stpool), 1) * numst];
        end
        stpool = stpool(1:NP);
        stpool = stpool(randperm(NP));

        % Crossover mask uses the CR of the assigned strategy
        cr_used = cc(sub2ind([NP numst], (1:NP)', stpool));
        mui = rand(NP, D) < cr_used(:, ones(1, D));
        dd  = ceil(D * rand(NP, 1));
        mui(sub2ind([NP D], (1:NP)', dd)) = true;
        mpo = ~mui;

        % Trial generation
        Fv = normrnd(0.5, 0.3, NP, 1);
        F  = Fv(:, ones(1, D));
        ui = popold;

        s1 = (stpool == 1);
        if any(s1)
            v = pm3 + F .* (pm1 - pm2);
            ui(s1, :) = popold(s1, :) .* mpo(s1, :) + v(s1, :) .* mui(s1, :);
        end
        s2 = (stpool == 2);
        if any(s2)
            v = popold + F .* (bm - popold) + F .* (pm1 - pm2 + pm3 - pm4);
            ui(s2, :) = popold(s2, :) .* mpo(s2, :) + v(s2, :) .* mui(s2, :);
        end
        s3 = (stpool == 3);
        if any(s3)
            v = pm5 + F .* (pm1 - pm2 + pm3 - pm4);
            ui(s3, :) = popold(s3, :) .* mpo(s3, :) + v(s3, :) .* mui(s3, :);
        end
        s4 = (stpool == 4);
        if any(s4)
            % current-to-rand/1: no crossover, so the trial is the mutant itself
            v = popold + rand .* (pm5 - popold) + F .* (pm1 - pm2);
            ui(s4, :) = v(s4, :);
        end

        % Violating components are reinitialised uniformly
        L   = repmat(lb, NP, 1);
        S   = repmat(span, NP, 1);
        out = (ui < L) | (ui > repmat(ub, NP, 1));
        if any(out(:))
            ui(out) = L(out) + S(out) .* rand(sum(out(:)), 1);
        end

        [tempval, FE] = calculate_fitness(ui', problem, FE);
        tempval = tempval(:);

        for i = 1:NP
            if tempval(i) < bsf
                bsf          = tempval(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - NP + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, popold, val, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection (ties count as a success, as in the reference)
        succ = (tempval <= val);
        pop(succ, :) = ui(succ, :);
        val(succ)    = tempval(succ);

        % Success / failure bookkeeping
        ns_row = zeros(1, numst);
        nf_row = zeros(1, numst);
        for j = 1:numst
            inj       = (stpool == j);
            ns_row(j) = sum(inj & succ);
            nf_row(j) = sum(inj & ~succ);
            hit       = find(inj & succ);
            if ~isempty(hit)
                aaaa{j} = [aaaa{j}; cc(hit, j), iter * ones(numel(hit), 1)];
            end
        end
        ns = [ns; ns_row];
        nf = [nf; nf_row];

        if iter >= learngen
            for i = 1:numst
                tot = sum(ns(:, i)) + sum(nf(:, i));
                if tot == 0
                    pfit(i) = 0.01;
                else
                    pfit(i) = sum(ns(:, i)) / tot + 0.01;
                end
            end
            ns(1, :) = [];
            nf(1, :) = [];
        end

        iter = iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

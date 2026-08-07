% ----------------------------------------------------------------------- %
% Differential Evolution with an Ensemble of Parameters and Mutation Strategies (EPSDE)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP      = 50                 % Population size
%   F  pool = 0.4 : 0.1 : 0.9    % Six discrete scaling factors
%   CR pool = 0.1 : 0.1 : 0.9    % Nine discrete crossover rates
%   n_strategies = 3             % best/2/bin, rand/1/bin, current-to-rand/1
%   F jitter sigma = 0.001       % Per-dimension noise added to the drawn F
%
% Algorithm Concept:
%   - Every individual carries a complete TUPLE (strategy, CR, F) drawn from
%     three small discrete pools, so the same tuple recurs often enough for its
%     track record to mean something
%   - A tuple producing a successful offspring is pushed onto a pool of WINNING
%     TUPLES; later an individual either re-draws at random or, with
%     probability RATE, copies one from that pool
%   - RATE is the FAILURE rate averaged over the last eleven generations, so the
%     harder the search gets the more the population falls back on what worked
%   - A tuple taken from the winning pool and successful again is removed from
%     its old slot and re-pushed at the front, which ages the pool
%   - The drawn F is perturbed per dimension by N(0, 0.001), so no two
%     dimensions get exactly the same step
%
% Reference:
% R. Mallipeddi, P. N. Suganthan, Q. K. Pan, M. F. Tasgetiren,
% Differential evolution algorithm with ensemble of parameters and mutation
% strategies, Applied Soft Computing, vol. 11, no. 2, pp. 1679-1696, 2011.
% https://doi.org/10.1016/j.asoc.2010.04.024
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the MATLAB release distributed with Y. Wang's CoDE package, whose
% Readme credits the source to Dr. R. Mallipeddi. Two reference properties are
% kept: the index recorded for eviction from the winning pool is a separate
% random draw from the one that selected the tuple, and the eviction is applied
% after this generation's winners were prepended, so the indices are stale.
% Both only shuffle which winning tuple gets evicted.
% ONE DEFECT CORRECTED: the bound repair tests `if (trial < Lbound)` on a 1-by-D
% array, which is true only if EVERY component violates, so the repair never
% fires and EPSDE runs unbounded. It uses `any(...)` here, the written intent.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = epsde(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    NP = 50;
    FF = (0.4:0.1:0.9)';
    CR = (0.1:0.1:0.9)';
    nStrat = 3;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    pop = repmat(lb, NP, 1) + rand(NP, D) .* repmat(span, NP, 1);

    [val, FE] = calculate_fitness(pop', problem, FE);
    val = val(:);

    bsf     = inf;
    bestmem = pop(1, :);
    for k = 1:NP
        if val(k) < bsf
            bsf     = val(k);
            bestmem = pop(k, :);
        end
        if k <= maxFE
            curve(k) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                k, pop, val, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    Para  = zeros(NP, 3);      % [strategy, CR, F] per individual
    PPara = zeros(0, 3);       % pool of tuples that have produced a success
    RR    = zeros(NP, 1);
    rate  = [];
    RATE  = 0;
    iter  = 1;
    rows  = (1:NP)';

    % Main loop
    while FE < maxFE
        % Tuple assignment
        if iter == 1
            Para = [randi(nStrat, NP, 1), CR(randi(numel(CR), NP, 1)), FF(randi(numel(FF), NP, 1))];
            RR(:) = 0;
        else
            fresh = [randi(nStrat, NP, 1), CR(randi(numel(CR), NP, 1)), FF(randi(numel(FF), NP, 1))];
            RR(:) = 0;
            Para  = fresh;
            if ~isempty(PPara)
                reuse = rand(NP, 1) <= RATE;
                if any(reuse)
                    nr = sum(reuse);
                    % Two independent draws, as in the reference: recorded and used index need not agree
                    RR(reuse)     = randi(size(PPara, 1), nr, 1);
                    Para(reuse,:) = PPara(randi(size(PPara, 1), nr, 1), :);
                end
            end
        end

        popold = pop;

        % Per-dimension scaling factors
        Fmat = normrnd(Para(:, 3) * ones(1, D), 0.001);

        % Crossover mask; jrand is forced only when the mask came out empty
        mui   = rand(NP, D) < Para(:, 2 * ones(1, D));
        empty = ~any(mui, 2);
        if any(empty)
            ei = find(empty);
            mui(sub2ind([NP D], ei, ceil(rand(numel(ei), 1) * D))) = true;
        end

        % Four donor indices per individual, drawn from the whole population
        [~, ord] = sort(rand(NP, NP), 2);
        d1 = ord(:, 1);  d2 = ord(:, 2);  d3 = ord(:, 3);  d4 = ord(:, 4);

        ui = popold;

        s1 = (Para(:, 1) == 1);   % DE/best/2/bin
        if any(s1)
            v = repmat(bestmem, NP, 1) + ...
                (popold(d1, :) - popold(d2, :) + popold(d3, :) - popold(d4, :)) .* Fmat;
            ui(s1, :) = popold(s1, :) .* ~mui(s1, :) + v(s1, :) .* mui(s1, :);
        end

        s2 = (Para(:, 1) == 2);   % DE/rand/1/bin
        if any(s2)
            v = popold(d1, :) + Fmat .* (popold(d2, :) - popold(d3, :));
            ui(s2, :) = popold(s2, :) .* ~mui(s2, :) + v(s2, :) .* mui(s2, :);
        end

        s3 = (Para(:, 1) == 3);   % DE/current-to-rand/1, no crossover
        if any(s3)
            v = popold + rand(NP, D) .* (popold(d1, :) - popold) + ...
                Fmat .* (popold(d2, :) - popold(d3, :));
            ui(s3, :) = v(s3, :);
        end

        % Bound repair: a violating trial is reinitialised as a whole
        bad = any(ui < repmat(lb, NP, 1) | ui > repmat(ub, NP, 1), 2);
        if any(bad)
            nb = sum(bad);
            ui(bad, :) = repmat(lb, nb, 1) + rand(nb, D) .* repmat(span, nb, 1);
        end

        [tempval, FE] = calculate_fitness(ui', problem, FE);
        tempval = tempval(:);

        for k = 1:NP
            if tempval(k) < bsf
                bsf     = tempval(k);
                bestmem = ui(k, :);
            end
            ec = FE - NP + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, popold, val, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection
        succ = (tempval < val);
        pop(succ, :) = ui(succ, :);
        val(succ)    = tempval(succ);

        % Equivalent to the reference: bsf <= min(val), so beating bsf implies beating the parent

        count = sum(~succ);

        % Winning-tuple pool: prepend this generation's winners
        si = find(succ);
        if ~isempty(si)
            PPara = [flipud(Para(si, :)); PPara];
        end
        RRR = RR(si);
        RRR = RRR(RRR ~= 0);
        if ~isempty(RRR)
            RRR = RRR(RRR <= size(PPara, 1));
            PPara(RRR, :) = [];
        end

        % RATE: mean failure rate over the last eleven generations
        rate(iter, 1) = count / NP;
        if iter > 10
            RATE = mean(rate((iter - 10):iter));
        else
            RATE = mean(rate);
        end

        iter = iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bestmem;
end

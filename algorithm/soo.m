% ----------------------------------------------------------------------- %
% Simultaneous Optimistic Optimization (SOO)
% CEC 2014 competition -- 13th of 17 by recomputed mean rank (no official table)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   K = 3                           % Children per split; the middle one keeps the parent's value
%   h_max = 10*sqrt(ln(maxFe)^3)    % Deepest level that may still be split
%   split dimension = mod(h+1, D)+1 % Coordinate cut at depth h (cycles, root cuts the 2nd)
%
% Algorithm Concept:
%   - The box is a tree of cells, each represented by the value at its centre; the
%     root cell is the whole box
%   - A sweep visits the depths from 0 to h_max-1 and splits the best leaf of each
%     depth only if it is strictly better than every leaf split earlier in the sweep
%   - A split trisects the cell along the next coordinate: the two outer centres are
%     evaluated, the middle child inherits the parent's centre and value
%   - Cells created in a sweep wait for the next one, so coarse (exploring) and fine
%     (exploiting) cells are expanded together without a smoothness constant
%   - Deterministic: every seed produces the same run
%
% Reference:
% Philippe Preux, Remi Munos, Michal Valko,
% Bandits attack function optimization,
% 2014 IEEE Congress on Evolutionary Computation (CEC), 2014, 2245-2252.
% https://doi.org/10.1109/CEC.2014.6900558
% SOO itself: R. Munos, Advances in Neural Information Processing Systems 24 (2011) 783-791.
% ----------------------------------------------------------------------- %
% Implementation Note:
% The authors' archive (soo.tar.gz, soo/main.c) holds only a 1-D demonstration; the
% multi-D CEC 2014 code is unreleased. main.c is generalised with the paper's
% settings, and three details the paper leaves open were fixed by reproducing the
% official CEC 2014 result files (deterministic, all 51 runs identical): the root
% cuts the second coordinate, the test is strict (main.c; the paper prints <=) and
% h_max uses ln. All 30 final D = 10 errors then equal the files to their 6 digits.
% History population: the best leaf of every depth (a sweep's candidate set).
% Memory: cells are kept in [0,1]^D, split cells store their centre and leaves only
% (parent, side, value), about 4*D+20 bytes per evaluation (1 GB at D = 20, 1e7 FEs).
% A sweep that splits nothing (all values Inf; main.c would stop) splits the shallowest leaf.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = soo(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    h_max = 10 * sqrt(log(max(maxFE, 2)) ^ 3);
    n_lev = ceil(h_max);
    hs    = 0:n_lev;
    jdim  = mod(hs + 1, D) + 1;
    % Offset of an outer child's centre along jdim(h), in units of the box
    offs  = 3 .^ -(floor(hs / D) + 1);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf = inf;
    bsf_solution = (lb + ub) / 2;

    % Split cells: unit-box centres; leaves per depth: value (NaN = split), parent, side
    Xint  = zeros(D, max(1, ceil((maxFE - 1) / 2) + 1));
    n_int = 0;
    Fd    = cell(1, n_lev + 1);
    Pd    = cell(1, n_lev + 1);
    Cd    = cell(1, n_lev + 1);
    cnt   = zeros(1, n_lev + 1);
    dead  = zeros(1, n_lev + 1);
    bestF = nan(1, n_lev + 1);
    bestI = zeros(1, n_lev + 1);
    for lv = 1:n_lev + 1
        Fd{lv} = nan(1, 16);
        Pd{lv} = zeros(1, 16, 'int32');
        Cd{lv} = zeros(1, 16, 'int8');
    end

    % History population: one row per non-empty depth, its best leaf
    Pop    = zeros(0, D);
    PopF   = zeros(0, 1);
    rowOf  = zeros(1, n_lev + 1);

    u0 = 0.5 * ones(D, 1);
    f0 = evaluate(lb + span .* u0');
    add_leaves(0, int32(0), int8(0), clean(f0));
    record(FE);

    deepest = 0;
    while FE < maxFE
        compact_levels(deepest);
        snapF = bestF;
        snapI = bestI;
        vmin = inf;
        did = false;
        top = min(deepest, n_lev - 1);
        for h = 0:top
            if FE >= maxFE
                break;
            end
            if h >= h_max || ~(snapF(h + 1) < vmin)
                continue;
            end
            vmin = snapF(h + 1);
            split_leaf(h, snapI(h + 1));
            did = true;
            deepest = max(deepest, min(h + 1, n_lev));
        end
        % Only when every candidate is Inf: keep spending the budget on the shallowest leaf
        if ~did && FE < maxFE
            lv = find(bestI(1:n_lev) > 0, 1);
            if isempty(lv)
                break;
            end
            split_leaf(lv - 1, bestI(lv));
            deepest = max(deepest, min(lv, n_lev));
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the rows of X (within the budget), tracking the best per evaluation
    function f = evaluate(X)
        [fr, FE_new] = calculate_fitness(X', problem, FE);
        f = fr(:);
        for q = 1:numel(f)
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = X(q, :);
            end
            curve(FE + q) = bsf;
        end
        FE = FE_new;
    end

    function record(ec)
        [population_history, fitness_history, history_index] = record_history( ...
            ec, Pop, PopF, population_history, fitness_history, history_index, maxFE);
    end

    % Unit-box centre of leaf k at depth h, rebuilt from its parent's centre and side
    function u = leaf_centre(h, k)
        p = Pd{h + 1}(k);
        if p == 0
            u = u0;
        else
            u = Xint(:, p);
            j = jdim(h);
            u(j) = u(j) + double(Cd{h + 1}(k)) * offs(h);
        end
    end

    % Trisects leaf k of depth h: left and right centres evaluated, the middle inherits
    function split_leaf(h, k)
        u = leaf_centre(h, k);
        fp = Fd{h + 1}(k);
        Fd{h + 1}(k) = NaN;
        dead(h + 1) = dead(h + 1) + 1;
        rescan(h);

        n_int = n_int + 1;
        Xint(:, n_int) = u;
        j = jdim(h + 1);
        UL = u;
        UL(j) = UL(j) - offs(h + 1);
        UR = u;
        UR(j) = UR(j) + offs(h + 1);
        n_ev = min(2, maxFE - FE);
        Xc = lb + span .* [UL'; UR'];
        fe0 = FE;
        fv = clean(evaluate(Xc(1:n_ev, :)));
        % Children deeper than h_max can never be split, so they are not stored
        if n_ev == 2 && h + 1 < h_max
            add_leaves(h + 1, int32([n_int, n_int, n_int]), int8([-1, 1, 0]), [fv(1), fv(2), fp]);
        end
        for q = 1:n_ev
            record(fe0 + q);
        end
    end

    % Appends leaves at depth h (C order: left, right, middle) and updates its best
    function add_leaves(h, par, side, fv)
        lv = h + 1;
        n = numel(fv);
        if cnt(lv) + n > numel(Fd{lv})
            grow = max(numel(Fd{lv}), n);
            Fd{lv} = [Fd{lv}, nan(1, grow)];
            Pd{lv} = [Pd{lv}, zeros(1, grow, 'int32')];
            Cd{lv} = [Cd{lv}, zeros(1, grow, 'int8')];
        end
        idx = cnt(lv) + (1:n);
        Fd{lv}(idx) = fv;
        Pd{lv}(idx) = par;
        Cd{lv}(idx) = side;
        cnt(lv) = cnt(lv) + n;
        [fb, ib] = min(fv);
        % Strict: on a tie the earlier leaf stays the depth's best
        if isnan(bestF(lv)) || fb < bestF(lv)
            bestF(lv) = fb;
            bestI(lv) = idx(ib);
            set_row(h);
        end
    end

    % Best live leaf of depth h after a removal (min skips the NaN of split leaves)
    function rescan(h)
        lv = h + 1;
        if dead(lv) >= cnt(lv)
            bestF(lv) = NaN;
            bestI(lv) = 0;
        else
            [bestF(lv), bestI(lv)] = min(Fd{lv}(1:cnt(lv)));
        end
        set_row(h);
    end

    % Keeps the population row of depth h in step with its best leaf
    function set_row(h)
        lv = h + 1;
        r = rowOf(lv);
        if bestI(lv) == 0
            if r > 0
                Pop(r, :) = [];
                PopF(r) = [];
                rowOf(lv) = 0;
                later = rowOf > r;
                rowOf(later) = rowOf(later) - 1;
            end
            return;
        end
        if r == 0
            r = numel(PopF) + 1;
            rowOf(lv) = r;
        end
        Pop(r, :) = lb + span .* leaf_centre(h, bestI(lv))';
        PopF(r, 1) = bestF(lv);
    end

    % Drops split leaves from mostly-dead levels; only between sweeps, as snapI holds positions
    function compact_levels(top)
        for lv = 1:min(top + 1, n_lev + 1)
            if dead(lv) > 64 && 2 * dead(lv) > cnt(lv)
                keep = ~isnan(Fd{lv}(1:cnt(lv)));
                nk = nnz(keep);
                Fd{lv}(1:nk) = Fd{lv}(keep);
                Pd{lv}(1:nk) = Pd{lv}(keep);
                Cd{lv}(1:nk) = Cd{lv}(keep);
                Fd{lv}(nk + 1:cnt(lv)) = NaN;
                cnt(lv) = nk;
                dead(lv) = 0;
                if nk > 0
                    [bestF(lv), bestI(lv)] = min(Fd{lv}(1:nk));
                else
                    bestF(lv) = NaN;
                    bestI(lv) = 0;
                end
            end
        end
    end
end

% A NaN value never wins a comparison in main.c; +Inf keeps that and frees NaN as the split mark
function f = clean(f)
    f(isnan(f)) = inf;
    f = f(:)';
end

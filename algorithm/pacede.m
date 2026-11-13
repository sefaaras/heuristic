% ----------------------------------------------------------------------- %
% Differential Evolution with Phase-Adaptive Control of Evaluations (PACE-DE)
% Stored as pacede; the acronym PACE-DE is not a valid MATLAB name
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop_size      = 18*D -> 4 (linear per DE run)       % population of one DE run
%   memory_size   = 5                                   % success history of step sizes and move shares
%   probing       = 2*(1 + D + D(D-1)/2) FE, if <= 0.05*maxFE   % interaction probing; blocks of <= 14
%   second_run    = 0.4 of the FE left (5 <= D <= 10)   % second exploration after the first DE run
%   ris_turn      = 0.1*maxFE after 0.05*maxFE without a 1e-9 relative gain (D >= 5)
%   fine_share    = 0.02*maxFE (D <= 20), finish_share = 0.03*maxFE   % held back from the restarts
%   run_end       = spread < 1e-8, 10 flat generations, or 0.10*maxFE plateau at spread < 1e-2
%   arm_rank      = (hits+1)/(runs+2) per FE spent      % choice among the restart arms
%
% Algorithm Concept:
%   - Interaction probing (global differential grouping) twice; when both agree on >= 2 blocks,
%     each block is optimised alone in two rounds, then the whole problem
%   - DE engine (DE-2LS): steps toward proven points with a success-rate step size, three-point
%     moves late in a run, linear population reduction, out-of-box trials repaired onto the edge
%   - A DE run ends on depletion (spread or flat generations) or on a plateau of a tight population
%   - Second exploration (5 <= D <= 10): a fresh DE run that redesigns one coordinate at first;
%     IPOP-CMA-ES restarts instead once the fitness reference is seen to move
%   - Restart portfolio: seeded DE, unseeded DE, subspace DE, CMA-ES and fmincon, chosen by
%     success per FE; it alternates with re-sampled inheritance search (RIS) when idle (D >= 5)
%   - Fine-scale phase (D <= 20) and finishing phase (edge snap, coordinate line search with
%     Brent's method, small-step CMA-ES, (1+1) steps) on shares held back from the restarts
%   - Under fitness drift the best point is re-evaluated on a doubling schedule, and at the end
%
% Reference:
% Sefa Aras,
% Differential Evolution with Phase-Adaptive Control of Evaluations,
% manuscript in preparation (2026).
% Components: the DE engine is that of DE-2LS (Dikshit Chauhan, arXiv:2606.27762, 2026,
%   https://doi.org/10.48550/arXiv.2606.27762), built on RDEx-SOP (Sichen Tao et al.,
%   arXiv:2603.27089, 2026); CMA-ES (N. Hansen, A. Ostermeier, Evolutionary Computation
%   9(2):159-195, 2001, https://doi.org/10.1162/106365601750190398) with the small-regime step
%   of BIPOP-CMA-ES (N. Hansen, GECCO 2009, https://doi.org/10.1145/1570256.1570333); global
%   differential grouping (Y. Mei, M. N. Omidvar, X. Li, X. Yao, ACM TOMS 42(2):13, 2016,
%   https://doi.org/10.1145/2791291), blocks optimised as in cooperative co-evolution (CCLSHADE:
%   M. El-Abd, IEEE CEC 2016; DECC: Z. Yang, K. Tang, X. Yao, Information Sciences
%   178(15):2985-2999, 2008); RIS of CMAES-RIS (F. Caraffini, G. Iacca, F. Neri, L. Picinali,
%   E. Mininno, IEEE CEC 2013, 1123-1130, https://doi.org/10.1109/CEC.2013.6557692; F. Caraffini,
%   F. Neri, B. N. Passow, G. Iacca, Soft Computing 17(12):2235-2256, 2013,
%   https://doi.org/10.1007/s00500-013-1106-7); IPOP-CMA-ES (A. Auger, N. Hansen, IEEE CEC 2005,
%   https://doi.org/10.1109/CEC.2005.1554902); the second exploration follows the binomial
%   schedule of NL-SHADE-RSP (V. Stanovov, S. Akhmedova, E. Semenkin, IEEE CEC 2021); the line
%   search follows MTS (L.-Y. Tseng, C. Chen, IEEE CEC 2008) with Brent's method (R. P. Brent,
%   Algorithms for Minimization without Derivatives, 1973)
% ----------------------------------------------------------------------- %
% Implementation Note:
% Final version (v9) of the development code, then named OPPO; bit-identical to it: only the function
% name and the comments differ. Identifiers keep the development terms: well = candidate solution, rig =
% population member, licence = evaluation budget, campaign = DE run, reservoir = variable block, survey =
% interaction probing, wildcat = second exploration, prospecting = restart portfolio (near-field, frontier,
% sidetrack, stimulation = CMA-ES, completion = fmincon), fine field = fine-scale phase, tail = finishing
% phase, edge wells = boundary repair, pressure checks = re-evaluation under fitness drift. The drift
% checks answer the harness's Eq. 6 running-max penalty on CEC2020RW (a moving fitness reference). Only
% the scalar fitness is used; re-tests and fmincon evaluations are counted and recorded.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = pacede(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    survey_cap   = 0.05;        % the structure survey only when it costs at most this share of the licence
    survey_eps   = 1e-12;       % interaction threshold, relative to the largest |yield| of the four wells
    effect_min   = 0.5;         % blocks only when at least this share of the coordinates moves the yield
    block_max    = 14;          % ... and when no group of interacting variables is larger
    block_rounds = [0.5 0.3];   % shares of the licence left after the survey, one block round each
    full_min     = 0.01;        % the continuation in full dimension only on at least this share of maxFE

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    n_pairs   = dim * (dim - 1) / 2;
    survey_fe = 2 * (1 + dim + n_pairs);
    if dim < 2 || survey_fe > survey_cap * maxFE
        [best_fitness, best_solution, curve, population_history, fitness_history] = engine(problem, [], [], ...
            maxFE, [], 0, maxFE, population_history, fitness_history, history_index);
        return;
    end

    % Structure survey; a second one at an independent base point when blocks are found, kept only if both agree
    n_surv = 0;   % surveys that read the interactions, and how many of them found a separable function
    n_sep  = 0;
    [blocks, X, fv] = survey();
    if numel(blocks) >= 2
        [blocks2, X2, fv2] = survey();
        if ~same_partition(blocks, blocks2, dim)
            blocks = {};
        end
        X  = [X; X2];
        fv = [fv fv2];
    end
    FE = numel(fv);
    [population_history, fitness_history, history_index] = record_history(FE, X, fv, ...
        population_history, fitness_history, history_index, maxFE);
    curve = zeros(1, maxFE);
    curve(1:FE) = cummin(fv);
    [f_ctx, ib] = min(fv);
    ctx = X(ib, :);

    if numel(blocks) < 2
        % No structure: the engine on the whole problem (a separable function keeps its tail for the line pass)
        [best_fitness, best_solution, cv, population_history, fitness_history] = engine(problem, [], [], ...
            maxFE - FE, [], FE, maxFE, population_history, fitness_history, history_index, ...
            n_surv > 0 && n_sep == n_surv);
        n_hold = max([find(cv < f_ctx, 1) - 1, 0]);   % the survey best holds the curve until the engine beats it
        curve(FE + 1:end) = cv;
        curve(FE + (1:n_hold)) = f_ctx;
        return;
    end

    % Block rounds: one licence per block, others held at ctx; kept only if better, both re-evaluated in one call
    left = maxFE - FE;
    for share = block_rounds
        budget = floor(share * left);
        for g = 1:numel(blocks)
            G = blocks{g};
            B = min(floor(budget * numel(G) / dim), maxFE - FE - 2);
            if B < 50 * numel(G)
                continue;
            end
            [~, bx, cv, population_history, fitness_history, history_index] = engine(problem, G, ctx, B, [], ...
                FE, maxFE, population_history, fitness_history, history_index);
            curve(FE + (1:B)) = min(f_ctx, cummin(cv(1:B)));
            FE = FE + B;
            cand = ctx;
            cand(G) = bx;
            ft = calculate_fitness([ctx; cand]', problem, 0);
            ft = ft(:)';
            [population_history, fitness_history, history_index] = record_history(FE + 2, [ctx; cand], ft, ...
                population_history, fitness_history, history_index, maxFE);
            if ft(2) < ft(1)
                ctx = cand;
            end
            f_ctx = min(ft);
            curve(FE + (1:2)) = f_ctx;
            FE = FE + 2;
        end
    end
    % Continuation in full dimension from the context vector (couplings the survey did not see)
    if maxFE - FE >= max(1, full_min * maxFE)
        seed = struct('x', ctx, 'f', f_ctx, 'range', max(fv) - min(fv));
        [f_ctx, bx, cv, population_history, fitness_history] = engine(problem, [], [], maxFE - FE, seed, ...
            FE, maxFE, population_history, fitness_history, history_index);
        ctx = bx(:)';
        curve(FE + 1:end) = cv;
        FE = maxFE;
    end
    curve(max(FE, 1):end) = min(curve(max(FE, 1)), f_ctx);
    best_fitness  = f_ctx;
    best_solution = ctx;

    % One survey: singles and pairs moved; a nonzero mixed second difference is an interaction; a moving reference ends it
    function [blocks, X, fv] = survey()
        x0 = lb + rand(1, dim) .* (ub - lb);
        y  = lb + rand(1, dim) .* (ub - lb);
        [P1, P2] = find(triu(true(dim), 1));
        X = repmat(x0, 1 + dim + n_pairs, 1);
        X(sub2ind(size(X), 1 + (1:dim)', (1:dim)')) = y;
        r = 1 + dim + (1:n_pairs)';
        X(sub2ind(size(X), r, P1)) = y(P1);
        X(sub2ind(size(X), r, P2)) = y(P2);
        blocks = {};
        f0a = calculate_fitness(x0', problem, 0);
        fs  = calculate_fitness(X(2:1 + dim, :)', problem, 0);
        f0b = calculate_fitness(x0', problem, 0);
        if ~isequaln(f0a, f0b)
            X  = [X(1:1 + dim, :); x0];
            fv = [f0a fs(:)' f0b];
            return;
        end
        fq = calculate_fitness(X(r, :)', problem, 0);
        X  = [X; x0];
        fv = [f0a fs(:)' fq(:)' f0b];
        if all(isfinite(fv))
            f0 = fv(1);
            fi = fv(1 + (1:dim));
            fp = fv(r);
            effect = fi ~= f0;
            delta  = fp - fi(P1) - fi(P2) + f0;
            scale  = max(max(abs(fp), abs(f0)), max(abs(fi(P1)), abs(fi(P2))));
            link   = abs(delta) > survey_eps * scale;
            n_surv = n_surv + 1;
            n_sep  = n_sep + (all(effect) && ~any(link));
            if mean(effect) >= effect_min
                [blocks, big] = components(dim, P1(link), P2(link));
                if big > block_max
                    blocks = {};
                end
            end
        end
    end
end

% The DE engine on a licence of maxFE FE; G: its coordinates (others at ctx); seed: start point; fe0: time offset
function [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
        engine(problem, G, ctx, maxFE, seed, fe0, total_fe, population_history, fitness_history, history_index, keep_tail)
    % keep_tail: the held tail is never released (no tail guard)
    if nargin < 11
        keep_tail = false;
    end

    if isempty(G)
        dim = problem.dimension;
        lb  = problem.lb(:)';
        ub  = problem.ub(:)';
    else
        dim = numel(G);
        lb  = problem.lb(G);
        ub  = problem.ub(G);
        lb  = lb(:)';
        ub  = ub(:)';
    end
    span  = ub - lb;

    plateau_window = 0.10;   % share of the licence without a real gain that makes a plateau
    tight_field    = 1e-2;   % a plateau only counts while the field spans less than this share of the box
    cost_aware     = true;   % rank prospect types per FE they cost, not by chance alone

    depletion_tol    = 1e-8;
    flat_field       = 10;    % generations of one exact yield across the field that end a campaign
    gain_tol         = 1e-8;  % relative gain below which a well is no real improvement
    maturity_age     = 0.7;
    min_rigs         = 10;    % smallest restart campaign
    sidetrack_shares = [0.1 0.2 0.5];
    stim_radius      = 0.3;   % unit-cube step before the 10^(-2U) shrink
    stim_tolx        = 1e-12;
    completion_share = 0.02;
    licence_floor    = max(0.002 * maxFE, 4 + floor(3 * log(dim)));
    surprise_scale   = 0.1 * (span / 200);
    edge_zone        = 0.05;  % the licence best this close to an edge (share of the box) makes it an edge
    edge_share       = 0.1;   % share of the other crossings that also land on the edge
    history_size     = 20;    % earlier licence bests re-tested at the end
    tail_share       = 0.03;  % last share of the licence held back for well finishing
    line_points      = 48;    % wells drilled along one coordinate in a line sweep
    line_basins      = 8;     % sampled basins of a line brought to their bottom
    line_brent_wells = 12;    % wells of Brent's method per basin
    line_sweeps      = 3;     % sweeps at most, another one only after a gain that jumped a basin
    line_share       = 0.7;   % share of the tail for the line pass at most
    snap_zone        = 0.05;  % a coordinate this close to a bound (share of the box) is tried on it
    finish_sigma     = 1e-8;  % first finishing step (share of the box)
    finish_cma       = 0.5;   % share of the tail after the line pass for CMA-ES; the rest for (1+1) step-outs
    guard_tol        = 1e-3;  % a gain this large (share of |yield|) just before the tail keeps the field producing
    wildcat_share    = 0.4;   % share of the FE left after the first campaign for the wildcat
    wildcat_ramp     = 0.3;   % age until which the wildcat's wells redesign one coordinate only
    wildcat_dmin     = 5;     % ... and of at least this dimension
    wildcat_dmax     = 10;    % the wildcat only on problems of at most this dimension
    wildcat_mode     = false; % the campaign running is the wildcat
    fine_share       = 0.02;  % share of the licence for the fine field, taken from prospecting
    fine_radius      = 1e-8;  % the fine field's first wells lie within this share of the box of the best
    fine_rigs        = 4;     % rigs of the fine field per dimension
    fine_tol         = 1e-13; % field spread that depletes the fine field
    fine_dmax        = 20;    % the fine field only on problems of at most this dimension
    fine_mode        = false; % the campaign running is the fine field
    ris_idle         = 0.05;  % prospecting pauses after this share of the licence without a real gain; RIS then takes a turn
    ris_chunk        = 0.1;   % share of the licence per RIS turn; prospecting and RIS alternate
    ris_tol          = 1e-9;  % relative gain below which the licence best has not really gained (for ris_idle)
    ris_dmin         = 5;     % RIS only on problems of at least this dimension (the whole problem, also for a block)
    ris_on           = problem.dimension >= ris_dmin;
    ipop_sigma       = 0.5;   % with a moving reference the wildcat is IPOP-CMA-ES restarts at this step (share of the box)

    FE    = 0;
    % Campaigns keep the whole licence's schedule but stop hold FE early; the tail finishes unless still gaining
    tail_fe = round(tail_share * maxFE);
    hold  = tail_fe;
    cap   = maxFE - hold;     % also lowered by the final re-test reserve once the reference is seen to move
    curve = zeros(1, maxFE);

    best_fitness  = inf;
    best_solution = lb;

    % Best of the running campaign, for its own plateau test
    campaign_best      = inf;
    campaign_last_gain = 0;

    % Scale a real gain is judged on: the yield range of the first wells
    yield_range = inf;

    % Pressure checks
    drift_seen = false;
    probe_gap  = 0;
    probe_next = inf;
    history_x  = zeros(0, dim);   % earlier licence bests, newest last

    % Licence best at its last gain of more than ris_tol (relative), and its FE
    ris_ref    = inf;
    ris_ref_fe = 0;

    % Prospect types: 1 near-field, 2 frontier, 3 sidetrack, 4 stimulation, 5 completion
    prospect_runs  = zeros(1, 5);
    prospect_hits  = zeros(1, 5);
    prospect_fe    = zeros(1, 5);
    completed_well = [];

    if isempty(seed)
        depleted = produce(0, maxFE, 18 * dim, [], [], true, []);
    else
        % Continuation from the caller's point: prospecting around it from the start
        log_well(seed.f, seed.x(:)');
        yield_range = seed.range;
        probe_gap   = 18 * dim;
        probe_next  = probe_gap;
        depleted    = true;
    end
    if depleted && isempty(seed) && dim >= wildcat_dmin && dim <= wildcat_dmax
        % Wildcat: a fresh young-field campaign; on a moving reference IPOP-CMA-ES restarts instead (population doubled, <= 32x)
        wildcat_mode = true;
        FEw = FE + round(wildcat_share * (min(maxFE - (tail_fe - hold), cap) - FE));
        if drift_seen
            kr = 0;
            while FEw - FE > 2 * (4 + floor(3 * log(dim))) * 2^min(kr, 5)
                stimulate(FEw, ipop_sigma, false, lb + rand(1, dim) .* span, 2^min(kr, 5));
                kr = kr + 1;
            end
        else
            produce(FE, FEw, 18 * dim, [], [], false, []);
        end
        wildcat_mode = false;
    end
    fine_fe = round(fine_share * maxFE) * (dim <= fine_dmax);
    if depleted
        FEp = maxFE - (tail_fe - hold);
        % The fine field's share is held back like the tail (a field still gaining releases both)
        hold = hold + fine_fe;
        cap  = cap - fine_fe;
        prospect(FEp);
        if hold >= fine_fe
            % Prospecting idle for ris_idle of the licence: RIS and prospecting take turns, each with a fresh idle clock
            if ris_on
                while min(FEp, cap) - FE >= licence_floor
                    ris_ref_fe = FE;
                    ris(min(cap, FE + round(ris_chunk * maxFE)));
                    if min(FEp, cap) - FE < licence_floor
                        break;
                    end
                    ris_ref_fe = FE;
                    prospect(FEp);
                end
            end
            hold = hold - fine_fe;
            cap  = cap + fine_fe;
        end
    end
    if fine_fe > 0
        % Fine field on its own share before the tail; what it leaves goes to the tail
        fine_field(min(cap, maxFE - (tail_fe - hold)));
    end

    % Finishing: edge snap, line pass, small-step CMA-ES on half of the rest, clamped (1+1) step-outs on the other
    cap  = cap + hold;
    hold = 0;
    edge_snap(cap);
    line_pass(min(cap, FE + round(line_share * (cap - FE))));
    s = finish_sigma;
    cma_end = FE + round(finish_cma * (cap - FE));
    while cma_end - FE > 4 + floor(3 * log(dim))
        stimulate(cma_end, s, true);
        s = 10^(-4 - 6 * rand);
    end
    finish(cap);

    if drift_seen
        final_retest();
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;

    % Prospect campaigns until FEend, each type drawn by its risked chance
    function prospect(FEend)
        while min(FEend, cap) - FE >= licence_floor
            if ris_on && FE - ris_ref_fe > ris_idle * maxFE
                return;   % the licence best has not really gained for a while: RIS takes the rest
            end
            if FE >= probe_next
                probe([]);
            end
            n = sum(prospect_runs);
            chance = (prospect_hits + 1) ./ (prospect_runs + 2);
            if cost_aware
                cost = max(1, (prospect_fe + sum(prospect_fe) / max(1, n)) ./ (prospect_runs + 1));
                chance = chance ./ cost;
            end
            % Completing a well that was already completed would only repeat itself
            if ~isempty(completed_well) && isequal(completed_well, best_solution)
                chance(5) = 0;
            end
            if ~any(chance)
                break;
            end
            k = find(rand * sum(chance) <= cumsum(chance), 1);
            if isempty(k)
                k = find(chance, 1, 'last');
            end
            before    = best_fitness;
            fe_before = FE;
            FEe       = min(FEend, cap + hold);
            rest      = FEe - FE;
            switch k
                case 1
                    rigs = max(min_rigs, min(18 * dim, round(18 * dim * rest / maxFE)));
                    produce(FE, FEe, rigs, best_solution, best_fitness, false, []);
                case 2
                    rigs = max(min_rigs, min(18 * dim, round(18 * dim * rest / maxFE)));
                    produce(FE, FEe, rigs, [], [], false, []);
                case 3
                    share = sidetrack_shares(randi(numel(sidetrack_shares)));
                    m = min(dim - 1, max(1, round(share * dim)));
                    S = sort(randperm(dim, m));
                    rigs = max(min_rigs, min(18 * m, round(18 * m * rest / maxFE)));
                    produce(FE, FE + min(rest, max(round(rest * m / dim), 2 * rigs)), ...
                            rigs, best_solution, best_fitness, false, S);
                case 4
                    stimulate(FEe, stim_radius * 10^(-2 * rand));
                case 5
                    completed_well = best_solution;
                    complete(FE + min(rest, ceil(completion_share * maxFE)));
            end
            prospect_runs(k) = prospect_runs(k) + 1;
            prospect_hits(k) = prospect_hits(k) + is_gain(before, best_fitness);
            prospect_fe(k)   = prospect_fe(k) + (FE - fe_before);
        end
    end

    % Fine field: small campaigns within fine_radius of the best, run to fine_tol; repeated while the last one gained
    function fine_field(FEend)
        rigs_f = max(min_rigs, round(fine_rigs * dim));
        while FEend - FE >= 4 * rigs_f
            before = best_fitness;
            fine_mode = true;
            produce(FE, FEend, rigs_f, best_solution, best_fitness, false, []);
            fine_mode = false;
            if ~(best_fitness < before)
                break;
            end
        end
    end

    % One production campaign over [FE0, FEend]; S drills only those coordinates around the discovery
    function depleted = produce(FE0, FEend, rigs_max, discovery, discovery_fit, first, S)
        depleted = false;
        licence  = FEend - FE0;
        nS       = numel(S);

        register_size  = 2 * rigs_max;
        log_size       = 5;
        stepout_sd     = 0.025;
        focus_xi       = 0.65;
        focus_k        = 6.5;
        final_rigs     = 4;
        tp_share_init  = 0.7;
        smoothing_age  = 0.75;
        smoothing_old  = 0.65;
        surprise_rate  = 0.4;

        rigs         = rigs_max;
        n_register   = rigs_max;
        hit_rate     = 0.5;
        log_redesign = ones(log_size, 1);
        log_stepout  = ones(log_size, 1);
        log_slot     = 1;
        tp_share     = tp_share_init;
        flat_gens    = 0;

        if fine_mode
            register = repmat(discovery, register_size, 1) + (2 * rand(register_size, dim) - 1) .* repmat(fine_radius * span, register_size, 1);
            register = min(max(register, repmat(lb, register_size, 1)), repmat(ub, register_size, 1));
        elseif nS == 0
            register = repmat(lb, register_size, 1) + rand(register_size, dim) .* repmat(span, register_size, 1);
        else
            register = repmat(discovery, register_size, 1);
            register(:, S) = repmat(lb(S), register_size, 1) + rand(register_size, nS) .* repmat(span(S), register_size, 1);
        end
        register_fit = inf(register_size, 1);
        i0 = 1;
        if first
            best_solution = register(1, :);
        else
            % The plateau test judges this campaign's own wells, not the discovery it carries
            campaign_best      = inf;
            campaign_last_gain = FE;
            if ~isempty(discovery)
                register(1, :)  = discovery;
                register_fit(1) = discovery_fit;
                i0 = 2;
            end
        end

        % All first wells in one call (one Eq. 6 reference on RW), then logged one by one
        n_first = min(rigs - i0 + 1, min(FEend, cap) - FE);
        if n_first > 0
            fv = fitness_of(register(i0:i0 + n_first - 1, :)', problem, FE);
            register_fit(i0:i0 + n_first - 1) = fv(:);
            if first
                ok = isfinite(fv);
                if any(ok)
                    yield_range = max(fv(ok)) - min(fv(ok));
                end
            end
        end
        for i = i0:i0 + n_first - 1
            FE = FE + 1;
            log_well(register_fit(i), register(i, :));
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, register(1:i, :), register_fit(1:i), population_history, fitness_history, ...
                history_index, maxFE);
        end
        if first
            probe_gap  = rigs;
            probe_next = FE + probe_gap;
        end

        [active_fit, order] = sort(register_fit(1:rigs), 'ascend');
        active          = register(order, :);
        active_fit_prev = active_fit;
        trial_fit       = inf(rigs_max, 1);
        was_tp          = false(rigs_max, 1);
        next_slot       = 1;

        while FE < min(FEend, cap)
            if FE >= probe_next
                rows = probe([active(1:rigs, :); register(1:n_register, :)]);
                if ~isempty(rows)
                    active_fit(1:rigs)         = rows(1:rigs);
                    register_fit(1:n_register) = rows(rigs + 1:end);
                    active_fit_prev(1:rigs)    = active_fit(1:rigs);
                    campaign_best = min(rows);
                    if FE >= min(FEend, cap)
                        break;
                    end
                end
            end

            stepout_mean = 0.4 + tanh(hit_rate * 5) * 0.25;

            [~, by_register] = sort(register_fit(1:rigs), 'ascend');
            [~, by_active]   = sort(active_fit(1:rigs), 'ascend');

            rank_w   = exp(-(0:rigs-1)' / rigs * 3);
            rank_cum = cumsum(rank_w / sum(rank_w));
            pressure = 3 * (rigs - (0:rigs-1)');

            focus   = min(rigs, max(2, floor(rigs * focus_xi * exp(-hit_rate * focus_k))));
            anchors = floor(rigs * 0.17 * (1 - 0.5 * (FE - FE0) / licence));
            if anchors <= 1
                anchors = 2;
            end

            hits         = 0;
            hit_redesign = zeros(rigs, 1);
            hit_stepout  = zeros(rigs, 1);
            hit_gain     = zeros(rigs, 1);

            for drill = 1:rigs
                if FE >= min(FEend, cap)
                    break;
                end
                age = (FE - FE0) / licence;

                base_i = randi(rigs);
                slot1  = randi(log_size);
                slot2  = randi(log_size + 1);

                draw_tp = rand;
                if age < maturity_age
                    draw_tp = 2;
                end
                three_point    = draw_tp < tp_share;
                was_tp(base_i) = three_point;

                base = active(base_i, :);

                if three_point
                    anchor = by_register(pick_weighted(pressure(1:anchors)));
                    while anchor == base_i
                        anchor = by_register(pick_weighted(pressure(1:anchors)));
                    end
                    w1 = by_active(pick_weighted(pressure));
                    while w1 == anchor
                        w1 = by_active(pick_weighted(pressure));
                    end
                    w2 = by_register(pick_weighted(pressure));
                    while w2 == anchor || w2 == w1
                        w2 = by_register(pick_weighted(pressure));
                    end
                    [up, mid, low] = three_point_order(register(anchor, :), register_fit(anchor), ...
                                                       active(w1, :), active_fit(w1), ...
                                                       register(w2, :), register_fit(w2));

                    % Slot log_size + 1 is the fixed 0.9 slot
                    stepout = -1;
                    while stepout < 0
                        if slot2 <= log_size
                            stepout = log_stepout(slot2) + 0.1 * tan(pi * (rand - 0.5));
                        else
                            stepout = 0.9 + 0.1 * tan(pi * (rand - 0.5));
                        end
                    end
                    stepout = min(stepout, 1);

                    if slot2 <= log_size
                        redesign = log_redesign(slot2) + 0.1 * randn;
                    else
                        redesign = 0.9 + 0.1 * randn;
                    end
                    redesign = min(max(redesign, 0), 1);

                    plan = base + stepout * (up - base) + stepout * (mid - low);
                else
                    anchor = by_register(randi(focus));
                    while anchor == base_i
                        anchor = by_register(randi(focus));
                    end
                    w1 = by_active(pick_cum(rank_cum));
                    while w1 == anchor
                        w1 = by_active(pick_cum(rank_cum));
                    end
                    w2 = by_register(randi(rigs));
                    while w2 == anchor || w2 == w1
                        w2 = by_register(randi(rigs));
                    end

                    stepout = stepout_mean + stepout_sd * randn();
                    while stepout < 0.0 || stepout > 1.0
                        stepout = stepout_mean + stepout_sd * randn();
                    end

                    redesign = log_redesign(slot1) + 0.05 * randn();
                    redesign = min(max(redesign, 0.0), 1.0);
                    if wildcat_mode
                        % Young field: wells redesign one coordinate only; the share ramps up to the log after
                        redesign = min(redesign, max(0, (age - wildcat_ramp) / (1 - wildcat_ramp)));
                    end

                    plan = base + stepout * (register(anchor, :) - base) ...
                                + stepout * (active(w1, :) - register(w2, :));
                end

                if nS == 0
                    take = rand(1, dim) < redesign;
                    take(randi(dim)) = true;
                    if rand < surprise_rate && ~fine_mode
                        well = base + surprise_scale .* tan(pi * (rand(1, dim) - 0.5));
                    else
                        well = base;
                    end
                    redesign_actual = sum(take) / dim;
                else
                    tk = rand(1, nS) < redesign;
                    tk(randi(nS)) = true;
                    take = false(1, dim);
                    take(S) = tk;
                    well = base;
                    if rand < surprise_rate && ~fine_mode
                        well(S) = base(S) + surprise_scale(S) .* tan(pi * (rand(1, nS) - 0.5));
                    end
                    redesign_actual = sum(tk) / nS;
                end
                well(take) = plan(take);

                % Edge wells: onto the crossed edge when the licence best lies near it (and in a small share of the rest), else re-sited
                outside = well < lb | well > ub | ~isfinite(well);
                if any(outside)
                    lo = well < lb;
                    hi = well > ub;
                    elo = lo & (best_solution - lb < edge_zone * span);
                    ehi = hi & (ub - best_solution < edge_zone * span);
                    well(elo) = lb(elo);
                    well(ehi) = ub(ehi);
                    rest = outside & ~elo & ~ehi;
                    ec = rest & (lo | hi) & (rand(1, dim) < edge_share);
                    well(ec & lo) = lb(ec & lo);
                    well(ec & hi) = ub(ec & hi);
                    rest = rest & ~ec;
                    well(rest) = lb(rest) + rand(1, sum(rest)) .* span(rest);
                end

                [fv, FE] = fitness_of(well', problem, FE);
                f_well   = fv(1);
                trial_fit(base_i) = f_well;

                log_well(f_well, well);

                if f_well <= active_fit(base_i)
                    slot = n_register + hits + 1;
                    register(slot, :)  = well;
                    register_fit(slot) = f_well;

                    if wildcat_mode
                        % One-to-one: the productive well takes its base's place (DE selection)
                        gain_i = active_fit(base_i) - f_well;
                        active(base_i, :)  = well;
                        active_fit(base_i) = f_well;
                    else
                        active(next_slot, :)  = well;
                        active_fit(next_slot) = f_well;
                    end

                    hits = hits + 1;
                    hit_redesign(hits) = redesign_actual;
                    hit_stepout(hits)  = stepout;
                    if wildcat_mode
                        hit_gain(hits) = gain_i;
                    else
                        % Read after the FIFO write, as the engine does: 0 when the slot was the base's own
                        hit_gain(hits) = abs(active_fit(base_i) - f_well);
                    end

                    next_slot = mod(next_slot, rigs) + 1;
                end

                curve(FE) = best_fitness;
                [population_history, fitness_history, history_index] = rec_history(...
                    FE, active(1:rigs, :), active_fit(1:rigs), ...
                    population_history, fitness_history, history_index, maxFE);
            end

            improved = trial_fit(1:rigs) <= active_fit_prev(1:rigs);
            gain     = active_fit_prev(1:rigs) - trial_fit(1:rigs);
            gain_tp  = sum(gain(improved & was_tp(1:rigs)));
            gain_std = sum(gain(improved & ~was_tp(1:rigs)));
            if gain_tp ~= 0 && gain_std ~= 0
                raw = gain_tp / (gain_tp + gain_std);
                if raw > 1
                    raw = 1;
                elseif raw < 0
                    raw = 0;
                end
                if (FE - FE0) / licence >= smoothing_age
                    tp_share = smoothing_old * tp_share + (1 - smoothing_old) * raw;
                else
                    tp_share = raw;
                end
            else
                tp_share = tp_share_init;
            end
            active_fit_prev(1:rigs) = active_fit(1:rigs);

            hit_rate = hits / rigs;

            % Rig release; the search for the worst keeps the pre-release count, as the engine does
            new_rigs = max(final_rigs, floor((final_rigs - rigs_max) / licence * (FE - FE0) + rigs_max));
            if new_rigs < rigs
                for released = 1:(rigs - new_rigs)
                    [~, worst] = max(active_fit(1:rigs));
                    active(worst:rigs-1, :)       = active(worst+1:rigs, :);
                    active_fit(worst:rigs-1)      = active_fit(worst+1:rigs);
                    active_fit_prev(worst:rigs-1) = active_fit_prev(worst+1:rigs);
                end
            end
            rigs = new_rigs;

            if hits > 0
                log_redesign(log_slot) = 0.5 * (lehmer_mean(hit_redesign(1:hits), hit_gain(1:hits)) ...
                                                + log_redesign(log_slot));
                log_stepout(log_slot)  = lehmer_mean(hit_stepout(1:hits), hit_gain(1:hits));
                log_slot = mod(log_slot, log_size) + 1;
            end

            n_register = rigs + hits;
            if n_register > rigs
                [sorted_fit, order] = sort(register_fit(1:n_register), 'ascend');
                n_register = rigs;
                register(1:n_register, :) = register(order(1:n_register), :);
                register_fit(1:n_register) = sorted_fit(1:n_register);
            end

            if FE < min(FEend, cap)
                if nS == 0
                    spread = max((max(active(1:rigs, :), [], 1) - min(active(1:rigs, :), [], 1)) ./ span);
                else
                    spread = max((max(active(1:rigs, S), [], 1) - min(active(1:rigs, S), [], 1)) ./ span(S));
                end
                % Flat field: every active well yields exactly the same, generation after generation
                if max(active_fit(1:rigs)) == min(active_fit(1:rigs))
                    flat_gens = flat_gens + 1;
                else
                    flat_gens = 0;
                end
                if flat_gens >= flat_field
                    depleted = true;
                    return;
                end
                on_plateau = FE - campaign_last_gain > plateau_window * maxFE && spread < tight_field;
                tol = depletion_tol;
                if fine_mode
                    tol = fine_tol;
                end
                if spread < tol || on_plateau
                    depleted = true;
                    return;
                end
            end
        end
    end

    % Stimulation: CMA-ES from the best well in the unit cube, mirrored into the block; in finishing only sigma ends it
    function stimulate(FEend, sigma, fine, x0, lam_mult)
        if nargin < 3
            fine = false;
        end
        if nargin < 4
            x0 = best_solution;
            lam_mult = 1;
        end
        N = dim;
        lambda = (4 + floor(3 * log(N))) * lam_mult;
        mu = floor(lambda / 2);
        w = log(mu + 1/2) - log(1:mu)';
        w = w / sum(w);
        mueff = sum(w)^2 / sum(w.^2);
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
        cs = (mueff + 2) / (N + mueff + 5);
        c1 = 2 / ((N + 1.3)^2 + mueff);
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)^2 + mueff));
        damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs;
        chiN = N^0.5 * (1 - 1 / (4 * N) + 1 / (21 * N^2));

        xmean = ((x0 - lb) ./ span)';
        pc = zeros(N, 1);
        ps = zeros(N, 1);
        B = eye(N);
        Dg = ones(N, 1);
        C = eye(N);
        invsqrtC = eye(N);
        eigeneval = 0;
        counteval = 0;

        campaign_best      = inf;
        campaign_last_gain = FE;
        hist_len = 10 + ceil(30 * N / lambda);
        gen_best = inf(hist_len, 1);
        g = 0;

        while FE < min(FEend, cap)
            m = min(lambda, min(FEend, cap) - FE);
            arz = randn(N, lambda);
            Y = xmean + sigma * (B * (Dg .* arz));
            if drift_seen
                Y = min(max(Y, 0), 1);       % a constrained problem: wells outside the box onto it
            else
                Y = 1 - abs(1 - mod(Y, 2));
            end
            X = lb' + span' .* Y;

            [fv, FE2] = fitness_of(X(:, 1:m), problem, FE);
            fit = fv(:)';
            fit(~isfinite(fit)) = inf;
            FE_start = FE;
            for j = 1:m
                FE = FE_start + j;
                log_well(fit(j), X(:, j)');
                curve(FE) = best_fitness;
            end
            FE = FE2;
            counteval = counteval + m;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, X(:, 1:m)', fit(1:m), population_history, fitness_history, ...
                history_index, maxFE);
            if m < lambda
                return;
            end

            [fs, idx] = sort(fit);
            xold = xmean;
            xmean = Y(:, idx(1:mu)) * w;
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
            hsig = norm(ps) / sqrt(1 - (1 - cs)^(2 * counteval / lambda)) / chiN < 1.4 + 2 / (N + 1);
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;
            artmp = (Y(:, idx(1:mu)) - xold) / sigma;
            C = (1 - c1 - cmu) * C + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
                + cmu * (artmp .* w') * artmp';
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));
            % Flat fitness among the best 70 %: widen the step
            if fs(1) == fs(ceil(0.7 * lambda))
                sigma = sigma * exp(0.2 + cs / damps);
            end

            if counteval - eigeneval > lambda / (c1 + cmu) / N / 10
                eigeneval = counteval;
                C = triu(C) + triu(C, 1)';
                [B, Dm] = eig(C);
                ev = diag(Dm);
                if any(~isfinite(ev)) || min(ev) <= 0
                    return;
                end
                Dg = sqrt(ev);
                invsqrtC = B * diag(1 ./ Dg) * B';
            end

            g = g + 1;
            gen_best(mod(g - 1, hist_len) + 1) = fs(1);
            if fine
                if ~isfinite(sigma) || sigma * max(Dg) < 1e-15 || max(Dg) / min(Dg) > 1e7
                    return;
                end
                continue;
            end
            if ~isfinite(sigma) || sigma * max(Dg) < stim_tolx || max(Dg) / min(Dg) > 1e7
                return;
            end
            if FE - campaign_last_gain > plateau_window * maxFE
                return;
            end
            if g >= hist_len && max(gen_best) - min(gen_best) <= gain_tol * max(1, min(abs(min(gen_best)), yield_range))
                return;
            end
        end
    end

    % Completion: fmincon (interior-point) from the best well, every evaluation counted
    function complete(FEend)
        if exist('fmincon', 'file') ~= 2
            return;
        end
        FEend = min(FEend, cap);
        budget = FEend - FE - (dim + 1);
        if budget < 2 * (dim + 1)
            return;
        end
        options = optimset('Display', 'off', 'Algorithm', 'interior-point', ...
                           'MaxFunEvals', budget, 'UseParallel', 'never');
        ws = warning('off', 'all');
        try
            fmincon(@well_test, best_solution(:), [], [], [], [], lb(:), ub(:), [], options);
        catch
        end
        warning(ws);

        function f = well_test(x)
            if FE >= FEend
                f = best_fitness;
                return;
            end
            x = min(max(x(:)', lb), ub);
            [fv, FE] = fitness_of(x', problem, FE);
            f = fv(1);
            log_well(f, x);
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, x, f, population_history, fitness_history, history_index, maxFE);
            if ~isfinite(f)
                f = realmax;
            end
        end
    end

    % Edge snap: coordinates within snap_zone of a bound set on it, together then alone; a better well is kept
    function edge_snap(FEend)
        x = best_solution;
        near = (x - lb < snap_zone * span & x > lb) | (ub - x < snap_zone * span & x < ub);
        idx = find(near);
        if isempty(idx)
            return;
        end
        tries = {idx};
        if numel(idx) > 1
            tries = [tries, num2cell(idx)];
        end
        for t = 1:numel(tries)
            if FE >= FEend
                return;
            end
            y = best_solution;
            j = tries{t};
            lo = y(j) - lb(j) < ub(j) - y(j);
            y(j(lo))  = lb(j(lo));
            y(j(~lo)) = ub(j(~lo));
            if isequal(y, best_solution)
                continue;
            end
            [fv, FE] = fitness_of(y', problem, FE);
            log_well(fv(1), y);
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, y, fv(1), population_history, fitness_history, history_index, maxFE);
        end
    end

    % Line pass: sweeps over the coordinates; another only after a gain that jumped a stratum, at most line_sweeps
    function line_pass(FEend)
        for sweep = 1:line_sweeps
            before   = best_fitness;
            x_before = best_solution;
            line_sweep(FEend);
            if ~is_gain(before, best_fitness) || ~any(abs(best_solution - x_before) > span / line_points)
                return;
            end
        end
    end

    % One sweep: each coordinate at line_points stratified wells (one call), the best basins finished by Brent's method
    function line_sweep(FEend)
        m = line_points;
        for i = randperm(dim)
            if FEend - FE < m + 10
                return;
            end
            Y = repmat(best_solution, m, 1);
            Y(:, i) = lb(i) + ((0:m - 1)' + rand(m, 1)) / m * span(i);
            [fv, FE2] = fitness_of(Y', problem, FE);
            fv = fv(:);
            for j = 1:m
                FE = FE + 1;
                log_well(fv(j), Y(j, :));
                curve(FE) = best_fitness;
            end
            FE = FE2;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, Y, fv, population_history, fitness_history, history_index, maxFE);
            iv = basins(i, Y(:, i), fv, min(line_basins, m), FEend);
            for t = 1:size(iv, 1)
                line_brent(i, iv(t, 1), iv(t, 2), line_brent_wells, FEend);
            end
        end
    end

    % Sampled local minima of the line, ranked by their parabola vertex; rows of iv: bracketing neighbours, lowest yield
    function iv = basins(i, y, fv, nq, FEend)
        m  = numel(fv);
        J  = find([true; fv(2:end) <= fv(1:end - 1)] & [fv(1:end - 1) <= fv(2:end); true]);
        nJ = numel(J);
        pred = fv(J); xv = y(J); a = y(max(1, J - 1)); b = y(min(m, J + 1));
        for t = 1:nJ
            j = J(t);
            if j == 1 || j == m
                continue;
            end
            x1 = y(j - 1); x2 = y(j); x3 = y(j + 1);
            f1 = fv(j - 1); f2 = fv(j); f3 = fv(j + 1);
            den = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1);
            if den <= 0 || ~isfinite(den)
                continue;
            end
            v = x2 - 0.5 * ((x2 - x1)^2 * (f2 - f3) - (x2 - x3)^2 * (f2 - f1)) / den;
            v = min(max(v, x1), x3);
            pv = f1 * (v - x2) * (v - x3) / ((x1 - x2) * (x1 - x3)) + f2 * (v - x1) * (v - x3) / ((x2 - x1) * (x2 - x3)) ...
               + f3 * (v - x1) * (v - x2) / ((x3 - x1) * (x3 - x2));
            xv(t) = v;
            pred(t) = min(f2, pv);
        end
        [~, s] = sort(pred);
        K  = min(nq, nJ);
        iv = zeros(K, 3);
        for t = 1:K
            k  = s(t);
            fx = fv(J(k));
            if xv(k) ~= y(J(k)) && FEend - FE >= 1
                w = best_solution; w(i) = xv(k);
                [fw, FE] = fitness_of(w', problem, FE);
                log_well(fw, w);
                curve(FE) = best_fitness;
                fx = min(fx, fw);
            end
            iv(t, :) = [a(k), b(k), fx];
        end
    end

    % Brent's method along coordinate i of the licence best on [a, b], at most n wells
    function line_brent(i, a, b, n, FEend)
        n = min(n, FEend - FE);
        if n < 3 || ~(b > a)
            return;
        end
        opt = optimset('TolX', 1e-12 * span(i), 'MaxFunEvals', n, 'MaxIter', n, 'Display', 'off');
        fminbnd(@(v) line_well(i, v, FEend), a, b, opt);
    end

    function f = line_well(i, v, FEend)
        if FE >= FEend
            f = inf;
            return;
        end
        w = best_solution;
        w(i) = v;
        [f, FE] = fitness_of(w', problem, FE);
        log_well(f, w);
        curve(FE) = best_fitness;
    end

    % Finishing (1+1) step-outs: step doubled on a hit, cut by 2^(-1/4) on a miss, wells clamped into the block
    function finish(FEend)
        x   = best_solution;
        fx  = best_fitness;
        sig = finish_sigma * span;
        while FE < FEend
            y = min(max(x + sig .* randn(1, dim), lb), ub);
            [fv, FE] = fitness_of(y', problem, FE);
            log_well(fv(1), y);
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = rec_history(...
                FE, y, fv(1), population_history, fitness_history, history_index, maxFE);
            if fv(1) <= fx
                x = y; fx = fv(1); sig = 2 * sig;
            else
                sig = sig * 2^(-0.25);
            end
            if max(sig ./ span) < 1e-16
                sig = finish_sigma * span;
            end
        end
    end

    % Pressure check: the best drilled again; if its yield moved, the rows are re-tested in ONE call (their yields, else [])
    function rows_fit = probe(rows)
        rows_fit = [];
        if cap - FE < 1
            return;
        end
        [fv, FE] = fitness_of(best_solution', problem, FE);
        curve(FE) = best_fitness;
        if fv(1) == best_fitness || (~isfinite(fv(1)) && ~isfinite(best_fitness))
            probe_gap  = 2 * probe_gap;
            probe_next = FE + probe_gap;
            return;
        end
        if ~drift_seen
            drift_seen = true;
            cap = max(FE + 1, maxFE - hold - (history_size + 1));
        end
        best_fitness = fv(1);
        probe_gap  = max(1, size(rows, 1));
        probe_next = FE + probe_gap;
        if isempty(rows)
            curve(FE) = best_fitness;
            return;
        end
        n = min(size(rows, 1), cap - FE);
        if n < size(rows, 1)
            return;
        end
        [fv, FE2] = fitness_of(rows', problem, FE);
        rows_fit = fv(:);
        for j = 1:n
            FE = FE + 1;
            if rows_fit(j) < best_fitness
                best_fitness  = rows_fit(j);
                history_x(end + 1, :) = best_solution; %#ok<AGROW>
                best_solution = rows(j, :);
            end
            curve(FE) = best_fitness;
        end
        FE = FE2;
        [population_history, fitness_history, history_index] = rec_history(...
            FE, rows, rows_fit, population_history, fitness_history, history_index, maxFE);
    end

    % Last re-test: the best and its predecessors in one call; the lowest yield is reported (ties keep the best)
    function final_retest()
        H = unique(history_x(max(1, end - history_size + 1):end, :), 'rows', 'stable');
        X = [best_solution; H];
        n = min(size(X, 1), maxFE - FE);
        if n < 2
            return;
        end
        X = X(1:n, :);
        [fv, FE2] = fitness_of(X', problem, FE);
        fv = fv(:);
        [fb, kb] = min(fv);
        best_fitness  = fb;
        best_solution = X(kb, :);
        for j = 1:n
            curve(FE + j) = best_fitness;
        end
        FE = FE2;
        [population_history, fitness_history, history_index] = rec_history(...
            FE, X, fv, population_history, fitness_history, history_index, maxFE);
    end

    % RIS of CMAES-RIS: a segment of the best re-sampled (exponential crossover), then the 3SOME short-distance search
    function ris(FEend)
        CR = 0.5^(2 / dim);
        while min(FEend, cap) - FE > 2 * dim
            if FE >= probe_next
                probe([]);
            end
            x = best_solution;
            k = randi(dim);
            for t = 1:dim
                x(k) = lb(k) + rand * span(k);
                k = mod(k, dim) + 1;
                if rand > CR
                    break;
                end
            end
            fx = ris_well(x);
            SR = 0.2 * span;
            gained = true;
            while max(SR ./ span) > 5e-10 && FE < min(FEend, cap)
                if ~gained
                    SR = SR / 2;
                end
                gained = false;
                for k = 1:dim
                    if FE >= min(FEend, cap)
                        break;
                    end
                    y = x;
                    y(k) = max(y(k) - SR(k), lb(k));
                    fy = ris_well(y);
                    if fy < fx
                        x = y; fx = fy; gained = true;
                        continue;
                    end
                    if FE >= min(FEend, cap)
                        break;
                    end
                    y = x;
                    y(k) = min(y(k) + 0.5 * SR(k), ub(k));
                    fy = ris_well(y);
                    if fy < fx
                        x = y; fx = fy; gained = true;
                    end
                end
            end
        end
    end

    function f = ris_well(x)
        [fv, FE] = fitness_of(x', problem, FE);
        f = fv(1);
        if ~isfinite(f)
            f = inf;
        end
        log_well(f, x);
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = rec_history(...
            FE, x, f, population_history, fitness_history, history_index, maxFE);
    end

    % Bookkeeping per well: licence best (and predecessors), last real gain (ris_idle), the campaign's plateau clock
    function log_well(f, x)
        if f < best_fitness
            if f < ris_ref - ris_tol * max(1, abs(ris_ref)) || ~isfinite(ris_ref)
                ris_ref    = f;
                ris_ref_fe = FE;
            end
            % A field still making large gains just before the held tail keeps the tail
            if hold > 0 && ~keep_tail && FE > maxFE - 2 * hold && guard_gain(best_fitness, f)
                cap  = cap + hold;
                hold = 0;
            end
            if isfinite(best_fitness)
                history_x(end + 1, :) = best_solution;
                if size(history_x, 1) > 4 * history_size
                    history_x = history_x(end - history_size + 1:end, :);
                end
            end
            best_fitness  = f;
            best_solution = x;
        end
        if is_gain(campaign_best, f)
            campaign_last_gain = FE;
        end
        if f < campaign_best
            campaign_best = f;
        end
    end

    % A gain large enough to keep the field producing through the tail
    function yes = guard_gain(old, new)
        if isfinite(old)
            yes = new < old - guard_tol * max(1, abs(old));
        else
            yes = is_gain(old, new);
        end
    end

    function yes = is_gain(old, new)
        yes = new < old - gain_tol * max(1, min(abs(old), yield_range));
        if ~isfinite(old)
            yes = isfinite(new);
        end
    end

    % Every evaluation: a block's wells are drilled into the context vector
    function [fv, FEo] = fitness_of(X, prob, FEi)
        if ~isempty(G)
            Xf_embed = repmat(ctx(:), 1, size(X, 2));
            Xf_embed(G, :) = X;
            X = Xf_embed;
        end
        [fv, FEo] = calculate_fitness(X, prob, FEi);
    end

    % Every record: a block's wells as points of the whole problem, on the run's time axis
    function [ph, fh, hi] = rec_history(fe, P, F, ph, fh, hi, ~)
        if ~isempty(G)
            Pf_embed = repmat(ctx, size(P, 1), 1);
            Pf_embed(:, G) = P;
            P = Pf_embed;
        end
        [ph, fh, hi] = record_history(fe0 + fe, P, F, ph, fh, hi, total_fe);
    end
end

function idx = pick_cum(cum_prob)
    idx = find(rand() <= cum_prob, 1);
    if isempty(idx)
        idx = numel(cum_prob);
    end
end

function idx = pick_weighted(w)
    c = cumsum(w);
    idx = find(rand() * c(end) <= c, 1);
    if isempty(idx)
        idx = numel(w);
    end
end

% Three wells ranked by yield: up-dip (best), middle, low
function [up, mid, low] = three_point_order(x1, f1, x2, f2, x3, f3)
    X = [x1; x2; x3];
    [~, o] = sort([f1; f2; f3]);
    up  = X(o(1), :);
    mid = X(o(2), :);
    low = X(o(3), :);
end

% Gain-weighted Lehmer mean of the productive wells' step-outs or redesign shares
function m = lehmer_mean(values, gains)
    w = gains / sum(gains);
    s = sum(w .* values);
    if abs(s) > 1e-8
        m = sum(w .* values .* values) / s;
    else
        m = 1.0;
    end
end

% Connected components of the interaction graph; lone variables form one block (a single one joins the smallest)
function [blocks, big] = components(dim, a, b)
    comp = zeros(1, dim);
    k = 0;
    for i = 1:dim
        if comp(i) == 0
            k = k + 1;
            comp(i) = k;
            q = i;
            while ~isempty(q)
                v = q(1);
                q(1) = [];
                nb = [b(a == v); a(b == v)]';
                nb = nb(comp(nb) == 0);
                comp(nb) = k;
                q = [q nb];
            end
        end
    end
    sizes  = accumarray(comp', 1)';
    big    = max(sizes);
    lone   = find(sizes == 1);
    blocks = {};
    for g = find(sizes > 1)
        blocks{end + 1} = find(comp == g);
    end
    single = find(ismember(comp, lone));
    if numel(single) >= 2
        blocks{end + 1} = single;
    elseif numel(single) == 1
        if isempty(blocks)
            blocks = {single};
        else
            [~, s] = min(cellfun(@numel, blocks));
            blocks{s} = sort([blocks{s} single]);
        end
    end
end

% Two block lists describe the same partition of 1..dim (block order free)
function yes = same_partition(a, b, dim)
    yes = numel(a) == numel(b);
    if yes && ~isempty(a)
        la = zeros(1, dim);
        lb = zeros(1, dim);
        for g = 1:numel(a), la(a{g}) = min(a{g}); end
        for g = 1:numel(b), lb(b{g}) = min(b{g}); end
        yes = isequal(la, lb);
    end
end

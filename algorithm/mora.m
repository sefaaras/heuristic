% ----------------------------------------------------------------------- %
% Morabaraba Optimization Algorithm (MOA)
% Stored as mora; the acronym MOA collides with moa
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N                   = 30     % Population size (players)
%   COWS_INITIAL        = 12     % Cows per player
%   FLYING_THRESHOLD    = 3      % Cow count that unlocks the flying phase
%   GAME_OVER_THRESHOLD = 2
%   MILL_TOL_REL = min(0.30, 0.05*(1+max(0,log10(dim/6))))
%   GS_c = 0.05, delta_stag = 8, F_de = 0.8, CR_de = 0.9
%
% Algorithm Concept:
%   - Three game phases map onto the search life cycle: PLACING (global
%     exploration), MOVING (guided local search along the 24-point board
%     topology) and FLYING (long-range escape)
%   - Mill formation: when three teammates align along a board line (a triplet
%     of decision variables) within MILL_TOL_REL, the mill "shoots" a cow off
%     the worst opposing player and drags its weakest dimension to the mill mean
%   - Supporting mechanisms: per-agent mood, velocity persistence, a mill
%     roster of multi-leader attractors, retreat-from-worst and a mill-rate
%     stagnation trigger
%   - Hybrid exploitation suite: chaotic and opposition-based initialisation,
%     Gaussian zoom, Levy and golden-section polish, coordinate line search,
%     Latin-hypercube multi-start, DE/best/1 rescue and a quarantined restart
%
% Reference:
% Bonginkosi A. Thango,
% Morabaraba Optimization Algorithm: A Novel Socio-Game-Inspired Meta-Heuristic
% for Global Optimization,
% Mathematics 2026, 14, 2171.
% https://doi.org/10.3390/math14122171
%
% Implementation Note:
%   The board-setup loop re-evaluates the position kept by opposition learning,
%   whose value min(f1, f2) is already in hand. It is kept: on CEC2020RW the
%   scalarisation reference is a running max, so re-reading the whole initial
%   population at one point makes those fitnesses comparable with each other,
%   and the duplicate costs 30 of the budget's 1e5 evaluations.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mora(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    MaxNFE = problem.maxFe;

    SearchAgents_no = 30;
    Max_iter = max(1, ceil(MaxNFE / 60));

    nfe   = 0;
    curve = zeros(1, MaxNFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Morabaraba game constants
    COWS_INITIAL     = 12;
    FLYING_THRESHOLD = 3;
    MILL_TOL_REL = min(0.30, 0.05 * (1 + max(0, log10(max(dim, 1) / 6))));

    % Hybrid-exploitation constants
    GS_c       = 0.05;
    delta_stag = 8;
    F_de       = 0.8;
    CR_de      = 0.9;
    n_worst    = max(1, floor(0.2 * SearchAgents_no));
    n_restart  = floor(0.5 * SearchAgents_no);

    % Board topology
    [POINT_ADJ, MILL_LINES] = morabaraba_board();
    N_POINTS = 24;
    N_LINES  = size(MILL_LINES, 1);

    LINE_DIMS = zeros(N_LINES, 3);
    for L = 1:N_LINES
        if dim >= 3
            LINE_DIMS(L, :) = randperm(dim, 3);
        else
            LINE_DIMS(L, :) = mod((L - 1) + (0:2), max(dim, 1)) + 1;
        end
    end

    % Team structure
    n_black = floor(SearchAgents_no / 2);
    n_white = SearchAgents_no - n_black;
    team = [zeros(1, n_black), ones(1, n_white)];

    % Initialisation: chaotic + opposition-based learning
    Positions = chaotic_init(SearchAgents_no, dim, lb, ub);
    OppPos = bsxfun(@plus, lb, bsxfun(@minus, ub, Positions));
    OppPos = max(min(OppPos, ub), lb);

    fv  = inf(1, SearchAgents_no);
    bsf = inf;
    bsx = Positions(1, :);

    for ii = 1:SearchAgents_no
        if nfe + 2 <= MaxNFE
            [f1, nfe] = calculate_fitness(Positions(ii, :)', problem, nfe);
            fv(ii) = f1;
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, f1, Positions(ii, :), bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            [f2, nfe] = calculate_fitness(OppPos(ii, :)', problem, nfe);
            if f2 < f1
                Positions(ii, :) = OppPos(ii, :);
            end
            fv(ii) = min(f1, f2);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, f2, OppPos(ii, :), bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
        end
    end

    cows_on_board  = zeros(1, SearchAgents_no);
    cows_to_place  = repmat(COWS_INITIAL, 1, SearchAgents_no);
    occupied_point = randi(N_POINTS, 1, SearchAgents_no);

    for ii = 1:SearchAgents_no
        Positions(ii, :) = max(min(Positions(ii, :), ub), lb);
        if nfe < MaxNFE
            [fv(ii), nfe] = calculate_fitness(Positions(ii, :)', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, fv(ii), Positions(ii, :), bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
        else
            fv(ii) = realmax;
        end
    end

    [BestScore, bidx] = min(fv);
    BestPosition = Positions(bidx, :);

    stag_count       = 0;
    prev_best        = BestScore;
    quarantine_bp    = [];
    quarantine_count = 0;

    agent_stag   = zeros(1, SearchAgents_no);
    agent_prev_f = fv;
    no_mill_iters = 0;

    agent_mood     = 2 * rand(1, SearchAgents_no) - 1;
    agent_last_dir = zeros(SearchAgents_no, dim);

    MILL_ROSTER_SIZE = 5;
    mill_roster_pos = zeros(MILL_ROSTER_SIZE, dim);
    mill_roster_q   = zeros(1, MILL_ROSTER_SIZE);
    mill_roster_idx = 0;
    mill_roster_cur = 1;

    worst_anchor = lb + 0.5 * (ub - lb);

    % Main game loop
    for t = 1:Max_iter
        if nfe >= MaxNFE, break; end

        if quarantine_count > 0
            CBP = quarantine_bp;
            quarantine_count = quarantine_count - 1;
        else
            CBP = BestPosition;
        end

        agent_mood = 2 * rand(1, SearchAgents_no) - 1;
        [~, worst_overall] = max(fv);
        worst_anchor = Positions(worst_overall, :);

        % Per-agent turn
        for ii = 1:SearchAgents_no
            if nfe >= MaxNFE, break; end

            x_cur = Positions(ii, :);
            f_cur = fv(ii);

            if cows_to_place(ii) > 0
                phase = 1;
            elseif cows_on_board(ii) > FLYING_THRESHOLD
                phase = 2;
            elseif cows_on_board(ii) == FLYING_THRESHOLD
                phase = 3;
            else
                phase = 0;
            end

            switch phase
                case 1
                    x_new = place_cow(x_cur, Positions, team, ii, LINE_DIMS, lb, ub, dim);
                    cows_to_place(ii) = cows_to_place(ii) - 1;
                    cows_on_board(ii) = cows_on_board(ii) + 1;
                    occupied_point(ii) = randi(N_POINTS);
                case 2
                    x_new = move_cow(x_cur, Positions, CBP, occupied_point(ii), POINT_ADJ, ...
                                     LINE_DIMS, lb, ub, dim, t, Max_iter, agent_mood(ii), ...
                                     agent_last_dir(ii, :), mill_roster_pos, mill_roster_q, ...
                                     mill_roster_idx, worst_anchor);
                    adj_list = POINT_ADJ{occupied_point(ii)};
                    if ~isempty(adj_list)
                        occupied_point(ii) = adj_list(randi(numel(adj_list)));
                    end
                case 3
                    x_new = fly_cow(x_cur, CBP, lb, ub, dim, agent_mood(ii), worst_anchor);
                    occupied_point(ii) = randi(N_POINTS);
                otherwise
                    x_new = lb + rand(1, dim) .* (ub - lb);
                    cows_on_board(ii) = 0;
                    cows_to_place(ii) = COWS_INITIAL;
                    occupied_point(ii) = randi(N_POINTS);
            end

            x_new = max(min(x_new, ub), lb);
            if nfe >= MaxNFE, break; end
            [f_new, nfe] = calculate_fitness(x_new', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, f_new, x_new, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);

            % Cheap-then-Levy dual attempt (moving / flying phases only)
            if (phase == 2 || phase == 3) && f_new >= f_cur && nfe < MaxNFE
                x_levy = x_new + rand() * 0.01 * (ub - lb) .* levy_step(dim);
                x_levy = max(min(x_levy, ub), lb);
                [f_levy, nfe] = calculate_fitness(x_levy', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, f_levy, x_levy, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if f_levy < f_new
                    x_new = x_levy;
                    f_new = f_levy;
                end
            end

            if f_new < f_cur
                move_vec = x_new - x_cur;
                move_norm = norm(move_vec);
                if move_norm > 1e-300
                    agent_last_dir(ii, :) = move_vec / move_norm;
                end
                Positions(ii, :) = x_new;
                fv(ii) = f_new;
                if f_new < BestScore
                    BestScore = f_new;
                    BestPosition = x_new;
                end
            end

            % Per-agent stagnation tracking
            if abs(fv(ii) - agent_prev_f(ii)) < 1e-12 * max(abs(agent_prev_f(ii)), 1)
                agent_stag(ii) = agent_stag(ii) + 1;
            else
                agent_stag(ii) = 0;
                agent_prev_f(ii) = fv(ii);
            end

            if agent_stag(ii) >= 15 && nfe < MaxNFE
                agent_stag(ii) = 0;
                jump_scale = 0.3 * (ub - lb);
                x_jump = Positions(ii, :) + jump_scale .* levy_step(dim);
                x_jump = max(min(x_jump, ub), lb);
                [f_jump, nfe] = calculate_fitness(x_jump', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, f_jump, x_jump, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if f_jump < fv(ii)
                    Positions(ii, :) = x_jump;
                    fv(ii) = f_jump;
                    agent_prev_f(ii) = f_jump;
                    if f_jump < BestScore
                        BestScore = f_jump;
                        BestPosition = x_jump;
                    end
                else
                    if rand() < 0.1
                        Positions(ii, :) = x_jump;
                        fv(ii) = f_jump;
                        agent_prev_f(ii) = f_jump;
                    end
                end
            end
        end

        if nfe >= MaxNFE, break; end

        % Mill formation + shooting
        mills_fired_this_iter = 0;
        for tm = [0, 1]
            team_idx = find(team == tm);
            opp_idx  = find(team ~= tm);
            n_team   = numel(team_idx);
            if n_team < 3, continue; end

            for L = 1:N_LINES
                if nfe >= MaxNFE, break; end
                d_idx = LINE_DIMS(L, :);
                n_trials = min(5, nchoosek(n_team, 3));

                for tr = 1:n_trials
                    if nfe >= MaxNFE, break; end
                    triple = team_idx(randperm(n_team, 3));
                    Vt = Positions(triple, d_idx);
                    spread = max(Vt, [], 1) - min(Vt, [], 1);
                    rng_d  = ub(d_idx) - lb(d_idx);
                    rel = spread ./ max(rng_d, 1e-300);

                    if all(rel < MILL_TOL_REL) && ~isempty(opp_idx)
                        mills_fired_this_iter = mills_fired_this_iter + 1;
                        [~, worst_in_opp] = max(fv(opp_idx));
                        victim = opp_idx(worst_in_opp);
                        cows_on_board(victim) = cows_on_board(victim) - 1;

                        mill_avg = mean(Vt, 1);
                        full_mean = mean(Positions(triple, :), 1);
                        quality = max(0, 1 - max(rel));
                        mill_roster_pos(mill_roster_cur, :) = full_mean;
                        mill_roster_q(mill_roster_cur) = quality;
                        mill_roster_cur = mod(mill_roster_cur, MILL_ROSTER_SIZE) + 1;
                        mill_roster_idx = min(mill_roster_idx + 1, MILL_ROSTER_SIZE);

                        diffs = abs(Positions(victim, d_idx) - mill_avg);
                        [~, hit_local] = max(diffs);
                        hit_dim = d_idx(hit_local);

                        x_v = Positions(victim, :);
                        x_v(hit_dim) = mill_avg(hit_local);
                        x_v = max(min(x_v, ub), lb);

                        [f_v, nfe] = calculate_fitness(x_v', problem, nfe);
                        [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                            stamp(nfe, MaxNFE, f_v, x_v, bsf, bsx, curve, Positions, fv, ...
                                  population_history, fitness_history, history_index);
                        if f_v < fv(victim)
                            Positions(victim, :) = x_v;
                            fv(victim) = f_v;
                            if f_v < BestScore
                                BestScore = f_v;
                                BestPosition = x_v;
                            end
                        end
                    end
                end
            end
        end

        % Mill-rate stagnation trigger
        if mills_fired_this_iter > 0
            no_mill_iters = 0;
        else
            no_mill_iters = no_mill_iters + 1;
        end

        if no_mill_iters >= 20 && nfe < MaxNFE
            no_mill_iters = 0;
            f_black = mean(fv(team == 0));
            f_white = mean(fv(team == 1));
            if f_black > f_white
                refresh_team = 0;
            else
                refresh_team = 1;
            end
            team_to_refresh = find(team == refresh_team);
            n_refresh = floor(numel(team_to_refresh) / 2);
            [~, sort_w] = sort(fv(team_to_refresh), 'descend');
            refresh_idx = team_to_refresh(sort_w(1:n_refresh));

            for kk = 1:numel(refresh_idx)
                if nfe >= MaxNFE, break; end
                ri = refresh_idx(kk);
                x_refresh = BestPosition + 0.4 * (ub - lb) .* levy_step(dim);
                x_refresh = max(min(x_refresh, ub), lb);
                [f_refresh, nfe] = calculate_fitness(x_refresh', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, f_refresh, x_refresh, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                Positions(ri, :) = x_refresh;
                fv(ri) = f_refresh;
                cows_on_board(ri) = 0;
                cows_to_place(ri) = COWS_INITIAL;
                agent_stag(ri) = 0;
                agent_prev_f(ri) = f_refresh;
                if f_refresh < BestScore
                    BestScore = f_refresh;
                    BestPosition = x_refresh;
                end
            end
        end

        % Hybrid exploitation suite

        % E1: dual-scale elite Gaussian zoom
        fg = max(BestScore, 0);
        gs_sig = max(min(GS_c * sqrt(fg / max(dim, 1)), GS_c * fg / max(dim, 1)), 1e-300);
        n_gs = max(10, round(200 * (t / Max_iter) ^ 2));
        for trial = 1:n_gs
            if nfe >= MaxNFE, break; end
            cand = max(min(BestPosition + gs_sig .* randn(1, dim), ub), lb);
            [fc, nfe] = calculate_fitness(cand', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            if fc < BestScore
                BestScore = fc;
                BestPosition = cand;
                fg = max(BestScore, 0);
                gs_sig = max(min(GS_c * sqrt(fg / max(dim, 1)), GS_c * fg / max(dim, 1)), 1e-300);
            end
        end

        % E14: deep-precision multiplicative polish
        if BestScore < 1e-3 && nfe + 50 < MaxNFE
            n_polish = min(100, max(20, floor((MaxNFE - nfe) / 100)));
            for trial = 1:n_polish
                if nfe >= MaxNFE, break; end
                eta = 10 ^ (-2 - 4 * rand());
                cand = BestPosition .* (1 - eta * randn(1, dim));
                cand = max(min(cand, ub), lb);
                [fc, nfe] = calculate_fitness(cand', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fc < BestScore, BestScore = fc; BestPosition = cand; end
            end
            for trial = 1:20
                if nfe >= MaxNFE, break; end
                shrink = 0.95 - 0.5 * rand();
                cand = max(min(BestPosition * shrink, ub), lb);
                [fc, nfe] = calculate_fitness(cand', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fc < BestScore, BestScore = fc; BestPosition = cand; end
            end
        end

        % E15: Levy-flight precision polish
        if BestScore < 1e-1 && nfe + 30 < MaxNFE
            for trial = 1:30
                if nfe >= MaxNFE, break; end
                lstep = levy_step(dim);
                step_scale = max(abs(BestScore) ^ 0.5, 1e-200);
                cand = max(min(BestPosition + step_scale * lstep .* abs(BestPosition), ub), lb);
                [fc, nfe] = calculate_fitness(cand', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fc < BestScore, BestScore = fc; BestPosition = cand; end
            end
        end

        % E2: medium Gaussian
        n_med = 20; if dim > 4, n_med = 5; end
        sm = (ub - lb) / 20;
        for trial = 1:n_med
            if nfe >= MaxNFE, break; end
            cand = max(min(BestPosition + sm .* randn(1, dim), ub), lb);
            [fc, nfe] = calculate_fitness(cand', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            if fc < BestScore, BestScore = fc; BestPosition = cand; end
        end

        % E3: random dimension-shrink
        nd = max(1, floor(dim / 5));
        for trial = 1:5
            if nfe >= MaxNFE, break; end
            d_idx2 = randperm(dim, nd);
            xnew = BestPosition; xnew(d_idx2) = xnew(d_idx2) * 0.5;
            xnew = max(min(xnew, ub), lb);
            [fc, nfe] = calculate_fitness(xnew', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, fc, xnew, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            if fc < BestScore, BestScore = fc; BestPosition = xnew; end
        end

        % E4: targeted max-dimension shrink
        if nfe < MaxNFE
            [~, max_dim] = max(abs(BestPosition));
            xnew = BestPosition; xnew(max_dim) = xnew(max_dim) * 0.1;
            xnew = max(min(xnew, ub), lb);
            [fc, nfe] = calculate_fitness(xnew', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, fc, xnew, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            if fc < BestScore, BestScore = fc; BestPosition = xnew; end
        end

        % E7: adaptive dimension-wise line search
        if mod(t, 20) == 0 && nfe + 6 * dim < MaxNFE && dim <= 100
            d_perm = randperm(dim);
            for di = d_perm
                if nfe + 6 >= MaxNFE, break; end
                rng_dd = ub(di) - lb(di);
                cur_v = BestPosition(di);
                probes = [lb(di) + 0.10 * rng_dd, lb(di) + 0.50 * rng_dd, lb(di) + 0.90 * rng_dd, ...
                          cur_v + 0.05 * rng_dd * randn(), cur_v - 0.05 * rng_dd * randn(), ...
                          cur_v + 0.20 * rng_dd * randn()];
                probes = max(min(probes, ub(di)), lb(di));
                for pi_ = 1:numel(probes)
                    if nfe >= MaxNFE, break; end
                    xnew = BestPosition; xnew(di) = probes(pi_);
                    xnew = max(min(xnew, ub), lb);
                    [fc, nfe] = calculate_fitness(xnew', problem, nfe);
                    [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                        stamp(nfe, MaxNFE, fc, xnew, bsf, bsx, curve, Positions, fv, ...
                              population_history, fitness_history, history_index);
                    if fc < BestScore, BestScore = fc; BestPosition = xnew; end
                end
            end
        end

        % E8: multi-attractor probe (low-dimensional multi-start)
        if dim <= 6 && mod(t, 8) == 0 && nfe + 200 < MaxNFE
            for ai = 1:15
                if nfe >= MaxNFE, break; end
                if mod(ai, 2) == 0
                    x_seed = lb + rand(1, dim) .* (ub - lb);
                else
                    perm_idx = randperm(dim);
                    strata = (perm_idx - 1 + rand(1, dim)) / dim;
                    x_seed = lb + strata .* (ub - lb);
                end
                x_seed = max(min(x_seed, ub), lb);
                [f_seed, nfe] = calculate_fitness(x_seed', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, f_seed, x_seed, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);

                zoom_sigma = 0.08 * (ub - lb);
                for zi = 1:10
                    if nfe >= MaxNFE, break; end
                    cand = max(min(x_seed + zoom_sigma .* randn(1, dim), ub), lb);
                    [fc, nfe] = calculate_fitness(cand', problem, nfe);
                    [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                        stamp(nfe, MaxNFE, fc, cand, bsf, bsx, curve, Positions, fv, ...
                              population_history, fitness_history, history_index);
                    if fc < f_seed
                        x_seed = cand; f_seed = fc;
                        zoom_sigma = zoom_sigma * 0.65;
                    end
                end
                if f_seed < BestScore
                    BestScore = f_seed;
                    BestPosition = x_seed;
                end
            end
        end

        % E9: boundary probe
        if mod(t, 25) == 0 && nfe + 6 < MaxNFE
            for trial = 1:6
                if nfe >= MaxNFE, break; end
                d_pick = randi(dim);
                xnew = BestPosition;
                if rand() < 0.5
                    xnew(d_pick) = ub(d_pick) - rand() * 0.1 * (ub(d_pick) - lb(d_pick));
                else
                    xnew(d_pick) = lb(d_pick) + rand() * 0.1 * (ub(d_pick) - lb(d_pick));
                end
                xnew = max(min(xnew, ub), lb);
                [fc, nfe] = calculate_fitness(xnew', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fc, xnew, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fc < BestScore, BestScore = fc; BestPosition = xnew; end
            end
        end

        % E10: golden-section coordinate-descent polish (late stage)
        if t > 0.7 * Max_iter && mod(t, 15) == 0 && nfe + 4 * dim < MaxNFE && dim <= 100
            phi_g = (sqrt(5) - 1) / 2;
            for di = randperm(dim)
                if nfe + 4 >= MaxNFE, break; end
                cur_v = BestPosition(di);
                rng_dd = ub(di) - lb(di);
                a = max(lb(di), cur_v - 0.05 * rng_dd);
                b = min(ub(di), cur_v + 0.05 * rng_dd);
                if b - a < 1e-15, continue, end
                x1 = b - phi_g * (b - a);
                x2 = a + phi_g * (b - a);
                xa = BestPosition; xa(di) = x1; xa = max(min(xa, ub), lb);
                xb = BestPosition; xb(di) = x2; xb = max(min(xb, ub), lb);
                [fa, nfe] = calculate_fitness(xa', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fa, xa, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                [fb, nfe] = calculate_fitness(xb', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fb, xb, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fa < BestScore, BestScore = fa; BestPosition = xa; end
                if fb < BestScore, BestScore = fb; BestPosition = xb; end
                if fa < fb
                    xc = BestPosition; xc(di) = (a + x1) / 2;
                else
                    xc = BestPosition; xc(di) = (x2 + b) / 2;
                end
                xc = max(min(xc, ub), lb);
                [fc, nfe] = calculate_fitness(xc', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, fc, xc, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if fc < BestScore, BestScore = fc; BestPosition = xc; end
            end
        end

        % E11: centroid-pull probe
        if mod(t, 12) == 0 && nfe + 6 < MaxNFE
            centroid = mean(Positions, 1);
            direction = BestPosition - centroid;
            dnorm = norm(direction);
            if dnorm > 1e-15
                direction = direction / dnorm;
                step_sizes = 0.05 * (1 - t / Max_iter) * (ub - lb);
                for sign_d = [1, -1]
                    if nfe >= MaxNFE, break; end
                    for scale = [0.5, 1.0, 2.0]
                        if nfe >= MaxNFE, break; end
                        xnew = max(min(BestPosition + sign_d * scale * step_sizes .* direction, ub), lb);
                        [fc, nfe] = calculate_fitness(xnew', problem, nfe);
                        [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                            stamp(nfe, MaxNFE, fc, xnew, bsf, bsx, curve, Positions, fv, ...
                                  population_history, fitness_history, history_index);
                        if fc < BestScore, BestScore = fc; BestPosition = xnew; end
                    end
                end
            end
        end

        % E12: adaptive-jump Levy burst
        if mod(t, 10) == 0 && nfe + 4 < MaxNFE
            jump_strength = 2 * (1 - rand());
            donor = Positions(randi(SearchAgents_no), :);
            X1 = max(min(BestPosition - jump_strength * (donor - BestPosition), ub), lb);
            [f1b, nfe] = calculate_fitness(X1', problem, nfe);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(nfe, MaxNFE, f1b, X1, bsf, bsx, curve, Positions, fv, ...
                      population_history, fitness_history, history_index);
            if f1b < BestScore
                BestScore = f1b; BestPosition = X1;
            elseif nfe < MaxNFE
                X2 = max(min(X1 + 0.1 * (ub - lb) .* levy_step(dim), ub), lb);
                [f2b, nfe] = calculate_fitness(X2', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, f2b, X2, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if f2b < BestScore, BestScore = f2b; BestPosition = X2; end
            end
        end

        % E5: stagnation DE/best/1 rescue
        if BestScore < prev_best * (1 - 1e-12)
            stag_count = 0;
        else
            stag_count = stag_count + 1;
        end
        prev_best = BestScore;

        if stag_count >= delta_stag && nfe < MaxNFE
            stag_count = 0;
            [~, wsidx] = sort(fv, 'descend');
            for kk = 1:n_worst
                if nfe >= MaxNFE, break; end
                wi = wsidx(kk);
                rest = 1:SearchAgents_no; rest(wi) = [];
                idx2 = rest(randperm(numel(rest), 2));
                mutant = max(min(BestPosition + F_de * (Positions(idx2(1), :) - Positions(idx2(2), :)), ub), lb);
                mask = rand(1, dim) < CR_de;
                if ~any(mask), mask(randi(dim)) = true; end
                tv = Positions(wi, :); tv(mask) = mutant(mask);
                tv = max(min(tv, ub), lb);
                [ft, nfe] = calculate_fitness(tv', problem, nfe);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(nfe, MaxNFE, ft, tv, bsf, bsx, curve, Positions, fv, ...
                          population_history, fitness_history, history_index);
                if ft < fv(wi)
                    Positions(wi, :) = tv;
                    fv(wi) = ft;
                    if ft < BestScore, BestScore = ft; BestPosition = tv; end
                end
            end
        end

        % E6: diversity restart + quarantine
        if mod(t, 10) == 0 && t > Max_iter / 4 && nfe < MaxNFE
            if std(fv) / (abs(mean(fv)) + 1e-300) < 0.05
                NewBlk = chaotic_init(n_restart, dim, lb, ub);
                [~, wsidx2] = sort(fv, 'descend');
                for kk = 1:n_restart
                    if nfe >= MaxNFE, break; end
                    Positions(wsidx2(kk), :) = NewBlk(kk, :);
                    [fnew, nfe] = calculate_fitness(NewBlk(kk, :)', problem, nfe);
                    fv(wsidx2(kk)) = fnew;
                    [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                        stamp(nfe, MaxNFE, fnew, NewBlk(kk, :), bsf, bsx, curve, Positions, fv, ...
                              population_history, fitness_history, history_index);
                    cows_on_board(wsidx2(kk)) = 0;
                    cows_to_place(wsidx2(kk)) = COWS_INITIAL;
                end
                if dim <= 6 && t < Max_iter * 0.75 && quarantine_count == 0
                    [~, bni] = min(fv(wsidx2(1:n_restart)));
                    bnp = Positions(wsidx2(bni), :);
                    if norm(bnp - BestPosition) > mean(ub - lb) * 0.05
                        quarantine_bp = bnp;
                        quarantine_count = 25;
                    end
                end
            end
        end
    end

    curve(min(nfe, MaxNFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Helper functions

function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, x, bsf, bsx, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        % +Inf is the not-yet-evaluated sentinel of fv, so the row waits for the
        % first full sweep; -Inf is a legitimate optimum and must not gate it
        if ~any(Fit == Inf)
            [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

function [adj, lines] = morabaraba_board()
    lines = [
        1  2  3;     3  4  5;     5  6  7;     7  8  1;
        9 10 11;    11 12 13;    13 14 15;    15 16  9;
       17 18 19;    19 20 21;    21 22 23;    23 24 17;
        2 10 18;     4 12 20;     6 14 22;     8 16 24;
    ];
    adj = cell(24, 1);
    for L = 1:size(lines, 1)
        a = lines(L, 1); b = lines(L, 2); c = lines(L, 3);
        adj{a} = unique([adj{a}, b]);
        adj{b} = unique([adj{b}, a, c]);
        adj{c} = unique([adj{c}, b]);
    end
    diags = [1 9 17; 3 11 19; 5 13 21; 7 15 23];
    for L = 1:size(diags, 1)
        a = diags(L, 1); b = diags(L, 2); c = diags(L, 3);
        adj{a} = unique([adj{a}, b]);
        adj{b} = unique([adj{b}, a, c]);
        adj{c} = unique([adj{c}, b]);
    end
    lines = [lines; diags];
end

function x_new = place_cow(x_cur, Positions, team, ii, LINE_DIMS, lb, ub, dim)
    teammates = find(team == team(ii) & (1:numel(team)) ~= ii);
    opponents = find(team ~= team(ii));
    x_new = x_cur;

    found_near_mill = false;
    if numel(teammates) >= 2
        N_LINES = size(LINE_DIMS, 1);
        line_order = randperm(N_LINES);
        for L = line_order
            d_idx = LINE_DIMS(L, :);
            pair = teammates(randperm(numel(teammates), min(2, numel(teammates))));
            if numel(pair) < 2, continue, end
            v1 = Positions(pair(1), d_idx);
            v2 = Positions(pair(2), d_idx);
            rng_d = ub(d_idx) - lb(d_idx);
            if all(abs(v1 - v2) ./ max(rng_d, 1e-300) < 0.15)
                target = 0.5 * (v1 + v2);
                x_new(d_idx) = target + 0.05 * rng_d .* randn(1, 3);
                found_near_mill = true;
                break
            end
        end
    end

    if ~found_near_mill
        x_rand = lb + rand(1, dim) .* (ub - lb);
        if ~isempty(teammates) && rand() < 0.5
            mate = teammates(randi(numel(teammates)));
            x_new = 0.5 * (x_cur + Positions(mate, :)) + 0.1 * (ub - lb) .* randn(1, dim);
        elseif ~isempty(opponents)
            opp_centroid = mean(Positions(opponents, :), 1);
            push_dir = x_rand - opp_centroid;
            nrm = norm(push_dir) + 1e-12;
            x_new = x_rand + 0.1 * (ub - lb) .* (push_dir / nrm);
        else
            x_new = x_rand;
        end
    end
end

function x_new = move_cow(x_cur, Positions, BestPosition, point, POINT_ADJ, LINE_DIMS, ...
                          lb, ub, dim, t, Max_iter, mood, last_dir, ...
                          mill_roster_pos, mill_roster_q, mill_roster_idx, worst_anchor)
    adj_list = POINT_ADJ{point};
    if isempty(adj_list)
        sigma = 0.05 * (ub - lb) * (1 - t / Max_iter);
        x_new = x_cur + sigma .* randn(1, dim);
        return
    end

    progress = t / Max_iter;
    boldness  = abs(mood);
    sign_mood = sign(mood + 1e-300);

    % Multi-leader attractor selection (mill roster), dim >= 10 only
    target = BestPosition;
    if dim >= 10 && mill_roster_idx > 0 && rand() < 0.4
        valid_q = mill_roster_q(1:mill_roster_idx);
        total_q = sum(valid_q);
        if total_q > 1e-300
            cumprob = cumsum(valid_q) / total_q;
            sel = find(rand() <= cumprob, 1, 'first');
            if isempty(sel), sel = mill_roster_idx; end
            target = mill_roster_pos(sel, :);
        end
    end

    % Candidate A: structural board-line move
    N_LINES = size(LINE_DIMS, 1);
    chosen_line = randi(N_LINES);
    step_factor = 0.5 * (1 + cos(pi * progress));
    sigma = step_factor * 0.3 .* (ub - lb) * (0.7 + 0.3 * boldness);

    d_idx = LINE_DIMS(chosen_line, :);
    x_A = x_cur;
    pull_A = (0.4 + 0.1 * boldness) * (target(d_idx) - x_cur(d_idx));
    x_A(d_idx) = x_cur(d_idx) + pull_A + sigma(d_idx) .* randn(1, 3);
    other_dims = setdiff(1:dim, d_idx);
    if ~isempty(other_dims)
        x_A(other_dims) = x_cur(other_dims) + 0.1 * sigma(other_dims) .* randn(1, numel(other_dims));
    end

    % Candidate B: fitness-driven full-dim DE-style pull
    n_pop = size(Positions, 1);
    r1 = randi(n_pop); r2 = randi(n_pop);
    while r1 == r2, r2 = randi(n_pop); end
    F_scale = 0.5 + 0.3 * rand();
    diff_vec = Positions(r1, :) - Positions(r2, :);
    pull_B = (1 - 0.5 * progress) * (BestPosition - x_cur);
    x_B = x_cur + pull_B + F_scale * diff_vec + 0.05 * (1 - progress) * (ub - lb) .* randn(1, dim);

    % Candidate C: logarithmic spiral around the leader
    b_spiral = 1;
    a2 = -1 - progress;
    l_spiral = (a2 - 1) * rand() + 1;
    dist_to_leader = abs(BestPosition - x_cur);
    x_C = dist_to_leader .* exp(b_spiral * l_spiral) .* cos(l_spiral * 2 * pi) + BestPosition;

    p_struct  = 0.50 - 0.40 * progress;
    p_fitness = 0.30;
    r = rand();
    if r < p_struct
        x_new = x_A;
    elseif r < p_struct + p_fitness
        x_new = x_B;
    else
        x_new = x_C;
    end

    % Inertia (velocity persistence), dim >= 10 only
    if dim >= 10 && any(last_dir ~= 0)
        inertia_mag = 0.15 * progress * boldness * mean(ub - lb);
        x_new = x_new + inertia_mag * last_dir;
    end

    % Retreat from worst, dim >= 10 only
    if sign_mood < 0 && dim >= 10
        to_worst = worst_anchor - x_new;
        dist_to_worst = norm(to_worst);
        threshold = 0.05 * mean(ub - lb);
        if dist_to_worst > 1e-300 && dist_to_worst < threshold
            deflect_mag = 0.05 * boldness * mean(ub - lb);
            x_new = x_new - deflect_mag * (to_worst / dist_to_worst);
        end
    end
end

function x_new = fly_cow(x_cur, BestPosition, lb, ub, dim, mood, worst_anchor)
    boldness  = abs(mood);
    sign_mood = sign(mood + 1e-300);
    jump_scale = 0.3 + 0.4 * boldness;

    if rand() < 0.7
        lstep = levy_step(dim);
        x_new = BestPosition + jump_scale .* (ub - lb) .* lstep;
    else
        x_new = lb + rand(1, dim) .* (ub - lb);
    end

    if sign_mood < 0 && dim >= 10
        to_worst = worst_anchor - x_new;
        d = norm(to_worst);
        threshold = 0.05 * mean(ub - lb);
        if d > 1e-300 && d < threshold
            x_new = x_new - 0.05 * boldness * mean(ub - lb) * (to_worst / d);
        end
    end
end

function step = levy_step(dim)
    alpha = 1.5;
    sigma = (gamma(1 + alpha) * sin(pi * alpha / 2) / ...
             (gamma((1 + alpha) / 2) * alpha * 2 ^ ((alpha - 1) / 2))) ^ (1 / alpha);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    raw = u ./ (abs(v) + 1e-100) .^ (1 / alpha);
    step = 0.01 * sign(raw) .* min(abs(raw), 2);
end

function X = chaotic_init(n, dim, lb, ub)
    X = zeros(n, dim);
    x = rand(1, dim);
    for i = 1:n
        x = 2 * x .* (x < 0.5) + 2 * (1 - x) .* (x >= 0.5);
        x = min(max(x, 1e-6), 1 - 1e-6);
        X(i, :) = lb + x .* (ub - lb);
    end
end

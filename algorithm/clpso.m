% ----------------------------------------------------------------------- %
% Comprehensive Learning Particle Swarm Optimizer (CLPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   ps = 40                      % Swarm size
%   c  = 1.49445                 % Single acceleration constant
%   w = 0.9 -> 0.2 (linear)      % Inertia weight, over the planned run
%   Vmax = 0.2 * (ub - lb)       % Velocity clamp
%   Pc_k = 0 -> 0.5              % Learning probability, exponential in particle index k
%   refreshing gap = 5           % Generations without pbest improvement
%
% Algorithm Concept:
%   - The global best is removed entirely; that term is what makes standard PSO
%     commit the whole swarm to one basin early
%   - EACH DIMENSION learns from a possibly different particle's pbest:
%         v_i^d = w*v_i^d + c*rand*( pbest_{f_i(d)}^d - x_i^d )
%     f_i(d) is drawn per dimension: with probability Pc_i a binary tournament
%     between two random particles, otherwise the particle's own pbest
%   - A particle can therefore be pulled towards up to D exemplars at once,
%     which is where the diversity comes from
%   - Pc_i grows with the particle index, spanning conservative to exploratory
%   - Exemplars are redrawn only after `refreshing gap` generations without a
%     pbest improvement; a particle outside the box is left unevaluated
%
% Reference:
% J. J. Liang, A. K. Qin, P. N. Suganthan, S. Baskar,
% Comprehensive Learning Particle Swarm Optimizer for Global Optimization of
% Multimodal Functions,
% IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, 2006.
% https://doi.org/10.1109/TEVC.2005.857610
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the MATLAB release distributed with Y. Wang's CoDE package, whose
% Readme credits the source to co-author P. N. Suganthan. That release carries a
% vestigial gbest term gated by a mask of size m(k) = 0, so it never contributes;
% it is correct CLPSO and is left as the authors wrote it.
% BUDGET: out-of-box particles are not evaluated, so the reference's fixed
% maxFES/ps generation loop under-spends. Its no-op `i = i-1` guard shows the
% intent was to run until the budget was consumed, so `while FE < maxFe` is used
% and the inertia index clamped, leaving w at its final 0.2.
% Clamping is NOT applied: it would put the particle on the bound, count it as
% inside and empty the skip set, so only evaluated particles are recorded.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = clpso(problem)

    n     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    ps = 40;
    cc = [1.49445 1.49445];
    m  = zeros(ps, 1);          % refreshing dimensions for the (inert) gbest term
    refresh_gap = 5;

    me  = max(2, floor(maxFE / ps));
    iwt = 0.9 - (1:me) * (0.7 / me);

    t  = 5 * (0:1/(ps-1):1);
    Pc = (exp(t) - exp(t(1))) ./ (exp(t(ps)) - exp(t(1))) * 0.5;

    mv   = 0.2 * (ub - lb);
    Vmin = repmat(-mv, ps, 1);
    Vmax = repmat(mv, ps, 1);
    LBm  = repmat(lb, ps, 1);
    UBm  = repmat(ub, ps, 1);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    pos = LBm + (UBm - LBm) .* rand(ps, n);
    [e, FE] = calculate_fitness(pos', problem, FE);
    e = e(:);

    vel = Vmin + 2 .* Vmax .* rand(ps, n);

    pbest    = pos;
    pbestval = e;

    [gbestval, gbestid] = min(pbestval);
    gbest = pbest(gbestid, :);

    bsf  = gbestval;
    bsfx = gbest;
    for i = 1:ps
        if i <= maxFE
            curve(i) = min(e(1:i));
            [population_history, fitness_history, history_index] = record_history(...
                i, pos, e, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    stay_num = zeros(ps, 1);
    ai       = zeros(ps, n);
    f_pbest  = repmat((1:ps)', 1, n);
    colidx   = repmat(1:n, ps, 1);

    for k = 1:ps
        [ai(k, :), f_pbest(k, :)] = drawExemplars(k, ps, n, m(k), Pc(k), pbestval);
    end

    it = 1;

    % Main loop
    while FE < maxFE
        it = it + 1;
        w  = iwt(min(it, me));

        % Refresh the exemplars of stagnated particles
        stale = find(stay_num >= refresh_gap);
        for kk = 1:numel(stale)
            k = stale(kk);
            stay_num(k) = 0;
            [ai(k, :), f_pbest(k, :)] = drawExemplars(k, ps, n, m(k), Pc(k), pbestval);
        end

        % Comprehensive learning velocity update
        pbest_f = pbest(sub2ind([ps n], f_pbest, colidx));
        gbestrep = repmat(gbest, ps, 1);

        aa  = cc(1) .* (1 - ai) .* rand(ps, n) .* (pbest_f - pos) ...
            + cc(2) .* ai       .* rand(ps, n) .* (gbestrep - pos);
        vel = w .* vel + aa;
        vel = min(max(vel, Vmin), Vmax);
        pos = pos + vel;

        % Only particles still inside the box are evaluated
        valid = find(all(pos <= UBm & pos >= LBm, 2));

        if ~isempty(valid)
            nv = numel(valid);
            [ev, FE] = calculate_fitness(pos(valid, :)', problem, FE);
            ev = ev(:);
            e(valid) = ev;

            for kk = 1:nv
                k = valid(kk);
                if pbestval(k) <= ev(kk)
                    stay_num(k) = stay_num(k) + 1;
                else
                    pbest(k, :)  = pos(k, :);
                    pbestval(k)  = ev(kk);
                    if pbestval(k) < gbestval
                        gbest    = pbest(k, :);
                        gbestval = pbestval(k);
                    end
                end
                if ev(kk) < bsf
                    bsf  = ev(kk);
                    bsfx = pos(k, :);
                end

                ec = FE - nv + kk;
                if ec >= 1 && ec <= maxFE
                    curve(ec) = bsf;
                    % An unevaluated particle has no current fitness, so it is
                    % left out rather than paired with an earlier position's
                    [population_history, fitness_history, history_index] = record_history(...
                        ec, pos(valid, :), e(valid), population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [ai_k, f_k] = drawExemplars(k, ps, n, mk, Pck, pbestval)
% Per-dimension exemplar: w.p. Pck a binary pbest tournament winner, else own pbest (>=1 forced)
    ai_k = zeros(1, n);
    if mk > 0
        ar = randperm(n);
        ai_k(ar(1:mk)) = 1;
    end

    fi1 = ceil(ps * rand(1, n));
    fi2 = ceil(ps * rand(1, n));
    fi  = (pbestval(fi1) < pbestval(fi2))' .* fi1 + (pbestval(fi1) >= pbestval(fi2))' .* fi2;

    bi = ceil(rand(1, n) - 1 + Pck);
    if all(bi == 0)
        rc = randperm(n);
        bi(rc(1)) = 1;
    end

    own = (1:ps)';
    f_k = bi .* fi + (1 - bi) .* (own(k) * ones(1, n));
end

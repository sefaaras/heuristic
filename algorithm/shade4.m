% ----------------------------------------------------------------------- %
% SHADE with Four Competing Strategies (SHADE4)
% CEC 2016 competition -- 8th place (Friedman ranking, CEC 2014 suite)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 100, H = 100               % Population size (fixed) and memory slots
%   MF = MCR = 0.5                 % Initial memory contents
%   CR ~ N(MCR, sqrt(0.1)), F ~ Cauchy(MF, 0.1)  % Parameter sampling
%   p = round(N*U(2/N, 0.2))       % pbest pool size, redrawn per trial
%   n0 = 2, delta = 1/(5*4)        % Strategy counters' start; reset below this share
%   |A| <= N                       % Archive of replaced parents
%
% Algorithm Concept:
%   - Four strategies compete per trial: current-to-pbest/1 with archive and
%     randrl/1 (base = best of three random members), each with bin or exp crossover
%   - A strategy is drawn by roulette on its success count + n0; all counts reset to
%     n0 once any strategy's share falls below 1/20
%   - SHADE memory: F and CR from a random slot, slot k rewritten after each
%     generation (CR by weighted mean, F by weighted Lehmer mean)
%   - Trials replace their parent in place, so later trials of the same generation
%     already see them; out-of-box coordinates are reflected until inside
%
% Reference:
% Petr Bujok, Josef Tvrdik, Radka Polakova,
% Evaluating the performance of SHADE with competing strategies on CEC 2014
% single-parameter test suite,
% 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 5002-5009.
% https://doi.org/10.1109/CEC.2016.7748322
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (web.osu.cz/~Bujok/files/shade4.zip:
% SHADE4.m and its operator files, written for cec15_func; N = H = 100 from its
% start script). Kept as released: the pbest pool is the first p ROWS of the
% population, not the p best -- `sortrows(pom, D + 1)` discards its result -- so the
% pbest term pulls towards random early-index members. The memory index wraps at N
% (equal to H here). Removed: the stop at the known optimum. Repeated reflection is
% computed in closed form (same point); a memory update that comes out non-finite
% keeps the old slot values.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = shade4(problem)

    D     = problem.dimension;
    a     = problem.lb(:)';
    b     = problem.ub(:)';
    maxFE = problem.maxFe;

    N  = 100;
    H  = N;
    n0 = 2;
    h  = 4;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    N  = min(N, maxFE);
    P  = a + rand(N, D) .* (b - a);
    [fv, FE] = calculate_fitness(P', problem, FE);
    fP = fv(:);

    bsf  = inf;
    bsfx = P(1, :);
    for e = 1:N
        if fP(e) < bsf
            bsf  = fP(e);
            bsfx = P(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, P, fP, population_history, fitness_history, history_index, maxFE);
    end

    MF  = 0.5 * ones(1, H);
    MCR = 0.5 * ones(1, H);
    k   = 1;
    A   = zeros(0, D);
    ni  = n0 * ones(1, h);

    while FE < maxFE
        Fpole    = -ones(1, N);
        CRpole   = -ones(1, N);
        deltafce = -ones(1, N);
        SCR = [];
        SF  = [];
        for i = 1:N
            if FE >= maxFE
                break;
            end
            r  = 1 + fix(rand * H);
            CR = MCR(r) + sqrt(0.1) * randn;
            CR = min(max(CR, 0), 1);
            F = -1;
            while F <= 0
                F = 0.1 * tan(rand * pi - pi / 2) + MF(r);
            end
            F = min(F, 1);
            Fpole(i)  = F;
            CRpole(i) = CR;

            % roulete.m on the success counts; the reset follows the draw, as released
            cp = cumsum(ni) / sum(ni);
            hh = 1 + fix(sum(cp < rand));
            if min(ni) / sum(ni) < 1 / (5 * h)
                ni = n0 * ones(1, h);
            end

            if hh <= 2
                y = cur_to_pbest(A, P, F, CR, i, hh == 2);
            else
                y = rand_rl(P, fP, F, CR, i, hh == 4);
            end
            y = reflect_fold(y, a, b);

            [fv, FE] = calculate_fitness(y', problem, FE);
            fy = fv(1);
            if fy < bsf
                bsf  = fy;
                bsfx = y;
            end

            if fy < fP(i)
                deltafce(i) = fP(i) - fy;
                ni(hh) = ni(hh) + 1;
                if size(A, 1) < N
                    A = [A; P(i, :)]; %#ok<AGROW>
                else
                    A(1 + fix(rand * size(A, 1)), :) = P(i, :);
                end
                SCR = [SCR, CRpole(i)]; %#ok<AGROW>
                SF  = [SF, Fpole(i)]; %#ok<AGROW>
            end
            if fy <= fP(i)
                P(i, :) = y;
                fP(i)   = fy;
            end

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, P, fP, population_history, fitness_history, history_index, maxFE);
        end

        if ~isempty(SF)
            delty = deltafce(deltafce ~= -1);
            w = delty / sum(delty);
            mcr = sum(w .* SCR);
            mf  = sum(w .* SF .* SF) / sum(w .* SF);
            if isfinite(mcr) && isfinite(mf)
                MCR(k) = mcr;
                MF(k)  = mf;
            end
            k = k + 1;
            if k > N
                k = 1;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function y = cur_to_pbest(A, P, F, CR, i, use_exp)
% currenttopbestbin_izrc / currenttopbestexp_izrc: pbest drawn from the first p rows (unsorted)
    [N, D] = size(P);
    pmin = 2 / N;
    p = round((pmin + (0.2 - pmin) * rand) * N);
    xpbest = P(1 + fix(p * rand), :);
    xi = P(i, :);
    vyb = draw_except(N, i);
    r1 = P(vyb, :);
    sjed = [P; A];
    r2 = sjed(draw_except(N + size(A, 1), [i, vyb]), :);
    v = xi + F * (xpbest - xi) + F * (r1 - r2);
    y = xi;
    if use_exp
        change = exp_positions(D, CR);
    else
        change = find(rand(1, D) < CR);
        if isempty(change)
            change = 1 + fix(D * rand);
        end
    end
    y(change) = v(change);
end

function y = rand_rl(P, fP, F, CR, i, use_exp)
% rand_RL_bin / rand_RL_exp: base vector is the best of three random members other than i
    [N, D] = size(P);
    y = P(i, :);
    vyb = draw_except(N, i, 3);
    [~, im] = min(fP(vyb));
    r1 = P(vyb(im), :);
    vyb(im) = [];
    v = r1 + F * (P(vyb(1), :) - P(vyb(2), :));
    if use_exp
        change = exp_positions(D, CR);
    else
        change = find(rand(1, D) < CR);
        if isempty(change)
            change = 1 + fix(D * rand);
        end
    end
    y(change) = v(change);
end

function change = exp_positions(D, CR)
% Exponential crossover: a run of consecutive (cyclic) coordinates from a random start
    L = 1 + fix(D * rand);
    change = L;
    position = L;
    while rand < CR && numel(change) < D
        position = position + 1;
        if position <= D
            change(end + 1) = position; %#ok<AGROW>
        else
            change(end + 1) = mod(position, D); %#ok<AGROW>
        end
    end
end

function idx = draw_except(n, expt, k)
% nahvyb_expt: k distinct draws from 1..n without the listed indices
    if nargin < 3
        k = 1;
    end
    pool = 1:n;
    pool(expt) = [];
    idx = zeros(1, k);
    for t = 1:k
        j = 1 + fix(rand * numel(pool));
        idx(t) = pool(j);
        pool(j) = [];
    end
end

function x = reflect_fold(x, lb, ub)
% Closed form of zrcad.m (reflect at the violated bound until inside): triangle wave of period 2*(ub - lb)
    out = x < lb | x > ub;
    if any(out)
        w = ub(out) - lb(out);
        y = mod(x(out) - lb(out), 2 * w);
        y(y > w) = 2 * w(y > w) - y(y > w);
        y(w == 0) = 0;
        x(out) = min(max(lb(out) + y, lb(out)), ub(out));
    end
end

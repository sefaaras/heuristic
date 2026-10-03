% ----------------------------------------------------------------------- %
% Competitive Differential Evolution with 12 strategies (b6e6rl)
% CEC 2013 competition -- 10th place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 30 (D <= 10), 50 (D > 10)  % Population size
%   h = 12, n0 = 2                  % Competing strategies; initial success count of each
%   delta = 1/(5*h)                 % All counts reset when one share drops below this
%   F = 0.5 or 0.8                  % Scale factor, fixed per strategy
%   CR = 0, 0.5, 1                  % Binomial crossover rates
%   CR1 < CR2 < CR3                 % Exponential crossover rates from p1 < p2 < p3
%
% Algorithm Concept:
%   - Twelve rand/1 strategies compete: binomial or exponential crossover, F in
%     {0.5, 0.8}, three CR values for each
%   - Each trial draws its strategy by roulette on the success counts n_i; a count
%     grows by one each time its trial replaces the parent
%   - Random localisation: the fittest of the three random donors is the base vector
%   - Exponential CR solves CR^D - D*p*CR + D*p - 1 = 0, p2 = (1 + 1/D)/2, p1 and p3
%     the midpoints towards 1/D and 1, so p is the expected share of mutant genes
%   - Out-of-box coordinates mirrored at the violated bound; generational greedy
%     replacement accepts ties (trial <= parent)
%
% Reference:
% Josef Tvrdik, Radka Polakova,
% Competitive differential evolution applied to CEC 2013 problems,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, 1651-1657.
% https://doi.org/10.1109/CEC.2013.6557759
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from DE_b6e6rl.m and its helpers in CEC2013_b6e6rl_source.zip, the authors'
% MATLAB release in the organiser archive (github.com/P-N-Suganthan/CEC2013). The
% release's drivers set NP = 30 at D = 10 and NP = 50 at D = 30, 50; other D take 30
% up to D = 10 and 50 above. Removed: the stop at the known optimum (DE_b6e6rl_zkr.m).
% Mirroring is the release's reflect-until-inside loop for 20 passes, then its closed
% form (same point), and non-finite coordinates are redrawn uniformly; the release
% loops forever on those. Same rng seed gives the release's run exactly.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = b6e6rl(problem)

    D = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    if D <= 10
        NP = 30;
    else
        NP = 50;
    end
    h = 12;
    n0 = 2;
    delta = 1 / (5 * h);
    ni = zeros(1, h) + n0;

    p2 = 0.5 * (1 + 1 / D);
    p1 = 0.5 * (p2 + 1 / D);
    p3 = 0.5 * (p2 + 1);
    CR1 = crexp_set(p1, D);
    CR2 = crexp_set(p2, D);
    CR3 = crexp_set(p3, D);
    % Strategies 1-6 binomial, 7-12 exponential, in the release's switch order
    Fs  = [0.5 0.5 0.5 0.8 0.8 0.8 0.5 0.5 0.5 0.8 0.8 0.8];
    CRs = [0 0.5 1 0 0.5 1 CR1 CR2 CR3 CR1 CR2 CR3];

    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;
    FE = 0;
    bsf = inf;

    pos = lb + (ub - lb) .* rand(NP, D);
    bsf_solution = pos(1, :);
    e = inf(NP, 1);
    n_eval = min(NP, maxFE);
    [fv, FE] = calculate_fitness(pos(1:n_eval, :)', problem, FE);
    e(1:n_eval) = fv(:);
    track(pos(1:n_eval, :), e(1:n_eval));

    while FE < maxFE
        poskon = zeros(NP, D);
        strat = zeros(NP, 1);
        for k = 1:NP
            [hh, p_min] = roulette(ni);
            if p_min < delta
                ni = zeros(1, h) + n0;
            end
            strat(k) = hh;
            y = de_rand_rl(pos, e, Fs(hh), CRs(hh), k, hh > 6);
            poskon(k, :) = mirror(y, lb, ub);
        end

        n_eval = min(NP, maxFE - FE);
        [fv, FE] = calculate_fitness(poskon(1:n_eval, :)', problem, FE);
        ekon = fv(:);
        for k = 1:n_eval
            if ekon(k) <= e(k)
                pos(k, :) = poskon(k, :);
                e(k) = ekon(k);
                ni(strat(k)) = ni(strat(k)) + 1;
            end
        end
        track(poskon(1:n_eval, :), ekon);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsf_solution;

    % Best-so-far per evaluation, then one history sample per evaluation with the settled population
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
                    ec, pos, e, population_history, fitness_history, history_index, maxFE);
            end
        end
    end
end

function CR = crexp_set(p, d)
% Smallest real root in [0, 1) of CR^d - d*p*CR + d*p - 1 (release CRexp_set.m)
    y = zeros(1, d + 1);
    y(1) = 1;
    y(d) = -d * p;
    y(d + 1) = d * p - 1;
    r = roots(y);
    r(imag(r) ~= 0) = [];
    r(r < 0 | r >= 1) = [];
    r = sort(r);
    CR = r(1);
end

function [res, p_min] = roulette(cutpoints)
% Index drawn with probability cutpoints(i)/sum(cutpoints); counts are integers, so cp(end) is exactly 1
    ss = sum(cutpoints);
    p_min = min(cutpoints) / ss;
    cp = cumsum(cutpoints) / ss;
    res = 1 + fix(sum(cp < rand));
end

function y = de_rand_rl(P, hod, F, CR, expt, use_exp)
% rand/1 with the fittest of three random donors as base (derand_RLe / derandexp_RLe)
    [N, d] = size(P);
    y = P(expt, :);
    opora = 1:N;
    opora(expt) = [];
    vyb = zeros(1, 3);
    for i = 1:3
        index = 1 + fix(rand * length(opora));
        vyb(i) = opora(index);
        opora(index) = [];
    end
    r123 = P(vyb, :);
    [~, indmin] = min(hod(vyb));
    r1 = r123(indmin, :);
    r123(indmin, :) = [];
    v = r1 + F * (r123(1, :) - r123(2, :));
    if use_exp
        L = 1 + fix(d * rand);
        change = L;
        position = L;
        while rand < CR && length(change) < d
            position = position + 1;
            if position <= d
                change(end + 1) = position; %#ok<AGROW>
            else
                change(end + 1) = mod(position, d); %#ok<AGROW>
            end
        end
    else
        change = find(rand(1, d) < CR);
        if isempty(change)
            change = 1 + fix(d * rand);
        end
    end
    y(change) = v(change);
end

function y = mirror(y, a, b)
% Reflect at the violated bound until inside (zrcad.m), per-dimension bounds
    bad = ~isfinite(y);
    if any(bad)
        y(bad) = a(bad) + rand(1, nnz(bad)) .* (b(bad) - a(bad));
    end
    for pass = 1:20
        hi = y > b;
        lo = y < a;
        if ~any(hi | lo)
            return;
        end
        y(hi) = 2 * b(hi) - y(hi);
        y(lo) = 2 * a(lo) - y(lo);
    end
    % Closed form of the same reflection: a triangle wave of period 2*(b - a)
    out = y < a | y > b;
    if any(out)
        w = b(out) - a(out);
        z = mod(y(out) - a(out), 2 * w);
        z(z > w) = 2 * w(z > w) - z(z > w);
        z(w == 0) = 0;
        y(out) = min(max(a(out) + z, a(out)), b(out));
    end
end

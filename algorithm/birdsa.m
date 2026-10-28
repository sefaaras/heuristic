% ----------------------------------------------------------------------- %
% Bird Swarm Algorithm (BSA)
% Stored as birdsa; the acronym BSA collides with bsa (Backtracking Search)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop     = 30            % Birds
%   FQ      = 10            % Every FQ-th iteration is a flight; the others forage or watch
%   prob    = U(0.8, 1)     % Foraging probability, per bird per iteration
%   c1 = c2 = 1.5           % Pulls toward the swarm best and the bird's own best, Eq. (1)
%   a1 = a2 = 1             % Vigilance coefficients, Eqs. (3)-(4)
%   FL      = U(0.5, 0.9)   % Scrounger's step toward a producer's best, per bird
%
% Algorithm Concept:
%   - Foraging (Eq. 1): x + c1*r1*(gbest - x) + c2*r2*(pbest - x), r1, r2 scalars
%   - Vigilance (Eqs. 2-4): x + r*A1*(mean(pbest) - x) + U(-1, 1)*A2*(pbest_k - x),
%     A1 = a1*exp(-N*f_i/sumF), A2 = a2*exp(sign(f_i - f_k)*N*f_k/sumF)
%   - Flight: the index half (1..N/2 or N/2+1..N) without the worst bird produces,
%     x*(1 + randn) with one scalar randn, a scaling of the whole vector about 0
%   - The best bird always produces; the rest of the other half scrounge,
%     x + FL*(pbest_k - x) with k drawn from the producing half
%   - Birds move from their current positions; personal bests and the swarm best
%     are replaced greedily after the sweep
%
% Reference:
% Xian-Bing Meng, X.Z. Gao, Lihua Lu, Yu Liu, Hengzhen Zhang,
% A new bio-inspired optimisation algorithm: Bird Swarm Algorithm,
% Journal of Experimental & Theoretical Artificial Intelligence 28(4) (2016)
% 673-687.
% https://doi.org/10.1080/0952813X.2015.1042530
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 51256 v1.0.0.0 (BSA.m,
% 21 Jun 2015); seeded runs are bit-identical to it. Release values kept over the
% paper's: FQ = 10 (Table 3: 3), N = 30 (5D-6D there), vigilance toward mean(pbest),
% scrounger step without Eq. (6)'s rand. Kept as released: a best or worst bird at
% index exactly N/2 matches none of the four producer cases, so that flight makes
% N/2+1..N produce and a best bird in 1..N/2 neither moves nor is evaluated. The
% producer move pulls toward the origin, not the box centre; kept (centre-bias
% control, Kudela 2023). A1, A2 assume f > 0 and could overflow on CEC2020RW (not
% seen on its 57 problems); a NaN coordinate would be kept at lb, where that
% evaluator clamps and scores it. Nothing depends on the iteration count.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = birdsa(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    pop = 30;
    FQ  = 10;
    c1  = 1.5;
    c2  = 1.5;
    a1  = 1;
    a2  = 1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Transposed draw: bird i takes the i-th rand(1, dim) of the stream, as released
    x = lb + (ub - lb) .* rand(dim, pop)';
    nev = min(pop, maxFE);
    [fit, FE] = calculate_fitness(x(1:nev, :)', problem, FE);
    fit = fit(:)';

    best_fitness  = Inf;
    best_solution = x(1, :);
    for e = 1:nev
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = x(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, x(1:nev, :), fit, population_history, fitness_history, ...
            history_index, maxFE);
    end
    pFit = fit;
    pX   = x;

    % The release's swarm best, refreshed only after a full sweep; it guides Eq. (1)
    [fMin, bestIndex] = min(fit);
    bestX = x(bestIndex, :);

    cand  = zeros(pop, dim);   % moves in the release's evaluation order
    order = zeros(1, pop);     % bird that each row of cand belongs to

    iteration = 0;
    while FE < maxFE
        iteration = iteration + 1;
        prob = rand(pop, 1) .* 0.2 + 0.8;
        n = 0;

        if mod(iteration, FQ) ~= 0
            sumPfit = sum(pFit);
            meanP = mean(pX);
            for i = 1:pop
                if rand < prob(i)
                    xi = x(i, :) + c1 * rand .* (bestX - x(i, :)) + ...
                        c2 * rand .* (pX(i, :) - x(i, :));   % Eq. (1)
                else
                    person = randiTabu(1, pop, i);
                    xi = x(i, :) + rand .* (meanP - x(i, :)) * a1 * ...
                        exp(-pFit(i) / (sumPfit + realmin) * pop) + a2 * ...
                        (rand * 2 - 1) .* (pX(person, :) - x(i, :)) * exp( ...
                        -(pFit(person) - pFit(i)) / (abs(pFit(person) - pFit(i)) ...
                        + realmin) * pFit(person) / (sumPfit + realmin) * pop);   % Eqs. (2)-(4)
                end
                n = n + 1;
                order(n) = i;
                cand(n, :) = bounds(xi, lb, ub);
            end
        else
            FL = rand(pop, 1) .* 0.4 + 0.5;
            [~, minIndex] = min(pFit);
            [~, maxIndex] = max(pFit);
            % Strict inequalities: an index of exactly pop/2 leaves choose = 0
            choose = 0;
            if minIndex < 0.5 * pop && maxIndex < 0.5 * pop
                choose = 1;
            end
            if minIndex > 0.5 * pop && maxIndex < 0.5 * pop
                choose = 2;
            end
            if minIndex < 0.5 * pop && maxIndex > 0.5 * pop
                choose = 3;
            end
            if minIndex > 0.5 * pop && maxIndex > 0.5 * pop
                choose = 4;
            end

            if choose < 3
                producers = (pop / 2 + 1):pop;
                scroungers = 1:(0.5 * pop);
                extra = (choose == 1);
                skipMin = (choose ~= 2);
                lo = 0.5 * pop + 1;
                hi = pop;
            else
                producers = 1:(0.5 * pop);
                scroungers = (0.5 * pop + 1):pop;
                extra = (choose == 4);
                skipMin = (choose ~= 3);
                lo = 1;
                hi = 0.5 * pop;
            end
            for i = producers
                n = n + 1;
                order(n) = i;
                cand(n, :) = bounds(x(i, :) * (1 + randn), lb, ub);   % Eq. (5)
            end
            if extra
                n = n + 1;
                order(n) = minIndex;
                cand(n, :) = bounds(x(minIndex, :) * (1 + randn), lb, ub);
            end
            for i = scroungers
                if ~skipMin || minIndex ~= i
                    person = randi([lo, hi], 1);
                    n = n + 1;
                    order(n) = i;
                    cand(n, :) = bounds(x(i, :) + (pX(person, :) - x(i, :)) * FL(i), lb, ub);   % Eq. (6)
                end
            end
        end

        nev = min(n, maxFE - FE);
        [fnew, FE] = calculate_fitness(cand(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for e = 1:nev
            % A NaN fitness (a NaN coordinate on the numeric suites) is a spent FE, never accepted
            if ~isnan(fnew(e))
                k = order(e);
                x(k, :) = min(max(cand(e, :), lb), ub);   % NaN -> lb, as CEC2020RW scores it
                fit(k)  = fnew(e);
                if fnew(e) < best_fitness
                    best_fitness  = fnew(e);
                    best_solution = x(k, :);
                end
            end
            fe = FE - nev + e;
            curve(fe) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                fe, x, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i)  = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin  = pFit(i);
                bestX = pX(i, :);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% The release's Bounds, lower face first; a NaN coordinate passes through
function s = bounds(s, lb, ub)
    I = s < lb;
    s(I) = lb(I);
    J = s > ub;
    s(J) = ub(J);
end

% One integer in [lo, hi] other than tabu, by rejection as released
function value = randiTabu(lo, hi, tabu)
    value = randi([lo, hi], 1, 1);
    while value == tabu
        value = randi([lo, hi], 1, 1);
    end
end

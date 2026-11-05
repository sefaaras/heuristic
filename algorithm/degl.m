% ----------------------------------------------------------------------- %
% Differential Evolution with Global and Local Neighborhoods (DEGL)
% Self-adaptive weight variant DEGL/SAW, the one the paper recommends
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP     = 10*D                    % Population size
%   F      = 0.8                     % alpha = beta = F, both scale factors
%   Cr     = 0.9                     % Binomial crossover rate
%   k      = round((0.1*NP - 1)/2)   % Ring radius: neighbourhood 2k+1 ~ 10 % of NP
%   w      ~ U(0, 1) per member      % Self-adaptive weight, trial w' clamped to [0.05, 0.95]
%
% Algorithm Concept:
%   - Local donor L = x + F*(x_nbest - x) + F*(x_p - x_q), with nbest, p, q from the
%     ring neighbourhood i-k..i+k of the member's index
%   - Global donor g = x + F*(x_gbest - x) + F*(x_r1 - x_r2), r1, r2 from the population
%   - Donor v = w'*g + (1 - w')*L; w' = w + F*(w_gbest - w) + F*(w_r1 - w_r2) is
%     mutated with the same r1, r2 and kept with the trial
%   - Binomial crossover; members are processed one by one and a trial with
%     f(u) <= f(x) replaces its target at once, updating gbest and the ring bests
%
% Reference:
% Swagatam Das, Ajith Abraham, Uday K. Chakraborty, Amit Konar,
% Differential Evolution Using a Neighborhood-Based Mutation Operator,
% IEEE Transactions on Evolutionary Computation 13(3) (2009) 526-553.
% https://doi.org/10.1109/TEVC.2008.2009457
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (Secs. II-IV, parameters of Sec. VI); no author code is public.
% Members are replaced in place, as Sec. IV-C's sequential update reads. Paper silent,
% chosen: k rounds 2k+1 = 0.1*NP to the nearest integer k >= 1; w starts U(0, 1) as
% Sec. IV-D says, Sec. VI's "constrained within [0.05, 0.95]" read as the clamp of
% Eq. (20); p, q may include nbest and r1, r2 may include gbest; no bound handling is
% given, so violating donor components are clamped to the box. The asymmetric
% initialisation of Sec. V is the paper's test protocol and is not used here.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = degl(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    NP = 10 * D;
    F  = 0.8;
    Cr = 0.9;
    k  = min(max(1, round((0.1 * NP - 1) / 2)), floor((NP - 1) / 2));

    offs   = -k:k;
    others = [1:k, k + 2:2 * k + 1];   % positions of the ring members other than i

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Eq. (2), and the SAW weights of Sec. IV-D
    pop = lb + rand(NP, D) .* (ub - lb);
    w   = rand(NP, 1);
    nEval = min(NP, maxFE);
    [fit, FE] = calculate_fitness(pop(1:nEval, :)', problem, FE);
    fit = fit(:);

    best_fitness  = inf;
    best_solution = pop(1, :);
    for e = 1:nEval
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = pop(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, pop(1:nEval, :), fit, population_history, fitness_history, history_index, maxFE);
    end
    [~, gb] = min(fit);

    while FE < maxFE
        for i = 1:NP
            if FE >= maxFE
                break;
            end
            nb = mod(i - 1 + offs, NP) + 1;
            [~, j] = min(fit(nb));
            nbest = nb(j);
            pq = nb(others(randperm(2 * k, 2)));
            r  = randperm(NP - 1, 2);
            r  = r + (r >= i);

            xi = pop(i, :);
            L  = xi + F * (pop(nbest, :) - xi) + F * (pop(pq(1), :) - pop(pq(2), :));   % Eq. (14)
            g  = xi + F * (pop(gb, :) - xi) + F * (pop(r(1), :) - pop(r(2), :));       % Eq. (15)
            wt = w(i) + F * (w(gb) - w(i)) + F * (w(r(1)) - w(r(2)));                   % Eq. (19)
            wt = min(max(wt, 0.05), 0.95);                                               % Eq. (20)
            V  = min(max(wt * g + (1 - wt) * L, lb), ub);                                % Eq. (16)

            % Eq. (9)
            mask = rand(1, D) <= Cr;
            mask(randi(D)) = true;
            U = xi;
            U(mask) = V(mask);

            [fu, FE] = calculate_fitness(U', problem, FE);
            fu = fu(1);
            if fu < best_fitness
                best_fitness  = fu;
                best_solution = U;
            end
            curve(FE) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE, pop, fit, population_history, fitness_history, history_index, maxFE);

            % Eqs. (11) and (21); a NaN target is replaced by any non-NaN trial
            if fu <= fit(i) || (isnan(fit(i)) && ~isnan(fu))
                pop(i, :) = U;
                fit(i) = fu;
                w(i) = wt;
                if fu < fit(gb)
                    gb = i;
                end
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

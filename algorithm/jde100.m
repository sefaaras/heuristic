% ----------------------------------------------------------------------- %
% Self-Adaptive Differential Evolution with Two Populations (jDE100)
% CEC 2019 "100-Digit Challenge" competition winner
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   bNP = 1000, sNP = 25          % Big and small population sizes (fixed)
%   F_init = 0.5, CR_init = 0.9   % Initial self-adaptive parameters
%   tao1 = tao2 = 0.1             % Probability of re-drawing F / CR
%   Fl = 0.15, Fu = 1.1           % F band
%   CRl = 0.0, CRu = 1.1          % CR band
%   myEqs = 0.25, eps = 1e-16     % Reinitialisation threshold and tolerance
%
% Algorithm Concept:
%   - Two cooperating populations evolved by DE/rand/1/bin with jDE
%     self-adaptation: with probability tao a fresh F or CR is drawn from the
%     band, otherwise the stored value is reused and passed on by a winner
%   - A large exploring population (1000) and a small exploiting one (25) share
%     the budget equally: the slot index cycles over 2*bNP positions, so the
%     small population is swept bNP/sNP = 40 times per cycle
%   - One migration slot: the big population may use the small population's
%     first member as a donor, and the big population's best is copied into the
%     small one whenever it leads
%   - Either population is reinitialised when more than 25 % of it has
%     collapsed onto the best fitness
%   - Violating components are wrapped around the box rather than clamped
%
% Reference:
% Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic,
% The 100-Digit Challenge: Algorithm jDE100,
% 2019 IEEE Congress on Evolutionary Computation (CEC), 2019, pp. 19-26.
% https://doi.org/10.1109/CEC.2019.8789904
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own C++ release (CEC_19402_Brest_jDE100-ver2.0).
% PARAMETER TUNING NOT TRANSFERRED. The competition entry tunes Fl and CRl PER
% FUNCTION of the CEC2019 suite; those indices mean nothing on the suites used
% here, so the DEFAULTS Fl = 0.15 and CRl = 0.0 are always used. On any other
% suite this is therefore the untuned jDE100, not the configuration that won.
% BUDGET REGIME. The fixed population of 1025 and the `age > 1e9` restart
% trigger are sized for the challenge's 1e12 budget, so at 1e5-1e6 the restart
% can never fire. Both are kept at their published settings rather than rescaled.
% The wrap repair is written as lb + mod(U-lb, ub-lb), the same map.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jde100(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';      % 1 x D
    ub    = problem.ub(:)';      % 1 x D
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    bNP = 1000;
    sNP = 25;
    NP  = bNP + sNP;

    Finit  = 0.5;
    CRinit = 0.9;
    Fl     = 0.15;    % default; the per-function tuning is not transferred
    Fu     = 1.1;
    CRl    = 0.0;     % default; see the note above
    CRu    = 1.1;
    tao1   = 0.1;
    tao2   = 0.1;
    myEqs  = 0.25;
    epsq   = 1e-16;
    ageLmt = 1e9;

    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    P     = repmat(lb, NP, 1) + rand(NP, D) .* repmat(span, NP, 1);
    parF  = Finit  * ones(NP, 1);
    parCR = CRinit * ones(NP, 1);

    [cost, FE] = calculate_fitness(P', problem, FE);
    cost = cost(:);

    bsf  = inf;
    bsfx = P(1, :);
    for i = 1:NP
        if cost(i) < bsf
            bsf  = cost(i);
            bsfx = P(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, P, cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    [~, indBest] = min(cost);
    age = 0;

    % Main loop: one evaluation per iteration
    while FE < maxFE
        it = FE;
        i0 = mod(it, 2 * bNP);         % 0-based slot selector

        % Reinitialise the big population
        if i0 == 0 && (tooManyEqual(cost(1:bNP), cost(indBest), myEqs, epsq) || age > ageLmt)
            P(1:bNP, :)  = repmat(lb, bNP, 1) + rand(bNP, D) .* repmat(span, bNP, 1);
            parF(1:bNP)  = Finit;
            parCR(1:bNP) = CRinit;
            cost(1:bNP)  = realmax;    % not evaluated, as in the reference
            age = 0;
            [~, rel] = min(cost(bNP+1:NP));
            indBest  = bNP + rel;
        end

        % Reinitialise the small population
        if i0 == bNP && indBest > bNP && ...
                tooManyEqual(cost(bNP+1:NP), cost(indBest), myEqs, epsq)
            for w = bNP+1:NP
                if w == indBest, continue; end
                P(w, :)  = lb + rand(1, D) .* span;
                parF(w)  = Finit;
                parCR(w) = CRinit;
                cost(w)  = realmax;
            end
        end

        % Copy the big population's best into the small one
        if i0 == bNP && indBest <= bNP
            cost(bNP+1) = cost(indBest);
            P(bNP+1, :) = P(indBest, :);
            indBest     = bNP + 1;
        end

        % Target and donor selection
        if i0 < bNP
            idx = i0 + 1;

            r1 = randi(bNP);
            while r1 == idx
                r1 = randi(bNP);
            end
            % bNP+1 is the first member of the small population: one migration slot
            r2 = randi(bNP + 1);
            while r2 == idx || r2 == r1
                r2 = randi(bNP + 1);
            end
            r3 = randi(bNP + 1);
            while r3 == idx || r3 == r2 || r3 == r1
                r3 = randi(bNP + 1);
            end
        else
            idx = mod(i0 - bNP, sNP) + bNP + 1;

            r1 = randi(sNP) + bNP;
            while r1 == idx
                r1 = randi(sNP) + bNP;
            end
            r2 = randi(sNP) + bNP;
            while r2 == idx || r2 == r1
                r2 = randi(sNP) + bNP;
            end
            r3 = randi(sNP) + bNP;
            while r3 == idx || r3 == r2 || r3 == r1
                r3 = randi(sNP) + bNP;
            end
        end

        % jDE self-adaptation
        if rand() < tao1
            F = Fl + rand() * Fu;
        else
            F = parF(idx);
        end
        if rand() < tao2
            CR = CRl + rand() * CRu;
        else
            CR = parCR(idx);
        end

        % DE/rand/1/bin
        jrand = randi(D);
        take  = rand(1, D) < CR;
        take(jrand) = true;

        U = P(idx, :);
        U(take) = P(r1, take) + F * (P(r2, take) - P(r3, take));

        % Wrap-around repair, applied to the crossed components only
        oob = take & (U < lb | U > ub);
        if any(oob)
            U(oob) = lb(oob) + mod(U(oob) - lb(oob), span(oob));
        end

        [c, FE] = calculate_fitness(U', problem, FE);
        c = c(1);

        if idx <= bNP
            age = age + 1;
        end

        % Selection (no crowding: the parent itself is the competitor)
        if c < cost(indBest)
            age = 0;
            if c < bsf
                bsf  = c;
                bsfx = U;
            end
            cost(idx)  = c;
            P(idx, :)  = U;
            parF(idx)  = F;
            parCR(idx) = CR;
            indBest    = idx;
        elseif c <= cost(idx)
            cost(idx)  = c;
            P(idx, :)  = U;
            parF(idx)  = F;
            parCR(idx) = CR;
        end

        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            FE, P, cost, population_history, fitness_history, ...
            history_index, maxFE);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function tf = tooManyEqual(costs, cBest, myEqs, epsq)
% True when > myEqs individuals sit within epsq of the best; no "and > 2" guard unlike j2020
    tf = sum(abs(costs - cBest) < epsq) > myEqs * numel(costs);
end

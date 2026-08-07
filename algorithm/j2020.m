% ----------------------------------------------------------------------- %
% Self-Adaptive Differential Evolution with Two Populations (j2020)
% CEC 2020 competition -- 3rd place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   bNP = 7 * D, sNP = D          % Big and small population sizes
%   F_init = 0.5, CR_init = 0.9   % Initial self-adaptive parameters
%   tao1 = tao2 = 0.1             % Probability of re-drawing F / CR
%   big   pop: Fl = 0.01, Fu = 1.1, CRl = 0.0, CRu = 1.0
%   small pop: Fl = 0.17, Fu = 1.1, CRl = 0.1, CRu = 0.7
%   myEqs = 0.25                  % Reinitialisation threshold
%   eps   = 1e-12                 % "Same fitness" tolerance
%
% Algorithm Concept:
%   - Two cooperating populations evolved by DE/rand/1/bin with classic jDE
%     self-adaptation: with probability tao a fresh F or CR is drawn from that
%     population's own band, otherwise the stored value is reused and passed on
%   - The big population explores and the small one exploits, on equal shares of
%     the budget: the index cycles over 2*bNP slots, sweeping the small
%     population bNP/sNP = 7 times per cycle
%   - Migration: the big population draws donors from the first 1, 2 or 3
%     members of the small one, widening over the thirds of the budget, and its
%     own best is copied into the small population when it leads
%   - CROWDING: a big trial competes with its NEAREST individual, not its parent
%   - Either population is reinitialised when more than 25 % of it has collapsed
%     onto the best fitness; the big one also restarts after maxFE/10 stagnant
%
% Reference:
% Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic,
% Differential Evolution Algorithm for Single Objective Bound-Constrained
% Optimization: Algorithm j2020,
% 2020 IEEE Congress on Evolutionary Computation (CEC), 2020, pp. 1-8.
% https://doi.org/10.1109/CEC48606.2020.9185551
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own C++ release (main.cpp of the j2020 entry).
% Three reference properties are kept: the big-population r1 rejection reads
% `while (r1 == i && r1 == indBest)` where `||` was meant, so r1 may coincide
% with the target; `if (mRandom() < 0.0) border = true;` is dead code, so a
% violating component is wrapped (lb + mod(U-lb, ub-lb)) rather than clamped;
% and reinitialised individuals get infinite cost and are NOT evaluated, which
% is what makes a restart free. Crowding replacement is sequential: one trial
% at a time. Deliberate deviation: sNP = max(4, D), not sNP = D -- the small
% population's donor draw needs four distinct indices, so sNP < 4 makes that
% rejection loop hang. Identical at D >= 4; binds only on D = 2, 3.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = j2020(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';      % 1 x D
    ub    = problem.ub(:)';      % 1 x D
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    bNP = 7 * D;
    sNP = max(4, D);   % 4 distinct indices are drawn from it; see the header note
    NP  = bNP + sNP;

    Finit  = 0.5;
    CRinit = 0.9;
    Fu     = 1.1;
    tao1   = 0.1;
    tao2   = 0.1;
    myEqs  = 0.25;
    epsq   = 1e-12;

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
        it = FE;                       % the reference's iteration counter
        i0 = mod(it, 2 * bNP);         % 0-based slot selector

        % Reinitialise the big population
        if i0 == 0 && (tooManyEqual(cost(1:bNP), cost(indBest), myEqs, epsq) || age > maxFE / 10)
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
                P(w, :)   = lb + rand(1, D) .* span;
                parF(w)   = Finit;
                parCR(w)  = CRinit;
                cost(w)   = realmax;
            end
        end

        % Copy the big population's best into the small one
        if i0 == bNP && indBest <= bNP
            cost(bNP+1)  = cost(indBest);
            P(bNP+1, :)  = P(indBest, :);
            indBest      = bNP + 1;
        end

        % Target and donor selection
        if i0 < bNP
            isBig = true;
            idx   = i0 + 1;

            Fl = 0.01; CRl = 0.0; CRu = 1.0;

            if it < maxFE / 3
                mig = 1;
            elseif it < 2 * maxFE / 3
                mig = 2;
            else
                mig = 3;
            end

            % Reproduced verbatim: "&&" makes this rejection all but inert
            r1 = randi(bNP + 1);
            while r1 == idx && r1 == indBest
                r1 = randi(bNP + 1);
            end
            r2 = randi(bNP + mig);
            while r2 == idx || r2 == r1
                r2 = randi(bNP + mig);
            end
            r3 = randi(bNP + mig);
            while r3 == idx || r3 == r2 || r3 == r1
                r3 = randi(bNP + mig);
            end
        else
            isBig = false;
            idx   = mod(i0 - bNP, sNP) + bNP + 1;

            Fl = 0.17; CRl = 0.1; CRu = 0.7;

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

        if isBig
            age = age + 1;
            % Crowding: compete with the NEAREST big-population member
            [~, idx] = min(sum((P(1:bNP, :) - U) .^ 2, 2));
        end

        % Selection
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
% Reference collapse detector: > myEqs individuals (and > 2) within epsq of the best fitness
    eqs = sum(abs(costs - cBest) < epsq);
    tf  = (eqs > myEqs * numel(costs)) && (eqs > 2);
end

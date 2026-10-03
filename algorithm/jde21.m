% ----------------------------------------------------------------------- %
% Self-Adaptive DE with Population Size Reduction (jDE21, j21)
% CEC 2021 competition -- 2nd shifted, 1st non-rotated shifted, 5th provisional overall
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   bNP = 160 -> 80 -> 40 -> 20, sNP = 10  % Big population halved at 1/4, 2/4, 3/4 of maxFe
%   F_init = 0.5, CR_init = 0.9    % Initial self-adaptive parameters
%   tau1 = tau2 = 0.1              % Probability of re-drawing F / CR
%   big   pop: F = 0.1 + 1.1*U, CR = 0.0 + 1.1*U  % CR > 1 copies the whole mutant
%   small pop: F = 0.17 + 1.1*U, CR = 0.1 + 0.8*U
%   ageLmt = maxFe/10              % Big-population restart after this many stagnant FE
%   myEqs = 0.25, eps = 1e-12      % Restart when > 25 % share the best fitness
%
% Algorithm Concept:
%   - Two populations evolved by jDE DE/rand/1/bin, one trial at a time; the small
%     one runs bNP/sNP generations per big-population generation (equal FE shares)
%   - Big population: r2, r3 may come from the first 1, 2, 3 small members over the
%     thirds of the budget; the trial competes with its NEAREST big member (crowding)
%   - After each big generation its best is copied into the small population
%   - Either population restarts when more than 25 % of it sits on the best fitness;
%     the big one also after ageLmt FE without improving the best
%   - Big population halved three times: member i of the first half meets member
%     i of the second half and the better survives
%   - Violating coordinates are wrapped around the box
%
% Reference:
% Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic,
% Self-adaptive Differential Evolution Algorithm with Population Size Reduction for
% Single Objective Bound-Constrained Optimization: Algorithm j21,
% 2021 IEEE Congress on Evolutionary Computation (CEC), 2021, pp. 817-824.
% https://doi.org/10.1109/CEC45853.2021.9504782
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author code exists. Ported from the THIRD-PARTY Python port in MetaBox v2.0.0
% (src/baseline/bbo/jde21.py) and checked against the paper (Alg. 2, Table II, Sec.
% III) and the authors' j2020 C++. All constants match Table II (eps 1e-12 as in the
% table; the text says 1e-16). The paper overrides MetaBox on: one trial at a time
% (MetaBox is generational), CR > 1 keeps every mutant gene (MetaBox sets it to 0),
% small-population CR = 0.1 + 0.8*U (MetaBox 1.1), halving by pairwise competition
% (Sec. III; MetaBox drops the first half), restart checks first in each cycle,
% selection <=. MetaBox bugs fixed: age reset every generation, restarts above ub.
% Kept from MetaBox and j2020: the big r1 rejection `r1 == i && r1 == best`, restarted
% members cost realmax unevaluated, and the wrap repair (the paper says "reflected").
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jde21(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    bNP    = 160;
    sNP    = 10;
    Finit  = 0.5;
    CRinit = 0.9;
    tau1   = 0.1;
    tau2   = 0.1;
    Fl_b   = 0.1;
    Fl_s   = 0.17;
    Fu     = 1.1;
    CRl_b  = 0.0;
    CRl_s  = 0.1;
    CRu_b  = 1.1;
    CRu_s  = 0.8;
    ageLmt = maxFE / 10;
    epsq   = 1e-12;
    myEqs  = 0.25;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    NP    = bNP + sNP;
    P     = lb + rand(NP, D) .* span;
    parF  = Finit * ones(NP, 1);
    parCR = CRinit * ones(NP, 1);
    nInit = min(NP, maxFE);
    cost  = realmax * ones(NP, 1);
    [fv, FE] = calculate_fitness(P(1:nInit, :)', problem, FE);
    cost(1:nInit) = fv(:);

    bsf  = inf;
    bsfx = P(1, :);
    for e = 1:nInit
        if cost(e) < bsf
            bsf  = cost(e);
            bsfx = P(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, P, cost, population_history, fitness_history, history_index, maxFE);
    end

    [~, indBest] = min(cost);
    age  = 0;
    nRed = 0;

    while FE < maxFE
        % Alg. 2 lines 5-6: restart checks of both populations
        if tooManyEqual(cost(1:bNP), cost(indBest), myEqs, epsq) || age > ageLmt
            P(1:bNP, :)  = lb + rand(bNP, D) .* span;
            parF(1:bNP)  = Finit;
            parCR(1:bNP) = CRinit;
            cost(1:bNP)  = realmax;             % not evaluated, as in j2020
            age = 0;
            [~, rel] = min(cost(bNP+1:NP));
            indBest  = bNP + rel;
        end
        if indBest > bNP && tooManyEqual(cost(bNP+1:NP), cost(indBest), myEqs, epsq)
            for w = bNP+1:NP
                if w == indBest
                    continue;
                end
                P(w, :)  = lb + rand(1, D) .* span;
                parF(w)  = Finit;
                parCR(w) = CRinit;
                cost(w)  = realmax;
            end
        end

        % One generation of the big population (lines 7-15)
        for i = 1:bNP
            if FE >= maxFE
                break;
            end
            if FE <= maxFE / 3
                mig = 1;
            elseif FE <= 2 * maxFE / 3
                mig = 2;
            else
                mig = 3;
            end
            % `&&` as in MetaBox and j2020: r1 may still equal i
            r1 = randi(bNP);
            while r1 == i && r1 == indBest
                r1 = randi(bNP);
            end
            r2 = randi(bNP + mig);
            while r2 == i || r2 == r1
                r2 = randi(bNP + mig);
            end
            r3 = randi(bNP + mig);
            while r3 == i || r3 == r2 || r3 == r1
                r3 = randi(bNP + mig);
            end
            [U, F, CR] = jde_trial(P, i, r1, r2, r3, parF(i), parCR(i), ...
                                   Fl_b, Fu, CRl_b, CRu_b, tau1, tau2, lb, span);

            [fv, FE] = calculate_fitness(U', problem, FE);
            c = fv(1);
            age = age + 1;
            % Crowding: the trial competes with the nearest big-population member
            [~, idx] = min(sum((P(1:bNP, :) - U) .^ 2, 2));
            [P, cost, parF, parCR, indBest, age, bsf, bsfx] = jde_select(P, cost, parF, parCR, ...
                indBest, age, bsf, bsfx, idx, U, c, F, CR);

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, P, cost, population_history, fitness_history, history_index, maxFE);
        end

        % Lines 16-18: copy the big population's best into the small one
        if indBest <= bNP
            cost(bNP+1)  = cost(indBest);
            P(bNP+1, :)  = P(indBest, :);
            indBest      = bNP + 1;
        end

        % Lines 19-25: m = bNP/sNP generations of the small population
        for g = 1:bNP / sNP
            for i = bNP+1:NP
                if FE >= maxFE
                    break;
                end
                r1 = randi(sNP) + bNP;
                while r1 == i
                    r1 = randi(sNP) + bNP;
                end
                r2 = randi(sNP) + bNP;
                while r2 == i || r2 == r1
                    r2 = randi(sNP) + bNP;
                end
                r3 = randi(sNP) + bNP;
                while r3 == i || r3 == r2 || r3 == r1
                    r3 = randi(sNP) + bNP;
                end
                [U, F, CR] = jde_trial(P, i, r1, r2, r3, parF(i), parCR(i), ...
                                       Fl_s, Fu, CRl_s, CRu_s, tau1, tau2, lb, span);

                [fv, FE] = calculate_fitness(U', problem, FE);
                [P, cost, parF, parCR, indBest, age, bsf, bsfx] = jde_select(P, cost, parF, parCR, ...
                    indBest, age, bsf, bsfx, i, U, fv(1), F, CR);

                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, P, cost, population_history, fitness_history, history_index, maxFE);
            end
        end

        % Line 26: halve the big population at 1/4, 2/4 and 3/4 of the budget
        while nRed < 3 && FE >= (nRed + 1) * maxFE / 4
            nRed = nRed + 1;
            half = bNP / 2;
            first = (1:half)';
            second = first + half;
            win = cost(second) < cost(first);
            P(first(win), :)  = P(second(win), :);
            cost(first(win))  = cost(second(win));
            parF(first(win))  = parF(second(win));
            parCR(first(win)) = parCR(second(win));
            keep  = [first; (bNP+1:NP)'];
            P     = P(keep, :);
            cost  = cost(keep);
            parF  = parF(keep);
            parCR = parCR(keep);
            bNP   = half;
            NP    = bNP + sNP;
            [~, indBest] = min(cost);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [U, F, CR] = jde_trial(P, i, r1, r2, r3, Fi, CRi, Fl, Fu, CRl, CRu, tau1, tau2, lb, span)
% jDE self-adaptation, DE/rand/1/bin (CR > 1 takes every gene) and the wrap of crossed genes
    if rand < tau1
        F = Fl + rand * Fu;
    else
        F = Fi;
    end
    if rand < tau2
        CR = CRl + rand * CRu;
    else
        CR = CRi;
    end
    D = size(P, 2);
    take = rand(1, D) < CR;
    take(randi(D)) = true;
    U = P(i, :);
    U(take) = P(r1, take) + F * (P(r2, take) - P(r3, take));
    oob = take & (U < lb | U > lb + span);
    if any(oob)
        U(oob) = lb(oob) + mod(U(oob) - lb(oob), span(oob));
    end
end

function [P, cost, parF, parCR, indBest, age, bsf, bsfx] = jde_select(P, cost, parF, parCR, ...
                                                                   indBest, age, bsf, bsfx, idx, U, c, F, CR)
% j2020 selection: a new overall best resets age; otherwise replace when not worse
    if c < bsf
        bsf  = c;
        bsfx = U;
    end
    if c < cost(indBest)
        age = 0;
        indBest = idx;
    elseif c > cost(idx)
        return;
    end
    cost(idx)  = c;
    P(idx, :)  = U;
    parF(idx)  = F;
    parCR(idx) = CR;
end

function tf = tooManyEqual(costs, cBest, myEqs, epsq)
% More than myEqs of the population (and more than 2) within epsq of the best fitness
    eqs = sum(abs(costs - cBest) < epsq);
    tf  = (eqs > myEqs * numel(costs)) && (eqs > 2);
end

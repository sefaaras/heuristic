% ----------------------------------------------------------------------- %
% Colliding Bodies Optimization (CBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popSize = 20               % Colliding bodies (even: the two halves pair up)
%   COR     = 1 - iter/maxIt   % Coefficient of restitution, Eq. (6), 1 -> 0
%
% Algorithm Concept:
%   - Bodies sorted by cost; the better half is stationary, and moving body
%     i + n/2 hits stationary body i with velocity v = x_i - x_(i+n/2)
%   - Mass m = 1/cost; a 1-D collision with restitution COR gives the stationary
%     body (1 + COR)*m_M*v/(m_S + m_M), the moving one (m_M - COR*m_S)*v/(m_S + m_M)
%   - Both new positions start from the stationary body's old position plus
%     rand(-1, 1) .* (velocity after collision), Eqs. (7)-(8)
%   - Every body is replaced and re-evaluated each iteration (no selection);
%     as COR falls to 0 the collision turns inelastic, one velocity per pair
%
% Reference:
% A. Kaveh, V.R. Mahdavi,
% Colliding bodies optimization: A novel meta-heuristic method,
% Computers & Structures 139 (2014) 18-27.
% https://doi.org/10.1016/j.compstruc.2014.04.005
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB listing (CBO, Appendix 1) in A. Kaveh, M. Ilchi
% Ghazaan, Computer codes for colliding bodies optimization and its enhanced
% version, Int. J. Optim. Civil Eng. 4(3) (2014) 321-339. maxIt = ceil(maxFe/20),
% so COR reaches 0 at the budget; the last batch is truncated to it. As listed:
% masses are 1/cost unnormalised (Eq. (1) normalises, same ratios) and bodies are
% sorted by cost, not by mass (they differ for negative costs). The listing has
% no bound handling; positions are clamped to the box here. Guard: where a mass
% ratio is not finite (zero, Inf or NaN cost, m_S + m_M = 0) it is taken in costs,
% (1+COR)f_S/(f_S+f_M) and (f_S-COR f_M)/(f_S+f_M), else equal masses. The
% reported best is updated at every evaluation, not once per iteration.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cbo(problem)

    nVar  = problem.dimension;
    xMin  = problem.lb(:)';
    xMax  = problem.ub(:)';
    maxFE = problem.maxFe;

    popSize = 20;
    half    = popSize / 2;
    maxIt   = max(1, ceil(maxFE / popSize));   % every body is re-evaluated each iteration

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    CB = xMin + rand(popSize, nVar) .* (xMax - xMin);

    best_fitness  = inf;
    best_solution = CB(1, :);

    iter = 0;
    while FE < maxFE
        iter = iter + 1;

        m = min(popSize, maxFE - FE);
        [fit, FE] = calculate_fitness(CB(1:m, :)', problem, FE);
        fit = fit(:)';

        % Recorded population: the initial batch whole, later bodies overwrite last iteration's slots
        if iter == 1
            Xrec = CB(1:m, :);
            frec = fit;
        end
        for g = 1:m
            if fit(g) < best_fitness
                best_fitness  = fit(g);
                best_solution = CB(g, :);
            end
            Xrec(g, :) = CB(g, :);
            frec(g)    = fit(g);
            ec = FE - m + g;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, frec, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        % Ties keep the body index order, NaN costs sort last
        agentCost = sortrows([fit', (1:popSize)']);
        mass = 1 ./ agentCost(:, 1);
        COR = 1 - iter / maxIt;   % Eq. (6)

        for e = 1:half
            iS = agentCost(e, 2);
            iM = agentCost(half + e, 2);
            mS = mass(e);
            mM = mass(half + e);
            velMb = CB(iS, :) - CB(iM, :);   % Eq. (3), stationary bodies at rest, Eq. (2)
            kS = ((1 + COR) * mM) / (mS + mM);   % Eq. (4)
            kM = (mM - COR * mS) / (mS + mM);    % Eq. (5)
            if ~isfinite(kS) || ~isfinite(kM)
                fS = agentCost(e, 1);
                fM = agentCost(half + e, 1);
                kS = ((1 + COR) * fS) / (fS + fM);   % the same ratios with m = 1/f multiplied out
                kM = (fS - COR * fM) / (fS + fM);
                if ~isfinite(kS) || ~isfinite(kM)
                    kS = (1 + COR) / 2;   % equal masses
                    kM = (1 - COR) / 2;
                end
            end
            velSa = kS * velMb;
            velMa = kM * velMb;
            CB(iM, :) = CB(iS, :) + 2 * (0.5 - rand(1, nVar)) .* velMa;   % Eq. (8)
            CB(iS, :) = CB(iS, :) + 2 * (0.5 - rand(1, nVar)) .* velSa;   % Eq. (7)
        end
        CB = min(max(CB, xMin), xMax);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

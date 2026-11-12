% ----------------------------------------------------------------------- %
% Shuffled Frog Leaping Algorithm (SFLA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   m = 20                      % Memeplexes
%   n = 20                      % Frogs per memeplex, population F = m*n = 400
%   q = 15                      % Frogs per sub-memeplex
%   N = 15                      % Evolution steps per memeplex between shuffles
%   Smax = 0.45*(ub - lb)       % Maximum leap per dimension (45 % of the range)
%
% Algorithm Concept:
%   - The ranked population is dealt round-robin into m memeplexes, so rank k
%     joins memeplex mod(k-1, m) + 1 and every memeplex spans the whole ranking
%   - A sub-memeplex of q frogs is drawn without replacement with triangular rank
%     weights p_j = 2(n+1-j)/(n(n+1)); its best Pb and worst Pw are recorded
%   - Pw leaps by S = min(max(rand*(Pb - Pw), -Smax), Smax), one scalar rand,
%     and keeps the new position only if it is better
%   - Failing that, the same leap towards the global best Px; failing that too,
%     censorship replaces Pw by a random frog unconditionally
%   - After N steps per memeplex the population is re-ranked and re-dealt
%
% Reference:
% Muzaffar Eusuff, Kevin Lansey, Fayzul Pasha,
% Shuffled frog-leaping algorithm: a memetic meta-heuristic for discrete
% optimization, Engineering Optimization 38(2) (2006) 129-154.
% https://doi.org/10.1080/03052150500384759
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from this paper's manuscript, Appendix A of Eusuff's PhD dissertation
% (University of Arizona 2004, hdl.handle.net/10150/191266); the paywalled article
% and the 2003 JWRPM introduction were not consulted. The paper rounds each leap
% to integers (NINT) for discrete variables; the continuous form drops only that.
% Parameters are the base set of the authors' sensitivity study (Table A.2), not
% the recommended ranges (m 100-150, n 30-100, N 20-30, q = 20, Smax 100 %), whose
% 3000 frogs made a CEC2020RW F34 run take ~2.5 h in the recorder. A leap stays
% between two in-box frogs, so the paper's infeasible-position branch never fires;
% the box clip only removes rounding. The convergence test is replaced by maxFe;
% Px is refreshed only at the shuffle.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sfla(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    m    = 20;
    n    = 20;
    q    = 15;
    Nev  = 15;
    Smax = 0.45 * span;
    F    = m * n;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X  = lb + rand(F, D) .* span;
    nf = min(F, maxFE);
    [fv, FE] = calculate_fitness(X(1:nf, :)', problem, FE);
    X  = X(1:nf, :);
    fX = fv(:);

    bsf          = inf;
    bsf_solution = X(1, :);
    for i = 1:nf
        % A NaN incumbent could never be beaten, so it is overwritten
        if fX(i) < bsf || (isnan(bsf) && ~isnan(fX(i)))
            bsf          = fX(i);
            bsf_solution = X(i, :);
        end
        curve(i) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            i, X, fX, population_history, fitness_history, history_index, maxFE);
    end

    % Triangular selection weights over memeplex ranks, best first
    w = 2 * (n + 1 - (1:n)) / (n * (n + 1));

    while FE < maxFE
        % Shuffle: sort places NaN last, i.e. as the worst frogs
        [~, order] = sort(fX);
        Px  = X(order(1), :);
        mem = reshape(order, m, n);

        for k = 1:m
            for step = 1:Nev
                idx  = mem(k, :);
                pick = sub_memeplex(w, q);
                iB = idx(pick(1));
                iW = idx(pick(end));
                xw = X(iW, :);
                fw = fX(iW);

                % Moves 1-3: leap to Pb, leap to Px, censorship
                for move = 1:3
                    if move < 3
                        if move == 1
                            target = X(iB, :);
                        else
                            target = Px;
                        end
                        S = rand * (target - xw);
                        S = min(max(S, -Smax), Smax);
                        U = min(max(xw + S, lb), ub);
                    else
                        U = lb + rand(1, D) .* span;
                    end
                    [fu, FE] = calculate_fitness(U', problem, FE);
                    taken = (move == 3) || (fu < fw);
                    if taken
                        X(iW, :) = U;
                        fX(iW)   = fu;
                    end
                    if fu < bsf || (isnan(bsf) && ~isnan(fu))
                        bsf          = fu;
                        bsf_solution = U;
                    end
                    curve(FE) = bsf;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, X, fX, population_history, fitness_history, history_index, maxFE);
                    if taken || FE >= maxFE
                        break;
                    end
                end

                % Re-rank the memeplex after the change (stable for ties)
                [~, o] = sort(fX(idx));
                mem(k, :) = idx(o);
                if FE >= maxFE
                    break;
                end
            end
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% q distinct memeplex ranks drawn with the triangular weights, returned sorted
function pick = sub_memeplex(w, q)
    avail = true(size(w));
    pick  = zeros(1, q);
    for t = 1:q
        c = cumsum(w .* avail);
        j = find(rand * c(end) < c, 1, 'first');
        if isempty(j)
            % rounding can leave the draw on the last edge
            j = find(avail, 1, 'last');
        end
        pick(t)  = j;
        avail(j) = false;
    end
    pick = sort(pick);
end

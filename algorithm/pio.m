% ----------------------------------------------------------------------- %
% Pigeon-Inspired Optimization (PIO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Np      = 150 -> ceil(Np/2)   % Pigeons; the landmark operator halves them, Eq. (7)
%   R       = 0.2                 % Map and compass factor: velocity decay exp(-R*t)
%   Nc1     = from maxFe          % Map and compass iterations, then Nc1/3 landmark ones
%   epsilon = eps                 % Landmark weight fitness = 1/(f + epsilon)
%
% Algorithm Concept:
%   - Map and compass, Eqs. (5)-(6): V = V*exp(-R*t) + rand*(Xg - X), X = X + V,
%     Xg the best point found, one rand per pigeon
%   - A pigeon leaving the box draws its rand again, the pseudocode's while loop
%   - Landmark, Eq. (7): pigeons ranked by f, the worse half abandoned for good
%   - Eqs. (8)-(9): the rest move X = X + rand*(Xc - X) toward
%     Xc = sum(w.*X)/(Np*sum(w)), w = 1/(f + epsilon)
%
% Reference:
% Haibin Duan, Peixin Qiao,
% Pigeon-inspired optimization: a new swarm intelligence optimizer for air robot
% path planning,
% International Journal of Intelligent Computing and Cybernetics 7(1) (2014) 24-37.
% https://doi.org/10.1108/IJICC-02-2014-0005
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper (the authors' copy on hbduan.buaa.edu.cn/papers, via the
% Wayback Machine); the lab's Code_of_Basic_PIO.zip was never archived and the site
% no longer resolves, so nothing is bit-compared. Np = 150, R = 0.2 and Nc1max:Nc2max
% = 150:200 are Sec. 6's; Nc1 is the largest map phase that fits maxFe with Nc1/3
% landmark iterations counted, and the landmark phase then runs to the budget.
% Kept as printed: Eq. (8) also divides by Np (pseudocode: 'average'), pulling Xc
% toward the origin until one pigeon is left:
%    CEC2014 D=10, 1e5 FE, median error of 5 seeds   F1      F4    F10    F23    F24
%    Eq. (8) as printed, Np included                 3.77e7  83.0  894.9  206.2  157.5
%    without the extra Np                            3.77e7  83.0  894.9  365.8  161.2
% Paper silent, chosen here: Np = ceil(Np/2), so the last pigeon re-evaluates its
% point; v0 ~ U(lb-x, ub-x) (SPSO 2011); rand drawn on the values keeping X in the
% box, X clamped when none does; f <= -epsilon weighs f - min f; NaN counts as +Inf.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = pio(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    Np      = 150;
    R       = 0.2;
    epsilon = eps;
    Nc1     = map_iterations(maxFE, Np);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    X = lb + rand(Np, dim) .* (ub - lb);
    V = lb + rand(Np, dim) .* (ub - lb) - X;   % x + v starts inside the box
    m = min(Np, maxFE);
    [fu, FE] = calculate_fitness(X(1:m, :)', problem, FE);
    fu = fu(:)';
    fu(isnan(fu)) = inf;
    f = inf(1, Np);
    f(1:m) = fu;
    best_fitness  = inf;
    best_solution = X(1, :);
    for e = 1:m
        if f(e) < best_fitness
            best_fitness  = f(e);
            best_solution = X(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, f, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Map and compass operator, Eqs. (5)-(6), for Nc1 iterations
    t = 0;
    while t < Nc1 && FE < maxFE
        t = t + 1;
        decay = exp(-R * t);
        G = best_solution - X;
        r = rand_in_box(X + V * decay, G, lb, ub);
        V = V * decay + r .* G;
        Xnew = min(max(X + V, lb), ub);
        f = evaluate(Xnew, X, f);
        X = Xnew;
    end

    % Landmark operator, Eqs. (7)-(9), to the end of the budget
    while FE < maxFE
        [f, ord] = sort(f);
        Np = ceil(Np / 2);
        X = X(ord(1:Np), :);
        f = f(1:Np);
        w = landmark_weights(f, epsilon);
        Xc = sum(w(:) .* X, 1) / (Np * sum(w));   % Eq. (8) as printed, Np included
        G = Xc - X;
        r = rand_in_box(X, G, lb, ub);
        Xnew = min(max(X + r .* G, lb), ub);
        f = evaluate(Xnew, X, f);
        X = Xnew;
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Evaluates Xnew (truncated to the budget); history rows switch from (Xold, fold) one by one
    function fnew = evaluate(Xnew, Xold, fold)
        m = min(size(Xnew, 1), maxFE - FE);
        [fu, FE] = calculate_fitness(Xnew(1:m, :)', problem, FE);
        fu = fu(:)';
        fu(isnan(fu)) = inf;
        Xrec = Xold;
        fnew = fold;
        for g = 1:m
            if fu(g) < best_fitness
                best_fitness  = fu(g);
                best_solution = Xnew(g, :);
            end
            Xrec(g, :) = Xnew(g, :);
            fnew(g) = fu(g);
            ec = FE - m + g;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Xrec, fnew, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end
end

% One rand per row, uniform over the values that keep Y + rand*G inside [lb, ub]
function r = rand_in_box(Y, G, lb, ub)
    u = rand(size(Y, 1), 1);
    A = (lb - Y) ./ G;
    B = (ub - Y) ./ G;
    lower = -inf(size(Y));
    upper = inf(size(Y));
    pos = G > 0;
    neg = G < 0;
    lower(pos) = A(pos);
    upper(pos) = B(pos);
    lower(neg) = B(neg);
    upper(neg) = A(neg);
    stuck = any(G == 0 & (Y < lb | Y > ub), 2);
    lo = max(0, max(lower, [], 2));
    hi = min(1, min(upper, [], 2));
    ok = lo <= hi & ~stuck;   % otherwise no rand helps and the caller clamps
    r = u;
    r(ok) = lo(ok) + u(ok) .* (hi(ok) - lo(ok));
end

% fitness = 1/(f + epsilon) for minimisation; re-based on the best when f <= -epsilon
function w = landmark_weights(f, epsilon)
    if any(f == -inf)
        w = double(f == -inf);
    else
        fin = isfinite(f);
        base = 0;
        if any(fin) && min(f(fin)) + epsilon <= 0
            base = min(f(fin));
        end
        w = zeros(size(f));
        w(fin) = 1 ./ (f(fin) - base + epsilon);
    end
    if ~(sum(w) > 0 && isfinite(sum(w)))
        w = ones(size(f));   % every kept pigeon at +Inf: plain average
    end
end

% Largest Nc1 with Np + Nc1*Np + the cost of Nc1/3 landmark iterations <= maxFE (Sec. 6: 150:200)
function Nc1 = map_iterations(maxFE, Np)
    Nc1 = max(0, floor((maxFE - Np) / Np));
    while Nc1 > 0 && Np + Nc1 * Np + landmark_cost(round(Nc1 / 3), Np) > maxFE
        Nc1 = Nc1 - 1;
    end
end

function c = landmark_cost(L, Np)
    c = 0;
    for t = 1:L
        Np = ceil(Np / 2);
        c = c + Np;
    end
end

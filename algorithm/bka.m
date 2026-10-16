% ----------------------------------------------------------------------- %
% Black-winged Kite Algorithm (BKA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 30                          % Population size (kites)
%   p   = 0.9                         % Attack-branch threshold, tested against r
%   n   = 0.05*exp(-2*(t/T)^2)        % Attack step, 0.05 -> 0.0068 over the budget
%
% Algorithm Concept:
%   - Attack (Eq. 5-6): multiplicative jitter x.*(1 + n*(2*rand-1)); when p < r
%     the deterministic expansion x*(1 + n*(1 + sin r)) instead
%   - Migration (Eq. 7-8): a Cauchy-scaled step along x - L when the kite beats a
%     random rival, else along L - m*x, with L the leader, m = 2*sin(r + pi/2)
%   - Leader L is the best kite at the start of the iteration
%   - Greedy replacement after each of the two moves, kite by kite
%
% Reference:
% Jun Wang, Wen-chuan Wang, Xiao-xue Hu, Lin Qiu, Hong-fei Zang,
% Black-winged kite algorithm: a nature-inspired meta-heuristic for solving
% benchmark functions and engineering problems,
% Artificial Intelligence Review 57(4) (2024) 98.
% https://doi.org/10.1007/s10462-024-10723-4
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' File Exchange 161401 BKA.m (the paper names it); FX
% downloads need a login, so the copy used is the one byte-identical in two
% GitHub mirrors (hungtrannam/BKIFF, lexiangrui/CUMCM2025). Kept as released:
% r is drawn once per run, so the attack branch (p < r, 1 run in 10) and m are
% fixed for the whole run, and the Cauchy factor is one scalar per move (the
% code draws D values and uses the last). T is counted from the budget,
% ceil((maxFe - pop)/(2*pop)), so n reaches its final value at maxFe. The
% reference reports the leader of its last iteration's start; here the best is
% tracked at every evaluation.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bka(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    pop = 30;
    p   = 0.9;
    T   = max(1, ceil((maxFE - pop) / (2 * pop)));   % attack + migration: two evaluations per kite

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    r = rand;   % drawn once per run, before the population, as in the reference
    XPos = initialization(pop, dim, ub, lb);
    [XFit, FE] = calculate_fitness(XPos', problem, FE);
    XFit = XFit(:)';

    [fbest, bidx] = min(XFit);
    Xbest = XPos(bidx, :);
    for e = 1:min(pop, maxFE)
        curve(e) = fbest;
        [population_history, fitness_history, history_index] = record_history(...
            e, XPos, XFit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    m = 2 * sin(r + pi / 2);   % Eq. (8)
    t = 0;
    while FE < maxFE
        t = t + 1;
        [~, lead] = min(XFit);
        XLeader_Pos = XPos(lead, :);
        n = 0.05 * exp(-2 * (t / T)^2);   % Eq. (6)

        for i = 1:pop
            % Attacking behaviour, Eq. (5)
            if p < r
                Xnew = XPos(i, :) + n .* (1 + sin(r)) * XPos(i, :);
            else
                Xnew = XPos(i, :) .* (n * (2 * rand(1, dim) - 1) + 1);
            end
            Xnew = min(max(Xnew, lb), ub);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < XFit(i)
                XPos(i, :) = Xnew;
                XFit(i) = fnew;
                if fnew < fbest
                    fbest = fnew;
                    Xbest = Xnew;
                end
            end
            if FE <= maxFE
                curve(FE) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, XPos, XFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end

            % Migration behaviour, Eq. (7); the reference keeps only the last of the D Cauchy draws
            s = randi(pop);
            cauchy = tan((rand(1, dim) - 0.5) * pi);
            if XFit(i) < XFit(s)
                Xnew = XPos(i, :) + cauchy(dim) .* (XPos(i, :) - XLeader_Pos);
            else
                Xnew = XPos(i, :) + cauchy(dim) .* (XLeader_Pos - m .* XPos(i, :));
            end
            Xnew = min(max(Xnew, lb), ub);
            [fnew, FE] = calculate_fitness(Xnew', problem, FE);
            if fnew < XFit(i)
                XPos(i, :) = Xnew;
                XFit(i) = fnew;
                if fnew < fbest
                    fbest = fnew;
                    Xbest = Xnew;
                end
            end
            if FE <= maxFE
                curve(FE) = fbest;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, XPos, XFit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = fbest;
    best_fitness  = fbest;
    best_solution = Xbest;
end

% Initialization, one column per dimension as in the reference
function Positions = initialization(N, dim, ub, lb)
    Positions = zeros(N, dim);
    for i = 1:dim
        Positions(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

% ----------------------------------------------------------------------- %
% Vortex Search (VS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   psize  = 50                        % Candidate solutions per iteration
%   x      = 0.1                       % Probability argument of gammaincinv
%   a      = 1 -> 1/MaxItr (linear)    % Shape argument, a = itr/MaxItr
%   r      = (1/x)*gammaincinv(x, a)*(ub - lb)/2   % Radius, the Gaussian std
%   MaxItr = ceil(maxFe/psize)         % Iterations, counted from the budget
%
% Algorithm Concept:
%   - One centre Mu, first the middle of the box, is surrounded by psize
%     Gaussian candidates Mu + r.*randn: nested circles, the vortex
%   - Coordinates that leave the box are re-drawn uniformly inside it
%   - The centre then always moves to the best solution found so far
%   - The radius follows the inverse incomplete gamma function as a falls
%     linearly to 0: wide exploration first, then an ever tighter local search
%
% Reference:
% Berat Dogan, Tamer Olmez,
% A new metaheuristic for numerical function optimization: Vortex Search
% algorithm,
% Information Sciences 293 (2015) 125-145.
% https://doi.org/10.1016/j.ins.2014.08.053
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 74257 v1.0.0 (runVS.m,
% 13 Feb 2020, folder VSalgorithm/; the zip on his ITU page was never archived).
% Its scalar limits become per-dimension: Mu = (lb + ub)/2 and r scales
% (ub - lb)/2 per coordinate, not the paper's (max(ub) - min(lb))/2. MaxItr is
% counted from the budget, so a ends at 1/MaxItr at maxFe. Kept as released:
% ONE rand re-draws every coordinate below lb and one more every coordinate
% above ub (Eq. 5 draws per coordinate; the zip's runVS.asv clamps), and
% candidates equal to the centre once r underflows late in the run are still
% evaluated. The reported best is updated at every evaluation; the recorded
% population is the iteration's candidate set.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = vs(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    psize  = 50;
    x      = 0.1;
    MaxItr = max(1, ceil(maxFE / psize));   % one batch of psize evaluations per iteration
    itr    = MaxItr;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Mu   = 0.5 * (ub + lb);
    ginv = (1 / x) * gammaincinv(x, 1);   % a = 1 in the first iteration
    r    = ginv * ((ub - lb) / 2);

    % Until a candidate beats Inf the centre stays put (the release would stop on an undefined gbest)
    best_fitness  = Inf;
    best_solution = Mu;

    while FE < maxFE
        Cs = r .* randn(psize, dim) + Mu;

        % One draw per side, shared by every coordinate that left the box, as released
        out = Cs < lb;
        rep = repmat(rand * (ub - lb) + lb, psize, 1);
        Cs(out) = rep(out);
        out = Cs > ub;
        rep = repmat(rand * (ub - lb) + lb, psize, 1);
        Cs(out) = rep(out);

        nev = min(psize, maxFE - FE);
        [ObjVal, FE] = calculate_fitness(Cs(1:nev, :)', problem, FE);
        ObjVal = ObjVal(:)';

        for e = 1:nev
            if ObjVal(e) < best_fitness
                best_fitness  = ObjVal(e);
                best_solution = Cs(e, :);
            end
            ec = FE - nev + e;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Cs(1:nev, :), ObjVal, population_history, fitness_history, ...
                history_index, maxFE);
        end

        Mu = best_solution;   % the centre always moves to the best solution so far

        itr  = itr - 1;
        a    = itr / MaxItr;
        ginv = (1 / x) * gammaincinv(x, a);
        r    = ginv * ((ub - lb) / 2);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% ----------------------------------------------------------------------- %
% Squirrel Search Algorithm (SSA)
% Stored as sqsa; the acronym SSA collides with ssa (Salp Swarm)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n    = 50                        % Flying squirrels (1 hickory, 3 acorn, 46 normal)
%   Gc   = 1.9                       % Gliding constant
%   Pdp  = 0.1                       % Predator presence probability
%   dg   = 8*CL/(0.60*18)            % Gliding distance, CL ~ U[0.675, 1.5] -> [0.5, 1.11]
%   beta = 1.5                       % Levy index of the end-of-winter relocation
%   Smin = 1e-5/365^(2.5*t/tm)       % Seasonal threshold, tm counted from the budget
%
% Algorithm Concept:
%   - Sorted by fitness: the best squirrel sits on the hickory tree, the next
%     three on acorn trees, the rest on normal trees
%   - Acorn squirrels glide toward the hickory tree, x + dg*Gc*(x_ht - x)
%   - Normal squirrels are split at random: some glide toward a random acorn
%     tree, the others toward the hickory tree, with the same rule
%   - With probability Pdp a predator appears and the squirrel lands at a
%     uniformly random point instead
%   - Seasonal monitoring: once an acorn squirrel is within Smin of the hickory
%     one, the normal squirrels are relocated by a Levy flight (Eq. 14)
%
% Reference:
% Mohit Jain, Vijander Singh, Asha Rani,
% A novel nature-inspired algorithm for optimization: Squirrel search algorithm,
% Swarm and Evolutionary Computation 44 (2019) 148-175.
% https://doi.org/10.1016/j.swevo.2018.02.013
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the paper's equations and Algorithm 1: no author code was found;
% mealpy SquirrelSA served as a cross-check only (its Smin and Sc differ from
% Eqs. 12-13). Where the paper is silent: roles are re-ranked every iteration;
% the normal squirrels are split by a random permutation, 1..45 of them (count
% uniform) heading to the hickory tree and the rest each to one of the three
% acorn trees at random; Sc is taken at the start of the iteration, and
% Sc < Smin for any acorn squirrel relocates every normal squirrel; moves are
% clamped to the box. Kept as printed: Eq. (14) adds a small Levy step to the
% LOWER bound, so relocated squirrels land near or on lb. The unmoved hickory
% squirrel is not re-evaluated (49 FEs per iteration where the paper counts 50).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sqsa(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    n    = 50;
    nAc  = 3;      % acorn trees; the hickory tree is the 4th food source
    Gc   = 1.9;
    Pdp  = 0.1;
    beta = 1.5;
    hg   = 8;      % height lost in one glide, m
    CD   = 0.60;   % drag coefficient
    sf   = 18;     % scaling factor of the gliding distance
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);   % Eq. (17)
    tm = max(1, ceil((maxFE - n) / (n - 1)));   % iterations the budget allows at n - 1 FEs each

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    m = min(n, maxFE);
    X = lb + rand(m, dim) .* (ub - lb);   % Eq. (2)
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    bsf  = inf;
    bsfx = X(1, :);
    for e = 1:m
        if fit(e) < bsf
            bsf  = fit(e);
            bsfx = X(e, :);
        end
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    t = 0;
    while FE < maxFE
        t = t + 1;

        % Sorting and declaration (Section 4.3): hickory 1, acorn 2..4, normal 5..n
        [fit, ord] = sort(fit, 'ascend');
        X  = X(ord, :);
        ht = X(1, :);
        at = X(2:1 + nAc, :);
        nt = 2 + nAc:n;
        perm = nt(randperm(numel(nt)));
        nH   = randi(numel(nt) - 1);
        toH  = perm(1:nH);
        toA  = perm(nH + 1:end);

        Xnew = X;
        % Case 1, Eq. (4ab): acorn trees -> hickory tree
        for i = 2:1 + nAc
            dg = glide(hg, CD, sf);
            if rand >= Pdp
                Xnew(i, :) = X(i, :) + dg * Gc * (ht - X(i, :));
            else
                Xnew(i, :) = lb + rand(1, dim) .* (ub - lb);
            end
        end
        % Case 2, Eq. (5ab): normal trees -> acorn trees
        for i = toA
            target = at(randi(nAc), :);
            dg = glide(hg, CD, sf);
            if rand >= Pdp
                Xnew(i, :) = X(i, :) + dg * Gc * (target - X(i, :));
            else
                Xnew(i, :) = lb + rand(1, dim) .* (ub - lb);
            end
        end
        % Case 3, Eq. (6ab): normal trees -> hickory tree
        for i = toH
            dg = glide(hg, CD, sf);
            if rand >= Pdp
                Xnew(i, :) = X(i, :) + dg * Gc * (ht - X(i, :));
            else
                Xnew(i, :) = lb + rand(1, dim) .* (ub - lb);
            end
        end

        % Seasonal monitoring condition, Eqs. (12)-(13), then relocation by Eq. (14)
        Sc   = sqrt(sum((at - ht).^2, 2));
        Smin = 10e-6 / 365^(t / (tm / 2.5));
        if any(Sc < Smin)
            for i = nt
                step = 0.01 * randn(1, dim) * sigma ./ abs(randn(1, dim)).^(1 / beta);   % Eq. (16)
                Xnew(i, :) = lb + step .* (ub - lb);
            end
        end
        Xnew = min(max(Xnew, lb), ub);

        % Everyone but the unmoved hickory squirrel is evaluated
        idx = 2:n;
        m = min(numel(idx), maxFE - FE);
        idx = idx(1:m);
        [fnew, FE] = calculate_fitness(Xnew(idx, :)', problem, FE);
        fnew = fnew(:)';
        X(idx, :) = Xnew(idx, :);
        fit(idx)  = fnew;
        for g = 1:m
            if fnew(g) < bsf
                bsf  = fnew(g);
                bsfx = Xnew(idx(g), :);
            end
            ec = FE - m + g;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsfx;
end

% Gliding distance, Eqs. (7)-(11), scaled by sf: lift coefficient CL drawn in [0.675, 1.5]
function dg = glide(hg, CD, sf)
    CL  = 0.675 + (1.5 - 0.675) * rand;
    phi = atan(CD / CL);
    dg  = hg / tan(phi) / sf;
end

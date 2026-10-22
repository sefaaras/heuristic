% ----------------------------------------------------------------------- %
% Rao-3 algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 10                     % Population size, see Implementation Note
%
% Algorithm Concept:
%   - Like Rao-2, a best/worst step plus a random interaction with a partner k
%     drawn per candidate (k ~= i), with absolute values placed differently
%   - x' = x + r1.*(x_best - |x_worst|) + r2.*(|x_better| - x_worse), where
%     better/worse orders the pair (x, x_k) by fitness; r1, r2 per dimension
%   - Greedy acceptance per candidate; best and worst are recomputed at the
%     start of every generation
%
% Reference:
% Ravipudi Venkata Rao,
% Rao algorithms: Three metaphor-less simple algorithms for solving
% optimization problems,
% International Journal of Industrial Engineering Computations 11(1) (2020)
% 107-130.
% https://doi.org/10.5267/j.ijiec.2019.6.002
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's MATLAB code (Rao-3 unconstrained, PDF of 31 Jan 2020
% on sites.google.com/view/raoalgorithms, identical to the paper's Appendix 3).
% pop = 10 is that code's value; the paper tunes it per function (5 to 100,
% mostly 20 to 50 for Rao-3). As in the code, r1 and r2 are drawn per candidate
% and dimension, where the paper's r(1,j,i), r(2,j,i) carry no candidate index,
% and a tie f(i) = f(k) counts the partner as better. The code runs
% floor(maxFes/pop) generations after the initial population; here the last
% generation is cut at maxFe. The clamp is min/max, identical to the code's
% for finite points.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rao3(problem)

    dim   = problem.dimension;
    mini  = problem.lb;
    maxi  = problem.ub;
    maxFE = problem.maxFe;

    pop = 10;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = zeros(pop, dim);
    for i = 1:dim
        x(:, i) = mini(i)+(maxi(i)-mini(i))*rand(pop, 1);
    end
    [f, FE] = calculate_fitness(x', problem, FE);
    f = f(:);

    [best_fitness, bidx] = min(f);
    best_solution = x(bidx, :);
    for e = 1:min(pop, maxFE)
        curve(e) = min(f(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, x, f, population_history, fitness_history, ...
            history_index, maxFE);
    end

    xnew = zeros(pop, dim);
    while FE < maxFE
        [~, tindex] = min(f);
        Best = x(tindex, :);
        [~, windex] = max(f);
        worst = x(windex, :);

        for i = 1:pop
            k = randi(pop);
            while k == i
                k = randi(pop);
            end
            r = rand(2, dim);   % column j is the code's rand(1,2) for variable j
            % Eq. (3)
            if f(i) < f(k)
                xnew(i, :) = x(i, :)+r(1, :).*(Best-abs(worst))+r(2, :).*(abs(x(i, :))-x(k, :));
            else
                xnew(i, :) = x(i, :)+r(1, :).*(Best-abs(worst))+r(2, :).*(abs(x(k, :))-x(i, :));
            end
        end
        xnew = min(max(xnew, mini), maxi);

        n = min(pop, maxFE - FE);   % the last generation is cut at maxFe
        [fnew, FE] = calculate_fitness(xnew(1:n, :)', problem, FE);
        fnew = fnew(:);
        for i = 1:n
            if fnew(i) < f(i)
                x(i, :) = xnew(i, :);
                f(i) = fnew(i);
            end
            if fnew(i) < best_fitness
                best_fitness  = fnew(i);
                best_solution = xnew(i, :);
            end
            ec = FE - n + i;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, x, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

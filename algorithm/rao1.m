% ----------------------------------------------------------------------- %
% Rao-1 algorithm
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 10                     % Population size, see Implementation Note
%
% Algorithm Concept:
%   - One move per candidate and generation along the best-minus-worst
%     direction, x' = x + r.*(x_best - x_worst), r drawn per dimension
%   - No interaction between candidates beyond the shared best and worst
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
% Ported from the author's MATLAB code (Rao-1 unconstrained, PDF of 31 Jan 2020
% on sites.google.com/view/raoalgorithms, identical to the paper's Appendix 1).
% pop = 10 is that code's value; the paper tunes it per function (5 to 100,
% mostly 10 or 20). As in the code, r is drawn per candidate and dimension,
% where the paper's r(1,j,i) carries no candidate index. The code runs
% floor(maxFes/pop) generations after the initial population; here the last
% generation is cut at maxFe. The clamp is min/max, identical to the code's
% for finite points.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rao1(problem)

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

    while FE < maxFE
        [~, tindex] = min(f);
        Best = x(tindex, :);
        [~, windex] = max(f);
        worst = x(windex, :);

        % Eq. (1); rand(dim, pop)' keeps the code's candidate-by-candidate draw order
        xnew = x + rand(dim, pop)' .* (Best - worst);
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

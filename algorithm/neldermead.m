% ----------------------------------------------------------------------- %
% Nelder-Mead Simplex with Random Restarts (NM)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   rho = 1, chi = 2            % Reflection and expansion coefficients
%   gamma = 0.5, sigma = 0.5    % Contraction and shrink coefficients
%   step = 0.05                 % Initial simplex edge, a fraction of the box
%   tol = 1e-12                 % Simplex diameter that triggers a restart
%
% Algorithm Concept:
%   - Keeps D+1 vertices and moves the worst one through the centroid of the
%     others: reflection, then expansion if the reflection is the new best, or
%     contraction if it is no better than the second worst
%   - When neither helps, all vertices shrink towards the best one
%   - Derivative-free and strictly local: it converges on one basin and stops
%   - A collapsed simplex is restarted from a new random point, which is what
%     turns the local method into a global search that can spend a whole budget
%
% Reference:
% John A. Nelder, Roger Mead,
% A Simplex Method for Function Minimization,
% The Computer Journal 7 (1965) 308-313.
% https://doi.org/10.1093/comjnl/7.4.308
% ----------------------------------------------------------------------- %
% Implementation Note:
% The coefficients are the standard ones and the initial simplex follows
% fminsearch: the starting point plus 5 % of the box along each coordinate.
% Two additions the paper does not have, both needed to make a local method a
% benchmark entrant: vertices are clamped to the box, and a simplex whose
% diameter falls below 1e-12 is restarted around a fresh uniform point, so the
% run keeps spending its budget instead of stalling on the first basin it finds.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = neldermead(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    rho = 1;
    chi = 2;
    gamma = 0.5;
    shrink = 0.5;
    step = 0.05;
    tol = 1e-12;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    best_fitness = inf;
    best_solution = zeros(1, dim);

    V = [];
    fV = [];

    while FE < maxFE
        if isempty(V)
            x0 = lb + rand(1, dim) .* (ub - lb);
            V = repmat(x0, dim + 1, 1);
            for j = 1:dim
                V(j + 1, j) = x0(j) + step * (ub(j) - lb(j));
            end
            V = min(max(V, repmat(lb, dim + 1, 1)), repmat(ub, dim + 1, 1));
            [fV, FE] = evaluate(V, problem, FE);
            [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                track(V, fV, FE, size(V, 1), best_fitness, best_solution, curve, ...
                      population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE
                break;
            end
        end

        [fV, order] = sort(fV);
        V = V(order, :);

        centroid = mean(V(1:dim, :), 1);   % all but the worst vertex
        xr = clamp(centroid + rho * (centroid - V(end, :)), lb, ub);
        [fr, FE] = evaluate(xr, problem, FE);

        if fr < fV(1)
            xe = clamp(centroid + chi * (xr - centroid), lb, ub);
            [fe, FE] = evaluate(xe, problem, FE);
            if fe < fr
                V(end, :) = xe; fV(end) = fe;
            else
                V(end, :) = xr; fV(end) = fr;
            end
            new_points = [xr; xe]; new_fits = [fr; fe];
        elseif fr < fV(dim)
            V(end, :) = xr; fV(end) = fr;
            new_points = xr; new_fits = fr;
        else
            if fr < fV(end)
                xc = clamp(centroid + gamma * (xr - centroid), lb, ub);   % outside contraction
            else
                xc = clamp(centroid - gamma * (centroid - V(end, :)), lb, ub);  % inside
            end
            [fc, FE] = evaluate(xc, problem, FE);
            if fc < min(fr, fV(end))
                V(end, :) = xc; fV(end) = fc;
                new_points = [xr; xc]; new_fits = [fr; fc];
            else
                V(2:end, :) = V(ones(dim, 1), :) + shrink * (V(2:end, :) - V(ones(dim, 1), :));
                V = clamp(V, lb, ub);
                [fshrink, FE] = evaluate(V(2:end, :), problem, FE);
                fV(2:end) = fshrink;
                new_points = [xr; xc; V(2:end, :)]; new_fits = [fr; fc; fshrink];
            end
        end

        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            track(new_points, new_fits, FE, numel(new_fits), best_fitness, best_solution, ...
                  curve, population_history, fitness_history, history_index, maxFE);

        if max(max(V, [], 1) - min(V, [], 1)) < tol
            V = [];    % collapsed, restart somewhere else
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

function [f, FE] = evaluate(X, problem, FE)
    [f, FE] = calculate_fitness(X', problem, FE);
    f = f(:);
end

function X = clamp(X, lb, ub)
    n = size(X, 1);
    X = min(max(X, repmat(lb, n, 1)), repmat(ub, n, 1));
end

% The simplex is the population the recorder sees; new_points are the FEs just spent
function [bf, bx, curve, ph, fh, hi] = track(new_points, new_fits, FE, n_new, bf, bx, ...
                                             curve, ph, fh, hi, maxFE)
    for i = 1:n_new
        if new_fits(i) < bf
            bf = new_fits(i);
            bx = new_points(i, :);
        end
        eval_count = FE - n_new + i;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bf;
            [ph, fh, hi] = record_history(eval_count, new_points, new_fits', ph, fh, hi, maxFE);
        end
    end
end

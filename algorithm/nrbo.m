% ----------------------------------------------------------------------- %
% Newton-Raphson-Based Optimizer (NRBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                      % Population size
%   DF = 0.6                    % Probability of applying the trap avoidance operator
%   delta = (1 - 2t)^5          % Step damping, t the spent budget ratio
%
% Algorithm Concept:
%   - The Newton-Raphson search rule replaces the derivative with a finite
%     difference over the best and worst positions, so a step is taken towards
%     the root of the interpolated gradient rather than along a fitness gradient
%   - The rule is applied twice, the second time around two probe points yp and
%     yq straddling the incumbent, which refines the step
%   - Two candidates, one anchored at the individual and one at the best, are
%     mixed with a third damped by delta into the trial vector
%   - With probability DF a trap avoidance operator replaces the trial by a
%     random combination of the best, the individual and the population mean
%   - Greedy selection per individual; the worst position is tracked because the
%     search rule needs it
%
% Reference:
% R. Sowmya, M. Premkumar, Pradeep Jangir,
% Newton-Raphson-based optimizer: A new population-based metaheuristic algorithm
% for continuous optimization problems,
% Engineering Applications of Artificial Intelligence 128 (2024) 107532.
% https://doi.org/10.1016/j.engappai.2023.107532
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released code (github.com/mpremme/
% Newton-Raphson-Based-Optimizer-NRBO-), which schedules delta on it/MaxIt with
% MaxIt = 500; here it runs on the spent budget FE/maxFe, so the schedule is
% traversed whatever the budget and FE >= maxFe is the only terminator.
% Both search-rule steps divide by a difference that vanishes when the best, the
% worst and the incumbent are collinear in a coordinate, which the reference
% leaves to produce Inf or NaN. NaN passes a clamp untouched and CEC2020RW scores
% such a point at its clamped corner, so a non-finite coordinate is redrawn
% uniformly in its own box, as in avoa and gto.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nrbo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    DF = 0.6;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    Position = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    [Fitness, FE] = calculate_fitness(Position', problem, FE);
    Fitness = Fitness(:);

    [~, order] = sort(Fitness);
    best_fitness = Fitness(order(1));
    best_solution = Position(order(1), :);
    Worst_Cost = Fitness(order(end));
    Worst_Pos = Position(order(end), :);

    for i = 1:min(N, maxFE)
        curve(i) = min(Fitness(1:i));
        [population_history, fitness_history, history_index] = record_history(...
            i, Position, Fitness', population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        delta = (1 - 2 * FE / maxFE) ^ 5;   % damping, negative once the budget is half spent

        for i = 1:N
            if FE >= maxFE
                break;
            end

            P1 = randperm(N, 2);
            rho = rand * (best_solution - Position(i, :)) ...
                + rand * (Position(P1(1), :) - Position(P1(2), :));

            NRSR = search_rule(best_solution, Worst_Pos, Position(i, :), rho, dim);
            X1 = Position(i, :) - NRSR + rho;
            X2 = best_solution - NRSR + rho;

            X3 = Position(i, :) - delta * (X2 - X1);
            a1 = rand(1, dim);
            a2 = rand(1, dim);
            Xnew = a1 .* (a1 .* X1 + (1 - a2) .* X2) + (1 - a2) .* X3;

            if rand < DF
                theta1 = -1 + 2 * rand;
                theta2 = -0.5 + rand;
                beta = rand < 0.5;
                u1 = beta * 3 * rand + (1 - beta);
                u2 = beta * rand + (1 - beta);
                if u1 < 0.5
                    Xnew = Xnew + theta1 * (u1 * best_solution - u2 * Position(i, :)) ...
                         + theta2 * delta * (u1 * mean(Position) - u2 * Position(i, :));
                else
                    Xnew = best_solution + theta1 * (u1 * best_solution - u2 * Position(i, :)) ...
                         + theta2 * delta * (u1 * mean(Position) - u2 * Position(i, :));
                end
            end

            Xnew = bound_check(Xnew, lb, ub);

            [Xnew_Cost, FE] = calculate_fitness(Xnew', problem, FE);
            Xnew_Cost = Xnew_Cost(1);

            if Xnew_Cost < Fitness(i)
                Position(i, :) = Xnew;
                Fitness(i) = Xnew_Cost;
                if Fitness(i) < best_fitness
                    best_fitness = Fitness(i);
                    best_solution = Position(i, :);
                end
            end

            if Xnew_Cost > Worst_Cost
                Worst_Cost = Xnew_Cost;
                Worst_Pos = Xnew;
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = best_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Position, Fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Newton-Raphson search rule, Eq. (10) and Eq. (16) of the reference
function NRSR = search_rule(Best_Pos, Worst_Pos, Position, rho, dim)
    DelX = rand(1, dim) .* abs(Best_Pos - Position);
    NRSR = randn * ((Best_Pos - Worst_Pos) .* DelX) ./ (2 * (Best_Pos + Worst_Pos - 2 * Position));
    Xa = Position - NRSR + rho;
    r1 = rand;
    r2 = rand;
    yp = r1 * (mean(Xa + Position) + r1 * DelX);
    yq = r2 * (mean(Xa + Position) - r2 * DelX);
    NRSR = randn * ((yp - yq) .* DelX) ./ (2 * (yp + yq - 2 * Position));
end

% A NaN has no side to clamp to, so it is redrawn inside the box
function x = bound_check(x, lb, ub)
    nf = ~isfinite(x);
    x(nf) = lb(nf) + rand(1, sum(nf)) .* (ub(nf) - lb(nf));
    x = min(max(x, lb), ub);
end

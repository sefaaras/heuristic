% ----------------------------------------------------------------------- %
% Bacterial Foraging Optimization (BFO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   S = 50                      % Bacteria in the colony
%   Ns = 4                      % Longest swim in one direction
%   Nre = 4, Ned = 2            % Reproduction steps and elimination events
%   Ped = 0.25                  % Probability a bacterium is dispersed
%   C = 0.1 % of the box        % Chemotactic step length
%   Nc                          % Chemotactic steps, counted from the budget
%
% Algorithm Concept:
%   - Chemotaxis: a bacterium tumbles to a random unit direction and takes a
%     step; while the step keeps improving it swims on in the same direction,
%     up to Ns times, which is a cheap line search
%   - Health is the sum of the costs a bacterium met over its chemotactic life,
%     so a bacterium that spent its life in a good region is healthy
%   - Reproduction: the healthier half is copied over the other half, and the
%     copies start from the same place
%   - Elimination-dispersal: each bacterium is moved to a random position with
%     probability Ped, which is what keeps the colony from settling in one basin
%
% Reference:
% Kevin M. Passino,
% Biomimicry of bacterial foraging for distributed optimization and control,
% IEEE Control Systems Magazine 22 (2002) 52-67.
% https://doi.org/10.1109/MCS.2002.1004010
% ----------------------------------------------------------------------- %
% Implementation Note:
% Written from the paper's algorithm with its own example parameters
% (S = 50, Ns = 4, Nre = 4, Ned = 2, Ped = 0.25); the step length is 0.1 % of
% each coordinate's range rather than an absolute 0.1, so it means the same on
% every suite's box. Nc is not fixed but counted from the budget, since the
% paper's 100 belongs to its own evaluation count; a schedule that ends with
% budget left restarts, which is what makes Ned and Nre meaningful at any budget.
% Cell-to-cell signalling (Jcc) is left out, as in the comparisons this baseline
% is normally used for -- with it, the cost being minimised is no longer the
% objective alone.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bfo(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    S = 50;
    Ns = 4;
    Nre = 4;
    Ned = 2;
    Ped = 0.25;
    C = 0.001 * (ub - lb);                                  % 0.1 % of each range
    Nc = max(1, floor(maxFE / (Ned * Nre * S * (1 + Ns))));  % what the budget affords

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    theta = repmat(lb, S, 1) + rand(S, dim) .* repmat(ub - lb, S, 1);
    [J, FE] = calculate_fitness(theta', problem, FE);
    J = J(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    for i = 1:min(S, maxFE)
        if J(i) < best_fitness
            best_fitness = J(i);
            best_solution = theta(i, :);
        end
        curve(i) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            i, theta, J', population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        for ell = 1:Ned
            for k = 1:Nre
                health = zeros(S, 1);
                for j = 1:Nc
                    for i = 1:S
                        if FE >= maxFE
                            break;
                        end
                        Jlast = J(i);
                        delta = 2 * rand(1, dim) - 1;
                        direction = delta / sqrt(sum(delta .^ 2));

                        theta(i, :) = clamp(theta(i, :) + C .* direction, lb, ub);
                        [J(i), FE] = evaluate(theta(i, :), problem, FE);
                        health(i) = health(i) + J(i);
                        [best_fitness, best_solution, curve, population_history, ...
                         fitness_history, history_index] = track(theta, J, i, FE, ...
                            best_fitness, best_solution, curve, population_history, ...
                            fitness_history, history_index, maxFE);

                        m = 0;
                        while m < Ns && J(i) < Jlast && FE < maxFE
                            m = m + 1;
                            Jlast = J(i);
                            theta(i, :) = clamp(theta(i, :) + C .* direction, lb, ub);
                            [J(i), FE] = evaluate(theta(i, :), problem, FE);
                            health(i) = health(i) + J(i);
                            [best_fitness, best_solution, curve, population_history, ...
                             fitness_history, history_index] = track(theta, J, i, FE, ...
                                best_fitness, best_solution, curve, population_history, ...
                                fitness_history, history_index, maxFE);
                        end
                    end
                    if FE >= maxFE, break; end
                end

                % The healthier half replaces the other half
                [~, order] = sort(health);
                keep = order(1:floor(S / 2));
                theta = [theta(keep, :); theta(keep, :)];
                J = [J(keep); J(keep)];
                if size(theta, 1) < S
                    theta(end+1, :) = theta(1, :); %#ok<AGROW>
                    J(end+1, 1) = J(1);            %#ok<AGROW>
                end
                if FE >= maxFE, break; end
            end

            dispersed = rand(S, 1) < Ped;
            n_disp = sum(dispersed);
            if n_disp > 0 && FE < maxFE
                theta(dispersed, :) = repmat(lb, n_disp, 1) + ...
                    rand(n_disp, dim) .* repmat(ub - lb, n_disp, 1);
                [Jd, FE] = calculate_fitness(theta(dispersed, :)', problem, FE);
                J(dispersed) = Jd(:);
                idx = find(dispersed);
                for q = 1:n_disp
                    [best_fitness, best_solution, curve, population_history, ...
                     fitness_history, history_index] = track(theta, J, idx(q), ...
                        FE - n_disp + q, best_fitness, best_solution, curve, ...
                        population_history, fitness_history, history_index, maxFE);
                end
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

function [f, FE] = evaluate(x, problem, FE)
    [f, FE] = calculate_fitness(x', problem, FE);
    f = f(1);
end

function x = clamp(x, lb, ub)
    x = min(max(x, lb), ub);
end

function [bf, bx, curve, ph, fh, hi] = track(theta, J, i, FE, bf, bx, curve, ph, fh, hi, maxFE)
    if J(i) < bf
        bf = J(i);
        bx = theta(i, :);
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bf;
        [ph, fh, hi] = record_history(FE, theta, J', ph, fh, hi, maxFE);
    end
end

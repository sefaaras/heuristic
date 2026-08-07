% ----------------------------------------------------------------------- %
% Glider Snake Optimizer (GSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   sol_count     = 30    % Number of solutions (gliding snakes)
%   mutation_rate = 0.5   % Probability of the chain-index gliding move
%
% Algorithm Concept:
%   - The swarm is sorted every iteration, so solution 1 is the leader and
%     solution s-1 is the immediate predecessor in the chain
%   - Gliding move (default): each snake follows both the global leader and
%     its own predecessor, scaled by the linearly decaying amplitude
%     A = 1 - t/T
%   - Chain-index move (mutation branch, lower half of the ranking):
%     the snake is re-positioned along a randomly selected chain member,
%     modulated by the leader/non-leader fitness ratios
%
% Reference:
% El-Sayed M. El-kenawy, Nima Khodadadi, Seyedali Mirjalili et al.,
% Glider snake optimizer (GSO): a nature-inspired metaheuristic algorithm
% for global and engineering optimization problems,
% Artificial Intelligence Review 59, 91 (2026).
% https://doi.org/10.1007/s10462-026-11504-x
% ----------------------------------------------------------------------- %
% Implementation Note:
% The chain-index move adds two raw fitness ratios. The reference guards only the
% zero denominator of the non-leader ratio; the leader ratio is left open, and on
% CEC2020RW RC25, whose interior pole really returns Inf, Inf/Inf turned the
% swarm NaN. Both ratios now fall back to zero whenever they are not finite,
% which is the reference's own treatment of a ratio that carries no information.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gso(problem)

    dimensions  = problem.dimension;
    lower_bound = problem.lb;
    upper_bound = problem.ub;
    maxFE       = problem.maxFe;

    sol_count        = 30;
    mutation_rate    = 0.5;
    iterations_count = max(1, ceil(maxFE / sol_count));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    solutions = zeros(sol_count, dimensions);
    for s = 1:sol_count
        for d = 1:dimensions
            solutions(s, d) = lower_bound(d) + rand() * (upper_bound(d) - lower_bound(d));
        end
    end
    solutions_fitness = inf(sol_count, 1);

    bsf = inf;
    bsx = solutions(1, :);

    for t = 1:iterations_count
        if FE >= maxFE, break; end

        % Bound clamp then evaluate the whole swarm
        for s = 1:sol_count
            for d = 1:dimensions
                if solutions(s, d) > upper_bound(d)
                    solutions(s, d) = upper_bound(d);
                elseif solutions(s, d) < lower_bound(d)
                    solutions(s, d) = lower_bound(d);
                end
            end
        end

        [solutions_fitness, FE] = calculate_fitness(solutions', problem, FE);
        solutions_fitness = solutions_fitness(:);

        [solutions_fitness, sort_idx] = sort(solutions_fitness);
        solutions = solutions(sort_idx, :);

        leader_fitness = solutions_fitness(1);
        if leader_fitness < bsf
            bsf = leader_fitness;
            bsx = solutions(1, :);
        end

        for k = 1:sol_count
            ec = FE - sol_count + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, solutions, solutions_fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        A = 1 - t / iterations_count;

        for s = sol_count:-1:2
            m  = rand();
            rs = sort(randperm(sol_count, 3));
            rs1 = rs(1);
            rs2 = rs(2);
            rs3 = rs(3);

            for d = 1:dimensions
                if mutation_rate > m && s > 0.5 * sol_count
                    chain_index = rs1 / sol_count;
                    led_d = leader_fitness / solutions_fitness(s);
                    if solutions_fitness(rs3) ~= 0
                        nLed_d = solutions_fitness(rs2) / solutions_fitness(rs3);
                    else
                        nLed_d = 0;
                    end
                    % Same fallback the reference already uses for a meaningless ratio
                    if ~isfinite(led_d),  led_d  = 0; end
                    if ~isfinite(nLed_d), nLed_d = 0; end
                    solutions(s, d) = chain_index * solutions(rs1, d) + A * (led_d + nLed_d);
                else
                    sd = solutions(s, d);
                    d_gLeader = solutions(1, d) - sd;
                    d_mLeader = solutions(s - 1, d) - sd;
                    solutions(s, d) = sd + A * (d_gLeader + d_mLeader);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

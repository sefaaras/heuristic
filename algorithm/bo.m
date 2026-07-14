% ----------------------------------------------------------------------- %
% Bonobo Optimizer (BO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 30                     % Population size
%   p_xgm_initial = 0.03       % Initial extra-group mating probability
%   scab = 1.25, scsb = 1.3    % Sharing coefficients (alpha / selected bonobo)
%   rcpp = 0.0035              % Rate of change in phase probability
%   tsgs_factor_max = 0.05     % Max temporary sub-group size factor
%
% Algorithm Concept:
%   - Inspired by the fission-fusion social strategy and reproductive
%     behaviour of bonobos
%   - Self-adjusting phase/directional probabilities balance the four
%     mating strategies (promiscuous, restrictive, consortship, extra-group)
%
% Reference:
% Amit Kumar Das, Dilip Kumar Pratihar,
% Bonobo optimizer (BO): an intelligent heuristic with self-adjusting
% parameters over continuous spaces and its applications to engineering problems,
% Applied Intelligence 52 (2022) 2942-2974
% https://doi.org/10.1007/s10489-021-02444-w
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bo(problem)

    % Extract problem parameters
    d = problem.dimension;
    Var_min = problem.lb;
    Var_max = problem.ub;
    maxFE = problem.maxFe;

    N = 30;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, N, d);
    fitness_history = zeros(history_size, N);
    history_index = 1;

    % Algorithm-specific parameters
    p_xgm_initial = 0.03;
    scab = 1.25;
    scsb = 1.3;
    rcpp = 0.0035;
    tsgs_factor_max = 0.05;

    % Initialization
    cost = zeros(N, 1);
    bonobo = zeros(N, d);
    for i = 1:N
        bonobo(i, :) = unifrnd(Var_min, Var_max, [1 d]);
    end
    [cost, FE] = calculate_fitness(bonobo', problem, FE);
    cost = cost(:);

    [bestcost, ID] = min(cost);
    alphabonobo = bonobo(ID, :);
    pbestcost = bestcost;

    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = bestcost;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, bonobo, cost, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    % Other parameters
    npc = 0;
    ppc = 0;
    p_xgm = p_xgm_initial;
    tsgs_factor_initial = 0.5 * tsgs_factor_max;
    tsgs_factor = tsgs_factor_initial;
    p_p = 0.5;
    p_d = 0.5;

    while FE < maxFE
        FE_before = FE;
        tsgs_max = max(2, ceil(N * tsgs_factor));

        for i = 1:N
            if FE >= maxFE, break; end
            newbonobo = zeros(1, d);
            B = 1:N;
            B(i) = [];
            % Actual size of the temporary sub-group
            tsg = randi([2 tsgs_max]);
            % Selection of pth Bonobo (fission-fusion) and flag determination
            q = randsample(B, tsg);
            temp_cost = cost(q);
            [~, ID1] = min(temp_cost);
            p = q(ID1);
            if (cost(i) < cost(p))
                p = q(randi([1 tsg]));
                flag = 1;
            else
                flag = -1;
            end
            % Creation of newbonobo
            if (rand <= p_p)
                r1 = rand(1, d);   % Promiscuous or restrictive mating strategy
                newbonobo = bonobo(i, :) + scab * r1 .* (alphabonobo - bonobo(i, :)) + flag * scsb * (1 - r1) .* (bonobo(i, :) - bonobo(p, :));
            else
                for j = 1:d
                    if (rand <= p_xgm)
                        rand_var = rand;   % Extra group mating strategy
                        if (alphabonobo(1, j) >= bonobo(i, j))
                            if (rand <= (p_d))
                                beta1 = exp(((rand_var)^2) + rand_var - (2 / rand_var));
                                newbonobo(1, j) = bonobo(i, j) + beta1 * (Var_max(j) - bonobo(i, j));
                            else
                                beta2 = exp((-((rand_var)^2)) + (2 * rand_var) - (2 / rand_var));
                                newbonobo(1, j) = bonobo(i, j) - beta2 * (bonobo(i, j) - Var_min(j));
                            end
                        else
                            if (rand <= (p_d))
                                beta1 = exp(((rand_var)^2) + (rand_var) - 2 / rand_var);
                                newbonobo(1, j) = bonobo(i, j) - beta1 * (bonobo(i, j) - Var_min(j));
                            else
                                beta2 = exp((-((rand_var)^2)) + (2 * rand_var) - 2 / rand_var);
                                newbonobo(1, j) = bonobo(i, j) + beta2 * (Var_max(j) - bonobo(i, j));
                            end
                        end
                    else
                        if ((flag == 1) || (rand <= p_d))   % Consortship mating strategy
                            newbonobo(1, j) = bonobo(i, j) + flag * (exp(-rand)) * (bonobo(i, j) - bonobo(p, j));
                        else
                            newbonobo(1, j) = bonobo(p, j);
                        end
                    end
                end
            end
            % Clipping
            for j = 1:d
                if (newbonobo(1, j) > Var_max(j))
                    newbonobo(1, j) = Var_max(j);
                end
                if (newbonobo(1, j) < Var_min(j))
                    newbonobo(1, j) = Var_min(j);
                end
            end
            [newcost, FE] = calculate_fitness(newbonobo', problem, FE);
            % New bonobo acceptance criteria
            if ((newcost < cost(i)) || (rand <= (p_xgm)))
                cost(i) = newcost;
                bonobo(i, :) = newbonobo;
                if (newcost < bestcost)
                    bestcost = newcost;
                    alphabonobo = newbonobo;
                end
            end
        end

        % Parameters updating
        if (bestcost < pbestcost)
            npc = 0;   % Positive phase
            ppc = ppc + 1;
            cp = min(0.5, (ppc * rcpp));
            pbestcost = bestcost;
            p_xgm = p_xgm_initial;
            p_p = 0.5 + cp;
            p_d = p_p;
            tsgs_factor = min(tsgs_factor_max, (tsgs_factor_initial + ppc * (rcpp^2)));
        else
            npc = npc + 1;   % Negative phase
            ppc = 0;
            cp = -(min(0.5, (npc * rcpp)));
            p_xgm = min(0.5, p_xgm_initial + npc * (rcpp^2));
            tsgs_factor = max(0, (tsgs_factor_initial - npc * (rcpp^2)));
            p_p = 0.5 + cp;
            p_d = 0.5;
        end

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = bestcost;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, bonobo, cost, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = alphabonobo;
    best_fitness = bestcost;

end

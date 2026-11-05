% ----------------------------------------------------------------------- %
% Brain Storm Optimization (BSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_p = 100, n_c = 5           % Ideas and k-means clusters
%   p_replace = 0.2              % Chance one cluster centre becomes a random idea
%   p_one = 0.8                  % Chance a new idea comes from one cluster, else two
%   p_centre_one = 0.4           % One cluster: chance to start from its centre
%   p_centre_two = 0.5           % Two clusters: chance to blend their centres
%   k = 20                       % Slope of the logsig step schedule
%   T = ceil((maxFe - n_p)/n_p)  % Generations the budget allows, the schedule's horizon
%
% Algorithm Concept:
%   - Each generation k-means (city-block, started from the last centres)
%     groups the ideas; the best idea of each cluster is its centre
%   - With chance p_replace one centre is replaced by a random point in the box
%   - A new idea starts from a centre or member of one cluster, chosen by
%     cluster size, or from a random blend of two clusters' centres or members
%   - It moves by logsig((T/2 - t)/k)*U(0, 1)*N(0, 1) per dimension, a step
%     that collapses after half of the run
%   - It replaces the idea with the same index if better; afterwards each
%     cluster's old centre is written back over its own index
%
% Reference:
% Yuhui Shi,
% Brain Storm Optimization Algorithm,
% Advances in Swarm Intelligence (ICSI 2011), Lecture Notes in Computer Science,
% vol. 6728, Springer, 2011, pp. 303-309.
% https://doi.org/10.1007/978-3-642-21515-5_36
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from bso2.m, the code circulated from Shi's group (copies with its comments
% verbatim: Travis-MA/BSO-swarm-robot evaluate/bso.m, the base of NEU-EA/OLBSO); the
% paper and Cheng & Shi's "Codes in MATLAB for BSO" were out of reach, n_p and n_c
% come from restatements. Seeded runs are bit-identical to it. Kept as released:
% U(0, 1) per dimension in the step; the write-back that discards an improvement at
% a centre's index. Changed: the absolute step scales by (ub - lb)/200 (exact on
% [-100, 100]); the release has no bound handling, a new idea is clamped to the box;
% a cluster with no fitness below the release's start value 1e26 (RC12 reaches 1e43)
% takes its best member where the release reads a stale index (0 in generation 1, a
% crash); a cluster k-means left empty borrows a member of the largest.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bso(problem)

    n_d   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    n_p = 100;
    n_c = 5;
    prob_one_cluster = 0.8;
    scale = span / 200;   % the release's absolute step, per dimension
    max_iteration = max(1, ceil((maxFE - n_p) / n_p));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % All four draws as released, so the random stream stays aligned with it
    popu         = lb + span .* rand(n_p, n_d);
    popu_sorted  = lb + span .* rand(n_p, n_d); %#ok<NASGU>
    centers      = lb + span .* rand(n_c, n_d);
    centers_copy = lb + span .* rand(n_c, n_d); %#ok<NASGU>
    best = zeros(n_c, 1);
    prob = zeros(n_c, 1);
    indi_temp = zeros(1, n_d);

    n0 = min(n_p, maxFE);
    [f0, FE] = calculate_fitness(popu(1:n0, :)', problem, FE);
    fitness_popu = inf(n_p, 1);
    fitness_popu(1:n0) = f0(:);

    best_fitness  = inf;
    best_solution = popu(1, :);
    for q = 1:n0
        if fitness_popu(q) < best_fitness
            best_fitness  = fitness_popu(q);
            best_solution = popu(q, :);
        end
        curve(q) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            q, popu(1:n0, :), fitness_popu(1:n0)', population_history, fitness_history, ...
            history_index, maxFE);
    end

    ws = warning('off', 'stats:kmeans:FailedToConverge');
    restore = onCleanup(@() warning(ws));

    n_iteration = 0;
    while FE < maxFE
        cluster = kmeans(popu, n_c, 'Distance', 'cityblock', 'Start', centers, ...
                         'EmptyAction', 'singleton');
        number_in_cluster = accumarray(cluster, 1, [n_c, 1]);
        for c = find(number_in_cluster == 0)'
            [~, big] = max(number_in_cluster);
            cluster(find(cluster == big, 1)) = c;
            number_in_cluster = accumarray(cluster, 1, [n_c, 1]);
        end

        % Best idea per cluster, against the release's start value 1e26
        fit_values = 100000000000000000000000000.0 * ones(n_c, 1);
        found = false(n_c, 1);
        for idx = 1:n_p
            c = cluster(idx);
            if fit_values(c) > fitness_popu(idx)
                fit_values(c) = fitness_popu(idx);
                best(c) = idx;
                found(c) = true;
            end
        end
        for c = find(~found)'
            mine = find(cluster == c);
            [fit_values(c), w] = min(fitness_popu(mine));
            best(c) = mine(w);
        end

        % Ideas listed cluster by cluster, as the release's counting sort leaves them
        acculate_num_cluster = [0; cumsum(number_in_cluster(1:n_c-1))];
        [~, order] = sort(cluster);
        popu_sorted = popu(order, :);

        centers = popu(best, :);
        centers_copy = centers;

        if (rand() < 0.2)
            cenIdx = ceil(rand() * n_c);
            centers(cenIdx, :) = lb + span .* rand(1, n_d);
        end

        for idx = 1:n_c
            prob(idx) = number_in_cluster(idx) / n_p;
            if idx > 1
                prob(idx) = prob(idx) + prob(idx - 1);
            end
        end

        nb = min(n_p, maxFE - FE);
        NEW = zeros(nb, n_d);
        step = 1 / (1 + exp(-((0.5 * max_iteration - n_iteration) / 20)));   % logsig
        for idx = 1:nb
            r_1 = rand();
            if r_1 < prob_one_cluster
                r = rand();
                for idj = 1:n_c
                    if r < prob(idj)
                        if rand() < 0.4
                            indi_temp = centers(idj, :);
                        else
                            indi_1 = acculate_num_cluster(idj) + ceil(rand() * number_in_cluster(idj));
                            indi_temp = popu_sorted(indi_1, :);
                        end
                        break
                    end
                end
            else
                cluster_1 = ceil(rand() * n_c);
                indi_1 = acculate_num_cluster(cluster_1) + ceil(rand() * number_in_cluster(cluster_1));
                cluster_2 = ceil(rand() * n_c);
                indi_2 = acculate_num_cluster(cluster_2) + ceil(rand() * number_in_cluster(cluster_2));
                tem = rand();
                if rand() < 0.5
                    indi_temp = tem * centers(cluster_1, :) + (1 - tem) * centers(cluster_2, :);
                else
                    indi_temp = tem * popu_sorted(indi_1, :) + (1 - tem) * popu_sorted(indi_2, :);
                end
            end
            stepSize = step * rand(1, n_d);
            indi_temp = indi_temp + stepSize .* randn(1, n_d) .* scale;
            indi_temp = min(max(indi_temp, lb), ub);
            NEW(idx, :) = indi_temp;
        end

        [fv, FE_new] = calculate_fitness(NEW', problem, FE);
        fv = fv(:);
        for idx = 1:nb
            if fv(idx) < fitness_popu(idx)
                fitness_popu(idx) = fv(idx);
                popu(idx, :) = NEW(idx, :);
            end
            if fv(idx) < best_fitness
                best_fitness  = fv(idx);
                best_solution = NEW(idx, :);
            end
            curve(FE + idx) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE + idx, popu, fitness_popu', population_history, fitness_history, ...
                history_index, maxFE);
        end
        FE = FE_new;
        if FE >= maxFE, break; end

        % Keep the best for each cluster
        popu(best, :) = centers_copy;
        fitness_popu(best) = fit_values;
        n_iteration = n_iteration + 1;
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

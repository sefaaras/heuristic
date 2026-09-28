% ----------------------------------------------------------------------- %
% Mean-Variance Mapping Optimization, swarm hybrid (MVMO-SH)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n_par = 15*D                % Particles, each with its own archive
%   n_tosave = 5                % Archive size per particle
%   n_random = D/2+1 -> 1       % Coordinates mutated, falling over the budget
%   Indep_run = 2               % Evaluations before a particle joins the swarm
%   fs_factor = 1 -> 20         % Shape scaling, quadratic in the spent budget
%   delta_d = 0.05, d0 = 1      % Alternative shape and its random scaling
%   good share = 0.7 -> 0.2     % Fraction of particles counted as good
%
% Algorithm Concept:
%   - Search runs in the unit cube: every coordinate is normalised by its own
%     range, so one mapping serves every problem
%   - Each particle archives its best solutions; the mean and variance of each
%     coordinate over that archive bend a uniform draw towards the mean
%   - The variance enters as -log(variance), so a coordinate that has settled
%     gets a sharp mapping and one that has not stays broad
%   - Only a subset of coordinates is mutated, shrinking as the budget is spent,
%     and the rest are copied from the parent
%   - Particles are ranked each evaluation: good ones continue from their own best,
%     bad ones are rebuilt from three good ones, and a rare local search polishes
%
% Reference:
% Jose L. Rueda, Istvan Erlich,
% Hybrid Mean-Variance Mapping Optimization for solving the IEEE-CEC 2013
% competition problems,
% 2013 IEEE Congress on Evolutionary Computation (CEC), 2013, pp. 1664-1671.
% https://doi.org/10.1109/CEC.2013.6557761
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released MVMO-SH for CEC2013 (MVMOS_SH.m), the
% single-configuration version (their CEC2015 one tunes a parameter set per
% problem). Only the default variable-selection strategy, mode 4, is implemented;
% the other four are never selected by these settings. The local search keeps its
% fmincon interior-point call, so this algorithm needs the Optimization Toolbox; a
% non-finite start or a failure keeps the incumbent, unguarded in the release.
% fmincon is also cut to the remaining budget, which the release overran by up to
% 1.1 %. As released, local search stops after a second search ends on a nearly
% singular warning; that is read from lastwarn, cleared per run here so an earlier
% parfor job on the same worker cannot trip it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mvmo(problem)

    n_var = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;
    scaling = ub - lb;

    n_par = 15 * n_var;
    n_to_save = 5;
    n_randomly_ini = min(n_var, round(n_var / 2) + 1);
    n_randomly_last = 1;
    Indep_run = 2;
    fs_factor_start = 1;
    fs_factor_end = 20;
    delta_dddd0 = 0.05;
    ratio_gute_max = 0.7;
    ratio_gute_min = 0.2;
    l_vari = 2;
    local_search_prob = 1.5 / 100 / n_var;
    make_sense = maxFE - round(n_var * 100 / 2);

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    x_norm = rand(n_par, n_var);
    x_norm_best = x_norm;
    meann = x_norm;
    meann_app = x_norm;
    shape = zeros(n_par, n_var);
    dddd = ones(n_par, n_var);
    variance = ones(n_par, n_var);
    izm = zeros(n_par, 1);
    no_in = zeros(n_par, 1);
    no_inin = zeros(n_par, 1);
    independent_run_p = zeros(n_par, 1);

    archive_x = zeros(n_to_save, n_var, n_par);
    archive_f = inf(n_to_save, n_par);

    best_fitness = inf;
    best_solution = lb + scaling .* x_norm(1, :);
    delta_nrandomly = n_randomly_ini - n_randomly_last;
    n_singular_ls = 0;
    lastwarn('');

    while FE < maxFE
        ff = FE / maxFE;
        ff2 = ff * ff;
        shift = (1 - ff2 * 0.99) * 0.5;
        border_gute = max(3, round(n_par * (ratio_gute_max - ff * (ratio_gute_max - ratio_gute_min))));
        % nrand_method 41: a uniform draw between the schedule and the final count
        n_randomly_X = round(n_randomly_ini - ff * delta_nrandomly);
        n_randomly = max(1, round(n_randomly_last + rand * (n_randomly_X - n_randomly_last)));
        fs_factor0 = fs_factor_start + ff2 * (fs_factor_end - fs_factor_start);
        delta_dddd1 = 1 + delta_dddd0;

        for ipp = 1:n_par
            if FE >= maxFE
                break;
            end

            FE_prev = FE;
            if rand < local_search_prob && FE < make_sense && FE > 1
                [f_cur, x_norm(ipp, :), FE, singular] = local_search(x_norm(ipp, :), lb, ub, scaling, problem, FE, maxFE);
                % The release gives up local search once a second search runs into a nearly singular system
                n_singular_ls = n_singular_ls + singular;
                if n_singular_ls >= 2
                    local_search_prob = 0;
                end
            else
                [f_cur, FE] = evaluate_norm(x_norm(ipp, :), lb, scaling, problem, FE);
            end

            % A local search is charged as one block, over which the best so far holds
            curve(FE_prev + 1:min(FE, maxFE) - 1) = best_fitness;
            if f_cur < best_fitness
                best_fitness = f_cur;
                best_solution = lb + scaling .* x_norm(ipp, :);
            end
            if FE <= maxFE
                curve(FE) = best_fitness;
            end

            % Archive of this particle, and the mapping shape it implies
            [archive_x, archive_f, no_in, no_inin, meann, variance, shape, x_norm_best] = ...
                fill_archive(archive_x, archive_f, no_in, no_inin, meann, variance, shape, ...
                             x_norm_best, x_norm, ipp, f_cur, n_to_save, n_var, l_vari);

            % Population = each evaluated particle's archive leader, the point its fitness belongs to
            visited = no_in > 0;
            [population_history, fitness_history, history_index] = record_history( ...
                FE, lb + scaling .* x_norm_best(visited, :), archive_f(1, visited), ...
                population_history, fitness_history, history_index, maxFE);

            meann_app(ipp, :) = meann(ipp, :);

            independent_run_p(ipp) = independent_run_p(ipp) + 1;
            if independent_run_p(ipp) >= Indep_run
                [~, IX] = sort(archive_f(1, :));
                is_good = false(n_par, 1);
                is_good(IX(1:border_gute)) = true;
                onep = IX(randi(max(1, border_gute - 2)) + 1);
                bestp = IX(1);
                worstp = IX(border_gute);
                if ~is_good(ipp)
                    % A bad particle is rebuilt from three good ones
                    for jl = 1:n_var
                        beta = 2 * (rand - shift);
                        val = x_norm_best(onep, jl) + beta * (x_norm_best(bestp, jl) - x_norm_best(worstp, jl));
                        tries = 0;
                        while (val > 1 || val < 0) && tries < 100
                            beta = 2 * (rand - shift);
                            val = x_norm_best(onep, jl) + beta * (x_norm_best(bestp, jl) - x_norm_best(worstp, jl));
                            tries = tries + 1;
                        end
                        x_norm(ipp, jl) = min(max(val, 0), 1);
                    end
                    meann_app(ipp, :) = x_norm(ipp, :);
                else
                    x_norm(ipp, :) = x_norm_best(ipp, :);
                end
            else
                x_norm(ipp, :) = x_norm_best(ipp, :);
            end

            % mode 4: one coordinate in sequence, the rest drawn at random
            considered = false(1, n_var);
            izm(ipp) = izm(ipp) - 1;
            if izm(ipp) < 1
                izm(ipp) = n_var;
            end
            considered(izm(ipp)) = true;
            while sum(considered) < min(n_randomly, n_var)
                considered(randi(n_var)) = true;
            end

            x_norm(ipp, considered) = rand(1, sum(considered));

            for ivar = 1:n_var
                if considered(ivar) && shape(ipp, ivar) > 1e-50
                    sss1 = shape(ipp, ivar);
                    sss2 = sss1;
                    delta_ddd_x = delta_dddd0 * (rand - 0.5) * 2 + delta_dddd1;
                    if shape(ipp, ivar) > dddd(ipp, ivar)
                        dddd(ipp, ivar) = dddd(ipp, ivar) * delta_ddd_x;
                    else
                        dddd(ipp, ivar) = dddd(ipp, ivar) / delta_ddd_x;
                    end
                    if randi(2) == 2
                        sss1 = dddd(ipp, ivar);
                    else
                        sss2 = dddd(ipp, ivar);
                    end
                    fs_factor = fs_factor0 * (1 + rand);
                    sss1 = sss1 * fs_factor;
                    sss2 = sss2 * fs_factor;
                    x_norm(ipp, ivar) = h_function(meann_app(ipp, ivar), sss1, sss2, x_norm(ipp, ivar));
                end
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = best_fitness;
end

function [f, FE] = evaluate_norm(x_norm, lb, scaling, problem, FE)
    [fv, FE] = calculate_fitness((lb + scaling .* x_norm)', problem, FE);
    f = fv(1);
end

function [f, x_norm, FE, singular] = local_search(x_norm, lb, ub, scaling, problem, FE, maxFE)
    x0 = lb + scaling .* x_norm;
    singular = false;
    f = evaluate_for_ls(x0(:), problem);
    FE = FE + 1;
    if ~isfinite(f)
        return;
    end
    % interior-point's own default of 3000 evaluations, cut to what is left of the budget
    options = optimset('Display', 'off', 'algorithm', 'interior-point', 'UseParallel', 'never', ...
                       'MaxFunEvals', min(3000, maxFE - FE));
    lastwarn('');
    try
        [xls, fls, ~, output] = fmincon(@(xx) evaluate_for_ls(xx, problem), x0(:), ...
                                        [], [], [], [], lb, ub, [], options);
    catch
        return;
    end
    [~, warn_id] = lastwarn;
    singular = strcmp(warn_id, 'MATLAB:nearlySingularMatrix');
    FE = FE + output.funcCount;
    if all(isfinite(xls))
        f = fls;
        x_norm = ((xls(:)' - lb) ./ scaling);
    end
end

function f = evaluate_for_ls(x, problem)
    f = calculate_fitness(x(:), problem, 0);
    f = f(1);
end

% The archive holds this particle's n_to_save best, sorted, and their statistics
function [archive_x, archive_f, no_in, no_inin, meann, variance, shape, x_norm_best] = ...
        fill_archive(archive_x, archive_f, no_in, no_inin, meann, variance, shape, ...
                     x_norm_best, x_norm, ipp, f_cur, n_to_save, n_var, l_vari)
    no_in(ipp) = no_in(ipp) + 1;
    changed = false;
    i_position = 0;
    if no_in(ipp) == 1
        archive_x(:, :, ipp) = repmat(x_norm(ipp, :), n_to_save, 1);
        archive_f(:, ipp) = repmat(f_cur, n_to_save, 1);
        no_inin(ipp) = no_inin(ipp) + 1;
    elseif f_cur < archive_f(n_to_save, ipp)
        for i = 1:n_to_save
            if f_cur < archive_f(i, ipp)
                i_position = i;
                changed = true;
                if i < n_to_save
                    no_inin(ipp) = no_inin(ipp) + 1;
                end
                break;
            end
        end
    end

    if changed
        nn = min(n_to_save, no_inin(ipp));
        idx = nn:-1:(i_position + 1);
        archive_x(idx, :, ipp) = archive_x(idx - 1, :, ipp);
        archive_f(idx, ipp) = archive_f(idx - 1, ipp);
        archive_x(i_position, :, ipp) = x_norm(ipp, :);
        archive_f(i_position, ipp) = f_cur;

        if no_inin(ipp) >= l_vari
            for ivar = 1:n_var
                [m, v] = mv_noneq(archive_x(1:nn, ivar, ipp));
                meann(ipp, ivar) = m;
                variance(ipp, ivar) = v;
            end
            nz = variance(ipp, :) > 1.1e-100;
            shape(ipp, nz) = -log(variance(ipp, nz));
        end
    end
    x_norm_best(ipp, :) = archive_x(1, :, ipp);
end

% Mean and variance over the DISTINCT archive values, so repeats do not sharpen the shape
function [vmean, vvar] = mv_noneq(values)
    vals = values(:);
    keep = true(numel(vals), 1);
    for i = 2:numel(vals)
        if any(abs(vals(1:i-1) - vals(i)) < 1e-70)
            keep(i) = false;
        end
    end
    vals = vals(keep);
    vmean = mean(vals);
    if numel(vals) > 1
        vvar = sum((vals - vmean) .^ 2) / numel(vals);
    else
        vvar = 1e-100;
    end
end

% The mapping: a uniform draw is bent towards the archive mean, Eq. (2) of the paper
function x = h_function(x_bar, s1, s2, x_p)
    H0 = (1 - x_bar) * exp(-s2);
    x = (x_bar - x_bar / exp(x_p * s1)) + (1 - x_bar) / exp(s2 - x_p * s2) + ...
        (x_bar / exp(s1) + H0) * x_p - H0;
end

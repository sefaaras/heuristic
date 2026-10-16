% ----------------------------------------------------------------------- %
% Snake Optimizer (SO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N          = 30          % Population size, first half male, rest female
%   C1         = 0.5         % Food quantity Q = C1*exp((t - T)/T), Eq. (5)
%   C2         = 0.05        % Exploration step
%   C3         = 2           % Exploitation step
%   Threshold  = 0.25        % Food exists once Q reaches it
%   Threshold2 = 0.6         % Hot while Temp = exp(-t/T) exceeds it, Eq. (4)
%   P_fight    = 0.4         % Cold phase: fight if rand > 0.6, mate otherwise
%
% Algorithm Concept:
%   - No food (Q < 0.25): each snake moves around random members of its own
%     sex, step C2*exp(-f_rand/f_i) times a random box point
%   - Food, hot: around the food (best snake), x_food +- C3*Temp*r*(x_food - x)
%   - Food, cold, fight: toward Q times the best snake of the other sex, step
%     scaled by exp(-f_best,other/f_i)
%   - Food, cold, mating: male i and female i move toward Q times each other;
%     with probability 0.5 an egg replaces the worst candidate of each sex
%   - Greedy acceptance of every candidate, males first, then females
%
% Reference:
% Fatma A. Hashim, Abdelazim G. Hussien,
% Snake Optimizer: A novel meta-heuristic optimization algorithm,
% Knowledge-Based Systems 242 (2022) 108320.
% https://doi.org/10.1016/j.knosys.2022.108320
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 106465 v1.0.0 (SO.m,
% 9 Feb 2022). It draws the leader and sign of Eqs. (6), (8) per dimension,
% and an egg of Eqs. (19), (20) replaces the worst snake's candidate, which
% still has to win the greedy test. Reference defect kept: the egg draws one
% rand for all dimensions, so it lands on the box diagonal lb + r*(ub - lb).
% Not evaluated, though the release spends an FE on each: a candidate equal
% to its snake, which cannot win (the best snake in every hot iteration, about
% 0.7% of the budget), and a non-finite one, left by an overflowing exp(-f/f)
% when fitness is negative, near 0 or -Inf (CEC2020RW), which the release's
% clamp turns into NaN. T counts N evaluations per iteration; t/T is clamped at 1.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = so(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 30;
    vec_flag = [1, -1];
    Threshold = 0.25;
    Threshold2 = 0.6;
    C1 = 0.5;
    C2 = 0.05;
    C3 = 2;
    T = max(1, ceil((maxFE - N) / N));   % every iteration evaluates all N candidates

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [best_fitness, bidx] = min(fit);
    best_solution = X(bidx, :);
    Xfood = X(bidx, :);

    for e = 1:min(N, maxFE)
        curve(e) = min(fit(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Eqs. (2), (3): rows 1..Nm are the males, the rest the females
    Nm = round(N / 2);
    Nf = N - Nm;
    mi = 1:Nm;
    fi = Nm + 1:N;
    [fitnessBest_m, g1] = min(fit(mi));
    Xbest_m = X(mi(g1), :);
    [fitnessBest_f, g2] = min(fit(fi));
    Xbest_f = X(fi(g2), :);

    t = 0;
    while FE < maxFE
        t = t + 1;
        tt = min(t, T);
        Temp = exp(-((tt) / T));            % Eq. (4)
        Q = C1 * exp(((tt - T) / (T)));     % Eq. (5)

        Xm = X(mi, :);
        Xf = X(fi, :);
        fm = fit(mi);
        ff = fit(fi);
        if Q < Threshold
            Xnewm = explore(Xm, fm, Nm, dim, lb, ub, C2, vec_flag);
            Xnewf = explore(Xf, ff, Nf, dim, lb, ub, C2, vec_flag);
        elseif Temp > Threshold2
            % Hot exploitation, Eq. (10): the sign is drawn per snake, r per dimension
            Rm = rand(dim + 1, Nm);
            Flag = vec_flag(floor(2 * Rm(1, :) + 1))';
            Xnewm = Xfood + C3 .* Flag .* Temp .* Rm(2:end, :)' .* (Xfood - Xm);
            Rf = rand(dim + 1, Nf);
            Flag = vec_flag(floor(2 * Rf(1, :) + 1))';
            Xnewf = Xfood + Flag .* C3 .* Temp .* Rf(2:end, :)' .* (Xfood - Xf);
        elseif rand > 0.6
            % Fight, Eqs. (11)-(14)
            FM = exp(-(fitnessBest_f) ./ (fm' + eps));
            Xnewm = Xm + C3 .* FM .* rand(dim, Nm)' .* (Q .* Xbest_f - Xm);
            FF = exp(-(fitnessBest_m) ./ (ff' + eps));
            Xnewf = Xf + C3 .* FF .* rand(dim, Nf)' .* (Q .* Xbest_m - Xf);
        else
            % Mating, Eqs. (15)-(18): male i pairs with female i
            Mm = exp(-ff(1:Nm)' ./ (fm' + eps));
            Xnewm = Xm + C3 .* rand(dim, Nm)' .* Mm .* (Q .* Xf(1:Nm, :) - Xm);
            Mf = exp(-fm(1:Nf)' ./ (ff' + eps));
            Xnewf = Xf + C3 .* rand(dim, Nf)' .* Mf .* (Q .* Xm(1:Nf, :) - Xf);
            if vec_flag(floor(2 * rand() + 1)) == 1
                % Eqs. (19), (20): one scalar rand per egg, as released
                [~, gworst] = max(fm);
                Xnewm(gworst, :) = lb + rand * (ub - lb);
                [~, gworst] = max(ff);
                Xnewf(gworst, :) = lb + rand * (ub - lb);
            end
        end

        Xnew = [Xnewm; Xnewf];
        ok = all(isfinite(Xnew), 2);
        Xnew(ok, :) = min(max(Xnew(ok, :), lb), ub);
        % A candidate equal to its snake cannot pass the strict test below
        idx = find(ok & any(Xnew ~= X, 2));
        idx = idx(1:min(numel(idx), maxFE - FE));
        if isempty(idx)
            continue;
        end
        Xnew = Xnew(idx, :);
        [y, FE] = calculate_fitness(Xnew', problem, FE);
        y = y(:)';
        for k = 1:numel(idx)
            j = idx(k);
            if y(k) < fit(j)
                X(j, :) = Xnew(k, :);
                fit(j) = y(k);
                if y(k) < best_fitness
                    best_fitness = y(k);
                    best_solution = Xnew(k, :);
                end
            end
            ec = FE - numel(idx) + k;
            curve(ec) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end

        % The guides (best of each sex, food) are refreshed once per iteration
        [Ybest1, gbest1] = min(fit(mi));
        [Ybest2, gbest2] = min(fit(fi));
        if Ybest1 < fitnessBest_m
            Xbest_m = X(mi(gbest1), :);
            fitnessBest_m = Ybest1;
        end
        if Ybest2 < fitnessBest_f
            Xbest_f = X(fi(gbest2), :);
            fitnessBest_f = Ybest2;
        end
        if fitnessBest_m < fitnessBest_f
            Xfood = Xbest_m;
        else
            Xfood = Xbest_f;
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% Exploration, Eqs. (6)-(9): leader, sign and box point drawn per dimension, as released
function Xnew = explore(Xs, fs, n, dim, lb, ub, C2, vec_flag)
    R = rand(3, dim, n);
    L = reshape(floor(n * R(1, :, :) + 1), dim, n)';
    Flag = vec_flag(reshape(floor(2 * R(2, :, :) + 1), dim, n)');
    r3 = reshape(R(3, :, :), dim, n)';
    A = exp(-fs(L) ./ (fs(:) + eps));
    Xr = Xs(sub2ind([n, dim], L, repmat(1:dim, n, 1)));
    Xnew = Xr + Flag .* C2 .* A .* ((ub - lb) .* r3 + lb);
end

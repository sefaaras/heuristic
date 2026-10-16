% ----------------------------------------------------------------------- %
% Hippopotamus Optimization Algorithm (HO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents = 24     % Population size (hippopotamuses)
%   T_switch     = 0.6    % Phase 1 uses the leader-relative step while exp(-t/T) > 0.6
%   levy_beta    = 1.5    % Levy index of the predator step, scaled by 0.05
%
% Algorithm Concept:
%   - Phase 1 (first half of the herd): a male step towards the dominant
%     hippopotamus and a female/calf step relative to a random group mean
%   - Once exp(-t/T) <= 0.6 that step moves from the group mean away from the
%     leader, or half the time jumps to lb + r*(ub - lb) with one scalar r
%   - Phase 2 (second half): a random predator is evaluated; the new position
%     is a Levy step on the predator plus a 1/distance term
%   - Phase 3 (whole herd): escape by a random step drawn from the shrinking
%     local box lb/t .. ub/t
%   - Greedy acceptance after every evaluated candidate
%
% Reference:
% Mohammad Hussein Amiri, Nastaran Mehrabi Hashjin, Mohsen Montazeri,
% Seyedali Mirjalili, Nima Khodadadi,
% Hippopotamus optimization algorithm: a novel nature-inspired optimization
% algorithm,
% Scientific Reports 14 (2024) 5032.
% https://doi.org/10.1038/s41598-024-54910-3
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/inimakhan/HO, commit 4afd52a,
% HO.m + levy.m, same as the HO.zip there); N = 24 is the paper's experimental
% setting, the release's Main.m demo uses 16. The release evaluates and keeps
% the Phase-1 male step X_P1 unclamped, so the herd can leave the box; it is
% clamped here like the other candidates. Each candidate and each predator
% costs one evaluation, so an iteration spends 3*N and Max_iter is counted from
% that; a predator never joins the herd but counts towards the reported best.
% Without the X_P1 clamp and drawing as the release does (unifrnd, a full
% levy(N, dim) per hippo) a seeded run matches its curve bit for bit (cec14).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ho(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    N = 24;
    half = N / 2;
    Max_iter = max(1, ceil((maxFE - N) / (3 * N)));   % Phase 1: 2 per male, Phase 2: predator + candidate, Phase 3: 1 each

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(N, dim) .* (ub - lb);
    [fit, FE] = calculate_fitness(X', problem, FE);
    fit = fit(:)';

    [bsf, ib] = min(fit);
    bsx = X(ib, :);
    for e = 1:min(N, maxFE)
        curve(e) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            e, X, fit, population_history, fitness_history, ...
            history_index, maxFE);
    end

    fbest = inf;
    Xbest = X(1, :);
    t = 0;
    while FE < maxFE
        t = t + 1;

        % The dominant hippopotamus is refreshed once per iteration, as in the release
        [best, loc] = min(fit);
        if t == 1 || best < fbest
            fbest = best;
            Xbest = X(loc, :);
        end
        T = exp(-t / Max_iter);   % Eq. (5)

        for i = 1:half
            I1 = randi([1, 2], 1, 1);
            I2 = randi([1, 2], 1, 1);
            Ip1 = randi([0, 1], 1, 2);
            grp = randperm(N, randperm(N, 1));
            if isscalar(grp)
                MG = X(grp, :);
            else
                MG = mean(X(grp, :), 1);
            end
            h = cell(1, 5);   % Eq. (4)
            h{1} = I2 * rand(1, dim) + (~Ip1(1));
            h{2} = 2 * rand(1, dim) - 1;
            h{3} = rand(1, dim);
            h{4} = I1 * rand(1, dim) + (~Ip1(2));
            h{5} = rand;
            A = h{randi([1, 5], 1, 1)};
            B = h{randi([1, 5], 1, 1)};
            Xp1 = X(i, :) + rand(1, 1) .* (Xbest - I1 .* X(i, :));   % Eq. (3)
            if T > 0.6
                Xp2 = X(i, :) + A .* (Xbest - I2 .* MG);   % Eq. (6)
            elseif rand() > 0.5
                Xp2 = X(i, :) + B .* (MG - Xbest);   % Eq. (7)
            else
                Xp2 = (ub - lb) * rand + lb;
            end
            C = min(max([Xp1; Xp2], lb), ub);

            nb = min(2, maxFE - FE);
            [fc, FE] = calculate_fitness(C(1:nb, :)', problem, FE);
            fc = fc(:)';
            for k = 1:nb
                if fc(k) < fit(i)   % Eqs. (8)-(9), the female step is judged against the updated male
                    X(i, :) = C(k, :);
                    fit(i) = fc(k);
                end
                if fc(k) < bsf
                    bsf = fc(k);
                    bsx = C(k, :);
                end
                ec = FE - nb + k;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        for i = half + 1:N
            predator = lb + rand(1, dim) .* (ub - lb);   % Eq. (10)
            [fp, FE] = calculate_fitness(predator', problem, FE);
            if fp < bsf
                bsf = fp;
                bsx = predator;
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end

            dist = abs(predator - X(i, :));   % Eq. (11)
            b = 2 + 2 * rand;
            c = 1 + 0.5 * rand;
            d = 2 + rand;
            l = 4 * pi * rand - 2 * pi;
            RL = 0.05 * levy(dim, 1.5);   % Eq. (13)
            if fit(i) > fp
                Xp3 = RL .* predator + (b ./ (c - d * cos(l))) .* (1 ./ dist);   % Eq. (12)
            else
                Xp3 = RL .* predator + (b ./ (c - d * cos(l))) .* (1 ./ (2 .* dist + rand(1, dim)));
            end
            Xp3 = min(max(Xp3, lb), ub);   % max/min also map a NaN from a 1/0 term onto the box

            [f3, FE] = calculate_fitness(Xp3', problem, FE);
            if f3 < fit(i)   % Eq. (15)
                X(i, :) = Xp3;
                fit(i) = f3;
            end
            if f3 < bsf
                bsf = f3;
                bsx = Xp3;
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
            if FE >= maxFE, break; end
        end
        if FE >= maxFE, break; end

        lo = lb ./ t;   % Eq. (16)
        hi = ub ./ t;
        Xp4 = zeros(N, dim);
        for i = 1:N
            s = cell(1, 3);   % Eq. (18)
            s{1} = 2 * rand(1, dim) - 1;
            s{2} = rand(1, 1);
            s{3} = randn;
            D = s{randi([1, 3], 1, 1)};
            Xp4(i, :) = X(i, :) + rand(1, 1) .* (lo + D .* (hi - lo));   % Eq. (17)
        end
        Xp4 = min(max(Xp4, lb), ub);

        nb = min(N, maxFE - FE);
        [fc, FE] = calculate_fitness(Xp4(1:nb, :)', problem, FE);
        fc = fc(:)';
        for i = 1:nb
            if fc(i) < fit(i)   % Eq. (19)
                X(i, :) = Xp4(i, :);
                fit(i) = fc(i);
            end
            if fc(i) < bsf
                bsf = fc(i);
                bsx = Xp4(i, :);
            end
            ec = FE - nb + i;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                ec, X, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = bsf;
    best_fitness = bsf;
    best_solution = bsx;
end

% Mantegna Levy step, Eqs. (13)-(14) without the 0.05 scale
function z = levy(m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
    sigma_u = (num / den) ^ (1 / beta);
    u = randn(1, m) * sigma_u;
    v = randn(1, m);
    z = u ./ (abs(v) .^ (1 / beta));
end

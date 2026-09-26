% ----------------------------------------------------------------------- %
% Artificial Protozoa Optimizer (APO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   ps = 100                    % Population size
%   np = 1                      % Neighbour pairs in the foraging update
%   pf_max = 0.1                % Maximum share of members in dormancy/reproduction
%   pdr = (1+cos((1-i/ps)*pi))/2  % Dormancy chance of rank i (else reproduction)
%   pah = (1+cos(t*pi))/2       % Autotroph chance (else heterotroph), t the spent budget
%   f = rand*(1+cos(t*pi))      % Foraging factor
%
% Algorithm Concept:
%   - Each generation ranks the population by fitness and draws ceil(ps*rand*pf_max)
%     members for dormancy or reproduction; the rest forage
%   - Dormancy redraws the member uniformly in the box; reproduction adds a signed,
%     randomly scaled box point to a random subset of its coordinates
%   - Autotrophs move towards a random member plus a neighbour-pair difference weighted
%     by exp(-|f(k-)/f(k+)|), k- ranked better and k+ worse than the member
%   - Heterotrophs move towards a point scaled from their own position by
%     1 +- rand*(1-t), plus the same weighted difference of the adjacent ranks
%   - Foraging changes ceil(D*i/ps) randomly chosen coordinates of rank i, so
%     better-ranked members change fewer coordinates
%   - Coordinates are clamped to the box; a child replaces its parent only if better
%
% Reference:
% Xiaopeng Wang, Vaclav Snasel, Seyedali Mirjalili, Jeng-Shyang Pan, Lingping
% Kong, Hisham A. Shehadeh,
% Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic
% algorithm for engineering optimization,
% Knowledge-Based Systems 295 (2024) 111737.
% https://doi.org/10.1016/j.knosys.2024.111737
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' CEC2022 release (APO-CEC2022/APO_func.m in the
% "Matlab Code-Artificial Protozoa Optimizer" package). Its schedules run on
% iter/iter_max, the initial sweep being iteration 1; here that ratio is
% (FE + ps)/maxFe, so FE >= maxFe is the only terminator and the last
% generation evaluates only the children the budget still allows. A NaN or Inf
% coordinate (a NaN weight from an Inf fitness) is redrawn uniformly before the
% clamp, where the release's 0/1 clamp would carry it on as NaN.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = apo(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    maxFE = problem.maxFe;

    ps = 100;
    np = 1;
    pf_max = 0.1;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(ps, dim) .* (ub - lb);
    fit = inf(ps, 1);
    bsf = inf;
    bsf_solution = X(1, :);

    n = min(ps, maxFE);
    [fv, FE] = calculate_fitness(X(1:n, :)', problem, FE);
    fit(1:n) = fv(:);
    for k = 1:n
        if fit(k) < bsf
            bsf = fit(k);
            bsf_solution = X(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, X(1:n, :), fit(1:n), population_history, fitness_history, history_index, maxFE);
    end

    epn = zeros(np, dim);
    while FE < maxFE
        t = min(1, (FE + ps) / maxFE);                % iter/iter_max of the release
        [fit, order] = sort(fit);
        X = X(order, :);
        pf = pf_max * rand;
        in_dr = false(ps, 1);
        in_dr(randperm(ps, ceil(ps * pf))) = true;

        newX = zeros(ps, dim);
        for i = 1:ps
            if in_dr(i)
                pdr = 0.5 * (1 + cos((1 - i / ps) * pi));
                if rand < pdr                                                  % dormancy
                    newX(i, :) = lb + rand(1, dim) .* (ub - lb);
                else                                                           % reproduction
                    Mr = zeros(1, dim);
                    Mr(randperm(dim, ceil(rand * dim))) = 1;
                    newX(i, :) = X(i, :) + plus_minus() * rand * (lb + rand(1, dim) .* (ub - lb)) .* Mr;
                end
            else
                f = rand * (1 + cos(t * pi));
                Mf = zeros(1, dim);
                Mf(randperm(dim, ceil(dim * i / ps))) = 1;
                pah = 0.5 * (1 + cos(t * pi));
                if rand < pah                                                  % autotroph
                    j = randi(ps);
                    for k = 1:np
                        if i == 1
                            km = i;
                            kp = i + randi(ps - i);
                        elseif i == ps
                            km = randi(ps - 1);
                            kp = i;
                        else
                            km = randi(i - 1);
                            kp = i + randi(ps - i);
                        end
                        wa = exp(-abs(fit(km) / (fit(kp) + eps)));
                        epn(k, :) = wa * (X(km, :) - X(kp, :));
                    end
                    newX(i, :) = X(i, :) + f * (X(j, :) - X(i, :) + sum(epn, 1) / np) .* Mf;
                else                                                           % heterotroph
                    for k = 1:np
                        if i == 1
                            imk = i;
                            ipk = i + k;
                        elseif i == ps
                            imk = ps - k;
                            ipk = i;
                        else
                            imk = i - k;
                            ipk = i + k;
                        end
                        if imk < 1
                            imk = 1;
                        elseif ipk > ps
                            ipk = ps;
                        end
                        wh = exp(-abs(fit(imk) / (fit(ipk) + eps)));
                        epn(k, :) = wh * (X(imk, :) - X(ipk, :));
                    end
                    Xnear = (1 + plus_minus() * rand(1, dim) * (1 - t)) .* X(i, :);
                    newX(i, :) = X(i, :) + f * (Xnear - X(i, :) + sum(epn, 1) / np) .* Mf;
                end
            end
        end

        bad = ~isfinite(newX);
        if any(bad(:))
            R = lb + rand(ps, dim) .* (ub - lb);
            newX(bad) = R(bad);
        end
        newX = min(max(newX, lb), ub);

        n = min(ps, maxFE - FE);
        [fv, FE] = calculate_fitness(newX(1:n, :)', problem, FE);
        newfit = fv(:);
        better = newfit < fit(1:n);
        idx = find(better);
        X(idx, :) = newX(idx, :);
        fit(idx) = newfit(idx);

        for k = 1:n
            if newfit(k) < bsf
                bsf = newfit(k);
                bsf_solution = newX(k, :);
            end
            eval_count = FE - n + k;
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X, fit, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness = bsf;
    best_solution = bsf_solution;
end

% +1 or -1 with equal chance, the release's flag(ceil(2*rand))
function s = plus_minus()
    if rand <= 0.5
        s = 1;
    else
        s = -1;
    end
end

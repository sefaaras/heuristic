% ----------------------------------------------------------------------- %
% Chicken Swarm Optimization (CSO)
% Stored as chso; the acronym CSO collides with cso (Competitive Swarm)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop      = 100           % Chickens
%   G        = 10            % Ranks, groups and mothers rebuilt every G iterations
%   rPercent = 0.15          % Roosters RN = 15, one group head each
%   hPercent = 0.7           % Hens HN = 70; chicks CN = N - RN - HN = 15
%   mPercent = 0.5           % Mother hens MN = 35, as a share of the hens
%   FL       = U(0.5, 0.9)   % Chick's pull toward its mother, redrawn every iteration
%
% Algorithm Concept:
%   - Every G iterations the personal bests are ranked: best RN roosters, worst CN
%     chicks, the rest hens; each hen joins a random rooster, each chick a mother
%   - Rooster (Eqs. 1-2): pbest.*(1 + sigma*randn), sigma = 1 unless a random other
%     rooster is better, then exp((f_k - f_i)/(|f_i| + eps)); a scaling about 0
%   - Hen (Eqs. 3-5): pbest + S1*r1.*(rooster - pbest) + S2*r2.*(other - pbest),
%     S1 = exp((f_i - f_r1)/(|f_i| + eps)), S2 = exp(f_r2 - f_i), other ranked <= i
%   - Chick (Eq. 6): pbest + FL*(mother - pbest)
%   - All N move from personal bests, synchronously; a personal best is replaced
%     only by a better point after the sweep
%
% Reference:
% Xianbing Meng, Yu Liu, Xiaozhi Gao, Hengzhen Zhang,
% A New Bio-inspired Algorithm: Chicken Swarm Optimization,
% Advances in Swarm Intelligence (ICSI 2014), Lecture Notes in Computer Science
% 8794, Springer, 2014, pp. 86-94.
% https://doi.org/10.1007/978-3-319-11857-4_10
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the first author's release, File Exchange 48204 v1.3.0.0 (CSO.m,
% 24 Jun 2015); seeded runs are bit-identical to it. Release defaults are kept
% where Table 2 differs (RN = 0.2N, HN = 0.6N, MN = 0.1N). Kept as released:
% moves start from personal bests (the paper writes current positions); a hen's
% r2 is drawn from ranks 1..i, so it may be the hen itself; S2 takes the raw
% fitness difference and overflows once a hen's best has improved by over ~709
% since the last ranking, throwing it onto the box faces, and Inf*0 leaves NaN
% coordinates: the numeric suites score them NaN, a spent FE never accepted
% (0.04 % of FEs on CEC2014 F1-F3, D = 10); CEC2020RW's evaluator clamps NaN to lb
% and that point is kept (38 % of FEs on RW F5). No iteration-count parameter.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = chso(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    pop      = 100;
    G        = 10;
    rPercent = 0.15;
    hPercent = 0.7;
    mPercent = 0.5;

    rNum = round(pop * rPercent);
    hNum = round(pop * hPercent);
    cNum = pop - rNum - hNum;
    mNum = round(hNum * mPercent);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Transposed draw: chicken i takes the i-th rand(1, dim) of the stream, as released
    x = lb + (ub - lb) .* rand(dim, pop)';
    nev = min(pop, maxFE);
    [fit, FE] = calculate_fitness(x(1:nev, :)', problem, FE);
    fit = fit(:)';

    best_fitness  = Inf;
    best_solution = x(1, :);
    for e = 1:nev
        if fit(e) < best_fitness
            best_fitness  = fit(e);
            best_solution = x(e, :);
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, x(1:nev, :), fit, population_history, fitness_history, ...
            history_index, maxFE);
    end
    pFit = fit;
    pX   = x;

    xnew = zeros(pop, dim);   % row k is the move of the chicken ranked k

    t = 0;
    while FE < maxFE
        t = t + 1;
        FL = rand(pop, 1) .* 0.4 + 0.5;

        if mod(t, G) == 1 || t == 1
            [~, sortIndex] = sort(pFit);
            motherLib = randperm(hNum, mNum) + rNum;   % ranks of the mother hens
            mate = randpermF(rNum, hNum);              % rooster rank heading each hen's group
            mother = motherLib(randi(mNum, cNum, 1));
        end

        for i = 1:rNum
            anotherRooster = randiTabu(1, rNum, i);
            if pFit(sortIndex(i)) <= pFit(sortIndex(anotherRooster))
                tempSigma = 1;
            else
                tempSigma = exp((pFit(sortIndex(anotherRooster)) - ...
                    pFit(sortIndex(i))) / (abs(pFit(sortIndex(i))) + realmin));   % Eq. (2)
            end
            xnew(i, :) = pX(sortIndex(i), :) .* (1 + tempSigma .* randn(1, dim));   % Eq. (1)
            xnew(i, :) = bounds(xnew(i, :), lb, ub);
        end

        for i = (rNum + 1):(rNum + hNum)
            other = randiTabu(1, i, mate(i - rNum));
            c1 = exp((pFit(sortIndex(i)) - pFit(sortIndex(mate(i - rNum)))) / ...
                (abs(pFit(sortIndex(i))) + realmin));   % Eq. (4)
            c2 = exp((-pFit(sortIndex(i)) + pFit(sortIndex(other))));   % Eq. (5)
            xnew(i, :) = pX(sortIndex(i), :) + (pX(sortIndex(mate(i - rNum)), :) - ...
                pX(sortIndex(i), :)) .* c1 .* rand(1, dim) + (pX(sortIndex(other), :) - ...
                pX(sortIndex(i), :)) .* c2 .* rand(1, dim);   % Eq. (3)
            xnew(i, :) = bounds(xnew(i, :), lb, ub);
        end

        for i = (rNum + hNum + 1):pop
            xnew(i, :) = pX(sortIndex(i), :) + (pX(sortIndex(mother(i - rNum - hNum)), :) - ...
                pX(sortIndex(i), :)) .* FL(i);   % Eq. (6)
            xnew(i, :) = bounds(xnew(i, :), lb, ub);
        end

        % Evaluated in rank order, as released
        nev = min(pop, maxFE - FE);
        [fnew, FE] = calculate_fitness(xnew(1:nev, :)', problem, FE);
        fnew = fnew(:)';
        for e = 1:nev
            % A NaN fitness (a NaN coordinate on the numeric suites) is a spent FE, never accepted
            if ~isnan(fnew(e))
                k = sortIndex(e);
                x(k, :) = min(max(xnew(e, :), lb), ub);   % NaN -> lb, as CEC2020RW scores it
                fit(k)  = fnew(e);
                if fnew(e) < best_fitness
                    best_fitness  = fnew(e);
                    best_solution = x(k, :);
                end
            end
            fe = FE - nev + e;
            curve(fe) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                fe, x, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
        if FE >= maxFE, break; end

        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i)  = fit(i);
                pX(i, :) = x(i, :);
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% The release's Bounds, lower face first; a NaN coordinate passes through
function s = bounds(s, lb, ub)
    I = s < lb;
    s(I) = lb(I);
    J = s > ub;
    s(J) = ub(J);
end

% One integer in [lo, hi] other than tabu, by rejection as released
function value = randiTabu(lo, hi, tabu)
    value = randi([lo, hi], 1, 1);
    while value == tabu
        value = randi([lo, hi], 1, 1);
    end
end

% Every rooster heads at least one hen; the remaining hens pick a rooster at random
function result = randpermF(range, dim)
    temp   = randperm(range, range);
    temp2  = randi(range, dim, 1);
    index  = randperm(dim, (dim - range));
    result = [temp, temp2(index)'];
end

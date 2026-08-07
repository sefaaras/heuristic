% ----------------------------------------------------------------------- %
% Colony-based Search Algorithm (CSA)
% Stored as colsa; the acronym CSA collides with csa and chsa
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20    % Number of predators selected per epoch
%   T = 2     % Pattern-matrix expansion factor (colony size = T*N)
%
% Algorithm Concept:
%   - A colony of T*N patterns; each epoch a uniformly random, fully
%     displaced subset of N predators is drawn from it
%   - Direction-scaling factor: either a Cauchy-type ratio of centred
%     uniforms or a sign-flipped Levy-distributed scale
%   - Morphogenesis control matrix m: per pattern a random subset of
%     ceil(k*D) dimensions is activated, k = |randi([0 1]) - rand^randi([2 10])|
%   - Three interaction models for the evolutionary direction dx:
%     bilateral-bijective, bijective and swarmmic (top-N/5 attraction)
%   - Momentum term carried between epochs
%
% Reference:
% Pinar Civicioglu, Erkan Besdok,
% Colony-based search algorithm for numerical optimization,
% Applied Soft Computing (2023) 111162.
% https://doi.org/10.1016/j.asoc.2023.111162
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = colsa(problem)

    D     = problem.dimension;
    low   = problem.lb;
    up    = problem.ub;
    maxFE = problem.maxFe;

    N = 20;
    T = 2;                     % pattern matrix expanding factor (colony size)
    Epk = max(1, ceil((maxFE - T * N) / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    initindex = 1:N;
    moment = 0;

    % Initialise the colony
    p0 = zeros(T * N, D);
    for i = 1:T * N
        for j = 1:D
            p0(i, j) = rand .* (up(j) - low(j)) + low(j);
        end
    end
    [fitp0, FE] = calculate_fitness(p0', problem, FE);
    fitp0 = fitp0(:);

    [gmin, indbest] = min(fitp0);
    gbest = p0(indbest, :);
    bsf   = gmin;

    for eval_count = 1:(T * N)
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, p0, fitp0, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Iterative search phase
    for epoch = 1:Epk
        if FE >= maxFE, break; end

        % Selection of predators, p
        while 1
            index = randperm(T * N, N);              % uniform selection
            if sum(index == initindex) == 0
                initindex = index;
                break;
            end
        end
        p    = p0(index, :);
        fitp = fitp0(index);

        % Direction scaling factor
        if rand < rand, c = 1; else, c = D; end
        if rand < rand
            scale = (rand(N, c) - 0.50) ./ (rand(N, c) - 0.50);
        else
            t = [-1 1];
            scale = sign(rand(N, 1) - 0.50) .* ...
                    levy_dist(randi([2 5], N, 1), randi(10, N, 1) .^ t(randi(2)));
        end

        % Morphogenesis (mutation + crossover) control matrix, m
        m = zeros(N, D);
        for j = 1:N
            ind = randperm(D);
            k = abs(randi([0 1]) - rand ^ randi([2 10]));
            b = ind(1:ceil(k * D));
            m(j, b) = 1;
        end

        % Evolutionary direction vector, dx
        while 1
            v1 = randperm(N);
            v2 = randperm(N);
            if sum(v1 == (1:N)) == 0 && sum(v2 == v1) == 0 && sum(v2 == (1:N)) == 0
                break;
            end
        end
        [~, index0] = sort(fitp);
        v = randi(3);                                % interaction model
        switch v
            case 1, dx = p(v2, :) - p(v1, :);                          % bilateral-bijective
            case 2, dx = p(v1, :) - p;                                 % bijective
            case 3, dx = p(index0(randi([1 ceil(N/5)])), :) - p;        % swarmmic
        end

        % Morphogenesis pattern matrix, px
        s = (rand(N, 1) - 0.50) .* rand(N, 1) .^ randi([2 10], 1);
        px = p + scale .* m .* dx + s .* moment;

        % Boundary control
        for k = 1:N
            for l = 1:D
                if px(k, l) < low(l)
                    px(k, l) = low(l) + rand .^ randi([1 5], 1) * (up(l) - low(l));
                end
                if px(k, l) > up(l)
                    px(k, l) = up(l) + rand .^ randi([1 5], 1) * (low(l) - up(l));
                end
            end
        end

        % Greedy selection
        [fitpx, FE] = calculate_fitness(px', problem, FE);
        fitpx = fitpx(:);
        ind = fitpx < fitp;

        % Update clan and colony
        p(ind, :)     = px(ind, :);
        fitp(ind)     = fitpx(ind);
        p0(index, :)  = p;
        fitp0(index)  = fitp;

        % Update the global solution
        [gmin, indbest] = min(fitp0);
        gbest = p0(indbest, :);
        if gmin < bsf
            bsf = gmin;
        end

        for k = 1:N
            ec = FE - N + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, p0, fitp0, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Update momentum
        moment = (abs(randi([0 1], N, 1)) - m) .* dx;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = gmin;
    best_solution = gbest;
end

% Levy-distributed random numbers
function x = levy_dist(alpha, beta)
    z = rand() + 1;
    w = gamrnd(alpha, randi([2 5]));
    x = beta * z ./ w .^ (1 ./ alpha);
end

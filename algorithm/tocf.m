% ----------------------------------------------------------------------- %
% Tornado Optimizer with Coriolis Force (TOC)
% Stored as tocf
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   n   = 50        % Population size
%   nto = 4         % Thunderstorms + tornadoes
%   nt  = 3         % Thunderstorms  (tornadoes To = nto - nt = 1)
%   b_r = 1e5       % Radius scaling of the Coriolis term
%   chi = 4.10, eta = 2/|2-chi-sqrt(chi^2-4chi)|   % constriction factor
%
% Algorithm Concept:
%   - The swarm is split into one tornado, nt thunderstorms and nw windstorms
%   - Windstorm velocity: the Coriolis parameter f = 2*omega*sin(...) and the
%     sigmoid radii Rl/Rr build the Coriolis force CFl/CFr, which enters the
%     constricted velocity update (eta, mu)
%   - Exploration: windstorms evolve towards the tornado with the amplitude
%     alpha = |2*ay*rand - rand|
%   - Exploitation: windstorms merge into thunderstorms, thunderstorms evolve
%     into the tornado
%   - Random re-formation whenever a storm gets closer than nu to its leader
%
% Reference:
% Malik Braik, Heba Al-Hiary, Hussein Alzoubi, Abdelaziz Hammouri,
% Mohammed Azmi Al-Betar, Mohammed A. Awadallah,
% Tornado optimizer with Coriolis force: a novel bio-inspired meta-heuristic
% algorithm for solving engineering problems,
% Artificial Intelligence Review 58, 123 (2025).
% https://doi.org/10.1007/s10462-025-11118-9
%
% Implementation Note:
%   Random re-formation displaces a windstorm at the end of an iteration and the
%   box is only re-imposed when that storm is next evaluated, so a sample taken
%   in between holds out-of-box rows. Clamping at re-formation time is not
%   equivalent: the next iteration's step starts from that raw position.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tocf(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    n   = 50;
    nto = 4;
    nt  = 3;
    To  = nto - nt;      % tornadoes
    nw  = n - nto;       % windstorms

    max_it = max(1, ceil((maxFE - n) / (nw + nt)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initial population
    y = initialization(n, dim, ub, lb);
    [fit, FE] = calculate_fitness(y', problem, FE);
    fit = fit(:);

    [~, index] = sort(fit);

    Tornadoposition = y(index(1:To), :);
    TornadoCost     = fit(index(1:To));

    Thunderstormsposition = y(index(2:nto), :);
    ThunderstormsCost     = fit(index(2:nto))';
    bThunderstormsCost    = ThunderstormsCost;
    bThunderstormsposition = Thunderstormsposition;

    Windstormsposition = y(index(nto+1:nto+nw), :);
    WindstormsCost     = fit(index(nto+1:nto+nw))';
    bWindstormsCost     = WindstormsCost;
    bWindstormsposition = Windstormsposition;

    bsf = TornadoCost(1);
    bsx = Tornadoposition(1, :);

    for eval_count = 1:n
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, y, fit, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    vel_storm = 0.1 * Windstormsposition;
    phi = zeros(nw, dim);

    % Designate windstorms to thunderstorms and tornadoes
    nwindstorms = 1:nw;
    nwindstorms = nwindstorms(sort(randperm(nw, nto)));
    nWT = diff(nwindstorms)';
    nWT(end + 1) = nw - sum(nWT);
    nWT1 = nWT(1);
    nWH  = nWT(2:end);

    b_r = 100000;
    fdelta = [-1, 1];
    chi = 4.10;
    eta = 2 / abs(2 - chi - sqrt(chi ^ 2 - 4 * chi));

    % Main loop
    t = 1;
    while t <= max_it && FE < maxFE

        nu = (0.1 * exp(-0.1 * (t / max_it) ^ 0.1)) ^ 16;
        mu = 0.5 + rand / 2;
        ay = (max_it - (t ^ 2 / max_it)) / max_it;

        Rl =  2 / (1 + exp((-t + max_it / 2) / 2));
        Rr = -2 / (1 + exp((-t + max_it / 2) / 2));

        % Update the windstorm velocities
        for i = 1:nw
            for j = 1:dim
                if rand > 0.5
                    delta1 = fdelta(ceil(2 * rand()));
                    zeta = ceil((To) .* rand(1, To))';
                    wmin = 1; wmax = 4.0; rr = wmin + rand() * (wmax - wmin);
                    wr = (((2 * rand()) - (1 * rand() + rand())) / rr);
                    c = b_r * delta1 * wr;
                    omega = 0.7292115E-04;
                    f = 2 * omega * sin(-1 + 2 .* rand(1, 1));

                    phi(i, j) = Tornadoposition(zeta, j) - Windstormsposition(i, j);
                    if sign(Rl) >= 0
                        if sign(phi(i, j)) >= 0
                            phi(i, j) = -phi(i, j);
                        end
                    end
                    CFl = (((f ^ 2 * Rl ^ 2) / 4) - Rl * 1 * phi(i, j));
                    if sign(CFl) < 0
                        CFl = -CFl;
                    end
                    vel_storm(i, j) = eta * (mu * vel_storm(i, j) - c * (f * Rl) / 2 + (sqrt(CFl)));
                else
                    delta1 = fdelta(ceil(2 * rand()));
                    zeta = ceil((To) .* rand(1, To))';
                    rmin = 1; rmax = 4.0; rr = rmin + rand() * (rmax - rmin);
                    wr = (((2 * rand()) - (1 * rand() + rand())) / rr);
                    c = b_r * delta1 * wr;

                    phi(i, j) = Tornadoposition(zeta, j) - Windstormsposition(i, j);
                    if sign(Rr) <= 0
                        if sign(phi(i, j)) <= 0
                            phi(i, j) = -phi(i, j);
                        end
                    end
                    omega = 0.7292115E-04;
                    f = 2 * omega * sin(-1 + 2 .* rand(1, 1));
                    CFr = (((f ^ 2 * Rr ^ 2) / 4) - Rr * 1 * phi(i, j));
                    if sign(CFr) < 0
                        CFr = -CFr;
                    end
                    vel_storm(i, j) = eta * (mu * vel_storm(i, j) - c * (f * Rr) / 2 + (sqrt(CFr)));
                end
            end
        end

        % Exploration -- evolution of windstorms to tornadoes
        for i = 1:nWT1
            if FE >= maxFE, break; end
            rand_index = floor((nWT1) .* rand(1, nWT1)) + 1;
            rand_w = Windstormsposition(rand_index, :);
            alpha = abs(2 * ay * rand - 1 * rand);

            Windstormsposition(i, :) = Windstormsposition(i, :) + ...
                2 * alpha * (Tornadoposition - rand_w(i, :)) + vel_storm(i, :);

            ub_ = Windstormsposition(i, :) > ub;
            lb_ = Windstormsposition(i, :) < lb;
            Windstormsposition(i, :) = (Windstormsposition(i, :) .* (~(ub_ + lb_))) + ub .* ub_ + lb .* lb_;

            [WindstormsCost(i), FE] = calculate_fitness(Windstormsposition(i, :)', problem, FE);

            if WindstormsCost(i) < bWindstormsCost(i)
                bWindstormsposition(i, :) = Windstormsposition(i, :);
                bWindstormsCost(i)        = WindstormsCost(i);
            end
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, WindstormsCost(i), Windstormsposition(i, :), bsf, bsx, curve, ...
                      [Tornadoposition; Thunderstormsposition; Windstormsposition], ...
                      [TornadoCost; ThunderstormsCost'; WindstormsCost'], ...
                      population_history, fitness_history, history_index);
        end

        [minTornadoCost, in] = min(bWindstormsCost);
        if minTornadoCost < TornadoCost
            TornadoCost = minTornadoCost;
            Tornadoposition = bWindstormsposition(in, :);
        end

        % Exploitation -- windstorms merge into thunderstorms
        for i = 1:nt
            for j = 1:nWH(i)
                if FE >= maxFE, break; end
                idx = j + sum(nWT(1:i));

                Windstormsposition(idx, :) = Windstormsposition(idx, :) + ...
                    2 * rand * (Thunderstormsposition(i, :) - Windstormsposition(idx, :)) + ...
                    2 * rand * (Tornadoposition(1, :) - Windstormsposition(idx, :));

                ub_ = Windstormsposition(idx, :) > ub;
                lb_ = Windstormsposition(idx, :) < lb;
                Windstormsposition(idx, :) = (Windstormsposition(idx, :) .* (~(ub_ + lb_))) + ub .* ub_ + lb .* lb_;

                [WindstormsCost(idx), FE] = calculate_fitness(Windstormsposition(idx, :)', problem, FE);

                if WindstormsCost(idx) < ThunderstormsCost(i)
                    bThunderstormsposition(i, :) = Windstormsposition(idx, :);
                    Thunderstormsposition(i, :)  = Windstormsposition(idx, :);
                    ThunderstormsCost(i)         = WindstormsCost(idx);
                end
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, WindstormsCost(idx), Windstormsposition(idx, :), bsf, bsx, curve, ...
                          [Tornadoposition; Thunderstormsposition; Windstormsposition], ...
                          [TornadoCost; ThunderstormsCost'; WindstormsCost'], ...
                          population_history, fitness_history, history_index);
            end
        end

        [minTornadoCost, in] = min(ThunderstormsCost);
        if minTornadoCost < TornadoCost
            TornadoCost = minTornadoCost;
            Tornadoposition = bThunderstormsposition(in, :);
        end

        % Evolution of thunderstorms into the tornado
        for i = 1:nt
            if FE >= maxFE, break; end
            zeta  = ceil((To) .* rand(1, To));
            alpha = abs(2 * ay * rand - 1 * rand);
            p = floor((nt) .* rand(1, nt)) + 1;
            rand_w = Thunderstormsposition(p, :);

            Thunderstormsposition(i, :) = Thunderstormsposition(i, :) + ...
                2 .* alpha * (Thunderstormsposition(i, :) - Tornadoposition(zeta, :)) + ...
                2 .* alpha * (rand_w(i, :) - Thunderstormsposition(i, :));

            ub_ = Thunderstormsposition(i, :) > ub;
            lb_ = Thunderstormsposition(i, :) < lb;
            Thunderstormsposition(i, :) = (Thunderstormsposition(i, :) .* (~(ub_ + lb_))) + ub .* ub_ + lb .* lb_;

            [ThunderstormsCost(i), FE] = calculate_fitness(Thunderstormsposition(i, :)', problem, FE);

            if ThunderstormsCost(i) < bThunderstormsCost(i)
                bThunderstormsposition(i, :) = Thunderstormsposition(i, :);
                bThunderstormsCost(i)        = ThunderstormsCost(i);
            end
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, ThunderstormsCost(i), Thunderstormsposition(i, :), bsf, bsx, curve, ...
                      [Tornadoposition; Thunderstormsposition; Windstormsposition], ...
                      [TornadoCost; ThunderstormsCost'; WindstormsCost'], ...
                      population_history, fitness_history, history_index);
        end

        [minTornadoCost, in] = min(bThunderstormsCost);
        if minTornadoCost < TornadoCost
            TornadoCost = minTornadoCost;
            Tornadoposition = bThunderstormsposition(in, :);
        end

        % Random re-formation of windstorms
        for i = 1:nWT1
            if norm(Windstormsposition(i, :) - Tornadoposition) < nu
                delta2 = fdelta(floor(2 * rand() + 1));
                Windstormsposition(i, :) = Windstormsposition(i, :) - ...
                    (2 * ay * (rand * (lb - ub) - lb)) * delta2;
            end
        end
        for i = 1:nt
            if norm(Windstormsposition(i, :) - Thunderstormsposition(i, :)) < nu
                for j = 1:nWH(i)
                    delta2 = fdelta(floor(2 * rand() + 1));
                    idx = j + sum(nWT(1:i));
                    Windstormsposition(idx, :) = Windstormsposition(idx, :) - ...
                        (2 * ay * (rand * (lb - ub) - lb)) * delta2;
                end
            end
        end

        t = t + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Curve / history stamp for a single evaluation
function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, x, bsf, bsx, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = x;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

% Initialization
function pos = initialization(noP, dim, ub_, lb_)
    pos = zeros(noP, dim);
    for i = 1:dim
        pos(:, i) = rand(noP, 1) .* (ub_(i) - lb_(i)) + lb_(i);
    end
end

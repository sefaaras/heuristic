% ----------------------------------------------------------------------- %
% Dream Optimization Algorithm (DOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 50    % Population size (must be a multiple of 5: 5 dream groups)
%
% Algorithm Concept:
%   - Memory strategy: every individual is reset to its group's best
%     solution, i.e. the "remembered" part of the dream
%   - Forgetting and supplementation strategy: k randomly chosen dimensions
%     are perturbed with the cosine-annealed amplitude
%     (cos((i + T/10)*pi/T)+1)/2 scaled by the search range
%   - Dream sharing: with probability 0.1 the forgotten dimensions are
%     copied from another randomly chosen individual instead
%   - The run is split into a 90 % exploration phase (5 independent groups)
%     and a 10 % exploitation phase (single group around the global best)
%   - Dimension-dependent boundary handling: for D > 15 a violating value is
%     replaced by another individual's value, otherwise re-drawn uniformly
%
% Reference:
% Yifan Lang, Yuelin Gao,
% Dream Optimization Algorithm (DOA): A novel metaheuristic optimization
% algorithm inspired by human dreams and its applications to real-world
% engineering problems,
% Computer Methods in Applied Mechanics and Engineering 436 (2025) 117718.
% https://doi.org/10.1016/j.cma.2024.117718
% ----------------------------------------------------------------------- %
% Implementation Note:
% DOA is steady-state: the memory strategy moves a whole dream group without
% re-evaluating it, so the live population never matches its fitness. The
% history therefore records xrec, which holds each individual at the position
% its fcur entry was measured at, so a row is a population that existed rather
% than a mix of two sweeps. Rows start once the first sweep has given every
% individual a fitness.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = doa(problem)

    D     = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    pop = 50;
    T   = 10 * max(1, ceil(maxFE / (10 * pop)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = initialization(pop, D, ub, lb);
    SELECT = 1:pop;

    sbest  = ones(1, D);
    sbestd = ones(5, D);
    fbest  = inf;
    fbestd = inf(5, 1);

    fcur = inf(pop, 1);      % last known fitness of each individual (for history)
    xrec = x;                % position each fcur entry was measured at
    bsf  = inf;
    bsx  = x(1, :);

    % Exploration phase
    for i = 1:(9 * T / 10)
        if FE >= maxFE, break; end

        for m = 1:5                                    % five dream groups
            if FE >= maxFE, break; end
            k = randi([ceil(D/8/m), ceil(D/3/m)]);
            grp = (((m-1)/5*pop) + 1) : (m/5*pop);

            for j = grp
                if FE >= maxFE, break; end
                [fj, FE] = calculate_fitness(x(j, :)', problem, FE);
                fcur(j) = fj;
                xrec(j, :) = x(j, :);
                [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, fj, x(j, :), bsf, bsx, curve, xrec, fcur, population_history, ...
                          fitness_history, history_index);
                if fj < fbestd(m)
                    sbestd(m, :) = x(j, :);
                    if FE < maxFE
                        [fj2, FE] = calculate_fitness(x(j, :)', problem, FE);
                        [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                            stamp(FE, maxFE, fj2, x(j, :), bsf, bsx, curve, xrec, fcur, population_history, ...
                                  fitness_history, history_index);
                    else
                        fj2 = fj;
                    end
                    fbestd(m) = fj2;
                end
            end

            for j = grp
                x(j, :) = sbestd(m, :);                % memory strategy
                in = randperm(D, k);
                if rand < 0.9
                    for h = 1:k
                        % forgetting and supplementation strategy
                        x(j, in(h)) = x(j, in(h)) + ...
                            (rand * (ub(in(h)) - lb(in(h))) + lb(in(h))) * (cos((1 * i + T/10) * pi / T) + 1) / 2;
                        if (x(j, in(h)) > ub(in(h))) || (x(j, in(h)) < lb(in(h)))
                            if D > 15
                                select = SELECT;
                                select(j) = [];
                                sel = select(randi(pop - 1));
                                x(j, in(h)) = x(sel, in(h));
                            else
                                x(j, in(h)) = rand * (ub(in(h)) - lb(in(h))) + lb(in(h));
                            end
                        end
                    end
                else
                    for h = 1:k                        % dream sharing
                        x(j, in(h)) = x(randi(pop), in(h));
                    end
                end
            end

            if fbestd(m) < fbest
                fbest = fbestd(m);
                sbest = sbestd(m, :);
            end
        end
    end

    % Exploitation phase
    for i = ((9 * T / 10) + 1):T
        if FE >= maxFE, break; end

        for p = 1:pop
            if FE >= maxFE, break; end
            [fp, FE] = calculate_fitness(x(p, :)', problem, FE);
            fcur(p) = fp;
            xrec(p, :) = x(p, :);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, fp, x(p, :), bsf, bsx, curve, xrec, fcur, population_history, ...
                      fitness_history, history_index);
            if fp < fbest
                sbest = x(p, :);
                if FE < maxFE
                    [fp2, FE] = calculate_fitness(x(p, :)', problem, FE);
                    [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                        stamp(FE, maxFE, fp2, x(p, :), bsf, bsx, curve, xrec, fcur, population_history, ...
                              fitness_history, history_index);
                else
                    fp2 = fp;
                end
                fbest = fp2;
            end
        end

        for j = 1:pop
            if FE >= maxFE, break; end
            [fj, FE] = calculate_fitness(x(j, :)', problem, FE);
            fcur(j) = fj;
            xrec(j, :) = x(j, :);
            [bsf, bsx, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, fj, x(j, :), bsf, bsx, curve, xrec, fcur, population_history, ...
                      fitness_history, history_index);

            km = max(2, ceil(D/3));
            k  = randi([2, km]);
            x(j, :) = sbest;
            in = randperm(D, k);
            for h = 1:k
                x(j, in(h)) = x(j, in(h)) + ...
                    (rand * (ub(in(h)) - lb(in(h))) + lb(in(h))) * (cos((i) * pi / T) + 1) / 2;
                if (x(j, in(h)) > ub(in(h))) || (x(j, in(h)) < lb(in(h)))
                    if D > 15
                        select = SELECT;
                        select(j) = [];
                        sel = select(randi(pop - 1));
                        x(j, in(h)) = x(sel, in(h));
                    else
                        x(j, in(h)) = rand * (ub(in(h)) - lb(in(h))) + lb(in(h));
                    end
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Curve / history stamp for a single evaluation
function [bsf, bsx, curve, ph, fh, hi] = stamp(FE, maxFE, f, xc, bsf, bsx, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
        bsx = xc;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        % +Inf is the not-yet-evaluated sentinel of fcur, so the row waits for the
        % first full sweep; -Inf is a legitimate optimum and must not gate it
        if ~any(Fit == Inf)
            [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
        end
    end
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

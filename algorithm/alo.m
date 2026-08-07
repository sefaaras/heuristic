% ----------------------------------------------------------------------- %
% Ant Lion Optimizer (ALO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 40                       % Ants, and an equal number of antlions
%   I = 1 -> 1e6                 % Trap-shrinking ratio, stepped up at 10/50/75/90/95 % of the run
%
% Algorithm Concept:
%   - Two populations of the same size: ANTS, which move, and ANTLIONS, which
%     sit in traps and are the memory of the search
%   - An ant moves by a bounded RANDOM WALK: a +/-1 cumulative sum of length
%     Max_iter rescaled onto an interval centred on an antlion, of which only
%     the current step is used -- a normalised order statistic of the walk
%   - Each ant takes TWO such walks, one around an antlion picked by roulette on
%     inverse fitness and one around the elite, and lands on their average
%   - The walk interval is divided by I, stepped up by an order of magnitude at
%     fixed budget fractions: the trap's "sliding sand" collapses the reachable
%     region on a schedule, turning exploration into exploitation
%   - Ants and antlions are merged and the best N kept, so an ant beating an
%     antlion IS its new position; the elite is reinserted at rank 1
%
% Reference:
% Seyedali Mirjalili,
% The Ant Lion Optimizer,
% Advances in Engineering Software, vol. 83, pp. 80-98, 2015.
% https://doi.org/10.1016/j.advengsoft.2015.01.010
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's MATLAB release ("source codes demo version 1.0").
% Reproduced reference bug: the roulette weights are sorted once and never
% refreshed, which only fixes the shape of the rank preference since weights and
% population are both ordered best-first. Deliberate deviation -- the walk is
% capped at Lwalk = min(Max_iter, 500). The release rebuilds a full Max_iter
% walk per iteration to read one row, costing 2*D*maxFe^2/N draws, QUADRATIC in
% the budget (8 DAYS per cec2020_20 run); it consumes only the SCALE-FREE ratio
% R = (W(cur)-min W)/(max W-min W), whose distribution depends on cur/Max_iter,
% not on the length. Verified: kstest2 on R, 23 of 25 cells p>0.05; Wilcoxon on
% the algorithm (CEC2014 D=10/1e5, 15 runs) F1 .229 F6 .934 F15 .901 F23 .901.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = alo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    N        = 40;
    Max_iter = max(2, floor(maxFE / N));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    antlion_position = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    ant_position     = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);

    [antlions_fitness, FE] = calculate_fitness(antlion_position', problem, FE);
    antlions_fitness = antlions_fitness(:)';

    [sorted_antlion_fitness, sorted_indexes] = sort(antlions_fitness);
    Sorted_antlions = antlion_position(sorted_indexes, :);

    Elite_antlion_position = Sorted_antlions(1, :);
    Elite_antlion_fitness  = sorted_antlion_fitness(1);

    bsf  = Elite_antlion_fitness;
    bsfx = Elite_antlion_position;
    for i = 1:N
        if i <= maxFE
            curve(i) = min(antlions_fitness(1:i));
            [population_history, fitness_history, history_index] = record_history(...
                i, antlion_position, antlions_fitness', population_history, ...
                fitness_history, history_index, maxFE);
        end
    end

    antlions_fitness = sorted_antlion_fitness;

    % Frozen at the INITIAL antlion fitnesses, as released -- see the header note
    roulette_weights = 1 ./ sorted_antlion_fitness;

    % Walk length for drawing one normalised position, capped -- see the header note
    Lwalk = min(Max_iter, 500);

    % Block size for the vectorised random walks (~16 MB temporaries)
    blk = max(1, floor(2e6 / ((Lwalk + 1) * dim)));

    Current_iter = 2;

    % Main loop
    while FE < maxFE && Current_iter <= Max_iter
        % Roulette on inverse fitness picks one antlion per ant (weights NOT refreshed)
        rIdx = zeros(N, 1);
        for i = 1:N
            rIdx(i) = rouletteWheel(roulette_weights);
        end

        % Two random walks per ant: around its antlion and around the elite
        RA = walkStep(Sorted_antlions(rIdx, :), dim, Max_iter, lb, ub, Current_iter, blk, Lwalk);
        RE = walkStep(repmat(Elite_antlion_position, N, 1), dim, Max_iter, lb, ub, Current_iter, blk, Lwalk);

        ant_position = (RA + RE) / 2;                       % Eq. (2.13)

        % Clamp to the box and evaluate
        ant_position = min(max(ant_position, repmat(lb, N, 1)), repmat(ub, N, 1));

        [ants_fitness, FE] = calculate_fitness(ant_position', problem, FE);
        ants_fitness = ants_fitness(:)';

        for i = 1:N
            if ants_fitness(i) < bsf
                bsf  = ants_fitness(i);
                bsfx = ant_position(i, :);
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, ant_position, ants_fitness', population_history, ...
                    fitness_history, history_index, maxFE);
            end
        end

        % A caught ant becomes the antlion's new position
        double_population = [Sorted_antlions; ant_position];
        double_fitness    = [antlions_fitness ants_fitness];

        [double_fitness_sorted, I] = sort(double_fitness);
        Sorted_antlions        = double_population(I(1:N), :);
        antlions_fitness       = double_fitness_sorted(1:N);

        if antlions_fitness(1) < Elite_antlion_fitness
            Elite_antlion_position = Sorted_antlions(1, :);
            Elite_antlion_fitness  = antlions_fitness(1);
        end

        % Keep the elite in the population
        Sorted_antlions(1, :)  = Elite_antlion_position;
        antlions_fitness(1)    = Elite_antlion_fitness;

        Current_iter = Current_iter + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function P = walkStep(antlions, dim, Max_iter, lb, ub, cur, blk, Lwalk)
% Vectorised Random_walk_around_antlion: row `cur` of a bounded walk per antlion

    K = size(antlions, 1);
    P = zeros(K, dim);

    % Trap-shrinking ratio I, Eqs. (2.10) and (2.11)
    I = 1;
    if cur > Max_iter / 10,       I = 1 + 100      * (cur / Max_iter); end
    if cur > Max_iter / 2,        I = 1 + 1000     * (cur / Max_iter); end
    if cur > Max_iter * (3 / 4),  I = 1 + 10000    * (cur / Max_iter); end
    if cur > Max_iter * 0.9,      I = 1 + 100000   * (cur / Max_iter); end
    if cur > Max_iter * 0.95,     I = 1 + 1000000  * (cur / Max_iter); end

    lbs = lb / I;
    ubs = ub / I;

    % One sign draw per antlion, applied to every dimension, Eqs. (2.8), (2.9)
    sl = (rand(K, 1) < 0.5) * 2 - 1;      % +1 -> lb+antlion, -1 -> -lb+antlion
    su = (rand(K, 1) >= 0.5) * 2 - 1;

    C = sl .* repmat(lbs, K, 1) + antlions;
    D = su .* repmat(ubs, K, 1) + antlions;

    for s = 1:blk:K
        e   = min(s + blk - 1, K);
        nb  = e - s + 1;
        cols = nb * dim;

        W = [zeros(1, cols); cumsum(2 * (rand(Lwalk, cols) > 0.5) - 1, 1)];  % Eq. (2.1)
        a = min(W, [], 1);
        b = max(W, [], 1);

        Cb = reshape(C(s:e, :), 1, cols);
        Db = reshape(D(s:e, :), 1, cols);

        span = b - a;
        span(span == 0) = 1;                 % a degenerate walk maps to its lower end
        % Same RELATIVE position in the capped walk as cur is in the full one
        ridx = min(Lwalk + 1, max(1, 1 + round(Lwalk * (cur - 1) / Max_iter)));
        x = ((W(ridx, :) - a) .* (Db - Cb)) ./ span + Cb;   % Eq. (2.7)

        P(s:e, :) = reshape(x, nb, dim);
    end
end

function idx = rouletteWheel(weights)
% Mirjalili's RouletteWheelSelection, with his -1 fallback resolved to 1 as ALO.m does
    acc = cumsum(weights);
    p   = rand() * acc(end);
    idx = find(acc > p, 1, 'first');
    if isempty(idx)
        idx = 1;
    end
end

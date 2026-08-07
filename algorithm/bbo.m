% ----------------------------------------------------------------------- %
% Biogeography-Based Optimization (BBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popsize = 50                 % Habitats
%   Keep    = 2                  % Elitism: best habitats carried over intact
%   pmodify = 1                  % Probability a habitat is considered at all
%   pmutate = 0.005              % Base mutation probability
%   I = E = 1                    % Max immigration / emigration rate
%   lambdaLower = 0, lambdaUpper = 1, dt = 1
%
% Algorithm Concept:
%   - Each solution is a HABITAT. Sorted best-first, habitat k gets species
%     count S_k = P - k, hence immigration lambda_k = I*(1 - S_k/P) (low for
%     good habitats) and emigration mu_k = E*S_k/P (high for good habitats)
%   - MIGRATION is the search operator and is variable-wise, not a crossover:
%     for each habitat k and variable j, with probability lambda_k that variable
%     is REPLACED by its value in a habitat drawn by roulette on mu, so one
%     habitat can draw each of its D variables from a different donor
%   - Species-count PROBABILITY is integrated over time and MUTATION applied in
%     inverse proportion to it, pushing the population off improbable states
%   - Mutation touches only the worse half; the best two habitats are restored
%     intact at the end of every generation
%
% Reference:
% Dan Simon,
% Biogeography-Based Optimization,
% IEEE Transactions on Evolutionary Computation, vol. 12, no. 6, pp. 702-713, 2008.
% https://doi.org/10.1109/TEVC.2008.919004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from Dan Simon's own BBO software (bbo.m with GetSpeciesCounts,
% GetLambdaMu, ClearDups, PopSort), including the caveat in his own header that
% ClearDups replaces a duplicate's gene without recomputing that habitat's cost,
% so a habitat can carry a stale fitness for one generation.
% ProbFlag, which Simon defaults OFF, is switched ON: with it off BBO is
% migration-only and contradicts the paper, whose Section IV.B derives mutation
% from exactly these probabilities.
% Simon's release is INTEGER-CODED; the mutation and ClearDups draws become
% continuous lb + (ub-lb)*rand, forced by the problem class rather than chosen.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bbo(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Control parameters
    popsize = 50;
    Keep    = 2;
    pmodify = 1;
    pmutate = 0.005;
    lambdaLower = 0.0;
    lambdaUpper = 1.0;
    dt = 1;
    I  = 1;
    E  = 1;
    P  = popsize;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LBm = repmat(lb, popsize, 1);
    UBm = repmat(ub, popsize, 1);

    % Initialisation
    chrom = LBm + rand(popsize, dim) .* repmat(span, popsize, 1);
    chrom = clearDups(chrom, lb, span);

    [cost, FE] = calculate_fitness(chrom', problem, FE);
    cost = cost(:);

    [cost, ord] = sort(cost, 'ascend');
    chrom = chrom(ord, :);

    bsf  = cost(1);
    bsfx = chrom(1, :);
    for i = 1:popsize
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, chrom, cost, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    Prob = ones(1, popsize) / popsize;

    % Main loop
    while FE < maxFE
        % Save the elites
        chromKeep = chrom(1:Keep, :);
        costKeep  = cost(1:Keep);

        % Species counts and migration rates
        SpeciesCount = P - (1:popsize)';
        SpeciesCount(~isfinite(cost)) = 0;

        lambda = I * (1 - SpeciesCount / P);
        mu     = E * SpeciesCount / P;

        % Integrate the species-count probabilities
        lambdaMinus = I * (1 - (SpeciesCount - 1) / P);
        muPlus      = E * (SpeciesCount + 1) / P;
        ProbMinus   = [Prob(2:end) 0];
        ProbPlus    = [0 Prob(1:end-1)];
        ProbDot     = -(lambda' + mu') .* Prob + lambdaMinus' .* ProbMinus + muPlus' .* ProbPlus;
        Prob        = max(Prob + ProbDot * dt, 0);
        if sum(Prob) > 0
            Prob = Prob / sum(Prob);
        else
            Prob = ones(1, popsize) / popsize;
        end

        % Migration
        lambdaMin = min(lambda);
        lambdaMax = max(lambda);
        Island    = chrom;

        if lambdaMax > lambdaMin
            lambdaScale = lambdaLower + (lambdaUpper - lambdaLower) * ...
                          (lambda - lambdaMin) / (lambdaMax - lambdaMin);
        else
            lambdaScale = zeros(popsize, 1);
        end

        muCum = cumsum(mu);
        for k = 1:popsize
            if rand > pmodify
                continue;
            end
            take = rand(1, dim) < lambdaScale(k);
            if any(take)
                cols = find(take);
                for jj = 1:numel(cols)
                    sel = rouletteMu(muCum);
                    Island(k, cols(jj)) = chrom(sel, cols(jj));
                end
            end
        end

        % Mutation, in inverse proportion to the species-count probability
        Pmax = max(Prob);
        if Pmax > 0
            MutationRate = pmutate * (1 - Prob / Pmax);
        else
            MutationRate = pmutate * ones(1, popsize);
        end
        for k = round(popsize / 2):popsize
            hit = MutationRate(k) > rand(1, dim);
            if any(hit)
                Island(k, hit) = lb(hit) + span(hit) .* rand(1, sum(hit));
            end
        end

        % Feasibility, evaluation, sort
        chrom = min(max(Island, LBm), UBm);

        [cost, FE] = calculate_fitness(chrom', problem, FE);
        cost = cost(:);

        for i = 1:popsize
            if cost(i) < bsf
                bsf  = cost(i);
                bsfx = chrom(i, :);
            end
            ec = FE - popsize + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, chrom, cost, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);

        % The previous generation's elites replace the worst
        for k = 1:Keep
            chrom(popsize - k + 1, :) = chromKeep(k, :);
            cost(popsize - k + 1)     = costKeep(k);
        end

        % Simon's caveat: the replaced gene's cost is deliberately NOT recomputed
        chrom = clearDups(chrom, lb, span);

        [cost, ord] = sort(cost, 'ascend');
        chrom = chrom(ord, :);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function chrom = clearDups(chrom, lb, span)
% Simon's ClearDups: permutation-invariant duplicate test; one gene redrawn, cost not recomputed
    n   = size(chrom, 1);
    dim = size(chrom, 2);
    for i = 1:n
        c1 = sort(chrom(i, :));
        for j = i+1:n
            if isequal(c1, sort(chrom(j, :)))
                p = ceil(dim * rand);
                chrom(j, p) = lb(p) + span(p) * rand;
            end
        end
    end
end

function sel = rouletteMu(muCum)
% Roulette on the emigration rates: good habitats are copied from most often.
    r   = rand * muCum(end);
    sel = find(muCum >= r, 1, 'first');
    if isempty(sel)
        sel = numel(muCum);
    end
end

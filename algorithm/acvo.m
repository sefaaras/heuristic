% ----------------------------------------------------------------------- %
% Anti-Coronavirus Optimization (ACVO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   popSize = 25          % Population size
%   delta = 2             % Social-distancing threshold
%   R0 = 2.5              % Basic reproduction number
%   alpha = 1, ralpha = 0.98   % Adaptation factor and its damping
%   quarantineDuration = 5, isolationDuration = 10
%
% Algorithm Concept:
%   - Socio-inspired: a population of people slows the spread of the virus
%     through three operators: social distancing, quarantine and isolation
%   - The fittest (healthiest) individual is sought over the iterations
%
% Reference:
% Hojjat Emami,
% Anti-coronavirus optimization algorithm,
% Soft Computing 26 (2022) 4991-5023
% https://doi.org/10.1007/s00500-022-06903-5
% ----------------------------------------------------------------------- %
% Implementation Note:
% Positions are clamped to the box inside the shared evaluation helper, which
% also returns the clamped point so the stored individual matches what was
% evaluated. The reference applies no bound handling and the numeric CEC mex
% evaluates outside the box without complaint.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = acvo(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lowerbound = problem.lb;
    upperbound = problem.ub;
    maxFE = problem.maxFe;

    ProblemParams.lb = lowerbound;
    ProblemParams.ub = upperbound;
    ProblemParams.NPar = dim;
    ProblemParams.VarMin = ProblemParams.lb;
    ProblemParams.VarMax = ProblemParams.ub;
    ProblemParams.VarSize = [1 dim];

    % Algorithm parameters
    AlgorithmParams.popSize = 25;
    AlgorithmParams.MaxIt = (maxFE / (AlgorithmParams.popSize * AlgorithmParams.popSize));
    AlgorithmParams.landa = 1;
    AlgorithmParams.delta = 2;
    AlgorithmParams.R0 = 2.5;
    AlgorithmParams.m = 0;
    AlgorithmParams.alpha = 1;
    AlgorithmParams.ralpha = 0.98;
    AlgorithmParams.quarantineDuration = 5;
    AlgorithmParams.isolationDuration = 10;

    if isscalar(ProblemParams.VarMin) && isscalar(ProblemParams.VarMax)
        ProblemParams.dmax = (ProblemParams.VarMax - ProblemParams.VarMin) * sqrt(ProblemParams.NPar);
    else
        ProblemParams.dmax = norm(ProblemParams.VarMax - ProblemParams.VarMin);
    end

    popSize = AlgorithmParams.popSize;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    bf = inf;              % best-so-far cost (framework convention)
    bs = zeros(1, dim);    % best-so-far position

    % Initialization
    person.position = [];
    person.cost = [];
    person.status = [];
    recovered = repmat(person, 0, 1);

    [pop, bestSolution, FE, bf, bs] = initialization(ProblemParams, AlgorithmParams, problem, FE, bf, bs);

    list.position = [];
    list.cost = [];
    list.status = [];
    list.fs = [];
    list.days = [];
    Q = repmat(list, 0, 1);
    I = repmat(list, 0, 1);

    [population_history, fitness_history, history_index, curve] = record_block(...
        0, FE, maxFE, bf, pop, popSize, dim, curve, ...
        population_history, fitness_history, history_index);

    it = 1;
    while FE < maxFE
        AlgorithmParams.it = it;
        FE_before = FE;

        % Social distancing
        [pop, FE, bf, bs] = socialDistancing(ProblemParams, AlgorithmParams, pop, bestSolution, problem, FE, bf, bs);

        % Quarantine
        [pop, Q, I, FE, bf, bs] = quarantine(ProblemParams, AlgorithmParams, pop, Q, I, bestSolution, problem, FE, bf, bs);

        % Isolation
        [pop, I, FE, bf, bs] = isolation(ProblemParams, AlgorithmParams, pop, Q, I, bestSolution, problem, FE, bf, bs);

        pop = [pop; recovered];
        ss = size(pop, 1);

        [~, SortOrder] = sort([pop.cost]);
        pop = pop(SortOrder);
        pop = pop(1:ss);

        bestSolution = pop(1, :);
        AlgorithmParams.alpha = AlgorithmParams.alpha * AlgorithmParams.ralpha;

        % Record convergence curve and history
        [population_history, fitness_history, history_index, curve] = record_block(...
            FE_before, FE, maxFE, bf, pop, popSize, dim, curve, ...
            population_history, fitness_history, history_index);

        it = it + 1;
    end

    best_solution = bs;
    best_fitness = bf;

end

% Objective evaluation: clamps to the box, threads FE and best-so-far
function [z, FE, bf, bs, pos] = evalp(pos, problem, FE, bf, bs)
    pos = min(max(pos, problem.lb), problem.ub);
    [z, FE] = calculate_fitness(pos', problem, FE);
    if z < bf
        bf = z;
        bs = pos;
    end
end

% Record helper for a block's FE range (snapshot best popSize)
function [pop_hist, fit_hist, hist_idx, curve] = record_block(FE_before, FE, maxFE, bestcost, pop, popSize, dim, curve, pop_hist, fit_hist, hist_idx)
    np = numel(pop);
    take = min(popSize, np);
    pos = zeros(popSize, dim);
    cost = zeros(1, popSize);
    for i = 1:take
        pos(i, :) = pop(i).position;
        cost(i) = pop(i).cost;
    end
    for i = take + 1:popSize
        pos(i, :) = pop(take).position;
        cost(i) = pop(take).cost;
    end
    for eval_count = (FE_before + 1):FE
        if eval_count <= maxFE
            curve(eval_count) = bestcost;
            [pop_hist, fit_hist, hist_idx] = record_history(...
                eval_count, pos, cost, pop_hist, fit_hist, hist_idx, maxFE);
        end
    end
end

% Initialization
function [pop, bestSolution, FE, bf, bs] = initialization(ProblemParams, AlgorithmParams, problem, FE, bf, bs)
    person.position = [];
    person.cost = [];
    person.status = [];

    pop = repmat(person, AlgorithmParams.popSize, 1);
    bestSolution.cost = inf;

    for i = 1:AlgorithmParams.popSize
        pop(i).position = unifrnd(ProblemParams.VarMin, ProblemParams.VarMax, ProblemParams.VarSize);
        [pop(i).cost, FE, bf, bs, pop(i).position] = evalp(pop(i).position, problem, FE, bf, bs);
        pop(i).status = 1;
        if pop(i).cost <= bestSolution.cost
            bestSolution = pop(i);
        end
    end
end

% Isolation phase
function [pop, I, FE, bf, bs] = isolation(ProblemParams, AlgorithmParams, pop, Q, I, bestSolution, problem, FE, bf, bs) %#ok<INUSD>
    list.position = [];
    list.cost = [];
    list.status = [];
    list.fs = [];
    list.days = [];

    R = repmat(list, 0, 1);
    person.position = [];
    person.cost = [];
    person.status = [];

    for i = 1:numel(I)
        if FE >= problem.maxFe, break; end   % individuals past this keep their previous cost
        if ((I(i).status == -1) && (I(i).days <= AlgorithmParams.isolationDuration))
            gamma = 1 - (I(i).days / AlgorithmParams.isolationDuration);
            r1 = 0.5 * rand;
            lq = ceil(r1 * ProblemParams.NPar);
            varPos = randi(ProblemParams.NPar, 1, lq);
            I(i).position(varPos) = gamma .* (I(i).position(varPos) + bestSolution.position(varPos) .* AlgorithmParams.alpha);
            I(i).days = I(i).days + 1;
            [I(i).cost, FE, bf, bs, I(i).position] = evalp(I(i).position, problem, FE, bf, bs);
        else
            [I(i).cost, FE, bf, bs, I(i).position] = evalp(I(i).position, problem, FE, bf, bs);
            if (I(i).cost <= I(i).fs)
                I(i).status = 1;
                R = [R; I(i)];
            end
        end
    end

    I = I([I.status] == -1);
    recovered = repmat(person, 0, 1);

    if (~isempty(R))
        [recovered(1:numel(R)).position] = R.position;
        [recovered(1:numel(R)).cost] = R.cost;
        [recovered(1:numel(R)).status] = R.status;
    end

    pop = [pop; recovered];
end

% Social distancing phase
function [pop, FE, bf, bs] = socialDistancing(ProblemParams, AlgorithmParams, pop, bestSolution, problem, FE, bf, bs)
    AlgorithmParams.m = AlgorithmParams.it / AlgorithmParams.MaxIt;
    if (AlgorithmParams.m < 0.5)
        AlgorithmParams.m = 0.5;
    else
        AlgorithmParams.m = AlgorithmParams.it / AlgorithmParams.MaxIt;
    end
    pSize = ceil(AlgorithmParams.m * AlgorithmParams.popSize); %#ok<NASGU>

    person.position = [];
    person.cost = [];
    person.status = [];
    [psize1, ~] = size(pop);
     % newpop sized to the current population to avoid phantom (empty) entries
    newpop = repmat(person, psize1, 1);
    nNew = 0;

    for i = 1:psize1
        if FE >= problem.maxFe, break; end
        bestCost = inf;
        bestSol = [];
        for j = i + 1:psize1
            if FE >= problem.maxFe, break; end   % partial sweep: i keeps whatever it found so far
            dij = norm(pop(i).position - pop(j).position) / ProblemParams.dmax;
            if (dij ~= 0 && (dij <= AlgorithmParams.delta))
                sd = AlgorithmParams.delta - dij;
                impact = exp(-dij / AlgorithmParams.delta);
                Delta1 = sd * impact * AlgorithmParams.alpha * unifrnd(-1, +1, ProblemParams.VarSize);

                sij = norm(pop(i).position - bestSolution.position) / ProblemParams.dmax;
                beta = 2 * exp(-(sij));
                L = Levy(ProblemParams.NPar);
                Delta2 = beta .* L .* (bestSolution.position - pop(i).position);

                newsol.position = pop(i).position + Delta1 + Delta2;
                newsol.position = max(newsol.position, ProblemParams.VarMin);
                newsol.position = min(newsol.position, ProblemParams.VarMax);
                [newsol.cost, FE, bf, bs, newsol.position] = evalp(newsol.position, problem, FE, bf, bs);
                newsol.status = 1;

                if newsol.cost <= bestCost
                    bestCost = newsol.cost;
                    bestSol = newsol;
                end
            end
        end
        if ~isempty(bestSol)
            nNew = nNew + 1;
            newpop(nNew) = bestSol;
        end
    end
    newpop = newpop(1:nNew);

    pop = [pop; newpop];
    [~, SortOrder] = sort([pop.cost]);
    pop = pop(SortOrder);
    pop = pop(1:min(AlgorithmParams.popSize, numel(pop)));
end

% Levy flight
function L = Levy(d)
    landa = 1.5;
    sigma = (gamma(1 + landa) * sin(pi * landa / 2) / (gamma((1 + landa) / 2) * landa * 2^((landa - 1) / 2)))^(1 / landa);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / landa);
    L = 0.1 * step;
end

% Quarantine phase
function [pop, Q, I, FE, bf, bs] = quarantine(ProblemParams, AlgorithmParams, pop, Q, I, bestSolution, problem, FE, bf, bs) %#ok<INUSD>
    list.position = [];
    list.cost = [];
    list.status = [];
    list.fs = [];
    list.days = [];

    person.position = [];
    person.cost = [];
    person.status = [];

    R = repmat(list, 0, 1);
    pSize = size(pop, 1);

    q = ceil((1 - (1 - (AlgorithmParams.landa^2)) * AlgorithmParams.m) * AlgorithmParams.R0);
    AlgorithmParams.landa = 1 - (AlgorithmParams.it / AlgorithmParams.MaxIt);
    qq = repmat(list, q, 1);

    [~, SortOrder] = sort([pop.cost]);
    pop = pop(SortOrder);
    pop = pop(1:pSize);

    if (size(pop, 1) > q)
        for i = 1:q
            qq(i).position = pop(pSize - q + i).position;
            qq(i).cost = pop(pSize - q + i).cost;
            pop(pSize - q + i).status = 0;
            qq(i).status = 0;
            qq(i).fs = pop(pSize - q + i).cost;
            qq(i).days = 0;
        end
    else
        qq = [];
    end
    pop = pop([pop.status] == 1);
    Q = [Q; qq];

    for i = 1:numel(Q)
        if FE >= problem.maxFe, break; end   % individuals past this keep their previous cost
        if ((Q(i).status == 0) && (Q(i).days < AlgorithmParams.quarantineDuration))
            r1 = 0.5 * rand;
            lq = ceil(r1 * ProblemParams.NPar);
            varPos = randi(ProblemParams.NPar, 1, lq);
            Q(i).position(varPos) = Q(i).position(varPos) + rand * unifrnd(-1, +1, size(varPos));
            Q(i).days = Q(i).days + 1;
        else
            [Q(i).cost, FE, bf, bs, Q(i).position] = evalp(Q(i).position, problem, FE, bf, bs);
            if (Q(i).cost <= Q(i).fs)
                Q(i).status = 1;
                R = [R; Q(i)];
            else
                Q(i).status = -1;
                Q(i).days = 0;
                Q(i).fs = Q(i).cost;
                I = [I; Q(i)];
            end
        end
    end
    Q = Q([Q.status] == 0);

    recovered = repmat(person, 0, 1);
    if (~isempty(R))
        [recovered(1:numel(R)).position] = R.position;
        [recovered(1:numel(R)).cost] = R.cost;
        [recovered(1:numel(R)).status] = R.status;
    end
    pop = [pop; recovered];
end

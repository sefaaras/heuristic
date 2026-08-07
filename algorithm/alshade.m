% ----------------------------------------------------------------------- %
% Adaptive L-SHADE (AL-SHADE)
% Variant of L-SHADE with two competing mutation strategies
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NPinit = 18*dim   % Initial population size
%   NPmin  = 4        % Final population size (linear reduction)
%   rarc   = 2.6      % Archive size factor
%   p      = 0.11     % pbest fraction
%   H      = 6        % Historical memory size (MCR(H)=MF(H)=0.9)
%   P      = 0.5      % Initial probability of the pbest mutation branch
%
% Algorithm Concept:
%   - Two competing mutation strategies:
%       current-to-pbest/1 with archive (probability P)
%       current-to-xmean/1 with archive, where xmean is the weighted mean of
%       the best half of the archive
%   - The probability P is adapted online from the relative success rates of
%     the two strategies, scaled by the consumed evaluation budget, and is
%     clamped to [0.1, 0.9]
%   - Cauchy-sampled scaling factors from a pre-generated pool, normal
%     crossover rates, Lehmer-mean memory updates and linear population size
%     reduction, as in L-SHADE
%
% Reference:
% Yintong Li, Tong Han, Huan Zhou, Shangqin Tang, Hui Zhao,
% A novel adaptive L-SHADE algorithm and its application in UAV swarm
% resource configuration problem,
% Information Sciences (2022).
% https://doi.org/10.1016/j.ins.2022.05.058
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = alshade(problem)

    dim    = problem.dimension;
    lb     = problem.lb(:);      % dim x 1
    ub     = problem.ub(:);
    FEsmax = problem.maxFe;

    SearchAgents_no = 18 * dim;
    NPinit = SearchAgents_no;
    NPmin  = 4;

    F  = 0.5;
    CR = 0.5;
    rarc = 2.6;
    p = 0.11;
    H = 6;
    P = 0.5;

    counteval = 0;
    curve = zeros(1, FEsmax);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialise the population
    X = lb + (ub - lb) .* rand(dim, SearchAgents_no);
    [fitness, counteval] = calculate_fitness(X, problem, counteval);
    fitness = fitness(:)';

    [fitness, fidx] = sort(fitness);
    X = X(:, fidx);

    bsf = fitness(1);
    for eval_count = 1:min(SearchAgents_no, FEsmax)
        curve(eval_count) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, X', fitness, population_history, fitness_history, ...
            history_index, FEsmax);
    end

    % Archive and adaptive memories
    Asize = round(rarc * SearchAgents_no);
    A = [];
    nA = 0;
    MF  = F  * ones(H, 1);
    MCR = CR * ones(H, 1);
    MCR(H) = 0.9;
    MF(H)  = 0.9;
    iM = 1;

    A(:, nA + 1) = X(:, 1);
    Afitness(nA + 1) = fitness(1);
    nA = nA + 1;

    V = X;
    U = X;
    S_CR = zeros(1, SearchAgents_no);
    S_F  = zeros(1, SearchAgents_no);
    S_df = zeros(1, SearchAgents_no);

    Chy  = cauchyrnd(0, 0.1, SearchAgents_no + 200);
    iChy = 1;

    % Main loop
    while counteval < FEsmax
        SEL = ceil(nA / 2);
        weights = log(SEL + 1/2) - log(1:SEL)';
        weights = weights / sum(weights);
        Xsel = A(:, 1:SEL);
        xmean = Xsel * weights;

        pbest = 1 + floor(max(2, round(p * SearchAgents_no)) * rand(1, SearchAgents_no));
        r = floor(1 + H * rand(1, SearchAgents_no));

        CR = MCR(r)' + 0.1 * randn(1, SearchAgents_no);
        CR((CR < 0) | (MCR(r)' == -1)) = 0;
        CR(CR > 1) = 1;

        F = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            while F(i) <= 0
                F(i) = MF(r(i)) + Chy(iChy);
                iChy = mod(iChy, numel(Chy)) + 1;
            end
        end
        F(F > 1) = 1;

        PA = [X, A];

        % Mutation and crossover
        memory = zeros(1, SearchAgents_no);
        for i = 1:SearchAgents_no
            r1 = floor(1 + SearchAgents_no * rand);
            while i == r1
                r1 = floor(1 + SearchAgents_no * rand);
            end
            r2 = floor(1 + (SearchAgents_no + nA) * rand);
            while i == r1 || r1 == r2
                r2 = floor(1 + (SearchAgents_no + nA) * rand);
            end

            if rand < P
                V(:, i) = X(:, i) + F(i) .* (X(:, pbest(i)) - X(:, i)) + F(i) .* (X(:, r1) - PA(:, r2));
                memory(i) = 1;
            else
                V(:, i) = X(:, i) + F(i) .* (xmean - X(:, i)) + F(i) .* (X(:, r1) - PA(:, r2));
                memory(i) = 3;
            end

            for j = 1:dim
                if V(j, i) < lb(j)
                    V(j, i) = 0.5 * (lb(j) + X(j, i));
                end
                if V(j, i) > ub(j)
                    V(j, i) = 0.5 * (ub(j) + X(j, i));
                end
            end

            jrand = floor(1 + dim * rand);
            for j = 1:dim
                if rand < CR(i) || j == jrand
                    U(j, i) = V(j, i);
                else
                    U(j, i) = X(j, i);
                end
            end
        end

        % Evaluation
        [fu, counteval] = calculate_fitness(U, problem, counteval);
        fu = fu(:)';

        % Adapt the strategy probability P
        elitism = fu <= fitness;
        LL = zeros(1, SearchAgents_no);
        LL(elitism) = 1;
        LLL = memory + LL;

        A1_ALL    = sum(find(memory == 1));
        A1_better = sum(find(LLL == 2));
        A2_ALL    = sum(find(memory == 3));
        A2_better = sum(find(LLL == 4));

        if A1_ALL ~= 0 && A2_ALL ~= 0
            P_A1 = A1_better / A1_ALL;
            P_A2 = A2_better / A2_ALL;
            P = P + 0.05 * (1 - P) * (P_A1 - P_A2) * counteval / FEsmax;
            P = min(0.9, P);
            P = max(0.1, P);
        end

        % Selection
        nS = 0;
        for i = 1:SearchAgents_no
            if fu(i) < fitness(i)
                nS = nS + 1;
                S_CR(nS) = CR(i);
                S_F(nS)  = F(i);
                S_df(nS) = abs(fu(i) - fitness(i));
                X(:, i)  = U(:, i);
                fitness(i) = fu(i);
                if nA < Asize
                    A(:, nA + 1) = X(:, i);
                    Afitness(nA + 1) = fu(i);
                    nA = nA + 1;
                else
                    ri = floor(1 + Asize * rand);
                    A(:, ri) = X(:, i);
                    Afitness(ri) = fu(i);
                end
            elseif fu(i) == fitness(i)
                X(:, i) = U(:, i);
            end
        end

        % Update MCR and MF
        if nS > 0
            w = S_df(1:nS) ./ sum(S_df(1:nS));
            if all(S_CR(1:nS) == 0)
                MCR(iM) = -1;
            elseif MCR(iM) ~= -1
                MCR(iM) = sum(w .* S_CR(1:nS) .* S_CR(1:nS)) / sum(w .* S_CR(1:nS));
            end
            MF(iM) = sum(w .* S_F(1:nS) .* S_F(1:nS)) / sum(w .* S_F(1:nS));
            iM = mod(iM, H - 1) + 1;
        end

        [fitness, fidx] = sort(fitness);
        X = X(:, fidx);

        if fitness(1) < bsf
            bsf = fitness(1);
        end
        for k = 1:SearchAgents_no
            ec = counteval - SearchAgents_no + k;
            if ec >= 1 && ec <= FEsmax
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X', fitness, population_history, fitness_history, ...
                    history_index, FEsmax);
            end
        end

        % Linear population size reduction
        SearchAgents_no = round(NPinit - (NPinit - NPmin) * counteval / FEsmax);
        SearchAgents_no = max(NPmin, SearchAgents_no);
        fitness = fitness(1:SearchAgents_no);
        X = X(:, 1:SearchAgents_no);
        U = U(:, 1:SearchAgents_no);
        V = V(:, 1:SearchAgents_no);

        [Afitness, Ax] = sort(Afitness);
        A = A(:, Ax);
        Asize = round(rarc * SearchAgents_no);
        if nA > Asize
            nA = Asize;
            A = A(:, 1:Asize);
            Afitness = Afitness(1:Asize);
        end
    end

    curve(min(counteval, FEsmax):end) = bsf;

    best_fitness  = fitness(1);
    best_solution = X(:, 1)';
end

% Cauchy random numbers: r = a + b*tan(pi*(rand(n)-0.5))
function r = cauchyrnd(a, b, n)
    b(b <= 0) = NaN;
    p = rand(n);
    x = a + b .* tan(pi * (p - 0.5));
    x(p == 0) = -Inf;
    x(p == 1) = Inf;
    r = x;
end

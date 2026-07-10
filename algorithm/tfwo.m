% ----------------------------------------------------------------------- %
% Turbulent Flow of Water-based Optimization (TFWO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nWh  = 3    % Number of whirlpools
%   nObW = 30   % Number of objects (particles) per whirlpool
%   nPop = nWh + nWh*nObW = 93   % Total population
%
% Algorithm Concept:
%   - Inspired by the whirlpool phenomenon in turbulent water flow
%   - Population split into whirlpools, each with a set of objects
%   - Whirlpools exert centripetal force pulling objects to the centre
%   - Whirlpools also interact; the best object may become the new centre
%   - Globals removed (parfor-safe); unifrnd replaced by rand sampling
%
% Reference:
% Mojtaba Ghasemi, Iraj Faraji Davoudkhani, Ebrahim Akbari, Abolfazl Rahimnejad, Sahand Ghavidel, Li Li,
% A novel and effective optimization algorithm for global optimization and its engineering applications: Turbulent Flow of Water-based Optimization (TFWO),
% Engineering Applications of Artificial Intelligence 92 (2020) 103666
% https://doi.org/10.1016/j.engappai.2020.103666
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tfwo(problem)

    % Extract problem parameters
    nVar   = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE  = problem.maxFe;

    % TFWO settings
    nWh  = 3;
    nObW = 30;
    nPop = nWh + nWh * nObW;
    nOb  = nPop - nWh;

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage (population = all whirlpool centres + objects = nPop)
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, nPop, nVar);
    fitness_history = zeros(history_size, nPop);
    history_index = 1;

    % Initialization
    [Whirlpool, FE] = Initialize(problem, FE, nVar, VarMin, VarMax, nPop, nWh, nObW, nOb);

    [P, C] = gather_population(Whirlpool, nPop, nVar);
    [best_cost, bi] = min(C);
    best_pos = P(bi, :);

    for eval_count = 1:nPop
        curve(eval_count) = best_cost;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, P, C, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    % Main loop
    while FE < maxFE
        FE_before = FE;

        [Whirlpool, FE] = Effectsofwhirlpools(Whirlpool, problem, FE, nVar, VarMin, VarMax);
        Whirlpool = Pseudocode6(Whirlpool);

        [P, C] = gather_population(Whirlpool, nPop, nVar);
        [gen_best, gi] = min(C);
        if gen_best < best_cost
            best_cost = gen_best;
            best_pos  = P(gi, :);
        end

        for eval_count = (FE_before + 1):FE
            if eval_count >= 1 && eval_count <= maxFE
                curve(eval_count) = best_cost;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, P, C, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_fitness  = best_cost;
    best_solution = best_pos;
end

%% --- Initialization: build whirlpools and assign objects ---
function [Whirlpool, FE] = Initialize(problem, FE, nVar, VarMin, VarMax, nPop, nWh, nObW, nOb)
    Pos = rand(nPop, nVar) .* (VarMax - VarMin) + VarMin;
    [Cost, FE] = calculate_fitness(Pos', problem, FE);
    Cost = Cost(:);

    EmptyObject.Position = [];
    EmptyObject.Cost = [];
    EmptyObject.delta = [];
    Objects = repmat(EmptyObject, nPop, 1);
    for k = 1:nPop
        Objects(k).Position = Pos(k, :);
        Objects(k).Cost = Cost(k);
        Objects(k).delta = 0;
    end

    [~, order] = sort([Objects.Cost]);
    Objects = Objects(order);

    EmptyWhirlpool.Position = [];
    EmptyWhirlpool.Cost = [];
    EmptyWhirlpool.TotalCost = [];
    EmptyWhirlpool.nObW = [];
    EmptyWhirlpool.delta = [];
    EmptyWhirlpool.Objects = [];
    Whirlpool = repmat(EmptyWhirlpool, nWh, 1);
    for i = 1:nWh
        Whirlpool(i).Position = Objects(i).Position;
        Whirlpool(i).Cost = Objects(i).Cost;
        Whirlpool(i).delta = Objects(i).delta;
    end

    Objects = Objects(nWh + 1:end);
    Objects = Objects(randperm(nOb));
    for i = 1:nWh
        Whirlpool(i).nObW = nObW;
        Whirlpool(i).Objects = Objects(1:nObW);
        Objects = Objects(nObW + 1:end);
    end
end

%% --- Gather all positions/costs into matrices ---
function [P, C] = gather_population(Whirlpool, nPop, nVar)
    P = zeros(nPop, nVar);
    C = zeros(nPop, 1);
    idx = 0;
    for i = 1:numel(Whirlpool)
        idx = idx + 1;
        P(idx, :) = Whirlpool(i).Position;
        C(idx) = Whirlpool(i).Cost;
        for j = 1:Whirlpool(i).nObW
            idx = idx + 1;
            P(idx, :) = Whirlpool(i).Objects(j).Position;
            C(idx) = Whirlpool(i).Objects(j).Cost;
        end
    end
end

%% --- Move the best object of each whirlpool to its centre (Pseudocode 6) ---
function Whirlpool = Pseudocode6(Whirlpool)
    for i = 1:numel(Whirlpool)
        cc = [Whirlpool(i).Objects.Cost];
        [min_cc, min_cc_index] = min(cc);
        if min_cc <= Whirlpool(i).Cost
            BestObject = Whirlpool(i).Objects(min_cc_index);
            Whirlpool(i).Objects(min_cc_index).Position = Whirlpool(i).Position;
            Whirlpool(i).Objects(min_cc_index).Cost = Whirlpool(i).Cost;
            Whirlpool(i).Position = BestObject.Position;
            Whirlpool(i).Cost = BestObject.Cost;
        end
    end
end

%% --- Effects of whirlpools on objects and on each other (Pseudocodes 1-5) ---
function [Whirlpool, FE] = Effectsofwhirlpools(Whirlpool, problem, FE, nVar, VarMin, VarMax)

    for i = 1:numel(Whirlpool)
        for j = 1:Whirlpool(i).nObW

            if numel(Whirlpool) ~= 1
                J = [];
                AA = 1:numel(Whirlpool);
                AA(i) = [];
                for t = 1:AA(1)
                    J(t) = (abs(Whirlpool(t).Cost) ^ 1) * ((abs(sum(Whirlpool(t).Position)) - (sum(Whirlpool(i).Objects(j).Position))) ^ 0.5);
                end
                S = min(J);
                [~, D] = find(S == J);
                d = rand(1, nVar) .* (Whirlpool(D(1)).Position - Whirlpool(i).Objects(j).Position);
                S2 = max(J);
                [~, D2] = find(S2 == J);
                d2 = rand(1, nVar) .* (Whirlpool(D2(1)).Position - Whirlpool(i).Objects(j).Position);
            else
                d = rand(1, nVar) .* (Whirlpool(i).Position - Whirlpool(i).Objects(j).Position);
                d2 = 0;
            end

            Whirlpool(i).Objects(j).delta = Whirlpool(i).Objects(j).delta + rand * rand * pi;
            eee = Whirlpool(i).Objects(j).delta;
            fr0 = cos(eee);
            fr10 = -sin(eee);
            x = ((fr0 .* d) + (fr10 .* d2)) * (1 + abs(fr0 * fr10 * 1));
            RR = Whirlpool(i).Position - x;
            RR = min(max(RR, VarMin), VarMax);

            [Cost, FE] = calculate_fitness(RR', problem, FE);
            Cost = Cost(1);
            if Cost <= Whirlpool(i).Objects(j).Cost
                Whirlpool(i).Objects(j).Cost = Cost;
                Whirlpool(i).Objects(j).Position = RR;
            end

            FE_i = (abs(cos(Whirlpool(i).Objects(j).delta) ^ 2 * sin(Whirlpool(i).Objects(j).delta) ^ 2)) ^ 2;
            if rand < FE_i
                k = randi([1 nVar]);
                Whirlpool(i).Objects(j).Position(k) = rand * (VarMax(k) - VarMin(k)) + VarMin(k);
                [c2, FE] = calculate_fitness(Whirlpool(i).Objects(j).Position', problem, FE);
                Whirlpool(i).Objects(j).Cost = c2(1);
            end
        end
    end

    % Pseudo-code 4: interaction between whirlpools
    J2 = zeros(1, numel(Whirlpool));
    for t = 1:numel(Whirlpool)
        J2(t) = Whirlpool(t).Cost;
    end
    S2 = min(J2);
    [~, D2] = find(S2 == J2);
    d2 = Whirlpool(D2(1)).Position;

    for i = 1:numel(Whirlpool)
        J = [];
        for t = 1:numel(Whirlpool)
            J(t) = Whirlpool(t).Cost * (abs((sum(Whirlpool(t).Position)) - (sum(Whirlpool(i).Position))));
            if t == i
                J(t) = inf;
            end
        end
        S = min(J);
        [~, D] = find(S == J);

        Whirlpool(i).delta = Whirlpool(i).delta + rand * rand * pi;
        d = Whirlpool(D(1)).Position - Whirlpool(i).Position;
        fr = abs(cos(Whirlpool(i).delta) + sin(Whirlpool(i).delta));
        x = fr * rand(1, nVar) .* d;

        WP1_Position = Whirlpool(D(1)).Position - x;
        WP1_Position = min(max(WP1_Position, VarMin), VarMax);
        [WP1_Cost, FE] = calculate_fitness(WP1_Position', problem, FE);
        WP1_Cost = WP1_Cost(1);

        % Pseudo-code 5: whirlpool selection
        if WP1_Cost <= Whirlpool(i).Cost
            Whirlpool(i).Position = WP1_Position;
            Whirlpool(i).Cost = WP1_Cost;
        end
    end

    % Kept for fidelity with the reference (condition never holds by construction)
    if S2 < Whirlpool(D2(1)).Cost
        Whirlpool(end).Position = d2;
        Whirlpool(end).Cost = S2;
    end
end

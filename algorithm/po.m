% ----------------------------------------------------------------------- %
% Political Optimizer (PO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   SearchAgents_no = 50    % Population = areas * parties
%   areas   = 10            % Number of constituencies
%   parties = 5             % Number of political parties
%   lambda  = 1             % Initial party-switching rate
%
% Algorithm Concept:
%   - Party formation & constituency allocation: the population is arranged
%     as a parties-by-areas grid; each area has a winner and each party a leader
%   - Election campaign: the recent-past-based position updating strategy
%     (RPPUS, Eq. 9-10) moves a member relative to its party leader and to
%     its area winner, using the previous position and the previous fitness
%     to pick one of six geometric cases
%   - Party switching: with probability psr = (1 - t/Max_iter)*lambda a member
%     swaps with the least-fit member of another party
%   - Parliamentary affairs: area winners refine themselves against another
%     randomly chosen area winner
%
% Reference:
% Qamar Askari, Irfan Younas, Mehreen Saeed,
% Political Optimizer: A novel socio-inspired meta-heuristic for global
% optimization,
% Knowledge-Based Systems 195 (2020) 105709.
% https://doi.org/10.1016/j.knosys.2020.105709
% ----------------------------------------------------------------------- %
% Implementation Note:
% The alternative-winner candidate is clamped like the two population-level
% sites already were; the reference leaves that one path unbounded, which put
% roughly 8% of a 1e5 budget outside the box.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = po(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    areas   = 10;
    parties = 5;
    lambda  = 1;
    SearchAgents_no = areas * parties;

    Max_iter = max(1, ceil((maxFE - SearchAgents_no) / (SearchAgents_no + areas)));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    Leader_pos   = zeros(1, dim);
    Leader_score = inf;

    Positions     = initialization(SearchAgents_no, dim, ub, lb);
    auxPositions  = Positions;
    prevPositions = Positions;
    fitness       = zeros(SearchAgents_no, 1);

    % Election (initial)
    for i = 1:SearchAgents_no
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
    end
    [fitness, FE] = calculate_fitness(Positions', problem, FE);
    fitness = fitness(:);
    for i = 1:SearchAgents_no
        if fitness(i, 1) < Leader_score
            Leader_score = fitness(i, 1);
            Leader_pos   = Positions(i, :);
        end
    end
    bsf = Leader_score;

    for eval_count = 1:SearchAgents_no
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, Positions, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    auxFitness  = fitness;
    prevFitness = fitness;

    % Government formation
    [aWinnerInd, aWinners, pLeaderInd, pLeaders] = ...
        government(Positions, fitness, areas, parties, dim, SearchAgents_no);

    t = 0;
    while t < Max_iter && FE < maxFE
        prevFitness   = auxFitness;
        prevPositions = auxPositions;
        auxFitness    = fitness;
        auxPositions  = Positions;

        % Election campaign (RPPUS)
        for whichMethod = 1:2
            for a = 1:areas
                for p = 1:parties
                    i = (p - 1) * areas + a;

                    for j = 1:dim
                        if whichMethod == 1
                            center = pLeaders(p, j);
                        else
                            center = aWinners(a, j);
                        end

                        if prevFitness(i) >= fitness(i)          % Eq. (9)
                            if (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) <= center) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) >= center)
                                radius = center - Positions(i, j);
                                Positions(i, j) = center + rand() * radius;
                            elseif (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) >= center && center >= prevPositions(i,j)) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) <= center && center <= prevPositions(i,j))
                                radius = abs(Positions(i, j) - center);
                                Positions(i, j) = center + (2 * rand() - 1) * radius;
                            elseif (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) >= center && center <= prevPositions(i,j)) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) <= center && center >= prevPositions(i,j))
                                radius = abs(prevPositions(i, j) - center);
                                Positions(i, j) = center + (2 * rand() - 1) * radius;
                            end

                        elseif prevFitness(i) < fitness(i)       % Eq. (10)
                            if (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) <= center) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) >= center)
                                radius = abs(Positions(i, j) - center);
                                Positions(i, j) = center + (2 * rand() - 1) * radius;
                            elseif (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) >= center && center >= prevPositions(i,j)) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) <= center && center <= prevPositions(i,j))
                                radius = Positions(i, j) - prevPositions(i, j);
                                Positions(i, j) = prevPositions(i, j) + rand() * radius;
                            elseif (prevPositions(i,j) <= Positions(i,j) && Positions(i,j) >= center && center <= prevPositions(i,j)) ...
                                    || (prevPositions(i,j) >= Positions(i,j) && Positions(i,j) <= center && center >= prevPositions(i,j))
                                center2 = prevPositions(i, j);
                                radius  = abs(center - center2);
                                Positions(i, j) = center + (2 * rand() - 1) * radius;
                            end
                        end
                    end
                end
            end
        end

        % Party switching
        psr = (1 - t * ((1) / Max_iter)) * lambda;
        for p = 1:parties
            for a = 1:areas
                if rand() < psr
                    toParty = randi(parties);
                    while toParty == p
                        toParty = randi(parties);
                    end

                    toPStInd    = (toParty - 1) * areas + 1;
                    toPEndIndex = toPStInd + areas - 1;
                    [~, toPLeastFit] = max(fitness(toPStInd:toPEndIndex));
                    toPInd = toPStInd + toPLeastFit - 1;

                    fromPInd = (p - 1) * areas + a;
                    temp = Positions(toPInd, :);
                    Positions(toPInd, :)   = Positions(fromPInd);
                    Positions(fromPInd, :) = temp;

                    temp = fitness(toPInd);
                    fitness(toPInd)   = fitness(fromPInd);
                    fitness(fromPInd) = temp;
                end
            end
        end

        % Election
        for i = 1:SearchAgents_no
            Flag4ub = Positions(i, :) > ub;
            Flag4lb = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        [fitness, FE] = calculate_fitness(Positions', problem, FE);
        fitness = fitness(:);
        for i = 1:SearchAgents_no
            if fitness(i, 1) < Leader_score
                Leader_score = fitness(i, 1);
                Leader_pos   = Positions(i, :);
            end
        end
        if Leader_score < bsf, bsf = Leader_score; end
        for k = 1:SearchAgents_no
            ec = FE - SearchAgents_no + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, Positions, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Government formation
        [aWinnerInd, aWinners, pLeaderInd, pLeaders] = ...
            government(Positions, fitness, areas, parties, dim, SearchAgents_no);

        % Parliamentarism
        for a = 1:areas
            if FE >= maxFE, break; end
            newAWinner = aWinners(a, :);
            i = aWinnerInd(a);

            toa = randi(areas);
            while toa == a
                toa = randi(areas);
            end
            toAWinner = aWinners(toa, :);
            for j = 1:dim
                distance = abs(toAWinner(1, j) - newAWinner(1, j));
                newAWinner(1, j) = toAWinner(1, j) + (2 * rand() - 1) * distance;
            end
            newAWinner(1, :) = min(max(newAWinner(1, :), lb), ub);
            [newAWFitness, FE] = calculate_fitness(newAWinner(1, :)', problem, FE);

            if newAWFitness < fitness(i)
                Positions(i, :) = newAWinner(1, :);
                fitness(i)      = newAWFitness;
                aWinners(a, :)  = newAWinner(1, :);
            end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Positions, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        t = t + 1;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = Leader_score;
    best_solution = Leader_pos;
end

% Government formation: area winners and party leaders
function [aWinnerInd, aWinners, pLeaderInd, pLeaders] = ...
        government(Positions, fitness, areas, parties, dim, SearchAgents_no)
    aWinnerInd = zeros(areas, 1);
    aWinners   = zeros(areas, dim);
    for a = 1:areas
        [~, aWinnerParty] = min(fitness(a:areas:SearchAgents_no));
        aWinnerInd(a, 1) = (aWinnerParty - 1) * areas + a;
        aWinners(a, :)   = Positions(aWinnerInd(a, 1), :);
    end

    pLeaderInd = zeros(parties, 1);
    pLeaders   = zeros(parties, dim);
    for p = 1:parties
        pStIndex  = (p - 1) * areas + 1;
        pEndIndex = pStIndex + areas - 1;
        [~, leadIndex] = min(fitness(pStIndex:pEndIndex));
        pLeaderInd(p, 1) = (pStIndex - 1) + leadIndex;
        pLeaders(p, :)   = Positions(pLeaderInd(p, 1), :);
    end
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

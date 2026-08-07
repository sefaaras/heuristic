% ----------------------------------------------------------------------- %
% Leader-Follower Optimizer (LFO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   numFollowers           = 200   % Number of followers
%   numLeaders             = 2     % Number of leaders (dynamic hierarchy)
%   initialSigma           = 2     % Initial influence dispersion
%   explorationProbability = 0.01  % Probability of following an external agent
%
% Algorithm Concept:
%   - Influence operator: x_new = (1+R1)*x_leader + R2*x_current with
%     R1, R2 ~ sigma*N(0,1); sigma decays linearly over the run
%   - Followers combine the influenced positions of all leaders with the
%     hierarchy weights (numLeaders:-1:1)/sum, or follow a random external
%     agent with probability explorationProbability
%   - Leaders are refined by influencing each other and are replaced greedily
%   - Global leadership reassessment: the best numLeaders of the union of
%     leaders and followers become the new leaders
%
% Reference:
% Bruno L. Pereira,
% Leader-Follower Optimizer,
% IEEE Access, vol. 14 (2026), pp. 81195-81216.
% https://doi.org/10.1109/ACCESS.2026.3697639
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lfo(problem)

    numVariables = problem.dimension;
    lowerBounds  = problem.lb;
    upperBounds  = problem.ub;
    maxFE        = problem.maxFe;

    numFollowers           = 200;
    numLeaders             = 2;
    initialSigma           = 2;
    explorationProbability = 0.01;
    absConvCounterLimit    = 10000;
    relConvCounterLimit    = 10000;

    maxIterations = max(2, ceil((maxFE - numFollowers) / (numFollowers + numLeaders)));

    N     = numFollowers + numLeaders;
    FE    = 0;
    curve = zeros(1, maxFE);

    % Capped at 100 rows so a large population does not reserve GBs upfront
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Population initialisation
    followers = repmat(lowerBounds, numFollowers, 1) + ...
                repmat(upperBounds - lowerBounds, numFollowers, 1) .* rand(numFollowers, numVariables);
    [followerFitness, FE] = calculate_fitness(followers', problem, FE);
    followerFitness = followerFitness(:);

    [followerFitness, sortedIndices] = sort(followerFitness, 'ascend');
    followers = followers(sortedIndices, :);

    leaders       = followers(1:numLeaders, :);
    leaderFitness = followerFitness(1:numLeaders);

    bsf = leaderFitness(1);
    bsx = leaders(1, :);

    for eval_count = 1:numFollowers
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, [leaders; followers], [leaderFitness; followerFitness], ...
                population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    candidateLeaders       = zeros(numLeaders, numVariables);
    candidateLeaderFitness = zeros(numLeaders, 1);
    influencedPositions    = zeros(numLeaders, numVariables);

    leaderHierarchy = numLeaders:-1:1;
    leaderWeights   = leaderHierarchy / sum(leaderHierarchy);

    iteration = 0;
    running   = true;
    absConvFlag = false; relConvFlag = false;
    absConvCounter = 0;  relConvCounter = 0;
    previousBestFitness = inf;

    % Main LFO loop
    while running
        iteration = iteration + 1;

        sigma = initialSigma * (maxIterations - iteration) / (maxIterations - 1);

        % Followers update
        for i = 1:numFollowers
            if rand < explorationProbability
                externalAgent = (upperBounds - lowerBounds) .* rand(1, numVariables) + lowerBounds;
                followers(i, :) = influenceOperator(sigma, externalAgent, followers(i, :));
            else
                for j = 1:numLeaders
                    influencedPositions(j, :) = influenceOperator(sigma, leaders(j, :), followers(i, :));
                end
                followers(i, :) = leaderWeights * influencedPositions;
            end

            followers(i, :) = max(followers(i, :), lowerBounds);
            followers(i, :) = min(followers(i, :), upperBounds);
        end

        [followerFitness, FE] = calculate_fitness(followers', problem, FE);
        followerFitness = followerFitness(:);

        [mf, mi] = min(followerFitness);
        if mf < bsf
            bsf = mf;
            bsx = followers(mi, :);
        end
        for k = 1:numFollowers
            ec = FE - numFollowers + k;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, [leaders; followers], [leaderFitness; followerFitness], ...
                    population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
        if FE >= maxFE, break; end

        % Leaders update
        for i = 1:numLeaders
            if FE >= maxFE, break; end
            for j = 1:numLeaders
                influencedPositions(j, :) = influenceOperator(sigma, leaders(j, :), leaders(i, :));
            end
            candidateLeaders(i, :) = leaderWeights * influencedPositions;

            candidateLeaders(i, :) = max(candidateLeaders(i, :), lowerBounds);
            candidateLeaders(i, :) = min(candidateLeaders(i, :), upperBounds);

            [candidateLeaderFitness(i), FE] = calculate_fitness(candidateLeaders(i, :)', problem, FE);

            if candidateLeaderFitness(i) < leaderFitness(i)
                leaderFitness(i) = candidateLeaderFitness(i);
                leaders(i, :)    = candidateLeaders(i, :);
            end

            if candidateLeaderFitness(i) < bsf
                bsf = candidateLeaderFitness(i);
                bsx = candidateLeaders(i, :);
            end
            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, [leaders; followers], [leaderFitness; followerFitness], ...
                    population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Global leadership reassessment
        totalPopulation = [leaders; followers];
        totalFitness    = [leaderFitness; followerFitness];
        [totalFitness, sortedIndices] = sort(totalFitness, 'ascend');
        totalPopulation = totalPopulation(sortedIndices, :);
        for i = 1:numLeaders
            leaders(i, :)    = totalPopulation(i, :);
            leaderFitness(i) = totalFitness(i);
        end

        % Convergence analysis
        if iteration >= maxIterations
            running = false;
        end
        if FE >= maxFE
            running = false;
        end

        currentBestFitness = leaderFitness(1);

        if abs(currentBestFitness - previousBestFitness) < 1e-5
            absConvCounter = absConvCounter + 1;
            if absConvCounter >= absConvCounterLimit
                absConvFlag = true;
            end
        else
            absConvCounter = 0;
            absConvFlag = false;
        end

        if abs(currentBestFitness - previousBestFitness) / max(abs(currentBestFitness), 1e-10) < 1e-3
            relConvCounter = relConvCounter + 1;
            if relConvCounter >= relConvCounterLimit
                relConvFlag = true;
            end
        else
            relConvCounter = 0;
            relConvFlag = false;
        end

        if absConvFlag && relConvFlag
            running = false;
        end

        previousBestFitness = currentBestFitness;
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Influence operator
function updatedPosition = influenceOperator(sigma, leaderPosition, currentPosition)
    R1 = sigma * randn;
    R2 = sigma * randn;
    updatedPosition = (1 + R1) * leaderPosition + R2 * currentPosition;
end

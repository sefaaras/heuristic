% ----------------------------------------------------------------------- %
% Enzyme Action Optimizer (EAO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   EnzymeCount = 50    % Population size (enzymes / substrates)
%   EC          = 0.1   % Enzyme concentration
%
% Algorithm Concept:
%   - Affinity factor AF = sqrt(t/MaxIter) grows over the run, shifting the
%     search from exploration to exploitation
%   - Candidate 1: sinusoidal displacement of the substrate towards the best
%   - Candidates A/B: differential step between two random substrates plus an
%     AF-scaled pull to the best substrate, once with per-dimension random
%     coefficients (A) and once with scalar coefficients (B)
%   - The better of the three replaces the substrate if it improves it
%
% Reference:
% Ali Rodan, Loai Alnemer, Abdullah Al-Tamimi, Seyedali Mirjalili, Peter Tino,
% Enzyme Action Optimizer: A Novel Bio-inspired Optimization Algorithm,
% The Journal of Supercomputing (2025).
% https://doi.org/10.1007/s11227-025-07052-w
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = eao(problem)

    ActiveSiteDimension = problem.dimension;
    LB    = problem.lb;
    UB    = problem.ub;
    maxFE = problem.maxFe;

    EnzymeCount = 50;
    MaxIter     = max(1, ceil(maxFE / (3 * EnzymeCount)));
    EC          = 0.1;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % 1) Initialisation
    SubstratePool = repmat(LB, EnzymeCount, 1) + ...
                    repmat(UB - LB, EnzymeCount, 1) .* rand(EnzymeCount, ActiveSiteDimension);

    [ReactionRate, FE] = calculate_fitness(SubstratePool', problem, FE);
    ReactionRate = ReactionRate(:);

    [OptimalCatalysis, idx] = min(ReactionRate);
    BestSubstrate = SubstratePool(idx, :);
    bsf = OptimalCatalysis;

    for eval_count = 1:EnzymeCount
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, SubstratePool, ReactionRate, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % 2) Main loop
    for t = 1:MaxIter
        if FE >= maxFE, break; end

        AF = sqrt(t / MaxIter);

        for i = 1:EnzymeCount
            if FE >= maxFE, break; end

            % 1) First substrate position
            FirstSubstratePosition = (BestSubstrate - SubstratePool(i, :)) + ...
                rand(1, ActiveSiteDimension) .* sin(AF * SubstratePool(i, :));
            FirstSubstratePosition = max(min(FirstSubstratePosition, UB), LB);
            [FirstEvaluation, FE] = calculate_fitness(FirstSubstratePosition', problem, FE);
            [bsf, curve, population_history, fitness_history, history_index] = ...
                stamp(FE, maxFE, FirstEvaluation, bsf, curve, SubstratePool, ReactionRate, ...
                      population_history, fitness_history, history_index);
            if FE >= maxFE
                UpdatedPosition = FirstSubstratePosition;
                UpdatedFitness  = FirstEvaluation;
            else
                % 2) Two random distinct substrates
                Substrates = randperm(EnzymeCount, 2);
                while any(Substrates == i)
                    Substrates = randperm(EnzymeCount, 2);
                end
                S1 = SubstratePool(Substrates(1), :);
                S2 = SubstratePool(Substrates(2), :);

                % 2.1) Vector-valued random factors
                scA1 = EC + (1 - EC) * rand(1, ActiveSiteDimension);
                exA  = (EC + (1 - EC) * rand(1, ActiveSiteDimension)) .* AF;
                CandidateA = SubstratePool(i, :) + scA1 .* (S1 - S2) + ...
                             exA .* (BestSubstrate - SubstratePool(i, :));
                CandidateA = max(min(CandidateA, UB), LB);
                [CandidateAFitness, FE] = calculate_fitness(CandidateA', problem, FE);
                [bsf, curve, population_history, fitness_history, history_index] = ...
                    stamp(FE, maxFE, CandidateAFitness, bsf, curve, SubstratePool, ReactionRate, ...
                          population_history, fitness_history, history_index);

                % 2.2) Scalar random factors
                scB1 = EC + (1 - EC) * rand();
                exB  = (EC + (1 - EC) * rand()) * AF;
                CandidateB = SubstratePool(i, :) + scB1 .* (S1 - S2) + ...
                             exB .* (BestSubstrate - SubstratePool(i, :));
                CandidateB = max(min(CandidateB, UB), LB);
                if FE < maxFE
                    [CandidateBFitness, FE] = calculate_fitness(CandidateB', problem, FE);
                    [bsf, curve, population_history, fitness_history, history_index] = ...
                        stamp(FE, maxFE, CandidateBFitness, bsf, curve, SubstratePool, ReactionRate, ...
                              population_history, fitness_history, history_index);
                else
                    CandidateBFitness = inf;
                end

                % 2.3) Pick the better candidate
                if CandidateAFitness < CandidateBFitness
                    SecondSubstratePosition = CandidateA;
                    SecondEvaluation        = CandidateAFitness;
                else
                    SecondSubstratePosition = CandidateB;
                    SecondEvaluation        = CandidateBFitness;
                end

                % 3) Compare first vs. second
                if FirstEvaluation < SecondEvaluation
                    UpdatedPosition = FirstSubstratePosition;
                    UpdatedFitness  = FirstEvaluation;
                else
                    UpdatedPosition = SecondSubstratePosition;
                    UpdatedFitness  = SecondEvaluation;
                end
            end

            % 4) Update the pool and the global best
            if UpdatedFitness < ReactionRate(i)
                SubstratePool(i, :) = UpdatedPosition;
                ReactionRate(i)     = UpdatedFitness;
                if UpdatedFitness < OptimalCatalysis
                    OptimalCatalysis = UpdatedFitness;
                    BestSubstrate    = UpdatedPosition;
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = OptimalCatalysis;
    best_solution = BestSubstrate;
end

% Curve / history stamp for a single evaluation
function [bsf, curve, ph, fh, hi] = stamp(FE, maxFE, f, bsf, curve, X, Fit, ph, fh, hi)
    if f < bsf
        bsf = f;
    end
    if FE >= 1 && FE <= maxFE
        curve(FE) = bsf;
        [ph, fh, hi] = record_history(FE, X, Fit, ph, fh, hi, maxFE);
    end
end

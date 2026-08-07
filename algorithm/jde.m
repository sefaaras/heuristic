% ----------------------------------------------------------------------- %
% Self-Adapting Control Parameters in Differential Evolution (jDE)
% The self-adaptation scheme behind jDE100, jDE21 and j2020
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP   = 100                   % Population size
%   F    = 0.5, CR = 0.9         % Initial per-individual control parameters
%   tau1 = tau2 = 0.1            % Probability of re-drawing F / CR
%   Fl = 0.1, Fu = 0.9           % F is re-drawn as Fl + Fu * rand
%   CRl = 0.0, CRu = 1.0         % CR is re-drawn as CRl + CRu * rand
%
% Algorithm Concept:
%   - Each individual CARRIES ITS OWN (F, CR). Before a trial is built, each
%     value is replaced by a fresh uniform draw with probability tau, otherwise
%     the stored value is reused
%   - The trial is generated with the parent's (possibly re-drawn) parameters
%     and the parameters are inherited ONLY if the trial wins; a losing
%     individual reverts to the values it had before the draw
%   - The parameters therefore ride along with the solutions under the same
%     survival pressure: no statistics, no meta-parameters beyond tau
%   - Mutation is the plain DE/rand/1 with binomial crossover
%   - Violating components are reflected about the violated bound and clamped
%     to the opposite bound if the reflection overshoots
%
% Reference:
% Janez Brest, Saso Greiner, Borko Boskovic, Marjan Mernik, Viljem Zumer,
% Self-Adapting Control Parameters in Differential Evolution: A Comparative
% Study on Numerical Benchmark Problems,
% IEEE Transactions on Evolutionary Computation, vol. 10, no. 6, pp. 646-657, 2006.
% https://doi.org/10.1109/TEVC.2006.872133
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the MATLAB release distributed with Y. Wang's CoDE package, whose
% Readme credits the jDE source to Dr. J. Zhang.
% ONE DEFECT CORRECTED: that release's boundConstraint indexes the NP-by-D
% logical mask `pos` with another logical vector of different length, so instead
% of clamping an overshooting reflection it silently writes to unrelated entries
% of the trial. On its own [-100,100] boxes with F <= 1 a reflection almost
% never overshoots, which is why it went unnoticed; on the per-dimension bounds
% used here it would corrupt the population, so the clamp is written correctly.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jde(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    % Control parameters
    popsize = 100;
    tau1    = 0.1;
    tau2    = 0.1;
    Fl      = 0.1;
    Fu      = 0.9;
    CRl     = 0.0;
    CRu     = 1.0;

    F  = 0.5 * ones(popsize, 1);
    CR = 0.9 * ones(popsize, 1);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Initialisation
    popold = repmat(lb, popsize, 1) + rand(popsize, dim) .* repmat(ub - lb, popsize, 1);

    [valParents, FE] = calculate_fitness(popold', problem, FE);
    valParents = valParents(:);

    bsf          = inf;
    bsf_solution = popold(1, :);
    for i = 1:popsize
        if valParents(i) < bsf
            bsf          = valParents(i);
            bsf_solution = popold(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, popold, valParents, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        pop = popold;

        % Self-adaptation: re-draw F and CR with probability tau
        Fold  = F;
        CRold = CR;

        IF  = rand(popsize, 1) < tau1;
        ICR = rand(popsize, 1) < tau2;

        F(IF)   = Fl  + Fu  * rand(sum(IF), 1);
        CR(ICR) = CRl + CRu * rand(sum(ICR), 1);

        % DE/rand/1
        r0 = 1:popsize;
        [r1, r2, r3] = gnR1R2R3_jde(popsize, r0);

        vi = pop(r1, :) + F(:, ones(1, dim)) .* (pop(r2, :) - pop(r3, :));
        vi = boundConstraint_jde(vi, lu);

        % Binomial crossover
        mask  = rand(popsize, dim) > CR(:, ones(1, dim));
        rows  = (1:popsize)';
        cols  = floor(rand(popsize, 1) * dim) + 1;
        jrand = sub2ind([popsize dim], rows, cols);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        [valOffspring, FE] = calculate_fitness(ui', problem, FE);
        valOffspring = valOffspring(:);

        for i = 1:popsize
            if valOffspring(i) < bsf
                bsf          = valOffspring(i);
                bsf_solution = ui(i, :);
            end
            ec = FE - popsize + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, pop, valParents, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Selection; the parameters are inherited only on a win
        [valParents, I] = min([valParents, valOffspring], [], 2);
        popold = pop;
        popold(I == 2, :) = ui(I == 2, :);

        F(I == 1)  = Fold(I == 1);
        CR(I == 1) = CRold(I == 1);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% Helper Functions

function vi = boundConstraint_jde(vi, lu)
% Reflect about the crossed bound, clamping if it overshoots; see the header note
    NP = size(vi, 1);
    xl = repmat(lu(1, :), NP, 1);
    xu = repmat(lu(2, :), NP, 1);

    low = vi < xl;
    vi(low) = 2 .* xl(low) - vi(low);

    high = vi > xu;
    vi(high) = 2 .* xu(high) - vi(high);

    vi = min(max(vi, xl), xu);
end

function [r1, r2, r3] = gnR1R2R3_jde(NP1, r0)
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = (r1 == r0);
        if sum(pos) == 0, break; end
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end

    r2 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0, break; end
        r2(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end

    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:1000
        pos = ((r3 == r1) | (r3 == r0) | (r3 == r2));
        if sum(pos) == 0, break; end
        r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
end

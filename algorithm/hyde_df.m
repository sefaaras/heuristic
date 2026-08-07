% ----------------------------------------------------------------------- %
% Hybrid-adaptive Differential Evolution with Decay Function (HyDE-DF)
% CEC 2019 / GECCO 2019 100-Digit Challenge entry (score 93)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP   = 50                    % Population size
%   F    = 0.5 (three per individual), CR = 0.5
%   tau  = 0.1                   % jDE re-draw probability for every parameter
%   Fl = 0.1, Fu = 0.9           % F is re-drawn as Fl + Fu * rand
%   T = ceil(maxFe/NP)           % Decay horizon of the DF term, in generations
%
% Algorithm Concept:
%   - Each individual carries THREE self-adapted scaling factors (jDE rule:
%     re-draw with probability tau, keep on success, revert on failure):
%         v_i = x_i + F1*( x_best .* (F2 + N(0,1)) - x_i ) + F3*( x_r1 - x_r2 )
%     The attractor is a PERTURBED best, scaled componentwise by F2 plus noise,
%     so how much the swarm trusts the incumbent is itself learned
%   - The DF part decays that term: a = (T-t)/T, ginv = exp(1 - 1/a^2), falling
%     from about e towards 0, so the pull to the best is switched off and the
%     search degenerates into a local difference perturbation of each parent
%   - Bound repair is bounce-back: a violating component is redrawn uniformly
%     between the violated bound and that component of x_best
%
% Reference:
% Fernando Lezama, Joao Soares, Ricardo Faia, Zita Vale,
% Hybrid-adaptive differential evolution with decay function (HyDE-DF) applied
% to the 100-digit challenge competition on single objective numerical
% optimization, Proceedings of the Genetic and Evolutionary Computation
% Conference Companion (GECCO 2019), 2019, pp. 7-8.
% https://doi.org/10.1145/3319619.3326747
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release (HyDE/HyDE.m, Main_Competition.m
% of "100-Digits-93Score"), configuration I_strategy = 3, I_strategyVersion = 2
% -- the branch that IS HyDE-DF. The 100-digit stopping rule is dropped.
% DECAY HORIZON, the one substantive deviation: the reference fixes the decay at
% an absolute 1e5 generations against a nominal 6e6, fifty times this budget,
% which would leave it a no-op, so it is tied to the run as T = ceil(maxFe/NP)
% -- the alternative the authors left commented out one line below theirs.
% The mutation scales x_best componentwise, assuming an origin-centred box; on
% CEC2020RW that works against the geometry -- a property of the algorithm.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hyde_df(problem)

    I_D   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters (Main_Competition.m of the reference release)
    I_NP = 50;
    F0   = 0.5;
    CR0  = 0.5;
    tau  = 0.1;
    Fl   = 0.1;
    Fu   = 0.9;

    T = max(1, ceil(maxFE / I_NP));   % decay horizon, in generations

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    minM = repmat(lb, I_NP, 1);
    maxM = repmat(ub, I_NP, 1);

    % Initialisation
    FM_pop = minM + rand(I_NP, I_D) .* (maxM - minM);

    [S_val, FE] = calculate_fitness(FM_pop', problem, FE);
    S_val = S_val(:);

    bsf          = inf;
    bsf_solution = FM_pop(1, :);
    for i = 1:I_NP
        if S_val(i) < bsf
            bsf          = S_val(i);
            bsf_solution = FM_pop(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, FM_pop, S_val, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    [~, ib]  = min(S_val);
    bestmem  = FM_pop(ib, :);

    % Three scaling factors and one crossover rate per individual
    F_old  = F0  * ones(I_NP, 3);
    F_w    = F_old;
    CR_old = CR0 * ones(I_NP, 1);
    CR     = CR_old;

    rot = 0:(I_NP - 1);
    gen = 1;

    % Main loop
    while FE < maxFE
        % Decay factor
        a    = max(0, (T - min(gen, T)) / T);
        ginv = exp(1 - 1 / a ^ 2);        % a = 0 gives exp(-Inf) = 0

        % jDE self-adaptation of the three F columns and of CR
        i1 = rand(I_NP, 3) < tau;
        i2 = rand(I_NP, 1) < tau;
        F_w(i1)  = Fl + Fu * rand(sum(i1(:)), 1);
        F_w(~i1) = F_old(~i1);
        CR(i2)   = rand(sum(i2), 1);
        CR(~i2)  = CR_old(~i2);

        % Shuffled donor populations (rotation scheme of the reference)
        ind = randperm(4);
        a1  = randperm(I_NP);
        a2  = a1(rem(rot + ind(1), I_NP) + 1);
        pm1 = FM_pop(a1, :);
        pm2 = FM_pop(a2, :);

        % Crossover mask; one position per row is always taken
        mui = rand(I_NP, I_D) <= CR(:, ones(1, I_D));
        mui(sub2ind([I_NP I_D], (1:I_NP)', randi(I_D, I_NP, 1))) = true;
        mpo = ~mui;

        % HyDE-DF mutation
        FM_bm = repmat(bestmem, I_NP, 1);
        FM_ui = FM_pop ...
              + F_w(:, 3 * ones(1, I_D)) .* (pm1 - pm2) ...
              + ginv * F_w(:, ones(1, I_D)) .* ...
                (FM_bm .* (F_w(:, 2 * ones(1, I_D)) + randn(I_NP, I_D)) - FM_pop);

        FM_ui = FM_pop .* mpo + FM_ui .* mui;

        % Bounce-back repair towards the current best
        low = FM_ui < minM;
        if any(low(:))
            FM_ui(low) = minM(low) + rand(sum(low(:)), 1) .* (FM_bm(low) - minM(low));
        end
        high = FM_ui > maxM;
        if any(high(:))
            FM_ui(high) = FM_bm(high) + rand(sum(high(:)), 1) .* (maxM(high) - FM_bm(high));
        end

        [S_val_temp, FE] = calculate_fitness(FM_ui', problem, FE);
        S_val_temp = S_val_temp(:);

        for i = 1:I_NP
            if S_val_temp(i) < bsf
                bsf          = S_val_temp(i);
                bsf_solution = FM_ui(i, :);
            end
            ec = FE - I_NP + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, FM_pop, S_val, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Elitist selection (ties replace, as in the reference)
        ok = (S_val_temp <= S_val);
        S_val(ok)     = S_val_temp(ok);
        FM_pop(ok, :) = FM_ui(ok, :);

        [~, ib] = min(S_val);
        bestmem = FM_pop(ib, :);

        % Successful individuals keep the parameters they were built with
        F_old(ok, :) = F_w(ok, :);
        CR_old(ok)   = CR(ok);

        gen = gen + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

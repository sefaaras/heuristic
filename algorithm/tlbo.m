% ----------------------------------------------------------------------- %
% Teaching-Learning-Based Optimization (TLBO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 50                      % Class size (population)
%   TF = randi([1 2])            % Teaching factor, drawn per learner
%
% Algorithm Concept:
%   - The selling point is that there are NO algorithm-specific parameters:
%     no crossover rate, no inertia, no temperature
%   - TEACHER PHASE: the best individual teaches and every learner moves along
%     the gap between the teacher and the class mean,
%         x_new = x + r .* (x_teacher - TF * mean)
%     with r per dimension and TF an INTEGER 1 or 2 per learner, so the move
%     either shifts the class mean onto the teacher or overshoots it
%   - LEARNER PHASE: each learner picks another at random and moves towards it
%     if fitter, away if not, which stops the class collapsing onto the teacher
%   - Both phases are greedy and in place, costing one evaluation per learner
%     each, so a generation costs 2*NP. The teacher phase batches (teacher and
%     mean are frozen); the learner phase cannot, as partners are drawn live
%
% Reference:
% R. V. Rao, V. J. Savsani, D. P. Vakharia,
% Teaching-learning-based optimization: A novel method for constrained
% mechanical design optimization problems,
% Computer-Aided Design, vol. 43, no. 3, pp. 303-315, 2011.
% https://doi.org/10.1016/j.cad.2010.12.015
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author MATLAB release could be located, so both phases are implemented
% from the paper's Eqs. (3) and (5)-(6), with Yarpiz's ypea_tlbo.m as a
% structural cross-check; it agrees with the paper on every operator.
% POPULATION SIZE: the paper reports several class sizes rather than fixing one
% and the Yarpiz template defaults to 100. NP = 50 is used, mid-range for Rao's
% studies, giving 100 evaluations per generation.
% The paper specifies no bound handling; violating components are clamped.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = tlbo(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    NP = 50;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LBm = repmat(lb, NP, 1);
    UBm = repmat(ub, NP, 1);

    % Initialisation
    X = LBm + rand(NP, D) .* (UBm - LBm);

    [f, FE] = calculate_fitness(X', problem, FE);
    f = f(:);

    bsf  = inf;
    bsfx = X(1, :);
    for i = 1:NP
        if f(i) < bsf
            bsf  = f(i);
            bsfx = X(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, X, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        % Teacher phase
        the_mean   = mean(X, 1);
        [~, ibest] = min(f);
        teacher    = X(ibest, :);

        TF   = randi([1 2], NP, 1);
        Xnew = X + rand(NP, D) .* (repmat(teacher, NP, 1) - TF(:, ones(1, D)) .* repmat(the_mean, NP, 1));
        Xnew = min(max(Xnew, LBm), UBm);

        [fn, FE] = calculate_fitness(Xnew', problem, FE);
        fn = fn(:);

        [X, f, bsf, bsfx, curve, population_history, fitness_history, history_index] = ...
            greedyAccept(X, f, Xnew, fn, bsf, bsfx, FE, NP, maxFE, curve, ...
                         population_history, fitness_history, history_index);

        if FE >= maxFE
            break;
        end

        % Learner phase, sequential as in the reference: partners already improved this pass are visible
        for i = 1:NP
            if FE >= maxFE
                break;
            end

            j = randi(NP - 1);
            if j >= i
                j = j + 1;                          % a random partner other than i
            end

            step = X(i, :) - X(j, :);
            if f(j) < f(i)                          % partner is better -> move towards it
                step = -step;
            end

            xnew = X(i, :) + rand(1, D) .* step;
            xnew = min(max(xnew, lb), ub);

            [fnew, FE] = calculate_fitness(xnew', problem, FE);
            fnew = fnew(1);

            if fnew < bsf
                bsf  = fnew;
                bsfx = xnew;
            end
            if fnew < f(i)
                X(i, :) = xnew;
                f(i)    = fnew;
            end

            if FE >= 1 && FE <= maxFE
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, X, f, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function [X, f, bsf, bsfx, curve, ph, fh, hidx] = greedyAccept( ...
        X, f, Xnew, fn, bsf, bsfx, FE, NP, maxFE, curve, ph, fh, hidx)
% Record the batch, then replace each individual by its trial when the trial is better
    for i = 1:NP
        if fn(i) < bsf
            bsf  = fn(i);
            bsfx = Xnew(i, :);
        end
        ec = FE - NP + i;
        if ec >= 1 && ec <= maxFE
            curve(ec) = bsf;
            [ph, fh, hidx] = record_history(ec, X, f, ph, fh, hidx, maxFE);
        end
    end

    better = fn < f;
    X(better, :) = Xnew(better, :);
    f(better)    = fn(better);
end

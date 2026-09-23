% ----------------------------------------------------------------------- %
% Differential Evolution with Success-based Parameter Adaptation (DEsPA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP = 4 -> 15*D -> 4         % Grows to the threshold, then shrinks linearly
%   H = 10                      % Memory slots for F and CR
%   p = 0.11                    % pbest rate
%   arc_rate = 1.4              % Archive size, a multiple of NP
%   growth stops at 0.75*maxFe  % Where the population turns from growing to shrinking
%
% Algorithm Concept:
%   - current-to-pbest/1 in which the binomial crossover is fused into the
%     mutation: a coordinate is either fully mutated or copied from the parent
%   - F comes from a Cauchy and CR from a Gaussian around a memory slot, and
%     both memories take a fitness-weighted Lehmer mean of what succeeded
%   - The population GROWS from four individuals, adding random newcomers every
%     generation until it reaches 15*D, which is the opposite of the L-SHADE
%     schedule and the point of the method
%   - After three quarters of the budget the growth stops and the population is
%     cut linearly back to four, so the run ends as pure exploitation
%   - A trial equal to its parent is accepted, which keeps the population moving
%     across plateaus
%
% Reference:
% Noor H. Awad, Mostafa Z. Ali, Robert G. Reynolds,
% A differential evolution algorithm with success-based parameter adaptation for
% CEC2015 learning-based optimization,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 1098-1105.
% https://doi.org/10.1109/CEC.2015.7257012
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' released Java (DE.java, CEC2015 submission). Two of
% its quirks are kept. The archive is written but never read: r2 is drawn from
% [0, NP + archive size) and then folded back into the population by subtracting
% NP, so a non-empty archive only biases r2 towards the low indices. And a
% successful trial is archived after it has overwritten its parent, so the
% archive holds children, not the parents they replaced. The newcomer count
% round((NP + 15*D)/maxFe * FE + 4) is also kept as written. The folded r2 is
% clamped to NP: the Java guards only the single case r2 == NP, and an archive
% larger than the shrunken population walks off the end of it.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = despa(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    NP_min = 4;
    NP_threshold = 15 * dim;
    mem_size = 10;
    p_rate = 0.11;
    arc_rate = 1.4;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    NP = NP_min;
    pop = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(ub - lb, NP, 1);
    [fitness, FE] = calculate_fitness(pop', problem, FE);
    fitness = fitness(:);

    best_fitness = inf;
    best_solution = zeros(1, dim);
    [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
        track(pop, fitness, NP, FE, best_fitness, best_solution, curve, ...
              population_history, fitness_history, history_index, maxFE);

    mem_F = 0.5 * ones(mem_size, 1);
    mem_CR = 0.5 * ones(mem_size, 1);
    mem_pos = 1;
    archive = zeros(0, dim);
    archive_size = round(arc_rate * NP);
    stop_growth = false;

    while FE < maxFE
        NP = size(pop, 1);
        [~, sorted] = sort(fitness);
        p_num = max(1, round(NP * p_rate));

        r = randi(mem_size, NP, 1);
        F = zeros(NP, 1);
        for i = 1:NP
            while F(i) <= 0
                F(i) = mem_F(r(i)) + 0.1 * tan(pi * (rand - 0.5));
            end
        end
        F = min(F, 1);
        CR = min(max(mem_CR(r) + 0.1 * randn(NP, 1), 0), 1);

        n_arch = size(archive, 1);
        U = pop;
        for i = 1:NP
            r1 = randi(NP);
            while r1 == i
                r1 = randi(NP);
            end
            % The draw reaches into the archive's index range but is folded back
            r2 = randi(NP + n_arch);
            while r2 == i || r2 == r1
                r2 = randi(NP + n_arch);
            end
            if r2 > NP
                r2 = r2 - NP;
            end
            r2 = min(r2, NP);   % the Java guard catches only r2 == NP, and the archive can be larger

            pbest = pop(sorted(randi(p_num)), :);
            jrand = randi(dim);
            mask = rand(1, dim) < CR(i);
            mask(jrand) = true;
            trial = pop(i, :) + F(i) * (pbest - pop(i, :)) + F(i) * (pop(r1, :) - pop(r2, :));
            U(i, mask) = trial(mask);

            below = U(i, :) < lb;
            above = U(i, :) > ub;
            U(i, below) = 0.5 * (lb(below) + pop(i, below));
            U(i, above) = 0.5 * (ub(above) + pop(i, above));
        end

        [fu, FE] = calculate_fitness(U', problem, FE);
        fu = fu(:);

        S_F = []; S_CR = []; S_df = [];
        for i = 1:NP
            if fu(i) == fitness(i)
                pop(i, :) = U(i, :);
                fitness(i) = fu(i);
            elseif fu(i) < fitness(i)
                S_df(end+1, 1) = abs(fitness(i) - fu(i)); %#ok<AGROW>
                S_F(end+1, 1) = F(i);                     %#ok<AGROW>
                S_CR(end+1, 1) = CR(i);                   %#ok<AGROW>
                pop(i, :) = U(i, :);
                fitness(i) = fu(i);
                if archive_size > 1
                    if size(archive, 1) < archive_size
                        archive(end+1, :) = pop(i, :); %#ok<AGROW>
                    else
                        archive(randi(archive_size), :) = pop(i, :);
                    end
                end
            end
        end

        [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
            track(U, fu, NP, FE, best_fitness, best_solution, curve, ...
                  population_history, fitness_history, history_index, maxFE);

        if ~isempty(S_df)
            w = S_df / sum(S_df);
            mem_F(mem_pos) = sum(w .* S_F .* S_F) / sum(w .* S_F);
            mem_CR(mem_pos) = sum(w .* S_CR .* S_CR) / sum(w .* S_CR);
            mem_pos = mod(mem_pos, mem_size) + 1;
        end

        if FE >= maxFE
            break;
        end

        % Growth: random newcomers join until the population reaches 15*D
        if ~stop_growth && size(pop, 1) <= NP_threshold
            extra = round(((size(pop, 1) + NP_threshold) / maxFE) * FE + NP_min);
            extra = min(extra, maxFE - FE);
            if extra > 0
                newcomers = repmat(lb, extra, 1) + rand(extra, dim) .* repmat(ub - lb, extra, 1);
                [f_new, FE] = calculate_fitness(newcomers', problem, FE);
                pop = [pop; newcomers];
                fitness = [fitness; f_new(:)];
                [best_fitness, best_solution, curve, population_history, fitness_history, history_index] = ...
                    track(newcomers, f_new(:), extra, FE, best_fitness, best_solution, curve, ...
                          population_history, fitness_history, history_index, maxFE);
            end
        end

        % Past three quarters of the budget the population shrinks back to four
        if FE >= maxFE - maxFE / 4
            stop_growth = true;
            NP_next = round(((NP_min - NP_threshold) / maxFE) * FE + NP_threshold);
            NP_next = max(NP_next, NP_min);
            if size(pop, 1) > NP_next
                [fitness, order] = sort(fitness);
                pop = pop(order, :);
                pop = pop(1:NP_next, :);
                fitness = fitness(1:NP_next);
                archive_size = round(NP_next * arc_rate);
                if size(archive, 1) > archive_size
                    archive = archive(1:archive_size, :);
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

function [bf, bx, curve, ph, fh, hi] = track(X, f, n, FE, bf, bx, curve, ph, fh, hi, maxFE)
    for i = 1:n
        if f(i) < bf
            bf = f(i);
            bx = X(i, :);
        end
        eval_count = FE - n + i;
        if eval_count >= 1 && eval_count <= maxFE
            curve(eval_count) = bf;
            [ph, fh, hi] = record_history(eval_count, X, f', ph, fh, hi, maxFE);
        end
    end
end

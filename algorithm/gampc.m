% ----------------------------------------------------------------------- %
% Genetic Algorithm with a Multi-Parent Crossover (GA-MPC)
% CEC 2011 (real-world track) -- 1st place
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PS = 90                     % Population size, a multiple of 3
%   m = PS/2                    % Archive of the best individuals
%   tc = 2 or 3                 % Tournament size, drawn per selection
%   beta = N(0.7, 0.1)          % Crossover step, one draw per parent triple
%   p = 0.1                     % Per-gene probability of the randomized operator
%   cr = 1                      % Crossover rate
%
% Algorithm Concept:
%   - Tournament selection fills a pool; each three consecutive picks, ranked
%     x1 <= x2 <= x3, give x1 + beta*(x2 - x3), x2 + beta*(x3 - x1), x3 + beta*(x1 - x2)
%   - Randomized operator: each offspring gene is replaced, with probability p,
%     by the same gene of a random archive member
%   - The next population is the best PS of the archive (the best half of the old
%     population) and the offspring together
%   - An individual identical to its neighbour in the sorted population is shifted
%     by 0.5*U + 0.25*U*N(0,1) per coordinate to keep the population diverse
%
% Reference:
% Saber M. Elsayed, Ruhul A. Sarker, Daryl L. Essam,
% GA with a new multi-parent crossover for solving IEEE-CEC2011 competition
% problems, 2011 IEEE Congress of Evolutionary Computation (CEC), 2011,
% pp. 1034-1040.
% https://doi.org/10.1109/CEC.2011.5949731
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' GA_MPC.m (GA-MPC-for Upload.rar, the code linked from
% Suganthan's CEC2011 competition page, whose CEC2011_ranking.pdf gives the
% placement); its crossover and randomized operator match SAMO_GA.m of the UMOEAs
% release. Its bound repair is chosen per problem: the rule most of its problems
% use, a uniform redraw of the violating coordinate, is applied everywhere. Its
% problem-specific branches (beta = N(0.5, 0.3) on four problems, NaN read as 0)
% are dropped. Kept as written: a size-2 tournament reuses the third entrant of
% the last size-3 one. A shifted duplicate is re-evaluated and the population
% re-sorted (the release keeps its old fitness), and the last generation
% evaluates only the offspring the budget allows.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gampc(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    PopSize   = 90;
    arch_size = PopSize / 2;
    p         = 0.1;
    beta_mu   = 0.7;
    beta_sd   = 0.1;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    n0 = min(PopSize, maxFE);
    x = repmat(lb, n0, 1) + rand(n0, dim) .* repmat(span, n0, 1);
    [fv, FE] = calculate_fitness(x', problem, FE);
    fitx = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for i = 1:n0
        if fitx(i) < bsf
            bsf          = fitx(i);
            bsf_solution = x(i, :);
        end
        curve(i) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            i, x(1:i, :), fitx(1:i), population_history, fitness_history, history_index, maxFE);
    end
    [fitx, order] = sort(fitx);
    x = x(order, :);

    % Tournament entrants; never cleared in the release, so a size-2 draw keeps a stale third
    tbuf = zeros(1, 3);
    nbuf = 0;

    while FE < maxFE
        archive     = x(1:arch_size, :);
        archive_fit = fitx(1:arch_size);

        % The population is sorted, so the smallest index wins the tournament
        best = zeros(1, PopSize * 3);
        for i = 1:PopSize * 3
            TcSize = randi([2, 3]);
            tbuf(1:TcSize) = randi(PopSize, 1, TcSize);
            nbuf = max(nbuf, TcSize);
            best(i) = min(tbuf(1:nbuf));
        end

        offspring = zeros(PopSize, dim);
        for i = 1:3:PopSize
            beta = beta_mu + beta_sd * randn;
            cons = sort(best(i:i+2));
            if cons(1) == cons(2)
                while cons(2) == cons(1) || cons(2) == cons(3)
                    cons(2) = randi(PopSize);
                end
                cons = sort(cons);
            end
            if cons(1) == cons(3)
                while cons(3) == cons(1) || cons(3) == cons(2)
                    cons(3) = randi(PopSize);
                end
                cons = sort(cons);
            end
            if cons(2) == cons(3)
                while cons(3) == cons(1) || cons(3) == cons(2)
                    cons(3) = randi(PopSize);
                end
                cons = sort(cons);
            end
            offspring(i, :)   = x(cons(1), :) + beta * (x(cons(2), :) - x(cons(3), :));
            offspring(i+1, :) = x(cons(2), :) + beta * (x(cons(3), :) - x(cons(1), :));
            offspring(i+2, :) = x(cons(3), :) + beta * (x(cons(1), :) - x(cons(2), :));
        end

        out = offspring < repmat(lb, PopSize, 1) | offspring > repmat(ub, PopSize, 1) | ~isfinite(offspring);
        redraw = repmat(lb, PopSize, 1) + rand(PopSize, dim) .* repmat(span, PopSize, 1);
        offspring(out) = redraw(out);

        % Randomized operator: the gene comes from a random archive member, one draw per gene
        swap = rand(PopSize, dim) < p;
        donor = randi(arch_size, PopSize, dim);
        genes = archive(sub2ind([arch_size, dim], donor, repmat(1:dim, PopSize, 1)));
        offspring(swap) = genes(swap);

        k = min(PopSize, maxFE - FE);
        FE0 = FE;
        [fv, FE] = calculate_fitness(offspring(1:k, :)', problem, FE);
        offspring_fit = fv(:);

        pool     = [archive; offspring(1:k, :)];
        pool_fit = [archive_fit; offspring_fit];
        [pool_fit, order] = sort(pool_fit);
        keep = min(PopSize, numel(pool_fit));
        x    = pool(order(1:keep), :);
        fitx = pool_fit(1:keep);

        for e = 1:k
            if offspring_fit(e) < bsf
                bsf          = offspring_fit(e);
                bsf_solution = offspring(e, :);
            end
            curve(FE0 + e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE0 + e, x, fitx, population_history, fitness_history, history_index, maxFE);
        end

        shifted = false;
        for i = 2:size(x, 1)
            if FE >= maxFE
                break;
            end
            if all(x(i, :) == x(i-1, :))
                g = 0.5 * rand(1, dim) + 0.25 * rand(1, dim) .* randn(1, dim);
                x(i, :) = min(ub, max(lb, x(i, :) + g));
                [fv, FE] = calculate_fitness(x(i, :)', problem, FE);
                fitx(i) = fv(1);
                if fitx(i) < bsf
                    bsf          = fitx(i);
                    bsf_solution = x(i, :);
                end
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, x, fitx, population_history, fitness_history, history_index, maxFE);
                shifted = true;
            end
        end
        if shifted
            [fitx, order] = sort(fitx);
            x = x(order, :);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

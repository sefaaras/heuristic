% ----------------------------------------------------------------------- %
% Estimation of Multivariate Normal Algorithm, global (EMNA_global)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   R = 2000                    % Population size
%   N = 1000                    % Truncation-selected individuals that fit the model
%   elite = 1                   % Best individual carried over, R - 1 sampled
%   tol_std = 1e-12             % Restart below this largest std, in box widths
%
% Algorithm Concept:
%   - No crossover and no mutation: each generation fits one full-covariance
%     Gaussian N(m, C) to the N best individuals and samples the next population
%   - m and C are the maximum-likelihood estimates (1/N) over the selected set,
%     learned from scratch every generation with no memory of earlier ones
%   - The best individual survives into the next population (elitism)
%   - C shrinks with the selected set, the premature convergence EMNA is known
%     for; the run restarts from a uniform population once C degenerates
%
% Reference:
% Pedro Larranaga,
% A Review on Estimation of Distribution Algorithms, in: Pedro Larranaga,
% Jose A. Lozano (Eds.), Estimation of Distribution Algorithms: A New Tool
% for Evolutionary Computation, Genetic Algorithms and Evolutionary
% Computation 2, Springer US, 2002, pp. 57-100.
% https://doi.org/10.1007/978-1-4615-1539-5_3
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author code exists. EMNA_global as restated by co-author E. Bengoetxea
% (PhD thesis, ENST Paris 2002, Fig. 4.13, Eq. (4.29)); the chapter itself was
% not accessible. Settings of the group's continuous-EDA experiments (same
% book, Ch. 8, as restated in that thesis and by Ortiz-Boyer et al., JAIR 24,
% 2005): uniform start, R = 2000, best 1000 selected, best one kept. The run
% restarts from a new uniform population when chol(C) fails or the largest
% coordinate std is below 1e-12 box widths, until maxFe. Samples are clamped
% to the box. PyPop7 emna.py (1/(N-1), R = 200, no elitism) was only a check.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = emna(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    w     = ub - lb;

    R       = 2000;
    Nsel    = 1000;
    tol_std = 1e-12;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        n = min(R, maxFE - FE);
        X = lb + rand(n, D) .* w;
        [fv, FE] = calculate_fitness(X', problem, FE);
        f = fv(:);
        for k = 1:n
            if isempty(bsfx) || f(k) < bsf || (isnan(bsf) && ~isnan(f(k)))
                bsf  = f(k);
                bsfx = X(k, :);
            end
            ec = FE - n + k;
            curve(ec) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                ec, X, f, population_history, fitness_history, history_index, maxFE);
        end
        if n < R
            break;
        end

        while FE < maxFE
            % Truncation selection; NaN fitness ranks last
            key = f;
            key(isnan(key)) = inf;
            [~, order] = sort(key);
            S = X(order(1:Nsel), :);

            % Maximum-likelihood mean and covariance, Eq. (4.29) of the restatement
            m  = sum(S, 1) / Nsel;
            Sc = S - m;
            C  = (Sc' * Sc) / Nsel;
            C  = (C + C') / 2;

            [Rc, p] = chol(C);
            if p > 0 || max(sqrt(diag(C))' ./ w) < tol_std
                break;
            end

            xe = X(order(1), :);
            fe = f(order(1));

            n  = min(R - 1, maxFE - FE);
            Xn = m + randn(n, D) * Rc;              % rows ~ N(m, C), Rc'*Rc = C
            bad = ~isfinite(Xn);
            if any(bad(:))
                draw = lb + rand(n, D) .* w;
                Xn(bad) = draw(bad);
            end
            Xn = min(max(Xn, lb), ub);

            [fv, FE] = calculate_fitness(Xn', problem, FE);
            fv = fv(:);

            X = [xe; Xn];
            f = [fe; fv];
            for k = 1:n
                if fv(k) < bsf || (isnan(bsf) && ~isnan(fv(k)))
                    bsf  = fv(k);
                    bsfx = Xn(k, :);
                end
                ec = FE - n + k;
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history( ...
                    ec, X, f, population_history, fitness_history, history_index, maxFE);
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

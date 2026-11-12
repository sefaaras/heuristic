% ----------------------------------------------------------------------- %
% Generalized Generation Gap model with Parent-Centric Crossover (G3-PCX)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 100                        % Population size
%   mu = 3                         % Parents per offspring: the best plus two random
%   lambda = 2                     % Offspring per generation
%   r = 2                          % Random members replaced per generation
%   sigma_eta = sigma_zeta = 0.1   % PCX spreads across and along d
%
% Algorithm Concept:
%   - PCX centres every offspring on the index parent x_p, the population best:
%     y = x_p + zeta*d + eta, with d = g - x_p and g the centroid of the mu parents
%   - zeta ~ N(0, 0.1^2) moves along d; eta ~ N(0, (0.1*Dbar)^2 I) has its d part
%     removed, Dbar the mean distance of the other parents from the line along d
%   - G3 survival: r random members and the lambda offspring form a family whose
%     r best take those members' places, so the population best is never lost
%   - Steady state: two evaluations per generation
%
% Reference:
% Kalyanmoy Deb, Ashish Anand, Dhiraj Joshi,
% A Computationally Efficient Evolutionary Algorithm for Real-Parameter
% Optimization, Evolutionary Computation 10 (4) (2002) 371-395.
% https://doi.org/10.1162/106365602760972767
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' C release g3pcx.tar (Deb's code page, last edited
% 10.07.2004); fed its exact random stream, the port reproduced it bit for bit
% up to a last-digit difference. Its Rosenbrock success (37/50) belongs to its
% Knuth generator (39/50 here driven by it, 49/50 with MATLAB's). The release
% draws eta in all D coordinates and projects out d, so eta spans the D-1
% directions perpendicular to d, not the mu-1 of Eq. (2); followed. Its
% uncounted re-evaluation of the two replaced members is skipped. It quits a run
% when a parent or the centroid lies within 1e-40 of the index parent; here a
% new uniform population starts, until maxFe. Harness: offspring are clamped to
% the box, NaN fitness ranks last, 1 - cos^2 is floored at 0 (sqrt goes complex).
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = g3pcx(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N        = 100;
    mu       = 3;
    lambda   = 2;
    r        = 2;
    sig_eta  = 0.1;
    sig_zeta = 0.1;
    tiny     = 1e-40;                       % EPSILON of the release

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        n = min(N, maxFE - FE);
        X = lb + rand(N, D) .* span;
        X = X(1:n, :);
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
        if n < N
            break;
        end

        key = f;
        key(isnan(key)) = inf;
        [~, b] = min(key);

        while FE < maxFE
            % The best is the index parent; mu-1 others by a partial Fisher-Yates draw
            arr = 1:N;
            arr([1 b]) = arr([b 1]);
            for i = 1:mu - 1
                k = min(floor(rand * (N - i)) + i + 1, N);
                arr([i + 1 k]) = arr([k i + 1]);
            end
            P  = X(arr(1:mu), :);
            xp = P(1, :);

            g = P(1, :);
            for i = 2:mu
                g = g + P(i, :);
            end
            g = g / mu;
            d = g - xp;
            dist  = sqrt(sum(d .^ 2));
            diffs = P(2:mu, :) - xp;
            nd    = sqrt(sum(diffs .^ 2, 2));
            % The release quits the run here; a fresh population is drawn instead
            if dist < tiny || any(nd < tiny)
                break;
            end

            % Perpendicular distance of each other parent from the line along d
            cs   = (diffs * d') ./ (nd * dist);
            Dp   = nd .* sqrt(max(1 - cs .^ 2, 0));
            Dbar = sum(Dp) / (mu - 1);

            nk = min(lambda, maxFE - FE);
            K  = zeros(nk, D);
            for j = 1:nk
                eta  = randn(1, D) * (Dbar * sig_eta);
                eta  = eta - ((eta * d') * d) / dist ^ 2;
                zeta = randn * sig_zeta;
                K(j, :) = xp + eta + zeta * d;
            end
            bad = ~isfinite(K);
            if any(bad(:))
                draw = lb + rand(nk, D) .* span;
                K(bad) = draw(bad);
            end
            K = min(max(K, lb), ub);

            [fk, FE] = calculate_fitness(K', problem, FE);
            fk = fk(:);

            if nk == lambda
                % Family: r distinct random members, the best included if drawn
                mom = 1:N;
                for i = 0:r - 1
                    k = min(floor(rand * (N - i)) + i + 1, N);
                    mom([i + 1 k]) = mom([k i + 1]);
                end
                mom = mom(1:r);

                SX = [K; X(mom, :)];
                SF = [fk; f(mom)];
                sk = SF;
                sk(isnan(sk)) = inf;
                % The release's selection sort, strict '<', kept for its tie order
                list = 1:numel(sk);
                for i = 1:numel(sk) - 1
                    dbest = sk(list(i));
                    for j = i + 1:numel(sk)
                        if sk(list(j)) < dbest
                            dbest = sk(list(j));
                            list([i j]) = list([j i]);
                        end
                    end
                end
                X(mom, :) = SX(list(1:r), :);
                f(mom)    = SF(list(1:r));

                key = f;
                key(isnan(key)) = inf;
                [~, b] = min(key);
            end

            for k = 1:nk
                if fk(k) < bsf || (isnan(bsf) && ~isnan(fk(k)))
                    bsf  = fk(k);
                    bsfx = K(k, :);
                end
                ec = FE - nk + k;
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

% ----------------------------------------------------------------------- %
% Ant Colony Optimization for Continuous Domains (ACOR)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   k = max(50, D)               % Solution archive size (Table 2: k = 50)
%   m = 2                        % Ants per iteration
%   q = 0.1                      % Locality of the search process
%   xi = 0.85                    % Speed of convergence
%
% Algorithm Concept:
%   - The pheromone table, which has nothing to index on continuous variables,
%     becomes a SOLUTION ARCHIVE of the k best solutions ranked best-first; per
%     dimension it defines a weighted sum of k Gaussians, and sampling that
%     kernel replaces the discrete pheromone choice
%   - Weights depend only on RANK, making ACOR invariant to monotone
%     transformations of the objective, and small q lets the best ranks dominate:
%         w_l = 1/(q*k*sqrt(2*pi)) * exp(-(l-1)^2 / (2*q^2*k^2))          Eq.(7)
%   - The spread comes from the archive's own diversity, xi acting as the
%     evaporation rate: sigma_i^l = xi * sum_e |s_i^e - s_i^l| / (k-1)    Eq.(9)
%   - An ant picks its Gaussian ONCE and samples every dimension from that same
%     archived solution, which preserves correlation between variables
%   - Pheromone update is archive maintenance: add m, drop the m worst
%
% Reference:
% Krzysztof Socha, Marco Dorigo,
% Ant colony optimization for continuous domains,
% European Journal of Operational Research, vol. 185, no. 3, pp. 1155-1173, 2008.
% https://doi.org/10.1016/j.ejor.2006.06.046
% ----------------------------------------------------------------------- %
% Implementation Note:
% Implemented from the paper itself (Eqs. 7-9 and Section 3.3), since no author
% MATLAB release exists. Parameters are the authors' own Table 2 values, except
% q: Table 2 gives 1e-4, but Section 5.2 records that the authors raised it to
% 0.1 on their MULTIMODAL set for robustness, and every suite here is multimodal.
% Footnote 4 forbids k smaller than the dimension, so k = max(50, D) replaces
% the flat 50 on CEC2014/2017 at D = 100 and CEC2020RW at D = 158.
% Yarpiz's ypea_acor.m draws the archive index inside the per-dimension loop;
% that is not followed, as the paper makes the choice once per ant to exploit
% the correlation between variables. Sampled points are clamped to the box.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = acor(problem)

    n     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters (Table 2, with the Section 5.2 multimodal q)
    k  = max(50, n);
    m  = 2;
    q  = 0.1;
    xi = 0.85;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Rank-based weights, Eqs. (7) and (8)
    l  = (1:k)';
    w  = 1 / (q * k * sqrt(2 * pi)) * exp(-((l - 1) .^ 2) / (2 * q ^ 2 * k ^ 2));
    p  = w / sum(w);
    pc = cumsum(p);

    % Initial archive: k uniform random solutions
    S = repmat(lb, k, 1) + rand(k, n) .* repmat(ub - lb, k, 1);

    [f, FE] = calculate_fitness(S', problem, FE);
    f = f(:);

    [f, ord] = sort(f, 'ascend');
    S = S(ord, :);

    bsf  = f(1);
    bsfx = S(1, :);
    for i = 1:k
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, S, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        na = min(m, maxFE - FE);
        X  = zeros(na, n);

        for a = 1:na
            % One Gaussian per ant, chosen by roulette on the rank weights
            sel = find(pc >= rand, 1, 'first');
            if isempty(sel)
                sel = k;
            end

            % Eq. (9): mean absolute distance from the chosen solution
            sigma = xi * sum(abs(S - S(sel, :)), 1) / (k - 1);

            X(a, :) = S(sel, :) + sigma .* randn(1, n);
        end

        X = min(max(X, repmat(lb, na, 1)), repmat(ub, na, 1));

        [fx, FE] = calculate_fitness(X', problem, FE);
        fx = fx(:);

        for a = 1:na
            if fx(a) < bsf
                bsf  = fx(a);
                bsfx = X(a, :);
            end
            ec = FE - na + a;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, S, f, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % Pheromone update: add the new solutions, drop the worst
        Sall = [S; X];
        fall = [f; fx];
        [fall, ord] = sort(fall, 'ascend');
        S = Sall(ord(1:k), :);
        f = fall(1:k);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

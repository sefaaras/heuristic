% ----------------------------------------------------------------------- %
% Grasshopper Optimisation Algorithm (GOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 100                      % Grasshoppers
%   cMax = 1, cMin = 0.00004     % Comfort-zone coefficient, linear decrease
%   f = 0.5, l = 1.5             % Social force s(r) = f*exp(-r/l) - exp(-r)
%
% Algorithm Concept:
%   - Every individual feels a SOCIAL FORCE from every other:
%     s(r) = f*e^(-r/l) - e^(-r), repulsive at short range, attractive at
%     medium range and decaying to nothing far away, with a "comfort zone"
%     between where the net force is zero
%   - Pairwise distances are mapped into [2,3] by 2 + rem(d,2) first, keeping
%     every pair inside the informative part of that curve
%   - The new position is NOT an increment on the old one:
%         x_i <- c * sum_j [ c*(ub-lb)/2 * s(|x_j-x_i|) * (x_j-x_i)/d_ij ] + T
%     with T the best so far, so each grasshopper is rebuilt every generation
%     as the target plus a swarm-shaped perturbation
%   - c appears TWICE and falls linearly 1 -> 4e-5, shrinking both the zone
%     between grasshoppers and the displacement around the target
%
% Reference:
% Shahrzad Saremi, Seyedali Mirjalili, Andrew Lewis,
% Grasshopper Optimisation Algorithm: Theory and application,
% Advances in Engineering Software, vol. 105, pp. 30-47, 2017.
% https://doi.org/10.1016/j.advengsoft.2017.01.004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the author's own MATLAB release ("source codes demo V1.0").
% ONE VESTIGIAL BLOCK DROPPED: the release pads an odd dimension with a
% [-100,100] variable "because this algorithm should be run with an even number
% of variables", but that requirement came from a `for k = 1:2:dim` inner loop
% which is COMMENTED OUT in the released code. The force is computed over all
% dimensions at once, so the padded variable is optimised and then discarded,
% and its hardcoded bound is wrong for any other box.
% Max_iteration is derived as floor(maxFe/N) so the run spends exactly the
% budget.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = goa(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    N        = 100;
    Max_iter = max(2, floor(maxFE / N));
    cMax     = 1;
    cMin     = 0.00004;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LBm = repmat(lb, N, 1);
    UBm = repmat(ub, N, 1);
    span = ub - lb;

    % Initialisation
    X = LBm + rand(N, dim) .* (UBm - LBm);

    [Fitness, FE] = calculate_fitness(X', problem, FE);
    Fitness = Fitness(:)';

    [TargetFitness, ib] = min(Fitness);
    TargetPosition = X(ib, :);

    bsf  = TargetFitness;
    bsfx = TargetPosition;
    for i = 1:N
        if i <= maxFE
            curve(i) = min(Fitness(1:i));
            [population_history, fitness_history, history_index] = record_history(...
                i, X, Fitness', population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    l = 2;
    while FE < maxFE && l <= Max_iter
        c = cMax - l * ((cMax - cMin) / Max_iter);       % Eq. (2.8)

        Xnew = zeros(N, dim);
        for i = 1:N
            diff = X - X(i, :);                          % xj - xi
            d    = sqrt(sum(diff .^ 2, 2));              % Euclidean, N x 1
            g    = S_func(2 + rem(d, 2)) ./ (d + eps);   % Eq. (2.7) inner part
            g(i) = 0;                                    % skip j == i

            S_i  = (span * c / 2) .* sum(g .* diff, 1);
            Xnew(i, :) = c * S_i + TargetPosition;       % Eq. (2.7)
        end

        X = min(max(Xnew, LBm), UBm);

        [Fitness, FE] = calculate_fitness(X', problem, FE);
        Fitness = Fitness(:)';

        for i = 1:N
            if Fitness(i) < TargetFitness
                TargetFitness  = Fitness(i);
                TargetPosition = X(i, :);
            end
            if Fitness(i) < bsf
                bsf  = Fitness(i);
                bsfx = X(i, :);
            end
            ec = FE - N + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, X, Fitness', population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        l = l + 1;
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% Helper Functions

function o = S_func(r)
% Social force, Eq. (2.3): repulsion at short range, attraction at medium.
    f = 0.5;
    l = 1.5;
    o = f * exp(-r / l) - exp(-r);
end

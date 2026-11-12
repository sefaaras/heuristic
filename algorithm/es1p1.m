% ----------------------------------------------------------------------- %
% (1+1) Evolution Strategy with One-Fifth Success Rule ((1+1)-ES)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   sigma0    = 0.2                        % Initial step, in box widths (2 on [-5,5])
%   inc = 1.5, dec = 1.5^(-1/4)            % Step factors after a success / failure
%   TolSigma  = 1e-16                      % Restart below this step, in box widths
%   MaxNoImp  = 16*ln(10)/ln(1.5) = 90.9   % Restart after this many non-improving steps
%
% Algorithm Concept:
%   - One parent x samples one offspring x + sigma*N(0, I), the step scaled to
%     each coordinate's box width
%   - The offspring replaces the parent when it is no worse, f(y) <= f(x)
%   - One-fifth rule: sigma times 1.5 on a success, times 1.5^(-1/4) otherwise,
%     so sigma is stationary at a success rate of 1/5
%   - Independent restarts from a uniform point in the central 80 % of the box,
%     once sigma or the run of non-improving steps reaches its limit
%
% Reference:
% Anne Auger,
% Benchmarking the (1+1) evolution strategy with one-fifth success rule on the
% BBOB-2009 function testbed, Proceedings of the 11th Annual Conference
% Companion on Genetic and Evolutionary Computation (GECCO 2009), 2009,
% pp. 2447-2452.
% https://doi.org/10.1145/1570256.1570342
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (no released code): Table 1, the variant it benchmarks, with
% its independent restarts. Its BBOB box [-5,5] is mapped onto each coordinate
% of [lb,ub]: x0 ~ U[-4,4] becomes the central 80 % of the box, sigma0 = 2 is
% 0.2 box widths and TolSigma = 1e-15 is 1e-16 box widths; the step is
% sigma*(ub-lb).*N(0, I), so the search is isotropic in box units. MaxNoImp is
% printed 4*4*ln(10)/ln(1.5): four sigma decades of failures. Harness: the
% offspring is clamped to the box, which the unbounded original never needs;
% restarts run until maxFe, as in the paper's 10^6*D runs.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = es1p1(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    w     = ub - lb;

    sigma0    = 0.2;
    inc       = 1.5;
    dec       = 1.5 ^ (-1 / 4);
    tol_sigma = 1e-16;
    max_noimp = 4 * 4 * log(10) / log(1.5);

    FE    = 0;
    curve = zeros(1, maxFE);

    % The (1+1)-ES population is its parent, recorded once per evaluation
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    bsf  = inf;
    bsfx = [];

    while FE < maxFE
        % Independent restart, Section 2.2: x0 ~ U[-4,4]^D on the box [-5,5]^D
        x = lb + w .* (0.1 + 0.8 * rand(1, D));
        [fx, FE] = calculate_fitness(x', problem, FE);
        fx = fx(1);
        if isempty(bsfx) || fx < bsf || (isnan(bsf) && ~isnan(fx))
            bsf  = fx;
            bsfx = x;
        end
        curve(FE) = bsf;
        [population_history, fitness_history, history_index] = record_history( ...
            FE, x, fx, population_history, fitness_history, history_index, maxFE);

        sigma = sigma0;
        noimp = 0;
        while FE < maxFE && sigma > tol_sigma && noimp < max_noimp
            y = min(max(x + sigma * w .* randn(1, D), lb), ub);
            [fy, FE] = calculate_fitness(y', problem, FE);
            fy = fy(1);

            % An improvement is a strictly lower value; a success also accepts ties (Table 1, line 4)
            if fy < fx
                noimp = 0;
            else
                noimp = noimp + 1;
            end
            if fy <= fx
                x     = y;
                fx    = fy;
                sigma = sigma * inc;
            else
                sigma = sigma * dec;
            end

            if fy < bsf || (isnan(bsf) && ~isnan(fy))
                bsf  = fy;
                bsfx = y;
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history( ...
                FE, x, fx, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

% ----------------------------------------------------------------------- %
% Bare Bones Particle Swarm Optimization (BBPSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 20                      % Swarm size, laid out as a 4 x 5 torus
%   topology = von Neumann      % Up, down, left and right neighbours, self excluded
%   IP = 0.5                    % Interaction probability, tested per dimension
%
% Algorithm Concept:
%   - No velocity: each coordinate is drawn from N((p_id + g_d)/2, |p_id - g_d|) with
%     probability IP and otherwise kept at the personal best p_id
%   - g is the personal best of one lattice neighbour picked at random for every
%     sample, not the best neighbour
%   - The spread is set by consensus: it shrinks as p_i and g agree, with no lag
%   - Particles move one at a time and a personal best is replaced on strict
%     improvement, so the next particle in the sweep already sees it
%
% Reference:
% James Kennedy,
% Bare bones particle swarms,
% Proceedings of the 2003 IEEE Swarm Intelligence Symposium (SIS'03) 80-87.
% https://doi.org/10.1109/SIS.2003.1202251
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Secs. I-VI, Table 1, Appendix); its SAS macros are not published. Table 1
% ranks the Gaussian version with IP = 0.50 and g = a random neighbour first and Sec. VI calls
% it perhaps the most reliable, so that one is ported, on the 4 x 5 square every run used.
% Sec. VI prints the mean as (p_id - p_gd)/2, a typo: Sec. III defines (p_id + p_gd)/2, and a
% minus pulls the swarm to the origin (Sphere 1.9e-34, Rastrigin 0.006 here). The paper leaves
% open whether the random neighbour may be the particle itself (excluded: it would resample
% p_i), the update order (Kennedy's per-particle loop, as in fips) and the box: x0 ~ U(lb, ub)
% replaces the asymmetric test initialisation, and samples are clamped to the box.
% Appendix (3000 iterations, 50 runs; D = 30, f6 D = 2) against 30 seeded runs, paper / here:
%      Sphere 1.32e-13 / 1.4e-14   Griewank 1.66e-9 / 1.4e-9   Rosenbrock 69.4 / 51.3
%      f6 1.05e-3 / 1.6e-5   Rastrigin 38.5 / 32.0
%      with the paper's asymmetric init and no clamp: Rosenbrock 100.7, Rastrigin 43.4
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bbpso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    rows = 4;
    cols = 5;
    N    = rows * cols;
    ip   = 0.5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(N, D) .* span;

    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(x(1:m, :)', problem, FE);
    f = inf(N, 1);
    f(1:m) = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for k = 1:m
        if f(k) < bsf || (isnan(bsf) && ~isnan(f(k)))
            bsf          = f(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:m, :), f(1:m), population_history, fitness_history, history_index, maxFE);
    end

    p  = x;
    fp = f;

    % von Neumann square: up, down, left and right on the torus, self excluded
    nb = zeros(N, 4);
    for i = 1:N
        r = floor((i - 1) / cols);
        c = mod(i - 1, cols);
        nb(i, :) = [mod(r - 1, rows) * cols + c + 1, mod(r + 1, rows) * cols + c + 1, ...
                    r * cols + mod(c - 1, cols) + 1, r * cols + mod(c + 1, cols) + 1];
    end

    while FE < maxFE
        for i = 1:N
            % Sec. II: a random neighbour's previous best, drawn anew for every sample
            g = p(nb(i, randi(4)), :);
            % Sec. VI: a coordinate takes the Gaussian with probability IP, else keeps p_id
            infl = rand(1, D) < ip;
            xi = p(i, :);
            xi(infl) = (p(i, infl) + g(infl)) / 2 + abs(p(i, infl) - g(infl)) .* randn(1, nnz(infl));
            xi = min(max(xi, lb), ub);

            [fi, FE] = calculate_fitness(xi', problem, FE);
            x(i, :) = xi;
            f(i)    = fi;
            % A NaN personal best could never be beaten, so it is overwritten
            if fi < fp(i) || isnan(fp(i))
                fp(i)   = fi;
                p(i, :) = xi;
            end
            if fi < bsf || (isnan(bsf) && ~isnan(fi))
                bsf          = fi;
                bsf_solution = xi;
            end
            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, x, f, population_history, fitness_history, history_index, maxFE);
            if FE >= maxFE
                break;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

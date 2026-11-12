% ----------------------------------------------------------------------- %
% Big Bang-Big Crunch (BB-BC)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50                      % Population, the elite included
%   smoothf = 10                % Iterations after which the spread is halved
%   spread = 0.5*(ub - lb)      % Per-dimension scale of the Big Bang at k = 0
%
% Algorithm Concept:
%   - Big Crunch: the population collapses to one representative point, the
%     fittest individual (the centre option active in the authors' code)
%   - Big Bang: N - 1 new candidates are drawn around it per dimension,
%     x = x_c + 0.5*(ub - lb)*randn/(1 + k/smoothf), then clipped to the box
%   - The spread decays with the iteration count k only; it has no horizon
%   - Elitism: the centre is carried into the next population as its last member
%
% Reference:
% Osman K. Erol, Ibrahim Eksin,
% A new optimization method: Big Bang-Big Crunch,
% Advances in Engineering Software 37(2) (2006) 106-111.
% https://doi.org/10.1016/j.advengsoft.2005.04.005
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' own MATLAB release bbbcfv1.m (REV 1.6, Erol and Genc
% 2008, web.itu.edu.tr/okerol/BBBC.html); the paywalled 2006 paper was not
% available. Its 1/f-weighted centre of mass is commented out there and the
% fittest individual is the active centre, so no 1/f weighting (undefined for
% f <= 0) occurs; the authors restate x_c + R*r/(1 + k/sm) in Turk J Elec Eng Comp
% Sci 21 (2013) 1359. Their driver is not released: the whole population, elite
% included, is evaluated (N FE per iteration; keeping the elite's cost instead
% falls 4-8 s.e. below their [-10,10] Ackley, Rastrigin, Rosenbrock means), k = 1
% is the first Big Bang, N = 50 is our choice (theirs: 20, 30). D = 2 means, their
% 2013 Tables 2-3 (1e4 runs) / here (4e3 runs):
%      [-10,10], N 20, 1000 FE: Ackley 0.65/0.64  Rastrigin 1.35/1.36  Rosenbrock 0.40/0.42
%                               Sphere (500 FE) 0.08/0.078  Griewank 0.0143/0.0141
%      [-50,50], N 30, 3000 FE: Ackley 1.14/1.17  Rastrigin 1.96/2.00  Rosenbrock 3.59/3.53
%                               Sphere (1500 FE) 0.23/0.24  Griewank 0.0097/0.0099
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = bbbc(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N       = 50;
    smoothf = 10;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    pop = lb + rand(N, D) .* span;
    m = min(N, maxFE);
    [fv, FE] = calculate_fitness(pop(1:m, :)', problem, FE);
    pop  = pop(1:m, :);
    cost = fv(:);

    bsf          = inf;
    bsf_solution = pop(1, :);
    for i = 1:m
        % A NaN incumbent could never be beaten, so it is overwritten
        if cost(i) < bsf || (isnan(bsf) && ~isnan(cost(i)))
            bsf          = cost(i);
            bsf_solution = pop(i, :);
        end
        curve(i) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            i, pop, cost, population_history, fitness_history, history_index, maxFE);
    end

    k = 0;
    while FE < maxFE
        k = k + 1;

        % Big Crunch: min skips NaN, so the centre is the fittest measured point
        [~, ib] = min(cost);
        centre = pop(ib, :);

        % Big Bang around the centre, one normal draw per coordinate
        newpop = centre + 0.5 * span .* randn(N - 1, D) / (1 + k / smoothf);
        newpop = min(max(newpop, lb), ub);

        % The elite closes the population and is evaluated again with it
        nb  = min(N, maxFE - FE);
        pop = [newpop; centre];
        pop = pop(1:nb, :);
        [fv, FE] = calculate_fitness(pop', problem, FE);
        cost = fv(:);

        for i = 1:nb
            if cost(i) < bsf || (isnan(bsf) && ~isnan(cost(i)))
                bsf          = cost(i);
                bsf_solution = pop(i, :);
            end
            fe_i = FE - nb + i;
            curve(fe_i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                fe_i, pop, cost, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

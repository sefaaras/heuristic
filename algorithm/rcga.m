% ----------------------------------------------------------------------- %
% Real-Coded Genetic Algorithm (RCGA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   Np   = 100                   % Population size (even, pairs are crossed)
%   Pc   = 0.9                   % Crossover probability, per pair
%   Pm   = 1/D                   % Mutation probability, PER VARIABLE
%   etac = 20                    % Distribution index of SBX
%   etam = 20                    % Distribution index of polynomial mutation
%
% Algorithm Concept:
%   - The real-variable GA in the form that became the standard baseline through
%     NSGA-II: binary tournament, SBX, polynomial mutation, elitist survival
%   - SIMULATED BINARY CROSSOVER spreads two children symmetrically about their
%     parents with a spread factor
%         beta = (2u)^(1/(etac+1))            if u <= 0.5
%         beta = (1/(2(1-u)))^(1/(etac+1))    otherwise
%     preserving the parents' mean; large etac keeps children close
%   - POLYNOMIAL MUTATION perturbs a variable by (ub-lb)*delta with delta from a
%     polynomial distribution around zero, so unlike Gaussian mutation the
%     perturbation is bounded by the variable's own range
%   - Survival is (mu + lambda): parents and offspring compete together, so the
%     best solution can never be lost
%
% Reference:
% Kalyanmoy Deb, Ram Bhushan Agrawal,
% Simulated Binary Crossover for Continuous Search Space,
% Complex Systems, vol. 9, no. 2, pp. 115-148, 1995.
% https://content.wolfram.com/sites/13/2018/02/09-2-2.pdf
% Kalyanmoy Deb, Mayank Goyal,
% A Combined Genetic Adaptive Search (GeneAS) for Engineering Design,
% Computer Science and Informatics, vol. 26, no. 4, pp. 30-45, 1996.
% ----------------------------------------------------------------------- %
% Implementation Note:
% Named rcga rather than ga, which is a Global Optimization Toolbox function
% that a file called ga.m in algorithm/ would shadow for the whole session.
% There is no single author release to port -- the real-coded GA is a
% combination of operators, each specified in its own paper -- so both are
% implemented from the defining equations with the defaults of Deb's own
% NSGA-II distribution. A teaching implementation supplied the structure but
% two of its choices are not copied: its SBX repair applies only the lower bound
% to the first child and only the upper to the second, letting children leave
% the box; and it gates mutation per individual rather than per variable.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = rcga(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    % Control parameters
    Np   = 100;
    Pc   = 0.9;
    Pm   = 1 / D;
    etac = 20;
    etam = 20;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LBm = repmat(lb, Np, 1);
    UBm = repmat(ub, Np, 1);

    % Initialisation
    P = LBm + rand(Np, D) .* (UBm - LBm);

    [f, FE] = calculate_fitness(P', problem, FE);
    f = f(:);

    bsf  = inf;
    bsfx = P(1, :);
    for i = 1:Np
        if f(i) < bsf
            bsf  = f(i);
            bsfx = P(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, P, f, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Main loop
    while FE < maxFE
        % Binary tournament selection
        a = randi(Np, Np, 1);
        b = randi(Np, Np, 1);
        pick = a;
        worse = f(b) < f(a);
        pick(worse) = b(worse);
        Parent = P(pick, :);

        % Simulated binary crossover
        Parent = Parent(randperm(Np), :);
        offspring = Parent;
        for i = 1:2:Np
            if rand >= Pc
                continue;
            end
            u    = rand(1, D);
            beta = zeros(1, D);
            lo   = u <= 0.5;
            beta(lo)  = (2 * u(lo)) .^ (1 / (etac + 1));
            beta(~lo) = (1 ./ (2 * (1 - u(~lo)))) .^ (1 / (etac + 1));

            p1 = Parent(i, :);
            p2 = Parent(i + 1, :);
            offspring(i, :)     = 0.5 * ((1 + beta) .* p1 + (1 - beta) .* p2);
            offspring(i + 1, :) = 0.5 * ((1 - beta) .* p1 + (1 + beta) .* p2);
        end

        % Polynomial mutation, per variable
        hit = rand(Np, D) < Pm;
        if any(hit(:))
            u     = rand(Np, D);
            delta = zeros(Np, D);
            lo    = u < 0.5;
            delta(lo)  = (2 * u(lo)) .^ (1 / (etam + 1)) - 1;
            delta(~lo) = 1 - (2 * (1 - u(~lo))) .^ (1 / (etam + 1));
            offspring(hit) = offspring(hit) + (UBm(hit) - LBm(hit)) .* delta(hit);
        end

        offspring = min(max(offspring, LBm), UBm);

        [fo, FE] = calculate_fitness(offspring', problem, FE);
        fo = fo(:);

        for i = 1:Np
            if fo(i) < bsf
                bsf  = fo(i);
                bsfx = offspring(i, :);
            end
            ec = FE - Np + i;
            if ec >= 1 && ec <= maxFE
                curve(ec) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    ec, P, f, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        % (mu + lambda) survival
        Combined = [P; offspring];
        [fall, ind] = sort([f; fo], 'ascend');
        f = fall(1:Np);
        P = Combined(ind(1:Np), :);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsfx;
end

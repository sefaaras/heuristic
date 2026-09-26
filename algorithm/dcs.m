% ----------------------------------------------------------------------- %
% Differentiated Creative Search (DCS)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NP      = 30                % Population size
%   ngS     = max(6, round(NP*phi/3))  % High performers, phi = 2/(1+sqrt(5))
%   pc      = 0.5               % Chance the worst member is re-created at random
%   phi_qKR = 0.25 + 0.55*(rank/NP)^0.5  % Knowledge-acquisition rate by rank
%   lambda  = 0.1 + 0.518*(1 - t^0.5)    % Social impact, t the spent budget ratio
%   alpha   = phi               % Linnik flight index
%   sigma   = 0.05              % Linnik flight scale
%
% Algorithm Concept:
%   - The population is ranked every iteration; a member rewrites each coordinate
%     with a rank-driven rate eta in {0, 0.5, 1}, one coordinate always
%   - High performers (the best ngS) take a random peer's coordinates, each
%     perturbed by a heavy-tailed Linnik flight
%   - The others step from the best along lambda*(x_r2 - x) + omega*(x_r1 - x),
%     x_r2 drawn from the non-elite ranks and omega uniform per member
%   - With probability pc the worst member is replaced by a random point on the
%     box diagonal (one scalar draw scales every coordinate)
%   - Greedy one-to-one replacement; violating coordinates go midway to the bound
%
% Reference:
% Poomin Duankhan, Khamron Sunat, Sirapat Chiewchanwattana, Patchara Nasa-ngium,
% The Differentiated Creative Search (DCS): Leveraging differentiated
% knowledge-acquisition and creative realism to address complex optimization
% problems,
% Expert Systems with Applications 252 (2024) 123734.
% https://doi.org/10.1016/j.eswa.2024.123734
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (github.com/minikku/
% Differentiated-Creative-Search, DCS.m, with NP = 30 from its Run.m). Its
% counter starts after the initial population and is tested once per iteration,
% so a run spent up to max_nfe + 2*NP - 1; here every evaluation counts against
% maxFe and is guarded, and lambda runs on FE/maxFe. The worst member's random
% re-creation draws one scalar for all coordinates, so it lands on the box
% diagonal; kept as released. The Linnik ratio R is non-negative in exact
% arithmetic but can round below zero, which would make R^(1/alpha) complex, so
% it is floored at 0.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = dcs(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    lu    = [lb; ub];

    NP           = 30;
    golden_ratio = 2 / (1 + sqrt(5));
    ngS          = max(6, round(NP * (golden_ratio / 3)));
    pc           = 0.5;
    sigma        = 0.05;
    phi_qKR      = 0.25 + 0.55 * (((1:NP) / NP) .^ 0.5);

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    pos = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(ub - lb, NP, 1);
    [fitness, FE] = calculate_fitness(pos', problem, FE);
    fitness = fitness(:);

    bsf          = inf;
    bsf_solution = pos(1, :);
    for i = 1:NP
        if fitness(i) < bsf
            bsf          = fitness(i);
            bsf_solution = pos(i, :);
        end
        if i <= maxFE
            curve(i) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                i, pos, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    fbest = min(fitness);        % the search's own best, which moves bestInd

    while FE < maxFE
        [fitness, order] = sort(fitness, 1, 'ascend');
        pos = pos(order, :);
        bestInd = 1;

        lamda_t = 0.1 + 0.518 * (1 - (FE / maxFE) ^ 0.5);

        for i = 1:NP
            if FE >= maxFE
                break;
            end

            % Differentiated knowledge-acquisition rate, 0, 0.5 or 1
            eta = (round(rand * phi_qKR(i)) + (rand <= phi_qKR(i))) / 2;
            jrand = floor(dim * rand + 1);
            next = pos(i, :);

            if i == NP && rand < pc
                next = lb + rand * (ub - lb);
            elseif i <= ngS
                while true
                    r1 = round(NP * rand + 0.5);
                    if r1 ~= i && r1 ~= bestInd, break; end
                end
                for d = 1:dim
                    if rand <= eta || d == jrand
                        next(d) = pos(r1, d) + linnik(golden_ratio, sigma);
                    end
                end
            else
                while true
                    r1 = round(NP * rand + 0.5);
                    if r1 ~= i && r1 ~= bestInd, break; end
                end
                while true
                    r2 = ngS + round((NP - ngS) * rand + 0.5);
                    if r2 ~= i && r2 ~= bestInd && r2 ~= r1, break; end
                end
                omega_it = rand;   % learning ability
                for d = 1:dim
                    if rand <= eta || d == jrand
                        next(d) = pos(bestInd, d) + (pos(r2, d) - pos(i, d)) * lamda_t ...
                                  + (pos(r1, d) - pos(i, d)) * omega_it;
                    end
                end
            end

            next = boundConstraint(next, pos(i, :), lu);

            [fv, FE] = calculate_fitness(next', problem, FE);
            fnew = fv(1);

            if fnew < bsf
                bsf          = fnew;
                bsf_solution = next;
            end
            if fnew <= fitness(i)
                pos(i, :)  = next;
                fitness(i) = fnew;
                if fnew < fbest
                    fbest   = fnew;
                    bestInd = i;
                end
            end

            curve(FE) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE, pos, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function Y = linnik(alpha, sigma)
% Linnik flight of Nasa-ngium et al. (IEEE Access 7, 2019), one scalar draw
    u1 = rand;
    u2 = rand;
    Z = log(u1 / u2);
    Z = sign(rand - 0.5) * Z;
    U = rand;
    R = sin(0.5 * pi * alpha) * tan(0.5 * pi * (1 - alpha * U)) - cos(0.5 * pi * alpha);
    Y = sigma * Z * max(R, 0) ^ (1 / alpha);
end

function vi = boundConstraint(vi, pop, lu)
% Violating component moved to the parent/bound midpoint (JADE's rule)
    NP = size(pop, 1);

    xl  = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    xu  = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

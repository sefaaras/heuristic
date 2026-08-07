% ----------------------------------------------------------------------- %
% Light Spectrum Optimizer (LSO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NoF_LightRays = 25   % Population size (light rays)
%   Ps=0.05  Pe=0.6  Ph=0.4  B=0.05
%   Refractive index range red=1.3318 .. violet=1.3435
%
% Algorithm Concept:
%   - Colorful dispersion via Snell's-law refraction/reflection (L1,L2,L3)
%   - Two scattering stages balance exploration/exploitation
%
% Reference:
% Mohamed Abdel-Basset, Reda Mohamed, Karam M. Sallam, Ripon K. Chakrabortty,
% Light Spectrum Optimizer: A Novel Physics-Inspired Metaheuristic
% Optimization Algorithm,
% Mathematics 2022, 10(19), 3466.
% https://doi.org/10.3390/math10193466
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = lso(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    red = 1.3318; violet = 1.3435;
    NoF_LightRays = 25;
    Ps = 0.05; Pe = 0.6; Ph = 0.4; B = 0.05;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    LightRays = zeros(NoF_LightRays, dim);
    for i = 1:NoF_LightRays
        LightRays(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
    [fitness, FE] = calculate_fitness(LightRays', problem, FE);
    fitness = fitness(:)';

    [GBest_fitness, index] = min(fitness);
    GBestRayColor = LightRays(index, :);
    NewWLightRays = LightRays;

    for e = 1:NoF_LightRays
        if e <= maxFE
            curve(e) = GBest_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                e, LightRays, fitness, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    while FE < maxFE
        for i = 1:NoF_LightRays
            nA = LightRays(randi(NoF_LightRays), :);
            nB = LightRays(i, :);
            nC = GBestRayColor;
            xbar = (sum(LightRays) / NoF_LightRays);
            norm_nA = nA / (norm(nA) + eps);
            norm_nB = nB / (norm(nB) + eps);
            norm_nC = nC / (norm(nC) + eps);
            Incid_norm = xbar / (norm(xbar) + eps);

            k = red + rand .* (violet - red);
            p = rand; q = rand;
            L1 = (1 ./ k) .* (Incid_norm - (norm_nA .* dot(norm_nA, Incid_norm))) - (norm_nA .* (abs((1 - (1 ./ k.^2) + ((1 ./ k.^2) .* dot(norm_nA, Incid_norm).^2)))).^(1 / 2));
            L2 = L1 - ((2 .* norm_nB) .* dot(L1, norm_nB));
            L3 = k .* (L2 - (norm_nC .* dot(norm_nC, L2))) + norm_nC .* (abs(1 - (k.^2) + (k.^2) .* ((dot(norm_nC, L2)).^2))).^(1 / 2);

            a = rand * (1 - FE / maxFE);
            ginv = gammaincinv(a, 1);
            GI = a * (1 / rand) * ginv;
            Epsln = a .* randn(1, dim);

            if p <= q
                NewWLightRays(i, :) = LightRays(i, :) + GI .* Epsln .* rand(1, dim) .* (L1 - L3) .* (LightRays(randi(NoF_LightRays), :) - LightRays(randi(NoF_LightRays), :));
            else
                NewWLightRays(i, :) = (LightRays(i, :)) + GI .* Epsln .* rand(1, dim) .* (L2 - L3) .* (LightRays(randi(NoF_LightRays), :) - LightRays(randi(NoF_LightRays), :));
            end
            NewWLightRays(i, :) = check_bounds(NewWLightRays(i, :), ub, lb, Ph);

            [Fnew, FE] = calculate_fitness(NewWLightRays(i, :)', problem, FE);
            if Fnew <= fitness(i)
                LightRays(i, :) = NewWLightRays(i, :);
                fitness(i) = Fnew;
            end
            if Fnew <= GBest_fitness
                GBestRayColor = NewWLightRays(i, :);
                GBest_fitness = Fnew;
            end
            if FE <= maxFE
                curve(FE) = GBest_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, LightRays, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end

            % Scattering stages
            in1 = sort(fitness);
            F = abs((fitness(i) - GBest_fitness) / (GBest_fitness - (in1(NoF_LightRays)) + eps));
            if F < rand || rand < Ps
                if rand < Pe
                    NewWLightRays(i, :) = (LightRays(i, :)) + (rand) * (LightRays(randi(NoF_LightRays), :) - LightRays(randi(NoF_LightRays), :)) + (rand < B) .* rand(1, dim) .* ((GBestRayColor - LightRays(i, :)));
                else
                    NewWLightRays(i, :) = (((2 * cos(rand * 180)) .* (GBestRayColor .* LightRays(i, :))));
                end
            else
                U = (rand(1, dim) > rand(1, dim));
                NewWLightRays(i, :) = U .* (LightRays(randi(NoF_LightRays), :) + abs(randn) .* (LightRays(randi(NoF_LightRays), :) - LightRays(randi(NoF_LightRays), :))) + (1 - U) .* LightRays(i, :);
            end
            NewWLightRays(i, :) = check_bounds(NewWLightRays(i, :), ub, lb, Ph);

            [Fnew, FE] = calculate_fitness(NewWLightRays(i, :)', problem, FE);
            if Fnew <= fitness(i)
                LightRays(i, :) = NewWLightRays(i, :);
                fitness(i) = Fnew;
            end
            if Fnew <= GBest_fitness
                GBestRayColor = NewWLightRays(i, :);
                GBest_fitness = Fnew;
            end
            if FE <= maxFE
                curve(FE) = GBest_fitness;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, LightRays, fitness, population_history, fitness_history, ...
                    history_index, maxFE);
            end
            if FE >= maxFE, break; end
        end
    end

    curve(min(FE, maxFE):end) = GBest_fitness;
    best_fitness = GBest_fitness;
    best_solution = GBestRayColor;
end

% Bound handling: clamp with prob Ph, else random relocation
function x = check_bounds(x, ub, lb, Ph)
    if rand < Ph
        U = x > ub; L = x < lb;
        x = (x .* (~(U + L))) + ub .* U + lb .* L;
    else
        for j = 1:numel(x)
            if x(j) > ub(j) || x(j) < lb(j)
                x(j) = lb(j) + rand * (ub(j) - lb(j));
            end
        end
    end
end

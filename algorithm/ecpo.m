% ----------------------------------------------------------------------- %
% Electric Charged Particles Optimization (ECPO) for unconstrained
% benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   ECPSize = 50          % Population size (charged particles)
%   V = [1 2 1]           % [Strategy, NPI, Archive divisor]
%
% Algorithm Concept:
%   - Inspired by the Coulomb/Gauss forces between electrically charged
%     particles: superior particles attract inferior ones and vice versa
%   - Only selected particles (NPI) interact, according to one of three
%     interaction strategies; an archive guides part of the search
%
% Reference:
% Houssem R.E.H. Bouchekara,
% Electric Charged Particles Optimization and its application to the optimal
% design of a circular antenna array,
% Artificial Intelligence Review 54 (2021) 1767-1802
% https://doi.org/10.1007/s10462-020-09890-x
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = ecpo(problem)

    % Extract problem parameters
    ProblemSize = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    ECPSize = 50;
    V = [1 2 1];               % Strategy, NPI, Archive size divisor
    Strategy = V(1);
    NPI = V(2);
    archSize = ECPSize / V(3);

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, ECPSize, ProblemSize);
    fitness_history = zeros(history_size, ECPSize);
    history_index = 1;

    % Initialization
    ECP = repmat(lb, ECPSize, 1) + rand(ECPSize, ProblemSize) .* repmat((ub - lb), ECPSize, 1);
    [F_ECP, FE] = calculate_fitness(ECP', problem, FE);
    F_ECP = F_ECP(:)';
    [F_ECP, ind] = sort(F_ECP);
    ECP = ECP(ind, :);

    for eval_count = 1:FE
        if eval_count <= maxFE
            curve(eval_count) = F_ECP(1);
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, ECP, F_ECP, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    if Strategy == 1
        pop_fac = 2 * nchoosek(NPI, 2);
    elseif Strategy == 2
        pop_fac = NPI;
    elseif Strategy == 3
        pop_fac = 2 * nchoosek(NPI, 2) + NPI;
    end

    while FE < maxFE
        FE_before = FE;

        pop_arch = ECP(1:archSize, :);
        F_arch = F_ECP(1:archSize);
        newECP = [];
        newECP1 = [];
        newECP2 = [];

        for i = 1:1:ceil(ECPSize / pop_fac)
            Force = normrnd(0.7, 0.2);
            SP = sort(randperm(ECPSize, NPI));
            SP = SP(1:NPI);

            if Strategy == 1
                for ii = 1:NPI
                    for jj = 1:NPI
                        S1 = ECP(SP(ii), :) + Force * (ECP(1, :) - ECP(SP(ii), :));
                        if jj < ii
                            S1 = S1 + Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            newECP(end + 1, :) = S1; %#ok<AGROW>
                        elseif jj > ii
                            S1 = S1 - Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            newECP(end + 1, :) = S1; %#ok<AGROW>
                        end
                    end
                end
            elseif Strategy == 2
                for ii = 1:NPI
                    S1 = ECP(SP(ii), :) + 0 * Force * (ECP(1, :) - ECP(SP(ii), :));
                    for jj = 1:NPI
                        if jj < ii
                            S1 = S1 + Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                        elseif jj > ii
                            S1 = S1 - Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                        end
                    end
                    newECP(end + 1, :) = S1; %#ok<AGROW>
                end
            elseif Strategy == 3
                for ii = 1:NPI
                    S2 = ECP(SP(ii), :) + 1 * Force * (ECP(1, :) - ECP(SP(ii), :));
                    for jj = 1:NPI
                        S1 = ECP(SP(ii), :) + Force * (ECP(1, :) - ECP(SP(ii), :));
                        if jj < ii
                            S1 = S1 + Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            S2 = S2 + Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            newECP1(end + 1, :) = S1; %#ok<AGROW>
                        elseif jj > ii
                            S1 = S1 - Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            S2 = S2 - Force * (ECP(SP(jj), :) - ECP(SP(ii), :));
                            newECP1(end + 1, :) = S1; %#ok<AGROW>
                        end
                    end
                    newECP2(end + 1, :) = S2; %#ok<AGROW>
                end
                newECP = [newECP1; newECP2];
            end
        end

        newECP = bound(newECP, lb, ub);

        for i1 = 1:size(newECP, 1)
            for j = 1:ProblemSize
                r = rand;
                if (r < 0.2)
                    pos = randi(archSize(1));
                    newECP(i1, j) = pop_arch(pos, j);
                end
            end
        end

        ECP_All = [pop_arch; newECP];
        [fitnew, FE] = calculate_fitness(newECP', problem, FE);
        fitnew = fitnew(:)';
        F_All = [F_arch fitnew];

        [F_All, index] = sort(F_All);
        ECP_All = ECP_All(index, :);
        ECP = ECP_All(1:ECPSize, :);
        F_ECP = F_All(1:ECPSize);

        % Record convergence curve and history
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = F_ECP(1);
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, ECP, F_ECP, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = ECP(1, :);
    best_fitness = F_ECP(1);

end

%% --- Boundary handling ---
function [x] = bound(x, l, u)
    for j = 1:size(x, 1)
        x(j, x(j, :) < l) = l(x(j, :) < l);
        x(j, x(j, :) > u) = u(x(j, :) > u);
    end
end

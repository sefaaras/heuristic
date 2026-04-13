% ----------------------------------------------------------------------- %
% Spherical Evolution (SE) Algorithm for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   q = 20                    % Population size
%   type = 4                  % Mutation strategy (SE/rand/1)
%
% Algorithm Concept:
%   - Uses hyper-spherical coordinate transformation for mutation
%   - Transforms Cartesian difference vectors into spherical coordinates
%   - Random direction with preserved magnitude enables diverse exploration
%   - Multiple mutation strategies analogous to DE variants
%
% Reference:
% Rahul Kumar Patel, Licheng Jiao, Fang Liu,
% Spherical evolution for solving multimodal and composition optimization problems,
% Knowledge-Based Systems 277 (2023) 110837
% https://doi.org/10.1016/j.knosys.2023.110837
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = se(problem)

    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    q = 20;
    type = 4;

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, q, dim);
    fitness_history = zeros(history_size, q);
    history_index = 1;

    % Initialize population
    Positions = initialization(q, dim, ub, lb);
    [fitness, FE] = calculate_fitness(Positions', problem, FE);

    [global_best_fit, best_idx] = min(fitness);
    global_best_pos = Positions(best_idx,:);

    for eval_count = 1:min(q, maxFE)
        curve(eval_count) = global_best_fit;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, Positions, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    while FE < maxFE

        [~, y] = sort(fitness);

        F_val = zeros(1, q);
        paraDim = zeros(1, q);
        for j = 1:q
            F_val(y(j)) = j / q + 0.1 * randn;
            if F_val(y(j)) > 1
                F_val(y(j)) = 1;
            elseif F_val(y(j)) < 0
                F_val(y(j)) = j / q;
            end
            if dim >= 10
                cc = randperm(5);
                paraDim(j) = 5 + cc(1);
            else
                paraDim(j) = 3;
            end
        end

        for j = 1:q
            if FE >= maxFE, break; end

            kk = randperm(q);

            b = Positions(kk(1),:);
            f = Positions(kk(2),:);
            m = Positions(kk(3),:);
            n = Positions(kk(4),:);
            g = Positions(kk(5),:);
            a = global_best_pos;
            h = Positions(j,:);

            pp2 = randperm(dim);
            pp4 = randperm(paraDim(j));
            mm = pp4(1);
            if mm > numel(pp2)
                mm = numel(pp2);
            end
            pp3 = sort(pp2(1:mm));

            if length(pp3) == 1
                s1 = F_val(j) * HyperSphereTransform_1D(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform_1D(m, n, pp3);
                s3 = F_val(j) * HyperSphereTransform_1D(h, a, pp3);
            elseif length(pp3) == 2
                s1 = F_val(j) * HyperSphereTransform_2D(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform_2D(m, n, pp3);
                s3 = F_val(j) * HyperSphereTransform_2D(h, a, pp3);
            else
                s1 = F_val(j) * HyperSphereTransform(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform(m, n, pp3);
                s3 = F_val(j) * HyperSphereTransform(h, a, pp3);
            end

            switch type
                case 1 % SE/current-to-best/1
                    h(pp3) = h(pp3) + s3 + s2;
                case 2 % SE/best/1
                    h(pp3) = global_best_pos(pp3) + s1;
                case 3 % SE/best/2
                    h(pp3) = global_best_pos(pp3) + s1 + s2;
                case 4 % SE/rand/1
                    h(pp3) = g(pp3) + s1;
                case 5 % SE/rand/2
                    h(pp3) = g(pp3) + s1 + s2;
                case 6 % SE/current/1
                    h(pp3) = h(pp3) + s1;
                case 7 % SE/current/2
                    h(pp3) = h(pp3) + s1 + s2;
            end

            temp = h;
            temp = bound(temp, ub, lb);

            [temp_fit, FE] = calculate_fitness(temp', problem, FE);

            if temp_fit < fitness(j)
                Positions(j,:) = temp;
                fitness(j) = temp_fit;
            end

            if temp_fit < global_best_fit
                global_best_fit = temp_fit;
                global_best_pos = temp;
            end

            if FE <= maxFE
                curve(FE) = global_best_fit;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    for idx = 2:maxFE
        if curve(idx) == 0
            curve(idx) = curve(idx - 1);
        end
    end

    best_fitness = global_best_fit;
    best_solution = global_best_pos;

end

%% --- Hyper-Sphere Transform (General D-dimensional) ---
function ss = HyperSphereTransform(c, d, pp)
    D = length(pp);
    A = c(pp) - d(pp);
    R = norm(A, 2);

    O = zeros(1, D - 1);
    O(D - 1) = 2 * pi * rand;
    for i = 1:D - 2
        O(i) = rand * pi;
    end

    C = zeros(1, D);
    C(1) = R * prod(sin(O));
    for i = 2:D - 1
        C(i) = R * cos(O(i - 1)) * prod(sin(O(i:D - 1)));
    end
    C(D) = R * cos(O(D - 1));
    ss = C;
end

%% --- Hyper-Sphere Transform (1D) ---
function ss = HyperSphereTransform_1D(c, d, pp)
    R = abs(c(pp) - d(pp));
    C = R * cos(2 * pi * rand);
    ss = C;
end

%% --- Hyper-Sphere Transform (2D) ---
function ss = HyperSphereTransform_2D(c, d, pp)
    A = c(pp) - d(pp);
    R = norm(A, 2);
    o1 = 2 * pi * rand;
    C = zeros(1, 2);
    C(1) = R * sin(o1);
    C(2) = R * cos(o1);
    ss = C;
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    end
    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

%% --- Boundary Handling ---
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

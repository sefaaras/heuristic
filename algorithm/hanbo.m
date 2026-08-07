% ----------------------------------------------------------------------- %
% Hannibal Barca Optimizer (HBO)
% Stored as hanbo; the acronym HBO collides with hbo
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   warriors_no = 30      % Total number of warriors
%   roman_rate  = 0.63    % Share of the population forming the Roman army
%   percent_3   = 3%      % Roman left/right wing size (best individuals)
%   percent_12  = 12%     % Carthaginian left/right wing size (worst first)
%   c           = 0.94    % Attack damping coefficient
%
% Algorithm Concept:
%   - Two armies (Romans, Carthaginians) are each split into a left wing,
%     a right wing and a centre
%   - right_attack: reflection about the plane midway between the two wings
%   - left_attack : mirroring about the opposing wing's centroid
%   - center_attack: Gaussian drift towards the Roman general / Hannibal
%   - parallaxe: warriors are paired and pushed along the direction to the
%     leader by a distance proportional to the pair-centre-to-leader distance
%   - Three battle phases (attacks, strategic retreats, encirclement) chosen
%     by the iteration fraction
%
% Reference:
% Mohamed Wajdi Ouertani, Ghaith Manita, Ouajdi Korbaa,
% Hannibal Barca optimizer: the power of the pincer movement for global
% optimization and multilevel image thresholding,
% Cluster Computing (2025).
% https://doi.org/10.1007/s10586-025-05134-1
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = hanbo(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    warriors_no = 30;
    Max_iter    = max(1, ceil(maxFE / warriors_no));

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    HA_pos    = zeros(1, dim);
    HA_score  = inf;
    Rom_Gen   = zeros(1, dim);
    Rom_score = inf;

    % Initialise both armies
    roman_rate  = 0.63;
    romans_no   = round(warriors_no * roman_rate);
    cartagos_no = warriors_no - romans_no;

    P = initialization(warriors_no, dim, ub, lb);
    [F, FE] = calculate_fitness(P', problem, FE);
    F = F(:);

    romans_fitness   = F(1:romans_no);
    cartagos_fitness = F(romans_no+1:end);

    [bsf, bi] = min(F);
    bsx = P(bi, :);

    for eval_count = 1:warriors_no
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, P, F, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    % Divide the Romans: best 3% left wing, next 3% right wing, rest centre
    [~, si] = sort(romans_fitness);
    rom_order = si(:)';                       % indices into 1..romans_no
    p3 = max(1, round(0.03 * romans_no));
    G.lr = rom_order(1:p3);
    G.rr = rom_order(p3+1:min(2*p3, romans_no));
    G.cr = rom_order(2*p3+1:end);

    % Divide the Carthaginians: worst 12% left wing, next 12% right, rest centre
    [~, si] = sort(cartagos_fitness, 'descend');
    car_order = romans_no + si(:)';           % indices into the full matrix
    p12 = max(1, round(0.12 * cartagos_no));
    G.lc = car_order(1:p12);
    G.rc = car_order(p12+1:min(2*p12, cartagos_no));
    G.cc = car_order(2*p12+1:end);

    % Main battle loop
    for it = 1:Max_iter
        if FE >= maxFE, break; end

        if it <= 0.33 * Max_iter
            phase = 1;
        elseif it <= 0.66 * Max_iter
            phase = 2;
        else
            phase = 3;
        end

        switch phase
            case 1   % Early phase: all attacks
                plan = {{'right', G.rc, G.lr}, {'left', G.lc, G.rr}, {'center', G.cc, G.cr}};
            case 2   % Middle phase: strategic retreats
                plan = {{'right', G.rc, G.lr}, {'left', G.lc, G.lr}, ...
                        {'retreat', G.rr, []}, {'center', G.cc, G.cr}};
            otherwise % Final phase: encirclement
                plan = {{'right', G.rc, G.cr}, {'retreat', G.lr, []}, ...
                        {'left', G.lc, G.cr}, {'retreat', G.rr, []}, ...
                        {'center', G.cc, G.cr}};
        end

        for s = 1:numel(plan)
            if FE >= maxFE, break; end
            act = plan{s}{1};
            iA  = plan{s}{2};
            iB  = plan{s}{3};

            switch act
                case 'right'
                    [A, B] = right_attack(P(iA, :), P(iB, :), Rom_Gen, HA_pos);
                    P(iA, :) = enforceBounds(A, ub, lb);
                    P(iB, :) = enforceBounds(B, ub, lb);
                    order = {iA, iB};
                case 'left'
                    [A, B] = left_attack(P(iA, :), P(iB, :), Rom_Gen, HA_pos);
                    P(iA, :) = enforceBounds(A, ub, lb);
                    P(iB, :) = enforceBounds(B, ub, lb);
                    order = {iA, iB};
                case 'center'
                    [A, B] = center_attack(P(iA, :), P(iB, :), Rom_Gen, HA_pos);
                    P(iA, :) = enforceBounds(A, ub, lb);
                    P(iB, :) = enforceBounds(B, ub, lb);
                    order = {iA, iB};
                otherwise   % retreat
                    P(iA, :) = retreat(P(iA, :), ub, lb);
                    order = {iA};
            end

            for q = 1:numel(order)
                idx = order{q};
                if FE >= maxFE || isempty(idx), break; end

                for r = 1:numel(idx)
                    P(idx(r), :) = enforceBounds(P(idx(r), :), ub, lb);
                end
                [f, FE] = calculate_fitness(P(idx, :)', problem, FE);
                f = f(:);
                F(idx) = f;
                n = numel(f);

                for r = 1:n
                    if f(r) < HA_score
                        HA_score = f(r);
                        HA_pos   = P(idx(r), :);
                    elseif f(r) < Rom_score
                        Rom_score = f(r);
                        Rom_Gen   = P(idx(r), :);
                    end
                    if f(r) < bsf
                        bsf = f(r);
                        bsx = P(idx(r), :);
                    end
                    ec = FE - n + r;
                    if ec >= 1 && ec <= maxFE
                        curve(ec) = bsf;
                    end
                    % every warrior in order was moved before this block, so F
                    % only describes P again once the last group is evaluated
                    if q == numel(order) && r == n && ec >= 1 && ec <= maxFE
                        [population_history, fitness_history, history_index] = record_history(...
                            ec, P, F, population_history, fitness_history, ...
                            history_index, maxFE);
                    end
                end
            end
        end
    end

    curve(min(FE, maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsx;
end

% Right attack: reflection about the mid-plane of the two wings
function [new_pos_right, new_pos_left] = right_attack(pos_right, pos_left, General, Hannibal)
    reflection_plane = (mean(pos_right, 1) + mean(pos_left, 1)) / 2;
    random_vector_right = randn(size(pos_right)) * 0.5;
    random_vector_left  = randn(size(pos_left))  * 0.5;
    new_pos_right = reflection_plane + random_vector_right .* (reflection_plane - pos_right);
    new_pos_left  = reflection_plane + random_vector_left  .* (reflection_plane - pos_left);

    new_pos_right = 0.94 * parallaxe(new_pos_right, Hannibal);
    new_pos_left  = 0.94 * parallaxe(new_pos_left,  General);
end

% Left attack: mirroring about the opposing centroid
function [new_pos_right, new_pos_left] = left_attack(pos_right, pos_left, General, Hannibal)
    random_vector_right = randn(size(pos_right)) * 0.5;
    random_vector_left  = randn(size(pos_left))  * 0.5;
    new_pos_right = 2 * mean(pos_left, 1)  - random_vector_right .* pos_right;
    new_pos_left  = 2 * mean(pos_right, 1) - random_vector_left  .* pos_left;

    new_pos_right = 0.94 * parallaxe(new_pos_right, Hannibal);
    new_pos_left  = 0.94 * parallaxe(new_pos_left,  General);
end

% Center attack: Gaussian drift towards the two leaders
function [new_pos_right, new_pos_left] = center_attack(pos_right, pos_left, General, Hannibal)
    c = 0.94;
    random_vector_right = randn(size(pos_right)) * c * rand;
    random_vector_left  = randn(size(pos_left))  * c * rand;
    new_pos_right = pos_right + random_vector_right .* (General  - pos_right);
    new_pos_left  = pos_left  + random_vector_left  .* (Hannibal - pos_left);
end

% Retreat manoeuvre
function retreaters = retreat(retreaters, ub, lb)
    for i = 1:size(retreaters, 1)
        retreaters(i, :) = retreaters(i, :) + randn(1, size(retreaters, 2)) .* (ub - lb);
    end
end

% Parallaxe: paired push towards the leader
function modified_pos_right = parallaxe(pos_right, Rom_Gen)
    N = size(pos_right, 1);
    if mod(N, 2) ~= 0
        couples = reshape(randperm(N - 1), [], 2);
    else
        couples = reshape(randperm(N), [], 2);
    end

    modified_pos_right = pos_right;

    for idx = 1:size(couples, 1)
        couple_center = (pos_right(couples(idx, 1), :) + pos_right(couples(idx, 2), :)) / 2;
        center_to_gen_dist = norm(couple_center - Rom_Gen(1, :));

        for j = 1:2
            warrior = pos_right(couples(idx, j), :);
            influence_direction = (Rom_Gen(1, :) - warrior) / norm(Rom_Gen(1, :) - warrior);
            adjusted_step_size = center_to_gen_dist * (0.5 + rand() * 2);
            modified_pos_right(couples(idx, j), :) = warrior + adjusted_step_size * influence_direction;
        end
    end
end

% Bound handling
function position = enforceBounds(position, ub, lb)
    flagUb = position > ub;
    flagLb = position < lb;
    position = position .* ~(flagUb + flagLb) + ub .* flagUb + lb .* flagLb;
end

% Initialization
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end

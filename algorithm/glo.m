% ----------------------------------------------------------------------- %
% Gray Langurs Optimizer (GLO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N     = 30     % Number of langurs (>= 3)
%   FP_om = 0.4    % Female percentage in the one-male group
%   FP_mm = 0.3    % Female percentage in the multi-male group
%   MP_mm = 0.3    % Male   percentage in the multi-male group
%   omega = 0.05   % Migration probability
%   phi   = 5      % Minimum group size
%
% Algorithm Concept:
%   - The troop is split into three social groups: one-male, multi-male and
%     all-male; each has its own alpha, males, females and children with a
%     dedicated position-update rule
%   - Adaptive scaling factor ASF = 0.1 + (1-ct/mt)^2*0.9 shrinks the steps
%   - Dynamism: langurs migrate between groups with probability omega
%   - Autonomy: every langur additionally roams with a Levy-scaled step
%
% Reference:
% Saeid Barshandeh, Nima Khodadadi, Benyamin Abdollahzadeh et al.,
% Gray langurs optimizer: a multi-group bio-inspired optimization algorithm,
% Artificial Intelligence Review (2026).
% https://doi.org/10.1007/s10462-026-11529-2
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = glo(problem)

    D     = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    N  = 30;
    mt = max(1, ceil((maxFE - N) / (2 * N)));

    FP_om = 0.4;
    FP_mm = 0.3;
    MP_mm = 0.3;
    omega = 0.05;
    phi   = 5;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Mirrors record_history's sampling gate so the population is assembled only when due
    T_samples = record_history('get_samples');
    hist_step = max(1, floor(max(1, maxFE) / T_samples));

    % Initialisation
    X0 = repmat(lb, N, 1) + repmat(ub - lb, N, 1) .* rand(N, D);
    [F0, FE] = calculate_fitness(X0', problem, FE);
    F0 = F0(:);

    Sol = struct('X', cell(1, N), 'Cost', cell(1, N));
    for i = 1:N
        Sol(i).X    = X0(i, :);
        Sol(i).Cost = F0(i);
    end
    [~, index] = min([Sol.Cost]);
    Best = Sol(index);

    bsf = Best.Cost;
    bsx = Best.X;

    for eval_count = 1:N
        if eval_count <= maxFE
            curve(eval_count) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, X0, F0, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    Group = init_groups(Sol, N);

    % Optimisation
    for ct = 1:mt
        if FE >= maxFE, break; end

        ASF = 0.1 + (1 - ct / mt) ^ 2 * .9;

        for g = 1:3
            if FE >= maxFE, break; end
            S = numel(Group(g).Member);

            switch g
                case 1
                    Nm = 1;
                    Nf = floor(S * FP_om);      % Eq. (11)
                case 2
                    Nm = max(floor(S * MP_mm), 1);   % Eq. (16)
                    Nf = max(floor(S * FP_mm), 1);   % Eq. (17)
                case 3
                    Nm = S;
                    Nf = 0;
            end

            for i = 1:S
                if FE >= maxFE, break; end

                if i == 1
                    newX = update_alpha(Group(g).Member(i).X, Best.X, ASF, ct, mt, lb, ub, D);
                elseif g == 1
                    if i <= Nf
                        newX = update_female(Group(g).Member, 1, i, D, ASF);
                    else
                        newX = update_child(Group(g).Member, 1, Nf, D, ASF, lb, ub);
                    end
                elseif g == 2
                    if i <= Nm
                        newX = update_male(Group(g).Member(i).X, Best, ct, mt, ub, lb);
                    elseif i <= Nm + Nf
                        newX = update_female(Group(g).Member, Nm, i, D, ASF);
                    else
                        newX = update_child(Group(g).Member, Nm, Nf, D, ASF, lb, ub);
                    end
                else
                    newX = update_male(Group(g).Member(i).X, Best, ct, mt, ub, lb);
                end

                newX = min(max(newX, lb), ub);
                [newCost, FE] = calculate_fitness(newX', problem, FE);

                if newCost < Group(g).Member(i).Cost
                    Group(g).Member(i).X    = newX;
                    Group(g).Member(i).Cost = newCost;
                end
                if newCost < Group(g).Alpha.Cost
                    Group(g).Alpha.X    = newX;
                    Group(g).Alpha.Cost = newCost;
                end
                if newCost < Best.Cost
                    Best.X    = newX;
                    Best.Cost = newCost;
                end
                if newCost < bsf
                    bsf = newCost;
                    bsx = newX;
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    if FE >= (history_index - 1) * hist_step
                        [P, Fv] = flatten(Group, N, D);
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, P, Fv, population_history, fitness_history, ...
                            history_index, maxFE);
                    end
                end
            end

            % Maintain the social structure (reorder the langurs)
            [~, sind] = sort([Group(g).Member.Cost]);
            Group(g).Member = Group(g).Member(sind);
        end

        % Dynamism (migration)
        Group = dynamism(Group, omega, phi);

        % Autonomy (roaming around)
        for g = 1:3
            if FE >= maxFE, break; end
            for j = 1:numel(Group(g).Member)
                if FE >= maxFE, break; end
                levy_scale = rand .* (ub - lb) * exp(-20 * (ct / mt) ^ 2);
                levy = 1 ./ (abs(randn(1, D)) .^ (1 / 1.5));
                step = levy_scale .* levy * .001;
                newX = Group(g).Member(j).X .* step;

                newX = min(max(newX, lb), ub);
                [newCost, FE] = calculate_fitness(newX', problem, FE);

                if newCost < Group(g).Member(j).Cost
                    Group(g).Member(j).X    = newX;
                    Group(g).Member(j).Cost = newCost;
                end
                if newCost < Group(g).Alpha.Cost
                    Group(g).Alpha.X    = newX;
                    Group(g).Alpha.Cost = newCost;
                end
                if newCost < Best.Cost
                    Best.X    = newX;
                    Best.Cost = newCost;
                end
                if newCost < bsf
                    bsf = newCost;
                    bsx = newX;
                end
                if FE <= maxFE
                    curve(FE) = bsf;
                    if FE >= (history_index - 1) * hist_step
                        [P, Fv] = flatten(Group, N, D);
                        [population_history, fitness_history, history_index] = record_history(...
                            FE, P, Fv, population_history, fitness_history, ...
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

% Flatten the three groups into a population matrix
function [P, Fv] = flatten(Group, N, D)
    P  = zeros(N, D);
    Fv = zeros(N, 1);
    k = 0;
    for g = 1:3
        for j = 1:numel(Group(g).Member)
            k = k + 1;
            if k > N, return; end
            P(k, :) = Group(g).Member(j).X;
            Fv(k)   = Group(g).Member(j).Cost;
        end
    end
    if k < N
        P  = P(1:k, :);
        Fv = Fv(1:k);
    end
end

% Group construction: one-male / multi-male / all-male
function Group = init_groups(Sol, N)
    Groupsizes = cumsum([floor(N/3), floor(N/3), N - 2 * floor(N/3)]);
    Group(1).Member = Sol(1:Groupsizes(1));
    Group(2).Member = Sol(Groupsizes(1)+1:Groupsizes(2));
    Group(3).Member = Sol(Groupsizes(2)+1:Groupsizes(3));

    for i = 1:3
        [~, ind] = sort([Group(i).Member.Cost]);
        Group(i).Member = Group(i).Member(ind);
        Group(i).Alpha  = Group(i).Member(1);
    end
end

% Alpha update (neighbourhood search or approach to the global best)
function newX = update_alpha(X, Best_X, ASF, ct, mt, lb, ub, D)
    r1 = rand();
    if r1 < 0.5
        levy = 1 ./ (abs(randn(1, D)) .^ (1 / 1.5));         % Levy flight step
        a = unifrnd(-5, 5);
        b = ASF * max((1 - ct / mt), 0.4) * (ub - lb) / 30;
        newX = X + (a * b .* levy);
    else
        mu = unifrnd(-.1, .1, 1, D);
        C1 = (10 - (ct / mt)) * mu;                          % Eq. (7)
        newX = X + C1 .* (X - Best_X);                       % Eq. (8)
    end
end

% Male update
function newX = update_male(X, Best, ct, mt, ub, lb)
    C2 = 1 + randn;
    inertia = max(.4, .9 * (1 - ct / mt));
    newX = X + inertia * (X - Best.X) + C2 * (randn * (ub - lb));
end

% Female update (roulette-selected mate among the males)
function newX = update_female(Member, Nm, i, D, ASF)
    mate = RouletteWheelSelection([Member(1:Nm).Cost]);
    newX = Member(i).X + (randn(1, D) * .8 * ASF) .* (Member(mate).X - Member(i).X);
end

% Child update (crossover of a father and a mother) -- Eq. (12)
function newX = update_child(Member, Nm, Nf, D, ASF, lb, ub)
    mother = randi([Nm + 1, max(Nm + Nf, 2)]);
    father = randi([1, Nm]);
    fi = randn(1, D) .* ASF;
    mi = 1 - fi;
    newX = (fi .* Member(father).X) + (mi .* Member(mother).X) + ...
           randn(1, D) .* 0.05 * ASF .* (ub - lb);
end

% Migration between groups
function Group = dynamism(Group, omega, phi)
    for i = 1:3
        members = [];
        for j = 1:numel(Group(i).Member)
            if rand <= omega && numel(Group(i).Member) > phi
                cand = setdiff(1:3, i);
                DG = cand(randi(numel(cand)));       % destination group

                PlaceIndex = numel(Group(DG).Member) + 1;
                Group(DG).Member(PlaceIndex).X    = Group(i).Member(j).X;
                Group(DG).Member(PlaceIndex).Cost = Group(i).Member(j).Cost;

                members = [members j];  %#ok<AGROW>
                if numel(Group(i).Member) - numel(members) < 5
                    break;
                end
            end
        end

        if numel(members) > 0
            Group(i).Member(members) = [];
        end
    end
end

% Roulette-wheel selection (guarded against an empty index)
function i = RouletteWheelSelection(p)
    p = abs(p);
    r = rand * sum(p);
    c = cumsum(p);
    i = find(r <= c, 1, 'first');
    if isempty(i)
        i = numel(p);
    end
end

% ----------------------------------------------------------------------- %
% Group Search Optimizer (GSO)
% Stored as groupso; the acronym GSO collides with gso (Glider Snake Optimizer)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 48                       % Members: one producer, the rest scrounge or range
%   p_scrounge = 0.8             % Chance a non-producer scrounges, else it ranges
%   a = round(sqrt(D + 1))       % Failed scans before the producer turns back
%   theta_max = pi/a^2           % Maximum pursuit angle of the lateral scans
%   alpha_max = theta_max/2      % Maximum turning angle per iteration
%   l_max = sqrt(D)*(ub - lb)    % Maximum pursuit distance, ||U - L|| on a cube
%   phi_0 = pi/4                 % Initial head angle in each of the D - 1 angles
%
% Algorithm Concept:
%   - The best member produces: it scans three points at distance N(0, 1)*l_max,
%     straight ahead and at head angles +/- U(0, 1)*theta_max/2
%   - It flies to the best scan if that beats its spot, else it turns its head by
%     U(0, 1)*alpha_max; after a failed scans it returns to its last good angle
%   - Scroungers copy area: x + U(0, 1) .* (x_producer - x) per dimension
%   - Rangers turn by U(0, 1)*alpha_max and walk a*N(0, 1)*l_max along their head
%   - Directions come from the D - 1 head angles in polar coordinates
%   - A coordinate leaving the box takes back the move that put it there
%
% Reference:
% S. He, Q. H. Wu, J. R. Saunders,
% Group Search Optimizer: An Optimization Algorithm Inspired by Animal Searching
% Behavior,
% IEEE Transactions on Evolutionary Computation 13(5) (2009) 973-990.
% https://doi.org/10.1109/TEVC.2009.2011992
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' GSO.m (Liverpool, last modified 26 Aug 2008), once GSO.zip on
% Shan He's Birmingham page, now known only from saswat0/SSO-Net "Matlab codes/GSO.m"
% (interface changed); seeded runs are bit-identical to it. Kept as released: two
% non-unit direction formulas where the paper's Eq. (1) is one unit formula (the
% producer's shifts its sine index, the rangers' is another form; both equal it only at
% D = 2); each non-producer scrounges with probability 0.8, not exactly 80 %; the
% producer re-evaluated with the group (51 FEs a generation, the paper's 48 give 50);
% the revert using a scrounger's stale direction (the paper restores the previous
% value); one fail counter for every producer. Changed: l_max (the scalar ||U - L||) is
% sqrt(D)*(ub - lb) per dimension, bit-exact on a cube; a clamp absorbs revert rounding.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = groupso(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    N = 48;
    a = round(((D + 1) ^ .5));
    max_pursuit_angle = (pi / (a ^ 2));
    max_turning_angle = max_pursuit_angle / 2;
    % The release's ||U - L||, times each range over the root mean square range
    l_max = (norm(span) * (span / sqrt(mean(span .^ 2))))';

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    LowerBound = repmat(lb', 1, N);
    UpperBound = repmat(ub', 1, N);
    population = (lb + rand(N, D) .* span)';   % D x N, one member per column

    angle     = pi / 4 .* ones(D - 1, N);
    distance  = repmat(l_max, 1, N);
    direction = zeros(D, N);
    for j = 1:N
        direction(:, j) = ranger_direction(angle(:, j), D);
    end

    n0 = min(N, maxFE);
    [f0, FE] = calculate_fitness(population(:, 1:n0), problem, FE);
    fvalue = inf(1, N);
    fvalue(1:n0) = f0(:)';
    Xrec = population';   % where each fvalue was measured, one member per row

    best_fitness  = inf;
    best_solution = population(:, 1)';
    for q = 1:n0
        if fvalue(q) < best_fitness
            best_fitness  = fvalue(q);
            best_solution = population(:, q)';
        end
        curve(q) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            q, Xrec(1:n0, :), fvalue(1:n0), population_history, fitness_history, ...
            history_index, maxFE);
    end
    if FE >= maxFE
        curve(min(FE, maxFE):end) = best_fitness;
        return;
    end

    % As released, after the evaluation
    OutFlag = population <= LowerBound | population >= UpperBound;
    population = population - OutFlag .* distance .* direction;

    [~, index] = min(fvalue);
    bestmember = population(:, index);
    oldangle   = angle;
    badcounter = 0;

    while FE < maxFE
        for j = 1:N
            R1 = randn(1);
            R2 = rand(D - 1, 1);
            R3 = rand(D, 1);

            if j == index
                distance(:, j) = l_max * R1;
                if badcounter >= a
                    angle(:, j) = oldangle(:, j);
                end

                % Straight, left and right scans, each move undone where it leaves the box
                direction(:, j) = producer_direction(angle(:, j), D);
                StraightPosition = population(:, j) + distance(:, j) .* direction(:, j);
                Outflag = (StraightPosition > UpperBound(:, j) | StraightPosition < LowerBound(:, j));
                StraightPosition = StraightPosition - Outflag .* distance(:, j) .* direction(:, j);

                leftangle = angle(:, j) + max_pursuit_angle .* R2 / 2;
                direction(:, j) = producer_direction(leftangle, D);
                LeftPosition = population(:, j) + distance(:, j) .* direction(:, j);
                Outflag = (LeftPosition > UpperBound(:, j) | LeftPosition < LowerBound(:, j));
                LeftPosition = LeftPosition - Outflag .* distance(:, j) .* direction(:, j);

                rightangle = angle(:, j) - max_pursuit_angle .* R2 / 2;
                direction(:, j) = producer_direction(rightangle, D);
                RightPosition = population(:, j) + distance(:, j) .* direction(:, j);
                Outflag = (RightPosition > UpperBound(:, j) | RightPosition < LowerBound(:, j));
                RightPosition = RightPosition - Outflag .* distance(:, j) .* direction(:, j);

                Scans = [StraightPosition, LeftPosition, RightPosition];
                nsc = min(3, maxFE - FE);
                [fs, FE_new] = calculate_fitness(Scans(:, 1:nsc), problem, FE);
                fs = fs(:)';
                for q = 1:nsc
                    if fs(q) < best_fitness
                        best_fitness  = fs(q);
                        best_solution = Scans(:, q)';
                    end
                    curve(FE + q) = best_fitness;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE + q, Xrec, fvalue, population_history, fitness_history, ...
                        history_index, maxFE);
                end
                FE = FE_new;
                if FE >= maxFE, break; end

                SamplePosition = [population(:, j), Scans];
                SampleAngle    = [angle(:, j), angle(:, j), leftangle, rightangle];
                SampleValue    = [fvalue(j), fs];
                [~, best_position_ID] = min(SampleValue);
                population(:, j) = SamplePosition(:, best_position_ID);

                if best_position_ID ~= 1
                    angle(:, j) = SampleAngle(:, best_position_ID);
                    oldangle(:, j) = angle(:, j);
                    badcounter = 0;
                else
                    badcounter = badcounter + 1;
                    angle(:, j) = angle(:, j) + max_turning_angle .* R2;
                end

            else
                angle(1:D-1, j) = angle(1:D-1, j) + max_turning_angle .* R2;

                if rand(1) > .2
                    distance(:, j) = R3 .* (bestmember - population(:, j));
                    population(:, j) = population(:, j) + distance(:, j);
                else
                    distance(:, j) = l_max * (a * R1);
                    direction(:, j) = ranger_direction(angle(:, j), D);
                    population(:, j) = population(:, j) + distance(:, j) .* direction(:, j);
                end
            end
        end
        if FE >= maxFE, break; end

        % The release's revert covers the producer too, with its last scan's direction
        OutFlag = population <= LowerBound | population >= UpperBound;
        population = population - OutFlag .* distance .* direction;
        population = min(max(population, LowerBound), UpperBound);

        nb = min(N, maxFE - FE);
        [fv, FE_new] = calculate_fitness(population(:, 1:nb), problem, FE);
        fv = fv(:)';
        for q = 1:nb
            fvalue(q)  = fv(q);
            Xrec(q, :) = population(:, q)';
            if fv(q) < best_fitness
                best_fitness  = fv(q);
                best_solution = Xrec(q, :);
            end
            curve(FE + q) = best_fitness;
            [population_history, fitness_history, history_index] = record_history(...
                FE + q, Xrec, fvalue, population_history, fitness_history, ...
                history_index, maxFE);
        end
        FE = FE_new;

        [~, index] = min(fvalue);
        bestmember = population(:, index);
    end

    curve(min(FE, maxFE):end) = best_fitness;
end

% The producer's scan direction, as released
function d = producer_direction(phi, D)
    c = cos(phi);
    s = sin(phi);
    d = zeros(D, 1);
    d(1) = prod(c(1:D-1));
    for i = 2:D-1
        d(i) = s(i) .* prod(c(i:D-1));
    end
    d(D) = s(D-1);
end

% The rangers' (and initial) direction, a different formula in the release
function d = ranger_direction(phi, D)
    c = cos(phi);
    s = sin(phi);
    d = zeros(D, 1);
    d(1) = c(1);
    for i = 2:D-1
        d(i) = c(i) .* prod(s(i:D-1));
    end
    d(D) = prod(s(1:D-1));
end

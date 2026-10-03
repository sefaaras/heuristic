% ----------------------------------------------------------------------- %
% Natural Survivor Method based Stochastic Fractal Search (NSM-SFS)
% Variant of sfs: NSM survivor selection
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = 50, MDN = 1, Walk = 1             % Population, diffusion number, walk choice (sfs)
%   w = [0.9882 0.6862 0.6868]            % NSM weights: fitness, from best, from centre
%   n_centre = ceil(15*chebyshev(FE))     % Best individuals averaged into the centre
%   p_NSM = 1 - sigmoid(15*FE/maxFE - 9.9) % NSM vs greedy test, ~1 early -> 0.006 at end
%
% Algorithm Concept:
%   - Diffusion by Gaussian walks around BestPoint with step log(g)/g*|x - BestPoint|,
%     g the FE count as released; the parent competes with its offspring unevaluated
%   - Two updating processes as in sfs; the second moves BestPoint to any point
%     that beats fbest
%   - First updating process: with probability p_NSM the greedy survivor test is
%     replaced by comparing the NSM scores of the parent and the candidate
%   - NSM score = w1*fitness + w2*distance from the best + w3*distance from the
%     centre of the n_centre best, each min-max normalised, L1 distances
%   - A candidate worse than the population's worst is rejected and one better
%     than its best is accepted, so the best is never replaced by a worse point
%
% Reference:
% Hamdi Tolga Kahraman, Mehmet Kati, Sefa Aras, Durdane Ayse Tasci,
% Development of the Natural Survivor Method (NSM) for designing an updating
% mechanism in metaheuristic search algorithms,
% Engineering Applications of Artificial Intelligence 122 (2023) 106121.
% https://doi.org/10.1016/j.engappai.2023.106121
% Components: base algorithm SFS --
% Salimi, H. (2015),
% Stochastic Fractal Search: A powerful metaheuristic algorithm,
% Knowledge-Based Systems, 75, 1-18.
% https://doi.org/10.1016/j.knosys.2014.07.025
% ----------------------------------------------------------------------- %
% Implementation Note:
% Follows the authors' group working copy as released (sfsNSM.m, its NSM score file
% and the group's Chebyshev map and sigmoid switch). Besides NSM, the release makes
% three changes of its own to SFS, all kept: the walk takes g = the FE count after
% initialisation in log(g)/g instead of the generation; diffusion reuses the
% parent's fitness instead of re-evaluating it; and the second updating process
% moves BestPoint to any point that beats fbest, read from the unsorted vector.
% They, not NSM, carry the F1 gain (CEC2014 D=10, 5 seeds): pool sfs+NSM 16, released 0.05.
% The release spends its 50 initial evaluations outside the budget; here one budget
% counts them, and the reported best is the running best of every evaluated point
% rather than the release's fbest.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = nsm_sfs(problem)

    % Extract problem parameters
    dim = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    % SFS Parameters
    N = 50;                       % Population size (Start_Point)
    MDN = 1;                      % Maximum Diffusion Number
    Walk = 1;                     % Walk probability

    % NSM parameters of the group's sfsNSM.m
    nsm_w = [0.9882 0.6862 0.6868];                         % fitness, distance from best, from centre
    centre_count = ceil(chebyshev_map(maxFE + 100) * 15);   % indexed by FE, 1..15

    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);

    % Initialize storage for population and fitness history
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Initialize population
    point = initialization(N, dim, ub, lb);

    % Evaluate initial population
    [fitness, FE] = calculate_fitness(point', problem, FE);

    % Sort population based on fitness
    [sorted_fitness, indices] = sort(fitness);
    point = point(indices, :);
    fitness = sorted_fitness;

    % Find initial best
    best_fitness_current = fitness(1);
    best_solution_current = point(1, :);

    % Record best fitness for each initial evaluation
    for eval_count = 1:N
        curve(eval_count) = best_fitness_current;
        [population_history, fitness_history, history_index] = record_history(...
            eval_count, point, fitness, population_history, fitness_history, ...
            history_index, maxFE);
    end

    % Walk centre of the release, carried across generations and moved in the second process
    BestPoint = point(1, :);
    while FE < maxFE

        New_Point = zeros(N, dim);
        FitVector = zeros(1, N);

        % Diffusion process occurs for all points in the group
        for i = 1:N
            if FE >= maxFE
                break;
            end
            % g is the release's post-initialisation FE counter (starts at 1), not the generation
            nfeval = FE - N + 1;
            [NP, fit, FE] = Diffusion_Process(point(i, :), fitness(i), dim, lb, ub, nfeval, MDN, Walk, BestPoint, problem, FE);
            New_Point(i, :) = NP;
            FitVector(i) = fit;

            % Update convergence curve
            if fit < best_fitness_current
                best_fitness_current = fit;
                best_solution_current = NP;
            end

            % Record for each diffusion (MDN evaluations per point; the parent is not re-evaluated)
            for eval_idx = 1:MDN
                eval_count = FE - MDN + eval_idx;
                if eval_count > 0 && eval_count <= maxFE
                    curve(eval_count) = best_fitness_current;
                end
            end
        end

        if FE >= maxFE
            break;
        end

        % Update sorting
        fit = FitVector';
        [~, sortIndex] = sort(fit);

        % Starting The First Updating Process
        Pa = zeros(1, N);
        for i = 1:N
            Pa(sortIndex(i)) = (N - i + 1) / N;
        end

        RandVec1 = randperm(N);
        RandVec2 = randperm(N);

        P = zeros(N, dim);
        for i = 1:N
            for j = 1:dim
                if rand > Pa(i)
                    P(i, j) = New_Point(RandVec1(i), j) - rand * (New_Point(RandVec2(i), j) - New_Point(i, j));
                else
                    P(i, j) = New_Point(i, j);
                end
            end
        end

        % Check bounds
        P = Bound_Checking(P, lb, ub);

        % Evaluate first process
        FE0 = FE;
        [Fit_FirstProcess, FE] = calculate_fitness(P', problem, FE);

        % Survivor selection in sequence, so each test sees the replacements before it
        for i = 1:N
            nfeval = FE0 + i - N;   % the release's counter at this decision
            if nsm_switch(maxFE, nfeval)
                [f_worst, i_worst] = max(fit);
                [f_best, i_best] = min(fit);
                if f_worst > Fit_FirstProcess(i) && nsm_accept(New_Point(i, :), fit(i), ...
                        P(i, :), Fit_FirstProcess(i), New_Point(i_worst, :), f_worst, ...
                        New_Point(i_best, :), f_best, New_Point, fit, centre_count(min(nfeval, end)), nsm_w)
                    New_Point(i, :) = P(i, :);
                    fit(i) = Fit_FirstProcess(i);
                end
            elseif Fit_FirstProcess(i) <= fit(i)
                New_Point(i, :) = P(i, :);
                fit(i) = Fit_FirstProcess(i);
            end

            % Track every evaluated candidate: an NSM survivor can be worse than its parent
            if Fit_FirstProcess(i) < best_fitness_current
                best_fitness_current = Fit_FirstProcess(i);
                best_solution_current = P(i, :);
            end
        end

        % Record convergence curve for first process evaluations
        for eval_idx = 1:N
            eval_count = FE - N + eval_idx;
            if eval_count > 0 && eval_count <= maxFE
                curve(eval_count) = best_fitness_current;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, New_Point, fit, population_history, fitness_history, ...
                    history_index, maxFE);
            end
        end

        FitVector = fit;
        fbest = FitVector(1);   % the release reads fbest before sorting, so it is row 1's fitness

        % Sort and update best point
        [~, SortedIndex] = sort(FitVector);
        New_Point = New_Point(SortedIndex, :);
        FitVector = FitVector(SortedIndex);
        BestPoint = New_Point(1, :);

        point = New_Point;
        fitness = FitVector;

        % Starting The Second Updating Process
        Pa = sort(SortedIndex / N, 'descend');

        for i = 1:N
            if FE >= maxFE
                break;
            end

            if rand > Pa(i)
                % Selecting two different points in the group
                R1 = ceil(rand * N);
                R2 = ceil(rand * N);
                while R1 == R2
                    R2 = ceil(rand * N);
                end
                if R1 == 0, R1 = 1; end
                if R2 == 0, R2 = 1; end

                if rand < 0.5
                    ReplacePoint = point(i, :) - rand * (point(R2, :) - BestPoint);
                else
                    ReplacePoint = point(i, :) + rand * (point(R2, :) - point(R1, :));
                end

                ReplacePoint = Bound_Checking(ReplacePoint, lb, ub);

                % Evaluate replacement point
                [fit_replace, FE] = calculate_fitness(ReplacePoint', problem, FE);

                % The release moves the walk centre to any replacement that beats fbest
                if fit_replace < fbest
                    fbest = fit_replace;
                    BestPoint = ReplacePoint;
                end

                if fit_replace < fitness(i)
                    point(i, :) = ReplacePoint;
                    fitness(i) = fit_replace;

                    % Update best
                    if fit_replace < best_fitness_current
                        best_fitness_current = fit_replace;
                        best_solution_current = ReplacePoint;
                    end
                end

                % Record convergence
                if FE <= maxFE
                    curve(FE) = best_fitness_current;
                    [population_history, fitness_history, history_index] = record_history(...
                        FE, point, fitness, population_history, fitness_history, ...
                        history_index, maxFE);
                end
            end
        end
    end

    % Fill remaining curve values
    for i = FE+1:maxFE
        curve(i) = best_fitness_current;
    end

    % Return best solution
    best_fitness = best_fitness_current;
    best_solution = best_solution_current;

end

% Initialization Function
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        X = zeros(SearchAgents_no, dim);
        for i = 1:dim
            X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

% Bound Checking Function
function p = Bound_Checking(p, lowB, upB)
    for i = 1:size(p, 1)
        upper = double(gt(p(i, :), upB));
        lower = double(lt(p(i, :), lowB));
        up = find(upper == 1);
        lo = find(lower == 1);
        if (size(up, 2) + size(lo, 2) > 0)
            for j = 1:size(up, 2)
                p(i, up(j)) = (upB(up(j)) - lowB(up(j))) * rand() + lowB(up(j));
            end
            for j = 1:size(lo, 2)
                p(i, lo(j)) = (upB(lo(j)) - lowB(lo(j))) * rand() + lowB(lo(j));
            end
        end
    end
end

% Diffusion Process Function
function [createPoint, best_fitness, FE] = Diffusion_Process(Point, PointFit, dim, lb, ub, g, MDN, Walk, BestPoint, problem, FE)
    % Creating new points based on diffusion process
    NumDiffusion = MDN;
    New_Point = zeros(NumDiffusion + 1, dim);
    New_Point(1, :) = Point;

    % Diffusing Part
    for i = 1:NumDiffusion
        % Consider which walks should be selected
        if rand < Walk
            % Gaussian walk 1 (Equation 11)
            sigma = (log(g) / g) * (abs(Point - BestPoint));
            sigma(sigma == 0) = 1e-10;  % Avoid zero std
            GeneratePoint = normrnd(BestPoint, sigma, [1 dim]) + (randn * BestPoint - randn * Point);
        else
            % Gaussian walk 2 (Equation 12)
            sigma = (log(g) / g) * (abs(Point - BestPoint));
            sigma(sigma == 0) = 1e-10;  % Avoid zero std
            GeneratePoint = normrnd(Point, sigma, [1 dim]);
        end
        New_Point(i + 1, :) = GeneratePoint;
    end

    % Check bounds of New Point
    New_Point = Bound_Checking(New_Point, lb, ub);

    % Evaluate only the diffused points; the parent keeps its stored fitness (as released)
    [fnew, FE] = calculate_fitness(New_Point(2:end, :)', problem, FE);
    fitness = [PointFit, fnew(:)'];

    % Find best point from diffusion; a tie keeps the parent
    [best_fitness, best_idx] = min(fitness);
    createPoint = New_Point(best_idx, :);
end

% Sigmoid_Func_2_Increase of the group's toolbox: true selects the NSM test
function use = nsm_switch(maxFE, fe)
    if maxFE < fe
        use = false;
    else
        use = 1 / (1 + exp(-(15 * fe / maxFE - 9.9))) < rand;
    end
end

% Chebyshev map of the group's chaos(1, n, 1): x1 = 0.7, x(k+1) = cos(k*acos(x(k)))
function g = chebyshev_map(n)
    g = zeros(1, n);
    x = 0.7;
    for k = 1:n
        g(k) = (x + 1) / 2;
        x = cos(k * acos(x));
    end
end

% NSM survivor test (FDDBPopulasyonGuncelleTest of the group's copy): true keeps xn over xo
function accept = nsm_accept(xo, fo, xn, fn, xw, fw, xb, fb, X, fit, n_centre, w)
    if fb == fw || fb > fn
        accept = true;
    elseif fb == fo
        accept = false;
    else
        [~, order] = sort(fit);
        n_centre = max(n_centre, 2);
        centre = sum(X(order(1:n_centre), :), 1) / n_centre;
        cand = [xo; xn; xw; xb];
        % Distance from the centre is min-max normalised over old, new, worst and best
        dc = sum(abs(cand - centre), 2)';
        dc = (dc - min(dc)) / (max(dc) - min(dc));
        % Distance from the best is divided by its maximum over old, new and worst
        db = sum(abs(cand(1:3, :) - xb), 2)';
        db = db / max(db);
        nf = 1 - ([fo, fn] - fb) / (fw - fb);
        score = w(1) * nf + w(2) * db(1:2) + w(3) * dc(1:2);
        accept = score(2) > score(1);
    end
end

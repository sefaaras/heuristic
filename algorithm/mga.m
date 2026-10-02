% ----------------------------------------------------------------------- %
% Material Generation Algorithm (MGA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NCompan = 100   % Number of "components" (population of materials)
%
% Algorithm Concept:
%   - Materials are built from components: one new material takes each
%     coordinate from a different component and adds a Gaussian perturbation
%   - A second new material is a combination of randomly chosen components
%     with random weights normalised to sum to one
%   - Both join the population and only the best NCompan materials survive
%
% Reference:
% Siamak Talatahari, Mahdi Azizi, Amir H. Gandomi,
% Material Generation Algorithm: A Novel Metaheuristic Algorithm for
% Optimization of Engineering Problems,
% Processes 9 (5) (2021) 859.
% https://doi.org/10.3390/pr9050859
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release (MGAforpublish.m, File Exchange 92065): two
% new materials per iteration, clamped to the box as whole rows. When a single
% component is drawn the release's sum runs across the dimensions, so that
% material is one repeated value; kept as released. The best-so-far is tracked
% with scalars instead of the release's per-iteration history. The first
% generation line draws one component per dimension without repeats, which
% randperm cannot satisfy once the dimension exceeds NCompan = 100 (CEC2020RW,
% D = 118..158); there the draw falls back to sampling with replacement.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = mga(problem)

    % Extract problem parameters
    Var_Number = problem.dimension;
    VarMin = problem.lb;
    VarMax = problem.ub;
    maxFE = problem.maxFe;

    NCompan = 100;

    FE = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Create initial components
    Compan.Position = unifrnd(repmat(VarMin, NCompan, 1), repmat(VarMax, NCompan, 1));
    [fit0, FE] = calculate_fitness(Compan.Position', problem, FE);
    Compan.Fun_Eval = fit0(:)';

    [bestFitnessVal, Index1] = min(Compan.Fun_Eval);
    best = Compan.Position(Index1, :);

    for eval_count = 1:min(FE, maxFE)
        curve(eval_count) = bestFitnessVal;
    end
    [population_history, fitness_history, history_index] = record_block(...
        Compan.Position, Compan.Fun_Eval, 1, min(FE, maxFE), ...
        population_history, fitness_history, history_index, maxFE);

    % Main loop (two new materials per iteration)
    while FE < maxFE
        CompnNew = zeros(2, Var_Number);

        % New component 1: mix a random component per dimension
        Index = NCompan .* (0:Var_Number - 1) + pick_components(NCompan, Var_Number);
        CompnNew(1, :) = Compan.Position(Index) + unifrnd(-1, 1) .* randn(1, Var_Number);

        % New component 2: weighted (sum-to-one) combination
        Index = randperm(NCompan, 1);
        Index2 = randperm(NCompan, Index);
        CMs = randn(Index, 1);
        CMs = CMs / sum(CMs);
        CompnNew(2, :) = sum(CMs .* Compan.Position(Index2, :));   % one row sums across dimensions, as released

        % Apply lower and upper bound limits
        CompnNew = max(CompnNew, VarMin);
        CompnNew = min(CompnNew, VarMax);

        % Evaluation, never past the budget
        nNew = min(2, maxFE - FE);
        CompnNew = CompnNew(1:nNew, :);
        [fitNew, FE] = calculate_fitness(CompnNew', problem, FE);
        fitNew = fitNew(:)';

        % Update components (keep the best NCompan)
        AllPosition = [Compan.Position; CompnNew];
        AllFun_Eval = [Compan.Fun_Eval, fitNew];
        [~, Index1] = sort(AllFun_Eval);
        Compan.Position = AllPosition(Index1(1:NCompan), :);
        Compan.Fun_Eval = AllFun_Eval(Index1(1:NCompan));

        % Update best-so-far one evaluation at a time
        for k = 1:nNew
            if fitNew(k) < bestFitnessVal
                bestFitnessVal = fitNew(k);
                best = CompnNew(k, :);
            end
            ec = FE - nNew + k;
            curve(ec) = bestFitnessVal;
            [population_history, fitness_history, history_index] = record_history(...
                ec, Compan.Position, Compan.Fun_Eval, population_history, fitness_history, ...
                history_index, maxFE);
        end
    end

    curve(min(FE, maxFE):end) = bestFitnessVal;

    best_solution = best;
    best_fitness = bestFitnessVal;
end

% One source component per dimension, distinct while the pool allows it
function idx = pick_components(NCompan, Var_Number)
    if Var_Number <= NCompan
        idx = randperm(NCompan, Var_Number);
    else
        idx = randi(NCompan, 1, Var_Number);
    end
end

% Record a fixed population snapshot over an FE block
function [pop_hist, fit_hist, hist_idx] = record_block(pop, costs, fe_from, fe_to, pop_hist, fit_hist, hist_idx, maxFE)
    if fe_to < fe_from, return; end
    for eval_count = fe_from:fe_to
        [pop_hist, fit_hist, hist_idx] = record_history(...
            eval_count, pop, costs, pop_hist, fit_hist, hist_idx, ...
            maxFE);
    end
end

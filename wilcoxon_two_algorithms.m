% WILCOXON_TWO_ALGORITHMS
% Pairwise Wilcoxon signed-rank test between two optimization algorithms.
%
% Reads results stored under:
%   results/<algorithm>/<experiment>/F<func>/run<k>/run_info.mat
%
% For every (experiment, function) pair, the runs of the two algorithms are
% paired by run index, and the Wilcoxon signed-rank test is applied. Only
% the +, =, - summary is reported (lower error/fitness is better).
%
%   +  : algorithm A is significantly better than algorithm B (p <= alpha)
%   =  : no significant difference (p > alpha) or test not applicable
%   -  : algorithm B is significantly better than algorithm A (p <= alpha)

clear; clc;

% ------------------------- Configuration -------------------------
algorithms   = {'fdb_sfs', 'sfs'}; % {A, B}: comparison direction is proposed vs baseline
base_dir     = 'results';          % root folder of saved runs
alpha        = 0.05;               % significance level
score_plus   = 5;                  % project score weight for '+' wins
score_equal  = 1;                  % project score weight for '=' ties
score_minus  = 0;                  % project score weight for '-' losses
% -----------------------------------------------------------------

assert(numel(algorithms) == 2, 'Exactly two algorithm names are required.');
alg_a = algorithms{1};
alg_b = algorithms{2};

experiments = list_subdirs(fullfile(base_dir, alg_a));
if isempty(experiments)
    fprintf('No experiments found under %s\n', fullfile(base_dir, alg_a));
    return;
end

fprintf('Wilcoxon signed-rank test: %s vs %s (alpha = %.3f)\n', alg_a, alg_b, alpha);
fprintf('%-20s %6s %6s %6s\n', 'Experiment', '+', '=', '-');
fprintf('%s\n', repmat('-', 1, 40));

grand_plus = 0;
grand_eq   = 0;
grand_min  = 0;

for ei = 1:numel(experiments)
    exp_name  = experiments{ei};
    func_nums = list_function_numbers(fullfile(base_dir, alg_a, exp_name));
    if isempty(func_nums)
        continue;
    end

    n_plus = 0;
    n_eq   = 0;
    n_min  = 0;

    for fi = 1:numel(func_nums)
        [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_nums(fi));
        sym = wilcoxon_symbol(v_a, v_b, alpha);
        switch sym
            case '+', n_plus = n_plus + 1;
            case '=', n_eq   = n_eq   + 1;
            case '-', n_min  = n_min  + 1;
        end
    end

    fprintf('%-20s %6d %6d %6d\n', exp_name, n_plus, n_eq, n_min);
    grand_plus = grand_plus + n_plus;
    grand_eq   = grand_eq   + n_eq;
    grand_min  = grand_min  + n_min;
end

fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-20s %6d %6d %6d\n', 'TOTAL', grand_plus, grand_eq, grand_min);

% --- Project score: weighted sum over all (experiment, function) outcomes ---
project_score = grand_plus  * score_plus + ...
                grand_eq    * score_equal + ...
                grand_min   * score_minus;

fprintf('\nProject score (%s vs %s): %d\n', alg_a, alg_b, project_score);

% =================== Local helper functions ===================

function names = list_subdirs(parent)
    % Return immediate subdirectory names of a folder.
    names = {};
    if ~isfolder(parent)
        return;
    end
    d = dir(parent);
    for k = 1:numel(d)
        if d(k).isdir && ~ismember(d(k).name, {'.', '..'})
            names{end+1} = d(k).name; %#ok<AGROW>
        end
    end
end

function nums = list_function_numbers(experiment_dir)
    % Extract function numbers from folders named "F<n>".
    nums = [];
    if ~isfolder(experiment_dir)
        return;
    end
    d = dir(experiment_dir);
    for k = 1:numel(d)
        if d(k).isdir
            tok = regexp(d(k).name, '^F(\d+)$', 'tokens', 'once');
            if ~isempty(tok)
                nums(end+1) = str2double(tok{1}); %#ok<AGROW>
            end
        end
    end
    nums = sort(nums);
end

function runs = list_run_indices(function_dir)
    % Extract run indices from folders named "run<k>".
    runs = [];
    d = dir(function_dir);
    for k = 1:numel(d)
        if d(k).isdir
            tok = regexp(d(k).name, '^run(\d+)$', 'tokens', 'once');
            if ~isempty(tok)
                runs(end+1) = str2double(tok{1}); %#ok<AGROW>
            end
        end
    end
    runs = sort(runs);
end

function [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_num)
    % Load metric vectors paired by run index for two algorithms.
    root_a = fullfile(base_dir, alg_a, exp_name, sprintf('F%d', func_num));
    root_b = fullfile(base_dir, alg_b, exp_name, sprintf('F%d', func_num));

    v_a = [];
    v_b = [];
    if ~isfolder(root_a) || ~isfolder(root_b)
        return;
    end

    common_runs = intersect(list_run_indices(root_a), list_run_indices(root_b));
    if isempty(common_runs)
        return;
    end

    v_a = nan(numel(common_runs), 1);
    v_b = nan(numel(common_runs), 1);
    for i = 1:numel(common_runs)
        r = common_runs(i);
        v_a(i) = read_run_metric(fullfile(root_a, sprintf('run%d', r)));
        v_b(i) = read_run_metric(fullfile(root_b, sprintf('run%d', r)));
    end

    valid = ~isnan(v_a) & ~isnan(v_b);
    v_a = v_a(valid);
    v_b = v_b(valid);
end

function value = read_run_metric(run_dir)
    % Prefer best_error; fall back to best_fitness when error is unavailable.
    value = NaN;
    info_file = fullfile(run_dir, 'run_info.mat');
    if ~isfile(info_file)
        return;
    end
    S = load(info_file, 'run_info');
    if ~isfield(S, 'run_info')
        return;
    end
    ri = S.run_info;
    if isfield(ri, 'best_error') && isnumeric(ri.best_error) && ~isnan(ri.best_error)
        value = double(ri.best_error);
    elseif isfield(ri, 'best_fitness') && isnumeric(ri.best_fitness) && ~isnan(ri.best_fitness)
        value = double(ri.best_fitness);
    end
end

function sym = wilcoxon_symbol(v_a, v_b, alpha)
    % Perform Wilcoxon signed-rank test and return +/=/- symbol.
    sym = '=';
    if numel(v_a) < 2 || numel(v_b) < 2 || numel(v_a) ~= numel(v_b)
        return;
    end

    diffs = v_a - v_b;
    if all(diffs == 0)
        return;
    end

    try
        p = signrank(v_a, v_b);
    catch
        return;
    end
    if isnan(p) || p > alpha
        return;
    end

    % Use median of differences as the direction indicator; fall back to
    % the mean when the median is exactly zero (rare with ties).
    direction = median(diffs);
    if direction == 0
        direction = mean(diffs);
    end

    if direction < 0
        sym = '+';   % A has lower (better) values
    elseif direction > 0
        sym = '-';   % B has lower (better) values
    end
end

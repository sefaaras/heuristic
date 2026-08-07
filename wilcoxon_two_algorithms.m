% ----------------------------------------------------------------------- %
% Pairwise comparison of two algorithms, over the runs saved under
% results/<algorithm>/<experiment>/F<func>/run<k>/run_info.mat. Runs are paired
% by index within each (experiment, function); lower is better.
%
%   +  A significantly better than B (p <= alpha)
%   =  no significant difference (p > alpha), or the test does not apply
%   -  B significantly better than A (p <= alpha)
%
% Unconstrained suites: Wilcoxon signed-rank on best_error. best_fitness is an
% accepted fallback there and only there -- it differs from best_error by the
% function's constant bias, so it induces the SAME ordering.
%
% CEC2020RW needs its own rule. best_error is NaN on that suite and best_fitness
% is NOT a fallback: it is the Eq.(6) scalarisation, whose reference f_worst is a
% RUNNING MAXIMUM over the run, so its value depends on what that particular run
% happened to evaluate. Two runs of the same problem carry different scales and
% the difference between them is not a difference of anything.
%
% What IS comparable there is what save_run persists per run: is_feasible,
% objective (raw f(x) of the reported solution) and constraint_violation (mean
% v(x)). The rule below follows the competition guidelines' own precedence --
% feasibility first, then objective among feasible, then violation among
% infeasible -- as two stages, so that every test runs on ONE homogeneous
% quantity rather than on a scalar invented to linearise the three:
%
%   1. Feasibility. Only the pairs where exactly one algorithm reached a
%      feasible solution carry information; an exact sign test on them (this is
%      McNemar's test) decides the function if it is significant.
%   2. Quality. Signed-rank on objective over the pairs where BOTH are
%      feasible; if there are none, on violation over the pairs where both are
%      infeasible. Comparing a feasible run's objective against an infeasible
%      run's is meaningless, so those pairs are excluded rather than folded in.
%
% RW_CRITERION picks which solution of the run is scored.
%   'reported'      the solution the algorithm returned (default; is_feasible,
%                   objective, constraint_violation).
%   'best_feasible' the best feasible point the EVALUATOR saw
%                   (best_feasible_objective), the quantity a competition table
%                   reports.
% They usually agree, because the Eq.(6) fitness ranks any feasible point below
% any infeasible one -- but f_worst grows during the run, so an infeasible point
% scored early can end up below a feasible point found later and be returned in
% its place. Seeing that is why 'reported' is the default. Violation stays the
% reported solution's under both settings; no per-run minimum is stored.
% ----------------------------------------------------------------------- %
clear; clc;

project_root = fileparts(mfilename('fullpath'));
if isempty(project_root)
    project_root = pwd;
end

algorithms   = {'fdb_ea4eig', 'ea4eig'}; % {A, B} = proposed vs baseline
base_dir     = 'results';          % root folder of saved runs
alpha        = 0.05;               % significance level
RW_CRITERION = 'reported';         % 'reported' | 'best_feasible' (see header)
rw_verbose   = false;              % print the deciding stage per RW function
score_plus   = 5;                  % project score weight for '+' wins
score_equal  = 1;                  % project score weight for '=' ties
score_minus  = 0;                  % project score weight for '-' losses
score_fdb_bonus = 10;              % bonus if the proposed algorithm uses FDB

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

    is_rw = contains(lower(exp_name), 'cec2020rw');

    for fi = 1:numel(func_nums)
        if is_rw
            [A, B] = load_paired_rw(base_dir, alg_a, alg_b, exp_name, ...
                                    func_nums(fi), RW_CRITERION);
            [sym, note] = constrained_symbol(A, B, alpha);
            if rw_verbose
                % UNRCH: rw_verbose is a configuration literal, so checkcode
                % reads this as dead when it is off. Flipping it at the top of
                % the file is the whole point.
                fprintf('    F%-3d %s  %s\n', func_nums(fi), sym, note); %#ok<UNRCH>
            end
        else
            [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_nums(fi));
            sym = wilcoxon_symbol(v_a, v_b, alpha);
        end
        switch sym
            case '+', n_plus = n_plus + 1;
            case '=', n_eq   = n_eq   + 1;
            case '-', n_min  = n_min  + 1;
        end
    end

    tag = '';
    if is_rw
        tag = sprintf('  [feasibility-first, %s]', RW_CRITERION);
    end
    fprintf('%-20s %6d %6d %6d%s\n', exp_name, n_plus, n_eq, n_min, tag);
    grand_plus = grand_plus + n_plus;
    grand_eq   = grand_eq   + n_eq;
    grand_min  = grand_min  + n_min;
end

fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-20s %6d %6d %6d\n', 'TOTAL', grand_plus, grand_eq, grand_min);

base_project_score = grand_plus  * score_plus + ...
                     grand_eq    * score_equal + ...
                     grand_min   * score_minus;

if algorithm_uses_fitness_distance_balance(project_root, alg_a)
    fitness_distance_balance_bonus = score_fdb_bonus;
else
    fitness_distance_balance_bonus = 0;
end

project_score = base_project_score + fitness_distance_balance_bonus;

fprintf('\nBase project score (%s vs %s): %d\n', alg_a, alg_b, base_project_score);
fprintf('Fitness-Distance Balance bonus: %+d\n', fitness_distance_balance_bonus);
fprintf('Project score with bonus (%s vs %s): %d\n', alg_a, alg_b, project_score);

function names = list_subdirs(parent)
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
    % Paired by run index; only runs both algorithms completed are kept.
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
    % UNCONSTRAINED suites only -- never called for CEC2020RW, which goes
    % through read_rw_record instead. best_fitness is a legitimate fallback
    % here because it differs from best_error by the function's constant bias
    % and so induces the same ordering. That is NOT true on CEC2020RW, where
    % best_fitness is a run-private scalarisation; routing that suite away from
    % this function is what keeps the fallback honest.
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

function [A, B] = load_paired_rw(base_dir, alg_a, alg_b, exp_name, func_num, criterion)
    % CEC2020RW counterpart of load_paired_runs: keeps the three comparable
    % quantities together, because feasibility and objective only mean anything
    % as a pair. A run either algorithm did not complete drops the whole pair.
    A = empty_rw(); B = empty_rw();
    root_a = fullfile(base_dir, alg_a, exp_name, sprintf('F%d', func_num));
    root_b = fullfile(base_dir, alg_b, exp_name, sprintf('F%d', func_num));
    if ~isfolder(root_a) || ~isfolder(root_b)
        return;
    end

    common_runs = intersect(list_run_indices(root_a), list_run_indices(root_b));
    for i = 1:numel(common_runs)
        r = common_runs(i);
        ra = read_rw_record(fullfile(root_a, sprintf('run%d', r)), criterion);
        rb = read_rw_record(fullfile(root_b, sprintf('run%d', r)), criterion);
        if ~ra.ok || ~rb.ok
            continue;
        end
        A = push_rw(A, ra);
        B = push_rw(B, rb);
    end
end

function s = empty_rw()
    s = struct('feas', false(0,1), 'obj', zeros(0,1), 'viol', zeros(0,1));
end

function s = push_rw(s, r)
    s.feas(end+1, 1) = r.feas;
    s.obj(end+1, 1)  = r.obj;
    s.viol(end+1, 1) = r.viol;
end

function rec = read_rw_record(run_dir, criterion)
    % ok = false marks a run that cannot be scored at all, so the PAIR is
    % dropped -- silently substituting best_fitness here is the defect this
    % whole path exists to avoid.
    rec = struct('ok', false, 'feas', false, 'obj', NaN, 'viol', NaN);
    info_file = fullfile(run_dir, 'run_info.mat');
    if ~isfile(info_file)
        return;
    end
    S = load(info_file, 'run_info');
    if ~isfield(S, 'run_info')
        return;
    end
    ri = S.run_info;

    % Violation is read from the reported solution under both criteria: no
    % per-run minimum violation is stored, and viol_mean_all averages over every
    % point evaluated, which is a search statistic and not a result.
    if isfield(ri, 'constraint_violation') && isnumeric(ri.constraint_violation) ...
            && isscalar(ri.constraint_violation)
        rec.viol = double(ri.constraint_violation);
    end

    switch lower(char(criterion))
        case 'best_feasible'
            if ~isfield(ri, 'best_feasible_objective')
                return;
            end
            o = double(ri.best_feasible_objective);
            rec.feas = isscalar(o) && ~isnan(o);   % NaN = the run never got feasible
            rec.obj  = o;
            rec.ok   = true;
        otherwise   % 'reported'
            if ~isfield(ri, 'objective') || ~isfield(ri, 'is_feasible')
                return;
            end
            rec.obj  = double(ri.objective);
            rec.feas = logical(ri.is_feasible);
            rec.ok   = isscalar(rec.obj) && isscalar(rec.feas);
    end
    if rec.ok && ~rec.feas && isnan(rec.viol)
        rec.ok = false;   % infeasible with no violation recorded: nothing to rank
    end
end

function [sym, note] = constrained_symbol(A, B, alpha)
    % Feasibility first, then a signed-rank test on ONE homogeneous quantity.
    % See the file header for why the three quantities are not collapsed into a
    % single scalar: any such scalar needs an arbitrary offset between the
    % feasible and the infeasible range, and the offset would decide functions.
    sym = '='; note = 'no comparable pairs';
    n = numel(A.feas);
    if n < 2
        return;
    end

    % --- 1. feasibility -------------------------------------------------- %
    % Concordant pairs cancel, so only the discordant ones are informative.
    a_only = nnz(A.feas & ~B.feas);
    b_only = nnz(B.feas & ~A.feas);
    nd = a_only + b_only;
    if nd > 0
        p = sign_test_p(a_only, nd);
        if ~isnan(p) && p <= alpha
            if a_only > b_only
                sym = '+';
            else
                sym = '-';
            end
            note = sprintf('feasibility %d:%d of %d pairs, p=%.3g', ...
                           a_only, b_only, n, p);
            return;
        end
    end

    % --- 2. quality ------------------------------------------------------ %
    both_f = A.feas & B.feas;
    if nnz(both_f) >= 2
        [sym, p] = paired_symbol(A.obj(both_f), B.obj(both_f), alpha);
        note = sprintf('objective over %d/%d feasible pairs, %s', ...
                       nnz(both_f), n, pstr(p, nnz(both_f)));
        return;
    end
    both_i = ~A.feas & ~B.feas;
    if nnz(both_i) >= 2
        [sym, p] = paired_symbol(A.viol(both_i), B.viol(both_i), alpha);
        note = sprintf('violation over %d/%d infeasible pairs, %s', ...
                       nnz(both_i), n, pstr(p, nnz(both_i)));
        return;
    end
    note = sprintf('%d pairs, none homogeneous', n);
end

function s = pstr(p, m)
    % A NaN p is not a failure: paired_symbol returns it when every pair is
    % identical. Worth separating from a genuine non-result, and worth saying
    % when the sample is too small to reach alpha at all -- the signed-rank
    % test's smallest two-sided p on m pairs is 2^(1-m), so at alpha = 0.05
    % nothing below m = 6 can ever be significant.
    if isnan(p)
        s = 'all pairs identical';
    elseif p > 0.05 && 2^(1 - m) > 0.05
        s = sprintf('p=%.3g (m=%d too few to reach alpha)', p, m);
    else
        s = sprintf('p=%.3g', p);
    end
end

function p = sign_test_p(k, n)
    % Exact two-sided binomial test against 0.5 -- McNemar's test on the
    % discordant pairs. Hand-rolled through gammaln rather than binocdf so the
    % script does not need the Statistics Toolbox for this stage, and so it
    % stays exact at the small discordant counts where the normal
    % approximation is worst.
    p = NaN;
    if n < 1
        return;
    end
    k = min(k, n - k);
    i = 0:k;
    lg = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1) + n * log(0.5);
    p = min(1, 2 * sum(exp(lg)));
end

function sym = wilcoxon_symbol(v_a, v_b, alpha)
    sym = paired_symbol(v_a, v_b, alpha);
end

function [sym, p] = paired_symbol(v_a, v_b, alpha)
    sym = '='; p = NaN;
    v_a = v_a(:); v_b = v_b(:);
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
        p = NaN;
        return;
    end
    if isnan(p) || p > alpha
        return;
    end

    % Median of the differences gives the direction; the mean covers the rare
    % tie where the median is exactly zero.
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

function uses_fdb = algorithm_uses_fitness_distance_balance(project_root, algorithm_name)
    % Detected by name in the source, so any spelling of the call counts.
    uses_fdb = false;
    src = resolve_algorithm_file(project_root, algorithm_name);
    if isempty(src) || ~isfile(src)
        return;
    end

    try
        text = fileread(src);
    catch
        return;
    end

    normalized = lower(regexprep(text, '[^a-zA-Z0-9]', ''));
    uses_fdb = contains(normalized, 'fitnessdistancebalance');
end

function src = resolve_algorithm_file(project_root, algorithm_name)
    candidates = {
        fullfile(project_root, 'proposed', [algorithm_name '.m'])
        fullfile(project_root, 'algorithm', [algorithm_name '.m'])
    };

    src = '';
    for i = 1:numel(candidates)
        if isfile(candidates{i})
            src = candidates{i};
            return;
        end
    end
end

function [T, L, P] = hold_summary(alg, varargin)
%HOLD_SUMMARY An algorithm's HOLD runs, as the statistics the literature reports.
%
%   hold_summary('pacede')                      prints one line per case
%   [T, L, P] = hold_summary('pacede', 'cases', 24:29)
%   hold_summary('pacede', 'cases', 12, 'target', 2174.332597)   a study's own target
%
%   T   one row per case
%       case, problem, instance, maxFE, runs   the protocol
%       found          runs on disk
%       feasible       runs whose reported solution is feasible
%       best, mean, median, worst, std
%                      objective of the reported solutions over the FEASIBLE runs
%                      (std with n - 1, as the references print it)
%       best_known, gap_pct   (best - best_known) / |best_known| * 100
%       target, success, fe_to_target
%                      runs whose best feasible objective reached the target within
%                      maxFE, and their mean FE to it
%       budget_max     largest n_eval / maxFE (above 1 = budget overspent)
%       stale          runs whose reported objective is worse than the best feasible
%                      point the evaluator saw (an algorithm that under-reports)
%       seconds        mean wall time of one run
%   L   the published results under the same protocol (problem, instance, maxFE
%       and runs all equal), from hold_literature, with the case number in front
%   P   one row per run: case, run, objective, is_feasible, n_eval, fe_to_target,
%       seconds
%
% Options
%   'cases'   default: every case with runs on disk
%   'root'    default 'results' (as in hold_run)
%   'target'  one value for all cases or one per case; NaN = the default target
%   'tol'     default 1e-4. The default target is best_known + tol * |best_known|:
%             a run succeeds when it matches the best known value to about four
%             significant digits, the precision most references print. A study
%             with its own rule (e.g. strictly below a value) is reproduced by
%             passing 'target'.
%
% T and P are also written to <root>/<alg>/hold_best/summary.csv and runs.csv.

    p = inputParser;
    p.addParameter('cases', [], @isnumeric);
    p.addParameter('root', 'results', @(x) ischar(x) || isstring(x));
    p.addParameter('target', NaN, @isnumeric);
    p.addParameter('tol', 1e-4, @(x) isscalar(x) && x >= 0);
    p.parse(varargin{:});
    o = p.Results;

    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(here));
    addpath(here);
    root = char(o.root);
    if ~java.io.File(root).isAbsolute()
        root = fullfile(repo, root);
    end
    base = fullfile(root, alg, 'hold_best');

    cases = o.cases(:)';
    if isempty(cases)
        d = dir(fullfile(base, 'F*'));
        d = d([d.isdir]);
        tok = regexp({d.name}, '^F(\d+)$', 'tokens', 'once');
        tok = tok(~cellfun(@isempty, tok));
        cases = sort(cellfun(@(c) str2double(c{1}), tok));
    end
    target = o.target(:)';
    if isscalar(target)
        target = repmat(target, 1, numel(cases));
    elseif numel(target) ~= numel(cases)
        error('hold_summary:target', 'Give one target, or one per case (%d).', numel(cases));
    end

    nc = numel(cases);
    c = struct('problem', {cell(nc, 1)}, 'instance', {cell(nc, 1)});
    num = nan(nc, 18);
    P = table();
    L = table();
    for i = 1:nc
        k = cases(i);
        par = hold_par(k);
        c.problem{i} = par.problem;
        c.instance{i} = par.instance;
        tgt = target(i);
        if isnan(tgt)
            tgt = par.best_known + o.tol * abs(par.best_known);
        end

        files = dir(fullfile(base, sprintf('F%d', k), 'run*.mat'));
        files = files(~cellfun(@isempty, regexp({files.name}, '^run\d+\.mat$', 'once')));
        n = numel(files);
        run = nan(n, 1); obj = nan(n, 1); feas = false(n, 1); ne = nan(n, 1);
        bfo = nan(n, 1); fe = nan(n, 1); sec = nan(n, 1);
        for j = 1:n
            S = load(fullfile(files(j).folder, files(j).name));
            run(j) = S.run;  obj(j) = S.objective;  feas(j) = S.is_feasible;
            ne(j) = S.n_eval;  bfo(j) = S.best_feas_obj;  sec(j) = S.seconds;
            tr = S.trace;
            hit = find(tr(:, 2) <= tgt & tr(:, 1) <= S.maxFE, 1);
            if ~isempty(hit)
                fe(j) = tr(hit, 1);
            end
        end
        [run, ix] = sort(run);
        obj = obj(ix); feas = feas(ix); ne = ne(ix); bfo = bfo(ix); fe = fe(ix); sec = sec(ix);

        of = obj(feas);
        stale = sum(~isnan(bfo) & (~feas | obj > bfo + 1e-9 * max(1, abs(bfo))));
        num(i, :) = [k, par.maxFE, par.runs, n, sum(feas), ...
                     min_or_nan(of), mean_or_nan(of), median_or_nan(of), max_or_nan(of), ...
                     std_or_nan(of), par.best_known, ...
                     (min_or_nan(of) - par.best_known) / abs(par.best_known) * 100, ...
                     tgt, sum(~isnan(fe)), mean_or_nan(fe(~isnan(fe))), ...
                     max_or_nan(ne / par.maxFE), stale, mean_or_nan(sec)];

        P = [P; table(repmat(k, n, 1), run, obj, feas, ne, fe, sec, 'VariableNames', ...
             {'case', 'run', 'objective', 'is_feasible', 'n_eval', 'fe_to_target', 'seconds'})]; %#ok<AGROW>

        Lk = hold_literature('problem', par.problem, 'instance', par.instance, ...
                             'max_fe', par.maxFE, 'runs', par.runs);
        if height(Lk) > 0
            Lk = addvars(Lk, repmat(k, height(Lk), 1), 'Before', 1, 'NewVariableNames', 'case');
            L = [L; Lk]; %#ok<AGROW>
        end
    end

    T = array2table(num, 'VariableNames', {'case', 'maxFE', 'runs', 'found', 'feasible', ...
        'best', 'mean', 'median', 'worst', 'std', 'best_known', 'gap_pct', ...
        'target', 'success', 'fe_to_target', 'budget_max', 'stale', 'seconds'});
    T = addvars(T, string(c.problem), string(c.instance), 'After', 'case', ...
                'NewVariableNames', {'problem', 'instance'});

    if exist(base, 'dir')
        writetable(T, fullfile(base, 'summary.csv'));
        writetable(P, fullfile(base, 'runs.csv'));
    end

    if nargout == 0
        fprintf('%-4s %-15s %-9s %5s %5s %14s %14s %11s %14s %7s %9s %8s\n', 'case', 'problem', ...
                'instance', 'runs', 'feas', 'best', 'mean', 'std', 'best known', 'success', ...
                'budget', 'sec/run');
        for i = 1:nc
            r = T(i, :);
            fprintf('%-4d %-15s %-9s %2d/%-2d %5d %14.8g %14.8g %11.4g %14.8g %3d/%-3d %9.4f %8.0f\n', ...
                    r.case, r.problem, r.instance, r.found, r.runs, r.feasible, r.best, r.mean, ...
                    r.std, r.best_known, r.success, r.found, r.budget_max, r.seconds);
            if r.stale > 0
                fprintf('     ^ %d run(s) report a solution worse than the best feasible point they evaluated\n', r.stale);
            end
        end
        if height(L) > 0
            fprintf('%d published results under the same protocols: [T, L] = hold_summary(...)\n', height(L));
        end
        clear T
    end
end

function v = min_or_nan(x)
    if isempty(x), v = NaN; else, v = min(x); end
end

function v = max_or_nan(x)
    if isempty(x), v = NaN; else, v = max(x); end
end

function v = mean_or_nan(x)
    if isempty(x), v = NaN; else, v = mean(x); end
end

function v = median_or_nan(x)
    if isempty(x), v = NaN; else, v = median(x); end
end

function v = std_or_nan(x)
    if numel(x) < 2, v = NaN; else, v = std(x); end
end

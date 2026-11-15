function hold_run(algorithms, varargin)
%HOLD_RUN Run algorithms on the HOLD pool, keeping what a literature comparison needs.
%
%   hold_run('pacede')                          every case, own budget and run count
%   hold_run('pacede', 'cases', 20:29)          a subset of the cases
%   hold_run('pacede', 'runs', 1:2)             the first two runs of every case
%   hold_run({'pacede', 'lshade'}, 'workers', 12, 'root', 'results')
%
% One small file per run, <root>/<alg>/hold_best/F<case>/run<k>.mat:
%
%   best_solution, best_fitness   what the algorithm reported. best_fitness is the
%                                 search value (Eq. 6, built on a run-private
%                                 reference) and is NOT comparable across runs
%   objective, is_feasible,       the reported solution re-evaluated by the problem
%   violation, max_violation      itself: THIS is what is compared with the literature
%   n_eval                        evaluations actually spent (ground truth for the budget)
%   best_feas_obj, fe_best_feas_obj   best feasible objective the evaluator saw, and
%                                 the FE it was seen at (a reported solution worse
%                                 than this is a stale report)
%   first_fe_feasible             FE of the first feasible point
%   trace                         [FE, objective], one row per improvement of the best
%                                 feasible objective: enough to compute any success
%                                 rate or FEs-to-target a study reports
%   seconds, seed, host, matlab, commit, date, and the case's identity
%                                 (case_no, run, problem, instance, D, maxFE)
%
% No population history is kept: the recorder runs with a single sample (the
% start), so a run costs what the algorithm and the problem cost. The seed is the
% run number, as in main.m, so run k is the same run however the jobs are split.
%
% Finished runs are skipped, so an interrupted call resumes where it stopped. A
% run is written under a temporary name and then renamed, so a crash cannot leave
% a half-written file that a resume would take as done. A run that fails prints
% its error and stays missing; the next call retries it.
%
% Jobs are handed out one at a time, the most expensive first: a few cases are
% long single runs (200-bar truss, CHPED at 1.6-2M FE, protein 2EWH, Balerma), and
% parfor's default chunking would queue dozens of them behind one worker.
%
% Summarise with hold_summary.

    p = inputParser;
    p.addParameter('cases', [], @isnumeric);
    p.addParameter('runs', [], @isnumeric);
    p.addParameter('root', 'results', @(x) ischar(x) || isstring(x));
    p.addParameter('workers', feature('numcores'), @(x) isscalar(x) && x >= 1);
    p.parse(varargin{:});
    o = p.Results;
    if ischar(algorithms) || isstring(algorithms)
        algorithms = cellstr(algorithms);
    end

    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(here));
    addpath(repo, fullfile(repo, 'algorithm'), here);
    root = char(o.root);
    if ~java.io.File(root).isAbsolute()
        root = fullfile(repo, root);
    end
    cases = o.cases;
    if isempty(cases)
        cases = 1:hold_par();
    end
    cases = check_networks(here, cases(:)');

    [status, commit] = system(sprintf('git -C "%s" rev-parse --short HEAD', repo));
    commit = strtrim(commit);
    if status ~= 0
        commit = '';
    end
    host = getenv('COMPUTERNAME');

    % ---- job list, finished runs skipped -----------------------------------
    jobs = struct('alg', {}, 'k', {}, 'run', {}, 'file', {}, 'cost', {}, ...
                  'commit', {}, 'host', {});
    total = 0;
    for a = 1:numel(algorithms)
        for k = cases(:)'
            par = hold_par(k);
            runs = 1:par.runs;
            if ~isempty(o.runs)
                runs = intersect(runs, o.runs(:)');
            end
            for r = runs
                total = total + 1;
                f = fullfile(root, algorithms{a}, 'hold_best', sprintf('F%d', k), ...
                             sprintf('run%d.mat', r));
                if ~exist(f, 'file')
                    jobs(end + 1) = struct('alg', algorithms{a}, 'k', k, 'run', r, ...
                        'file', f, 'cost', par.maxFE * cost_per_fe(par.problem, par.n), ...
                        'commit', commit, 'host', host); %#ok<AGROW>
                end
            end
        end
    end
    fprintf('HOLD: %d runs in %d cases, %d already done, %d to run\n', ...
            total, numel(cases), total - numel(jobs), numel(jobs));
    if isempty(jobs)
        return;
    end
    [~, order] = sort([jobs.cost], 'descend');
    jobs = jobs(order);
    fprintf('estimated serial work %.1f core-hours (rough, i9-13900K one thread)\n', ...
            sum([jobs.cost]) / 3600);

    % ---- pool, then one job per hand-out -----------------------------------
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= o.workers
        if ~isempty(pool)
            delete(pool);
        end
        pool = parpool('local', o.workers);
    end
    opts = parforOptions(pool, 'RangePartitionMethod', 'fixed', 'SubrangeSize', 1);
    parfor (i = 1:numel(jobs), opts)
        run_one(jobs(i));
    end
    fprintf('HOLD: done\n');
end

function run_one(job)
    try
        rng(job.run, 'twister');
        problem = suite_problem('hold', job.k);
        calculate_fitness('reset');
        calculate_fitness('trace_on');
        record_history('set_samples', 1);       % the start only: no history kept
        record_history('set_problem', problem);

        t = tic;
        [best_fitness, best_solution] = feval(job.alg, problem);
        seconds = toc(t);

        st = calculate_fitness('stats');
        trace = calculate_fitness('trace');
        % audit call: count_this = false keeps it out of n_eval and the trace
        [~, ~, feas, viol, obj, vmax] = calculate_fitness(best_solution(:), problem, 0, false);

        par = hold_par(job.k);
        R = struct();
        R.algorithm = job.alg;
        R.case_no = job.k;
        R.run = job.run;
        R.seed = job.run;
        R.problem = par.problem;
        R.instance = par.instance;
        R.D = par.n;
        R.maxFE = problem.maxFe;
        R.best_solution = best_solution(:)';
        R.best_fitness = best_fitness;
        R.objective = obj(1);
        R.is_feasible = logical(feas(1));
        R.violation = viol(1);
        R.max_violation = vmax(1);
        R.n_eval = st.n_eval;
        R.best_feas_obj = st.best_feas_obj;
        R.fe_best_feas_obj = st.fe_best_feas_obj;
        R.first_fe_feasible = st.first_fe_feasible;
        R.trace = trace;
        R.seconds = seconds;
        R.host = job.host;
        R.matlab = version;
        R.commit = job.commit;
        R.date = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

        d = fileparts(job.file);
        if ~exist(d, 'dir')
            mkdir(d);
        end
        [~, name, ext] = fileparts(job.file);
        tmp = fullfile(d, ['tmp_' name ext]);
        save(tmp, '-struct', 'R');
        movefile(tmp, job.file, 'f');
        fprintf('%s F%d run %d: objective %.10g, feasible %d, %d FE, %.0f s\n', ...
                job.alg, job.k, job.run, R.objective, R.is_feasible, R.n_eval, seconds);
    catch err
        fprintf(2, '%s F%d run %d FAILED: %s\n', job.alg, job.k, job.run, ...
                getReport(err, 'extended', 'hyperlinks', 'off'));
    end
end

function cases = check_networks(here, cases)
% The water-network cases need the Exeter files in wdn/ (not in the repository,
% see the README). A missing file is downloaded from Exeter once, here, before any
% worker starts. Every file is then checked against the copy the evaluator was
% validated with: a network that is missing or different is dropped from this
% call with a warning, its runs stay missing, and a later call picks them up.
    NET = {'hanoi',   'HAN.inp', '2d3ba8ada3cd566a7a7fa515ee55931f9bd7576fb52fad6094c70cc7bc6b77fe';
           'balerma', 'BIN.inp', '4fcd41b5da07b4e4b77cb90b86e97e1a6dea307a864269030d9e32172054d3af'};
    URL = 'https://www.exeter.ac.uk/v8media/universityofexeter/emps/research/cws/downloads/data/3-epanet/';
    net = cell(size(cases));
    for j = 1:numel(cases)
        par = hold_par(cases(j));
        if isfield(par, 'net')
            net{j} = par.net;
        else
            net{j} = '';
        end
    end
    drop = false(size(cases));
    for i = 1:size(NET, 1)
        use = strcmp(net, NET{i, 1});
        if ~any(use)
            continue;
        end
        f = fullfile(here, 'wdn', NET{i, 2});
        if ~exist(f, 'file')
            fprintf('%s is missing: downloading it from the Exeter CWS benchmark page\n', NET{i, 2});
            part = [f '.download'];
            try
                if ~exist(fileparts(f), 'dir')
                    mkdir(fileparts(f));
                end
                websave(part, [URL NET{i, 2}], weboptions('Timeout', 60));
                movefile(part, f, 'f');
            catch err
                if exist(part, 'file')
                    delete(part);
                end
                fprintf(2, 'download of %s failed: %s\n', NET{i, 2}, err.message);
            end
        end
        if ~exist(f, 'file')
            warning('hold_run:network', ['%s is not in %s, so cases %s are skipped. ' ...
                    'Copy it there (see README.md) and run again.'], NET{i, 2}, fileparts(f), ...
                    mat2str(cases(use)));
            drop = drop | use;
        elseif ~strcmp(sha256_file(f), NET{i, 3})
            warning('hold_run:network', ['%s differs from the file the evaluator was validated ' ...
                    'with (SHA-256 in README.md), so cases %s are skipped.'], f, mat2str(cases(use)));
            drop = drop | use;
        end
    end
    cases = cases(~drop);
end

function h = sha256_file(f)
    fid = fopen(f, 'r');
    b = fread(fid, inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    d = typecast(md.digest(typecast(b, 'int8')), 'uint8');
    h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function c = cost_per_fe(problem, D)
% Seconds per FE of one pacede run, i9-13900K, from short runs on 2026-11-15
% (protein, Balerma and Hanoi on one thread; the rest from full runs or the
% evaluator alone). Used only to hand out the expensive jobs first; a wrong value
% changes the order, never a result.
    switch problem
        case 'pv_ddm_rtc',     c = 4e-5;
        case 'protein_ab3d',   c = 1.2e-3 * (D / 63)^2.3;   % 1.2 ms at D = 63, 10.5 at 163
        case {'chped48', 'chped96'}, c = 1e-3;
        case 'truss200_freq',  c = 8e-3;
        case {'dome1180_freq', 'dome1410_freq'}, c = 14e-3;
        case 'gtop_cassini1',  c = 5e-5;
        case 'wdn_balerma',    c = 3.7e-3;
        case 'wdn_hanoi',      c = 5e-5;
        otherwise,             c = 1e-3;
    end
end

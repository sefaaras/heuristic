function main_cec2022(varargin)
% ----------------------------------------------------------------------- %
% CEC2022 campaign driver for the behavioural-fingerprint study.
%
% Runs the algorithm/ pool over the CEC2022 suite at the competition's own
% protocol -- 12 functions, D = 10 at 2e5 FE and D = 20 at 1e6 FE, 30 runs --
% and writes one results/<alg>/cec2022_<D>/F<n>/run<k>/ folder per run through
% save_run, exactly as main.m does. Nothing here changes the recorded format:
% the 44 population metrics + bsf on the fixed FE grid are what the analysis
% layer reads, addressed BY NAME from the stored 'labels'.
%
% Two deliberate differences from main.m.
%
% 1. The OUTER loop is over algorithms and the parfor is over that algorithm's
%    12*runs jobs. A whole-pool parfor interleaves every algorithm, so a hard
%    0xc0000005 leaves no usable trace of where the sweep was; per algorithm it
%    does, and 360 jobs still saturate 24 workers. Progress is appended to a
%    LOG FILE per algorithm, not just stdout -- stdout is buffered and is lost
%    when the process dies.
%
% 2. runs defaults to 21, not the 51 that experiment_factory carries for the
%    rest of the grid and not the 30 of the CEC2022 protocol. The override is
%    applied here so the shared grid definition stays untouched. 21 seeds is
%    ample for the behavioural signatures this study reads; the performance
%    table it also produces is then NOT at the competition protocol and has to
%    say so.
%
% 3. The default pool is 185 of the 198 files -- see default_pool() for the 13
%    that are out and what each costs.
%
% CHUNKING. A single MATLAB process that runs the whole pool back to back dies
% intermittently with an uncatchable access violation in MATLAB's own LXE
% thread -- accumulated process state, not any one algorithm, so try/catch
% cannot see it. Drive the full sweep as separate processes; 'dry_run' prints
% the loop for the configured chunk_size. Every chunk resumes, so a chunk that
% dies costs only the runs that were in flight.
%
% Usage
%   main_cec2022('dry_run', true)                            % plan, cost, disk
%   main_cec2022('algorithms', {'de','gwo','lshade','pso','ea4eig'}, ...
%                'runs', 3, 'experiments', {'cec2022_10'})    % pilot
%   main_cec2022('chunk', [1 25])                            % pool entries 1..25
%   main_cec2022()                                           % all, one process
% ----------------------------------------------------------------------- %

    opt = parse_options(varargin);

    % calculate_fitness cd's into problem/CEC2022 relative to the working folder
    % and save_run writes a relative results/, so the sweep only works from the
    % repository root.
    root = fileparts(mfilename('fullpath'));
    if isempty(root)
        root = pwd;
    end
    if ~strcmpi(pwd, root)
        cd(root);
        fprintf('Working folder set to %s\n', root);
    end

    addpath(fullfile(root, 'algorithm'));
    addpath(fullfile(root, 'problem', 'CEC2022'));

    names_all = resolve_algorithms(opt, root);
    [names, lo, hi] = apply_chunk(names_all, opt);

    configs = cell(size(opt.experiments));
    for e = 1:numel(opt.experiments)
        cfg = experiment_factory(opt.experiments{e});
        cfg.runs_per_experiment = opt.runs;   % CEC2022 protocol, see header
        configs{e} = cfg;
    end

    print_plan(names_all, names, lo, hi, configs, opt, root);
    if opt.dry_run
        return;
    end

    %% -------------------- pool ------------------------------------------
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= opt.workers
        if ~isempty(pool)
            fprintf('Restarting pool: %d workers -> %d\n', pool.NumWorkers, opt.workers);
            delete(pool);
        end
        pool = parpool('local', opt.workers);
    end
    % calculate_fitness cd's into problem/CEC2022 RELATIVE to the worker's own
    % folder, which is only the client's if the pool was created after the cd
    % above. Sync it rather than depend on the pool's age.
    wait(parfevalOnAll(pool, @cd, 0, root));
    fprintf('Parallel pool: %d workers\n\n', pool.NumWorkers);

    %% -------------------- provenance ------------------------------------
    pipeline_id = 'v7';
    [st, commit] = system('git rev-parse --short HEAD');
    if st ~= 0
        commit = 'nogit';
    end
    commit = strtrim(commit);
    provenance = struct('pipeline_id', pipeline_id, 'pipeline_commit', commit);
    stamp = sprintf('%s_%s_%s', pipeline_id, timestamp(), commit);

    %% -------------------- logs ------------------------------------------
    log_dir = fullfile(root, 'logs');
    if ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end
    tag = sprintf('%03d-%03d_%s', lo, hi, timestamp());
    log_file  = fullfile(log_dir, sprintf('cec2022_%s.log', tag));
    fail_file = fullfile(log_dir, sprintf('cec2022_%s_failures.csv', tag));
    logline(log_file, 'START chunk %d-%d of %d | %d workers | runs=%d samples=%d | commit %s', ...
            lo, hi, numel(names_all), pool.NumWorkers, opt.runs, opt.samples, commit);

    %% -------------------- sweep -----------------------------------------
    sweep_t0 = tic;
    done_runs = 0;
    total_pending = 0;
    for a = 1:numel(names)
        total_pending = total_pending + numel(pending_jobs(names{a}, configs, opt));
    end
    logline(log_file, 'PENDING %d runs across %d algorithms', total_pending, numel(names));

    for a = 1:numel(names)
        alg = names{a};
        write_marker(fullfile(opt.results_root, alg), pipeline_id, stamp);

        jobs = pending_jobs(alg, configs, opt);
        n = numel(jobs);
        if n == 0
            logline(log_file, '[%3d/%3d] %-16s SKIP (complete)', a, numel(names), alg);
            continue;
        end

        logline(log_file, '[%3d/%3d] %-16s START %d runs', a, numel(names), alg, n);
        alg_t0 = tic;
        fails = cell(n, 1);
        samples = opt.samples;
        rroot = opt.results_root;

        parfor j = 1:n
            jb = jobs(j);
            try
                % Seed = run index, so run k repeats identically everywhere.
                rng(jb.run, 'twister');

                problem = struct();
                problem.fhd       = str2func('cec22_func');
                problem.number    = jb.func_num;
                problem.dimension = jb.dim;
                problem.lb        = jb.lb_scalar * ones(1, jb.dim);
                problem.ub        = jb.ub_scalar * ones(1, jb.dim);
                problem.maxFe     = jb.maxFe;
                if ~strcmp(rroot, 'results')
                    problem.results_root = rroot;
                end

                % Drop the previous job's cached function and dimension.
                calculate_fitness('reset');
                % set_samples also drops the recorder's cached problem, so it
                % comes first.
                record_history('set_samples', samples);
                record_history('set_problem', problem);
                save_run('set_provenance', provenance);

                t0 = tic;
                [best_fitness, best_solution, curve, population_history, fitness_history] = ...
                    feval(alg, problem);
                exec_time = toc(t0);

                best_error = best_fitness - get_global_minimum(jb.experiment, jb.func_num);

                save_run(alg, jb.func_num, jb.run, ...
                         best_fitness, best_solution, curve, ...
                         exec_time, problem, jb.experiment, ...
                         population_history, fitness_history, ...
                         best_error, jb.id, true);

                fails{j} = '';
            catch ME
                fails{j} = sprintf('%s,%s,F%d,run%d,"%s"', ...
                                   alg, jb.experiment, jb.func_num, jb.run, ...
                                   strrep(ME.message, '"', ''''));
            end
        end

        alg_sec = toc(alg_t0);
        bad = fails(~cellfun(@isempty, fails));
        if ~isempty(bad)
            fid = fopen(fail_file, 'a');
            if fid > 0
                fprintf(fid, '%s\n', bad{:});
                fclose(fid);
            end
        end

        done_runs = done_runs + n;
        eta_sec = (toc(sweep_t0) / max(1, done_runs)) * max(0, total_pending - done_runs);
        logline(log_file, '[%3d/%3d] %-16s DONE %d runs in %s (%.2f s/run) | %d failed | ETA %s', ...
                a, numel(names), alg, n, dur(alg_sec), alg_sec / n, numel(bad), dur(eta_sec));
    end

    logline(log_file, 'END chunk %d-%d | %d runs in %s', lo, hi, done_runs, dur(toc(sweep_t0)));
    fprintf('\nLog:      %s\n', log_file);
    if exist(fail_file, 'file')
        fprintf('FAILURES: %s\n', fail_file);
    end
end

% ======================================================================= %
% options and pool resolution
% ======================================================================= %

function opt = parse_options(args)
    p = inputParser;
    p.FunctionName = 'main_cec2022';
    addParameter(p, 'experiments',  {'cec2022_10', 'cec2022_20'});
    addParameter(p, 'algorithms',   {});
    addParameter(p, 'runs',         21);
    addParameter(p, 'samples',      1000);
    addParameter(p, 'chunk',        []);
    addParameter(p, 'chunk_size',   25);
    addParameter(p, 'workers',      feature('numcores'));
    addParameter(p, 'results_root', 'results');
    addParameter(p, 'dry_run',      false);
    parse(p, args{:});
    opt = p.Results;

    if ischar(opt.experiments) || isstring(opt.experiments)
        opt.experiments = cellstr(opt.experiments);
    end
    if ischar(opt.algorithms) || isstring(opt.algorithms)
        opt.algorithms = cellstr(opt.algorithms);
    end
    opt.experiments  = cellstr(opt.experiments);
    opt.results_root = char(opt.results_root);
    opt.dry_run      = logical(opt.dry_run);

    for e = 1:numel(opt.experiments)
        if ~startsWith(lower(opt.experiments{e}), 'cec2022')
            error('main_cec2022:notCec2022', ...
                  '"%s" is not a CEC2022 experiment -- use main.m for the rest of the grid.', ...
                  opt.experiments{e});
        end
    end
end

function names = resolve_algorithms(opt, root)
    if ~isempty(opt.algorithms)
        names = opt.algorithms(:).';
    else
        names = default_pool();
    end

    % MATLAB gives the current folder precedence over the path, so an
    % old-signature copy of an algorithm left in the repo root is what feval
    % would call. Catch it here rather than as "Too many output arguments"
    % three hours into a chunk.
    alg_dir = fullfile(root, 'algorithm');
    for k = 1:numel(names)
        w = which(names{k});
        if isempty(w)
            error('main_cec2022:unknownAlgorithm', ...
                  '"%s" is not a function on the path (see algorithm/).', names{k});
        end
        if ~strcmpi(fileparts(w), alg_dir)
            error('main_cec2022:shadowed', ...
                  '"%s" resolves to %s, not algorithm/. Remove the shadowing copy.', ...
                  names{k}, w);
        end
    end
end

function names = default_pool()
% The 185 cheapest of the 198 files in algorithm/, cut at `dba` on the
% tools/pool_timing_multi.m ranking (CEC2014 D=30 / 3e5, mean over F1/F9/F23).
%
% CAVEAT ON THE CUT. That ranking is CEC2014 D=30; this campaign is CEC2022
% D=10/20. The two orderings disagree, and four of the thirteen are not
% expensive here at all -- measured per run at cec2022_10 by
% tools/cec2022_screen.m:
%
%   lfd  57.96   smo  48.13   mga  46.97   mfla 26.66   alo  25.75
%   bho  21.37   lsa  18.32   jde100 15.92 ea4eig 13.87
%   glo   7.56   olshade 5.60 cmaes 4.14   hgs   0.67
%
% Those last four are 18.0 s of the 293 s the cut saves (1.8 %), and several
% algorithms it KEEPS cost more than they do (ardo 15.85, bo 15.37, tfwo 13.38).
% hgs is the extreme case: its 46x blow-up is the O(n^2) tempPosition growth on
% CEC2014's composition functions and does not occur on CEC2022.
%
% What the cut costs the STUDY, not the clock: alo is one of the seven
% analytically-proved-non-novel negative controls, cmaes is the anchor of the
% CMA-ES positive control, ea4eig is the CEC2022 competition winner being
% dropped from a CEC2022 study, and olshade is a CEC podium entry. Adding those
% four back is 49.4 s per pool-run, i.e. ~51 min of wall clock at 21 runs.
% To do it, append them to the list below.
    names = { ...
    'abc', 'acor', 'acvo', 'aefa', 'aeo', 'aft', ...
    'agde', 'agro', 'agsk', 'ala', 'alshade', 'ao', ...
    'aoa', 'aosma', 'apgsk_imode', 'ardo', 'aro', 'aroa', ...
    'aso', 'avoa', 'ba', 'bbo', 'bco', 'bde', ...
    'bes', 'besd', 'bipop_cmaes', 'bmo', 'bo', 'boa', ...
    'bsa', 'bwo', 'bwoa', 'caoa', 'capsa', 'cddo', ...
    'cecmaes', 'ceo', 'cfo', 'cgo', 'choa', 'chsa', ...
    'clpso', 'co', 'coa', 'codede', 'colsa', 'coot', ...
    'crystal', 'cs', 'csa', 'da', 'daoa', 'dba', ...
    'ddao', 'de', 'dish', 'do', 'doa', 'dsa', ...
    'eao', 'ebocmar', 'ecpo', 'edo', 'eflo', 'efo', ...
    'elshade_spacma', 'ema', 'eo', 'epsde', 'eto', 'fa', ...
    'fbi', 'fda', 'fho', 'gbo', 'geo', 'gjo', ...
    'goa', 'gsa', 'gsk', 'gso', 'gto', 'gwca', ...
    'gwo', 'hanbo', 'hba', 'hbo', 'hfpso', 'hgso', ...
    'hho', 'hho_jos', 'hoa', 'hs', 'hses', 'hssogsa', ...
    'hyde_df', 'iao', 'igwo', 'imode', 'ipop_cmaes', 'is', ...
    'j2020', 'jade', 'jaya', 'jde', 'jso', 'keo', ...
    'koa', 'lee', 'lfo', 'lhho', 'lshade', 'lshade_cnepsin', ...
    'lshade_epsin', 'lshade_nd', 'lshade_rsp', 'lshade_spacma', 'lso', 'lsrtde', ...
    'madde', 'mao', 'mfo', 'mlshade_spa', 'mlshade_spacma', 'moa', ...
    'mora', 'mpa', 'mpso', 'mrfo', 'ms', 'msgo', ...
    'mshoa', 'mvo', 'nchho', 'nlshade_lbc', 'nlshade_rsp', 'nlshade_rsp_mid', ...
    'nmpa', 'ooa', 'pf', 'po', 'poa', 'ppa', ...
    'ppso', 'pso', 'rcga', 'rime', 'rsa', 'rso', ...
    'sa', 'sade', 'sao', 'sca', 'sdo', 'se', ...
    'sfo', 'sfs', 'shade', 'slshade_dp', 'sma', 'sns', ...
    'soa', 'sogwo', 'sos', 'spbo', 'ssa', 'sslo', ...
    'sso', 'swo', 'tfwo', 'tlabc', 'tlbo', 'tocf', ...
    'tsa', 'tso', 'ufia', 'wde', 'whoa', 'wo', ...
    'woa', 'woo', 'wsoa', 'wutp', 'yypo'};
end

function [names, lo, hi] = apply_chunk(names_all, opt)
    lo = 1;
    hi = numel(names_all);
    if ~isempty(opt.chunk)
        c = double(opt.chunk);
        lo = max(1, c(1));
        if numel(c) > 1
            hi = min(numel(names_all), c(2));
        else
            hi = lo;
        end
        if lo > hi
            error('main_cec2022:emptyChunk', 'Chunk [%d %d] selects nothing.', lo, hi);
        end
    end
    names = names_all(lo:hi);
end

% ======================================================================= %
% job construction
% ======================================================================= %

function jobs = pending_jobs(alg, configs, opt)
% Every (function, run) of this algorithm that is not already on disk. A
% truncated save leaves a 0-byte run_info.mat that exist() still sees, so the
% file SIZE is the completion test and such a job is rerun.
    proto = struct('id', 0, 'experiment', '', 'func_num', 0, 'run', 0, ...
                   'dim', 0, 'maxFe', 0, 'lb_scalar', 0, 'ub_scalar', 0);
    jobs = repmat(proto, 0, 1);
    id = 0;

    for e = 1:numel(configs)
        cfg = configs{e};
        for fi = 1:numel(cfg.function_numbers)
            fn = cfg.function_numbers(fi);
            for r = 1:cfg.runs_per_experiment
                id = id + 1;
                info = fullfile(opt.results_root, alg, cfg.name, ...
                                sprintf('F%d', fn), sprintf('run%d', r), 'run_info.mat');
                d = dir(info);
                if ~isempty(d) && d.bytes > 0
                    continue;
                end
                jb = proto;
                jb.id         = id;
                jb.experiment = cfg.name;
                jb.func_num   = fn;
                jb.run        = r;
                jb.dim        = cfg.dimensions;
                jb.maxFe      = cfg.maxFE;
                jb.lb_scalar  = cfg.bounds(1);
                jb.ub_scalar  = cfg.bounds(2);
                jobs(end + 1, 1) = jb; %#ok<AGROW>
            end
        end
    end
end

function write_marker(alg_dir, pipeline_id, stamp)
% Folder-level stamp: an empty file named <id>_<date>_<commit>. A folder with no
% marker at all predates the scheme and is v1.
    if ~exist(alg_dir, 'dir')
        mkdir(alg_dir);
    end
    m = dir(fullfile(alg_dir, '*_*'));
    m = {m(~[m.isdir]).name};
    if any(startsWith(m, [pipeline_id '_']))
        return;   % already stamped with this id; resuming
    end
    fid = fopen(fullfile(alg_dir, stamp), 'w');
    if fid > 0
        fclose(fid);
    end
end

% ======================================================================= %
% plan and cost model
% ======================================================================= %

function print_plan(names_all, names, lo, hi, configs, opt, root)
% Cost constants are the measured ones (R2024a, one core): search+eval
% 4.4e-6*(D/10)^0.65 s/FE, recorder 0.19*(D/10)^0.62 s/run at 1000 samples, and
% the 1.88 factor that turns the MEDIAN algorithm into the measured pool MEAN.
% They price a typical pool member; alo alone is ~1.08e-5 s per unit of D*maxFE
% (21 s/run at D=10, 216 s at D=20), so it dominates its own share.
    fprintf('\n=== main_cec2022 plan ===\n');
    fprintf('Pool          : %d algorithms, running %d-%d (%d)\n', ...
            numel(names_all), lo, hi, numel(names));
    fprintf('Runs per func : %d      History samples: %d\n', opt.runs, opt.samples);
    fprintf('Results root  : %s\n\n', opt.results_root);

    core_sec  = 0;
    bytes_est = 0;
    n_runs    = 0;
    fprintf('%-14s %5s %10s %6s %8s %14s %10s\n', ...
            'experiment', 'D', 'maxFE', 'funcs', 'runs', 'core-hours', 'disk');
    for e = 1:numel(configs)
        cfg = configs{e};
        nr  = numel(cfg.function_numbers) * cfg.runs_per_experiment * numel(names);
        sec = nr * est_run_seconds(cfg.dimensions, cfg.maxFE, opt.samples);
        by  = nr * est_run_bytes(cfg.dimensions, opt.samples);
        fprintf('%-14s %5d %10.3g %6d %8d %14.1f %10s\n', ...
                cfg.name, cfg.dimensions, cfg.maxFE, ...
                numel(cfg.function_numbers), nr, sec / 3600, bytes(by));
        core_sec  = core_sec + sec;
        bytes_est = bytes_est + by;
        n_runs    = n_runs + nr;
    end

    % MEASURED end-to-end throughput on this box (i9-13900K, 24 workers), not
    % worker count times an efficiency guess: 240 homogeneous jobs at
    % cec2022_10 gave 4.11x over the serial basis. The pool itself is busy
    % (15.25x parallelism on feval); the loss is a 3.71x per-worker slowdown
    % from L3/memory-bandwidth contention plus 36 % of wall spent outside feval.
    % Do NOT replace this with core_sec / workers -- that overstates by ~6x.
    % Re-measure on other hardware; a 6-process run measured only 1.12x
    % contention, so fewer workers may well be faster (unresolved).
    THROUGHPUT = 4.11;
    wall = core_sec / THROUGHPUT;
    fprintf('\nTotal         : %d runs, %.0f core-hours\n', n_runs, core_sec / 3600);
    fprintf('Wall clock    : ~%s at the measured %.2fx (%d workers)\n', ...
            dur(wall), THROUGHPUT, opt.workers);
    fprintf('Disk          : ~%s modelled', bytes(bytes_est));

    [meas, n_meas] = measured_run_bytes(opt.results_root, configs);
    if ~isnan(meas)
        fprintf('  |  ~%s projected from %d runs already on disk', ...
                bytes(meas * n_runs), n_meas);
    end
    fprintf('\n');

    free = free_bytes(root);
    if ~isnan(free)
        fprintf('Free on disk  : %s\n', bytes(free));
        if free < bytes_est * 1.2
            fprintf('WARNING: modelled footprint is within 20%% of free space.\n');
            fprintf('         Halving ''samples'' halves the dim_metrics share.\n');
        end
    end

    if opt.dry_run
        fprintf('\n--- chunked driver (bash), chunk_size = %d ---\n', opt.chunk_size);
        fprintf('for lo in');
        for k = 1:opt.chunk_size:numel(names_all)
            fprintf(' %d', k);
        end
        fprintf('; do\n');
        fprintf('  matlab -batch "main_cec2022(''chunk'', [$lo, $((lo+%d))])"\n', opt.chunk_size - 1);
        fprintf('done\n');
        fprintf('\nDry run: nothing was executed.\n\n');
    else
        fprintf('\n');
    end
end

function s = est_run_seconds(D, maxFe, samples)
    c   = 4.4e-6 * (D / 10)^0.65;
    rec = 0.19   * (D / 10)^0.62 * (samples / 1000);
    s   = (maxFe * c + rec) * 1.88;
end

function b = est_run_bytes(D, samples)
% pop_metrics is samples x 45 double; dim_metrics is samples x D x 7 with three
% layers double and four single (save_run's precision split). MATLAB's default
% save deflates, by 9-85 % depending on how fast the population changes relative
% to the sampling interval -- 0.5 is the middle of the measured band.
    raw = samples * 45 * 8 + samples * D * (3 * 8 + 4 * 4);
    b   = raw * 0.5 + 20e3;   % + run_info.mat and best_solution.mat
end

function [b, n] = measured_run_bytes(results_root, configs)
% Mean on-disk bytes of the runs already written, so the plan stops guessing
% once the pilot has landed. NaN when there are none. Walked level by level:
% dir() takes a wildcard only in the LAST path element, so a single glob across
% <alg>/<exp>/F*/run* silently returns nothing.
    LIMIT = 200;
    tot = 0;
    n = 0;
    algs = subdirs(results_root);
    for a = 1:numel(algs)
        for e = 1:numel(configs)
            funcs = subdirs(fullfile(results_root, algs{a}, configs{e}.name));
            for fi = 1:numel(funcs)
                base = fullfile(results_root, algs{a}, configs{e}.name, funcs{fi});
                runs = subdirs(base);
                for r = 1:numel(runs)
                    f = dir(fullfile(base, runs{r}, '*.mat'));
                    if isempty(f)
                        continue;
                    end
                    tot = tot + sum([f.bytes]);
                    n = n + 1;
                    if n >= LIMIT
                        b = tot / n;
                        return;
                    end
                end
            end
        end
    end
    if n == 0
        b = NaN;
    else
        b = tot / n;
    end
end

function names = subdirs(p)
    names = {};
    d = dir(p);
    if isempty(d)
        return;
    end
    d = d([d.isdir]);
    names = {d.name};
    names = names(~ismember(names, {'.', '..'}));
end

function b = free_bytes(root)
    b = NaN;
    try
        b = double(java.io.File(root).getUsableSpace());
    catch
    end
end

% ======================================================================= %
% small helpers
% ======================================================================= %

function logline(file, fmt, varargin)
% Opened and closed per line: stdout is buffered and is lost when the process
% dies, and a chunk that dies must still leave everything it recorded.
    msg = sprintf(fmt, varargin{:});
    fprintf('%s\n', msg);
    fid = fopen(file, 'a');
    if fid > 0
        fprintf(fid, '%s  %s\n', timestamp(), msg);
        fclose(fid);
    end
end

function s = timestamp()
    s = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
end

function s = dur(sec)
    if ~isfinite(sec)
        s = 'n/a';
    elseif sec < 90
        s = sprintf('%.0f s', sec);
    elseif sec < 5400
        s = sprintf('%.1f min', sec / 60);
    elseif sec < 172800
        s = sprintf('%.1f h', sec / 3600);
    else
        s = sprintf('%.1f d', sec / 86400);
    end
end

function s = bytes(b)
    u = {'B', 'KB', 'MB', 'GB', 'TB'};
    k = 1;
    while b >= 1024 && k < numel(u)
        b = b / 1024;
        k = k + 1;
    end
    s = sprintf('%.1f %s', b, u{k});
end

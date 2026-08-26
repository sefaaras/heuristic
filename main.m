% ----------------------------------------------------------------------- %
% Campaign driver: builds the algorithm x experiment x function x run job
% grid, skips the jobs already on disk and runs the rest under parfor.
%
% pipeline_id stamps every results/<alg>/ folder and every run_info, so a run
% can be told apart from one an earlier pipeline produced. Folders carrying an
% older id hold a narrower metric row and are not comparable with this one.
%
% History rows sit on a fixed FE grid anchored on the RIGHT edge of each cell:
% row h is the first sample at or after h*step, so h/T is the fraction of the
% budget spent and row T is the state at max_fe. The row is 44 columns; read
% them by name from the stored 'labels', never by index (see metric_labels.m).
% ----------------------------------------------------------------------- %
clear; clc; close all;
addpath('algorithm');
addpath('problem/CEC2014');
addpath('problem/CEC2017');
addpath('problem/CEC2020');
addpath('problem/CEC2020RW');
addpath('problem/CEDP');
addpath('problem/CEC2021');
addpath('problem/CEC2022');

algorithms = {''};

HISTORY_SAMPLES = 1000;

% An unknown name builds a job grid that can only fail, and fullfile('results', '')
% resolves to results/ itself, so the stamp below would land in the results root.
for alg_idx = 1:length(algorithms)
    alg_check = char(algorithms{alg_idx});
    if isempty(alg_check) || exist(alg_check, 'file') ~= 2
        error('main:unknownAlgorithm', ...
              'algorithms{%d} = "%s" is not a function on the path (see algorithm/).', ...
              alg_idx, alg_check);
    end
end

num_cores = feature('numcores');
fprintf('Detected %d CPU cores\n', num_cores);

optimal_workers = max(4, num_cores);
fprintf('Using %d parallel workers\n', optimal_workers);

poolobj = gcp('nocreate');
if isempty(poolobj)
    fprintf('Starting parallel pool...\n');
    poolobj = parpool('local', optimal_workers);
else
    if poolobj.NumWorkers ~= optimal_workers
        fprintf('Current pool has %d workers, but we need %d. Restarting pool...\n', poolobj.NumWorkers, optimal_workers);
        delete(poolobj);
        poolobj = parpool('local', optimal_workers);
    else
        fprintf('Parallel pool already running with %d workers (optimal)\n', poolobj.NumWorkers);
    end
end

all_experiments = {
    'cec2014_10', 'cec2014_30', 'cec2014_50', 'cec2014_100', ...
    'cec2017_10', 'cec2017_30', 'cec2017_50', 'cec2017_100', ...
    'cec2020_5', 'cec2020_10', 'cec2020_15', 'cec2020_20', ...
    'cec2020rw', ...
    'cedp', ...
    'cec2021_10', 'cec2021_20', ...
    'cec2022_10', 'cec2022_20'
};

total_jobs = 0;
for exp_idx = 1:length(all_experiments)
    experiment_name = all_experiments{exp_idx};
    config = experiment_factory(experiment_name);
    jobs_for_this_exp = length(algorithms) * length(config.function_numbers) * config.runs_per_experiment;
    fprintf('  %s: %d algorithms x %d functions x %d runs = %d jobs\n', ...
            experiment_name, length(algorithms), length(config.function_numbers), config.runs_per_experiment, jobs_for_this_exp);
    total_jobs = total_jobs + jobs_for_this_exp;
end
fprintf('Preallocating for %d total jobs...\n', total_jobs);

all_jobs = repmat(struct('id', 0, 'exp_idx', 0, 'experiment_name', '', 'alg_name', '', ...
                        'func_num', 0, 'func_idx', 0, 'run', 0, 'config', struct()), total_jobs, 1);

job_id = 0;

for exp_idx = 1:length(all_experiments)
    experiment_name = all_experiments{exp_idx};
    config = experiment_factory(experiment_name);

    for alg_idx = 1:length(algorithms)
        alg_name = algorithms{alg_idx};

        for func_idx = 1:length(config.function_numbers)
            func_num = config.function_numbers(func_idx);

            for run_idx = 1:config.runs_per_experiment
                job_id = job_id + 1;

                all_jobs(job_id).id = job_id;
                all_jobs(job_id).exp_idx = exp_idx;
                all_jobs(job_id).experiment_name = experiment_name;
                all_jobs(job_id).alg_name = alg_name;
                all_jobs(job_id).func_num = func_num;
                all_jobs(job_id).func_idx = func_idx;
                all_jobs(job_id).run = run_idx;
                all_jobs(job_id).config = config;
            end
        end
    end
end

fprintf('Total jobs created: %d\n', total_jobs);

% Folder-level stamp: an empty file named <id>_<date>_<commit>. A folder with
% no marker at all predates the scheme and is v1.
% Bump this and nothing else here; what the id means, and what it invalidates,
% is recorded in pipeline_versions.m.
pipeline_id = 'v8';
[git_status, pipeline_commit] = system('git rev-parse --short HEAD');
if git_status ~= 0
    pipeline_commit = 'nogit';
end
pipeline_commit = strtrim(pipeline_commit);
stamp_name = sprintf('%s_%s_%s', pipeline_id, ...
                     char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss')), ...
                     pipeline_commit);
for alg_idx = 1:length(algorithms)
    alg_dir = fullfile('results', algorithms{alg_idx});
    if ~exist(alg_dir, 'dir')
        mkdir(alg_dir);
    end
    marks = dir(fullfile(alg_dir, '*_*'));
    marks = {marks(~[marks.isdir]).name};
    if any(startsWith(marks, [pipeline_id '_']))
        continue;   % already stamped with this id; resuming a run, leave it be
    end
    if ~isempty(marks)
        % A folder carrying two ids is a mixed folder and has to stay visible.
        fprintf('NOTE: %s already carries %s -- adding %s alongside it\n', ...
                alg_dir, strjoin(marks, ', '), stamp_name);
    end
    fid = fopen(fullfile(alg_dir, stamp_name), 'w');
    if fid > 0
        fclose(fid);
    end
end

% Repeated into every run_info, so a single run traces back to the code that
% produced it and not just to the folder-level stamp.
provenance = struct('pipeline_id', pipeline_id, 'pipeline_commit', pipeline_commit);

fprintf('\nChecking for completed jobs...\n');
jobs_to_run = false(total_jobs, 1);
completed_count = 0;

for job_idx = 1:total_jobs
    job = all_jobs(job_idx);

    base_dir = 'results';
    run_dir = fullfile(base_dir, job.alg_name, job.experiment_name, ...
                      sprintf('F%d', job.func_num), sprintf('run%d', job.run));
    info_file = fullfile(run_dir, 'run_info.mat');

    % A truncated save leaves a 0-byte run_info.mat that exist() still sees, so
    % the file SIZE is the completion test and such a job is rerun.
    fi = dir(info_file);
    if ~isempty(fi) && fi.bytes > 0
        completed_count = completed_count + 1;
    else
        jobs_to_run(job_idx) = true;
    end
end

remaining_jobs = sum(jobs_to_run);
fprintf('Found %d completed jobs, %d remaining jobs to run\n', completed_count, remaining_jobs);

if remaining_jobs == 0
    fprintf('All jobs are already completed!\n');
    return;
end

jobs_array = all_jobs(jobs_to_run);

fprintf('\nStarting parallel execution of %d remaining jobs...\n', remaining_jobs);

parfor job_idx = 1:remaining_jobs
    job = jobs_array(job_idx);

    try
        % Seed = run index, so run k repeats identically across every experiment.
        rng(job.run, 'twister');

        % Evaluator, box and budget for this (experiment, function). Shared
        % with redo_runs and verify_campaign so a repaired or verified run is
        % measured against exactly the grid the campaign ran.
        problem = suite_problem(job.experiment_name, job.func_num);

        fprintf('Running Job %d (of %d remaining): %s | %s | F%d | Run %d (seed=%d)\n', ...
                job_idx, remaining_jobs, job.experiment_name, job.alg_name, ...
                job.func_num, job.run, job.run);

        % Drop the previous job's function, dimension and constraint-suite cache.
        calculate_fitness('reset');

        % set_samples also drops the recorder's cached problem, so it comes first.
        record_history('set_samples', HISTORY_SAMPLES);

        % Bounds for the box-normalised diversity metrics; on RW also the handle
        % the recorder uses to read the population's constraint violation.
        record_history('set_problem', problem);

        % Push the code identity onto THIS worker
        save_run('set_provenance', provenance);

        run_start = tic;
        [best_fitness, best_solution, curve, population_history, fitness_history] = feval(job.alg_name, problem);
        exec_time = toc(run_start);

        fprintf('Job %d (Exp:%s, Alg:%s, F%d, Run%d) completed in %.2f sec (fitness: %.6e)\n', ...
                job.id, job.experiment_name, job.alg_name, job.func_num, job.run, exec_time, best_fitness);

        if ~(contains(job.experiment_name, 'cec2020rw') || strcmpi(job.experiment_name, 'cedp'))
            global_min = get_global_minimum(job.experiment_name, job.func_num);
            best_error = best_fitness - global_min;
        else
            best_error = NaN;
        end

        % Recover the raw f(x) and the mean violation v(x) of the reported solution.
        % Constrained suites only: elsewhere f IS the objective and nothing is violated.
        if contains(job.experiment_name, 'cec2020rw') || strcmpi(job.experiment_name, 'cedp')
            % count_this = false: an audit call must not enter the FE counters
            [~, ~, is_feasible, violation, objective, max_violation] = ...
                calculate_fitness(best_solution(:), problem, 0, false);
            is_feasible   = is_feasible(1);
            violation     = violation(1);
            objective     = objective(1);
            max_violation = max_violation(1);
        else
            is_feasible   = true;   % Other problems are unconstrained
            violation     = 0;
            objective     = best_fitness;
            max_violation = 0;
        end

        if contains(job.experiment_name, 'cec2020rw') || strcmpi(job.experiment_name, 'cedp')
            save_run(job.alg_name, job.func_num, job.run, ...
                     best_fitness, best_solution, curve, ...
                     exec_time, problem, job.experiment_name, ...
                     population_history, fitness_history, ...
                     job.id, is_feasible, violation, objective, max_violation);
        else
            save_run(job.alg_name, job.func_num, job.run, ...
                     best_fitness, best_solution, curve, ...
                     exec_time, problem, job.experiment_name, ...
                     population_history, fitness_history, ...
                     best_error, job.id, is_feasible);
        end

    catch ME
        % A failed job is logged and skipped; the sweep carries on.
        fprintf('Job %d (Exp:%s, Alg:%s, F%d, Run%d) failed: %s\n', ...
                 job.id, job.experiment_name, job.alg_name, ...
                 job.func_num, job.run, ME.message);

        for k = 1:length(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
end

fprintf('\n=== Execution Complete ===\n');
fprintf('Completed %d new jobs\n', remaining_jobs);
fprintf('Total completed jobs: %d / %d\n', completed_count + remaining_jobs, total_jobs);

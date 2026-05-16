% VALIDATE_AND_PACKAGE_TWO_ALGORITHMS
% Time-check, score, and package a two-algorithm submission.
%
% The script samples one random function from each of these four pools:
%   cec2021_10, cec2021_20, cec2022_10, cec2022_20
%
% It first checks every saved run_info.max_fe against experiment_factory.
% If the check is valid, it runs the configured algorithms with the maxFE
% values from experiment_factory, estimates the full four-experiment
% completion time, computes the Wilcoxon-style project score from results/,
% and creates one submission zip containing only the two algorithm files and
% the generated reports, including all runs' best fitness values.

clear; clc;

project_root = fileparts(mfilename('fullpath'));
if isempty(project_root)
    project_root = pwd;
end
cd(project_root);
setup_paths(project_root);

% ------------------------- Configuration -------------------------
algorithms = {'fdb_ea4eig', 'ea4eig'};   % two algorithm files to compare and package
developed_algorithms = algorithms;       % algorithm files to include in the zip

base_dir = 'results';
experiments = {'cec2021_10', 'cec2021_20', 'cec2022_10', 'cec2022_20'};

sample_functions_per_pool = 1;           % 1 from each pool => 4 random problems
sample_seed = [];                        % [] => random seed each run; set a number to reproduce
use_parallel_timing = true;
save_timing_runs = false;                % true stores timing samples under results/
sample_run_number_base = 900000;         % used only when save_timing_runs = true

minimum_time_hours = 1.0;
alpha = 0.05;
score_plus = 5;
score_equal = 1;
score_minus = 0;
score_fdb_bonus = 10;

% -----------------------------------------------------------------

assert(numel(algorithms) == 2, 'Exactly two algorithm names are required.');
algorithm_sources = validate_algorithm_files(project_root, algorithms);
print_algorithm_sources(algorithm_sources);

if isempty(sample_seed)
    rng('shuffle');
    sample_seed = randi(2^31 - 1);
end

num_cores = detect_num_cores();
fprintf('Algorithms: %s vs %s\n', algorithms{1}, algorithms{2});
fprintf('Detected CPU cores: %d\n', num_cores);

[fe_check_table, fe_check_summary] = validate_result_max_fe(base_dir, algorithms, experiments, sample_run_number_base);
print_fe_check_summary(fe_check_summary, fe_check_table);
if ~fe_check_summary.is_valid
    warn_result_validation_blocked(fe_check_summary);
    return;
end

fprintf('\nMaxFE validation passed. Starting timing samples.\n');
fprintf('Timing sample seed: %d\n\n', sample_seed);

timing_jobs = build_timing_jobs(algorithms, experiments, sample_functions_per_pool, sample_seed);
fprintf('Timing samples: %d algorithms x %d selected problems = %d jobs\n', ...
        numel(algorithms), numel(experiments) * sample_functions_per_pool, numel(timing_jobs));

timing_results = run_timing_jobs(timing_jobs, project_root, ...
                                 use_parallel_timing, save_timing_runs, ...
                                 sample_run_number_base, num_cores);
timing_table = timing_results_to_table(timing_results);
print_timing_results(timing_results);

[estimate_table, estimate_summary] = estimate_completion_time( ...
    timing_results, algorithms, experiments, num_cores, minimum_time_hours);

fprintf('\nEstimated full four-experiment time: %.2f hours\n', estimate_summary.estimated_total_hours);
fprintf('Minimum required time: %.2f hours\n', minimum_time_hours);
if estimate_summary.is_valid
    fprintf('Time validation: VALID\n');
else
    fprintf('Time validation: NOT VALID\n');
end

score_summary = compute_wilcoxon_score(base_dir, algorithms{1}, algorithms{2}, ...
                                       experiments, alpha, ...
                                       score_plus, score_equal, score_minus, ...
                                       project_root, score_fdb_bonus);
print_score_summary(score_summary, algorithms, alpha);

overall_validation = estimate_summary.is_valid && fe_check_summary.is_valid;
if overall_validation
    fprintf('\nOverall validation: VALID\n');
else
    fprintf('\nOverall validation: NOT VALID\n');
end

if overall_validation
    package_info = create_submission_package(project_root, base_dir, algorithms, ...
        developed_algorithms, experiments, ...
        timing_table, estimate_table, estimate_summary, ...
        score_summary, fe_check_table, fe_check_summary, overall_validation);

    fprintf('Submission zip:\n  %s\n', package_info.zip_file);
else
    warn_submission_blocked(estimate_summary, fe_check_summary);
end

% =================== Local helper functions ===================

function setup_paths(project_root)
    addpath(fullfile(project_root, 'algorithm'));
    addpath(fullfile(project_root, 'proposed'));
    addpath(fullfile(project_root, 'problem', 'CEC2014'));
    addpath(fullfile(project_root, 'problem', 'CEC2017'));
    addpath(fullfile(project_root, 'problem', 'CEC2020'));
    addpath(fullfile(project_root, 'problem', 'CEC2020RW'));
    addpath(fullfile(project_root, 'problem', 'CEC2021'));
    addpath(fullfile(project_root, 'problem', 'CEC2022'));
end

function num_cores = detect_num_cores()
    try
        num_cores = feature('numcores');
    catch
        num_cores = 1;
    end
    if isempty(num_cores) || ~isfinite(num_cores) || num_cores < 1
        num_cores = 1;
    end
    num_cores = max(1, floor(num_cores));
end

function jobs = build_timing_jobs(algorithms, experiments, sample_count, sample_seed)
    rng(sample_seed, 'twister');
    jobs = repmat(empty_timing_job(), 0, 1);
    job_id = 0;

    for ei = 1:numel(experiments)
        exp_name = experiments{ei};
        config = experiment_factory(exp_name);
        funcs = config.function_numbers;
        pick_count = min(sample_count, numel(funcs));
        picked_funcs = funcs(randperm(numel(funcs), pick_count));

        for fi = 1:numel(picked_funcs)
            func_num = picked_funcs(fi);
            problem_seed = sample_seed + ei * 10000 + func_num * 100 + fi;

            for ai = 1:numel(algorithms)
                job_id = job_id + 1;
                job = empty_timing_job();
                job.id = job_id;
                job.experiment_name = exp_name;
                job.alg_name = algorithms{ai};
                job.func_num = func_num;
                job.config = config;
                job.seed = problem_seed;
                jobs(job_id, 1) = job;
            end
        end
    end
end

function job = empty_timing_job()
    job = struct('id', 0, ...
                 'experiment_name', '', ...
                 'alg_name', '', ...
                 'func_num', 0, ...
                 'config', struct(), ...
                 'seed', 0);
end

function algorithm_sources = validate_algorithm_files(project_root, algorithms)
    algorithm_sources = cell(numel(algorithms), 1);
    missing = {};

    for i = 1:numel(algorithms)
        src = resolve_algorithm_file(project_root, algorithms{i});
        algorithm_sources{i} = src;
        if isempty(src)
            missing{end+1} = algorithms{i}; %#ok<AGROW>
        end
    end

    if ~isempty(missing)
        error('Algorithm file(s) not found under proposed/ or algorithm/: %s', ...
              strjoin(missing, ', '));
    end
end

function print_algorithm_sources(algorithm_sources)
    fprintf('Algorithm source files:\n');
    for i = 1:numel(algorithm_sources)
        fprintf('  %s\n', algorithm_sources{i});
    end
end

function results = run_timing_jobs(jobs, project_root, use_parallel, ...
                                   save_runs, run_number_base, num_cores)
    results = repmat(empty_timing_result(), numel(jobs), 1);
    run_parallel = false;

    if use_parallel && numel(jobs) > 1
        requested_workers = min(num_cores, numel(jobs));
        run_parallel = prepare_parallel_pool(requested_workers);
    end

    if run_parallel
        fprintf('Running timing samples in parallel...\n');
        parfor ji = 1:numel(jobs)
            results(ji) = run_timing_job(jobs(ji), project_root, save_runs, run_number_base);
        end
    else
        fprintf('Running timing samples serially...\n');
        for ji = 1:numel(jobs)
            results(ji) = run_timing_job(jobs(ji), project_root, save_runs, run_number_base);
        end
    end
end

function ok = prepare_parallel_pool(requested_workers)
    ok = false;
    try
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            parpool('local', requested_workers);
        end
        ok = true;
    catch ME
        warning('validate_and_package:ParallelPoolUnavailable', ...
                'Parallel pool could not be started: %s', ME.message);
    end
end

function result = empty_timing_result()
    result = struct('id', 0, ...
                    'experiment_name', '', ...
                    'alg_name', '', ...
                    'func_num', 0, ...
                    'dimension', NaN, ...
                    'maxFE', NaN, ...
                    'seed', NaN, ...
                    'duration_seconds', NaN, ...
                    'best_fitness', NaN, ...
                    'best_error', NaN, ...
                    'success', false, ...
                    'message', '');
end

function result = run_timing_job(job, project_root, save_runs, run_number_base)
    result = empty_timing_result();
    result.id = job.id;
    result.experiment_name = job.experiment_name;
    result.alg_name = job.alg_name;
    result.func_num = job.func_num;
    result.seed = job.seed;

    original_dir = pwd;
    cleanup = onCleanup(@() cd(original_dir));

    try
        cd(project_root);
        setup_paths(project_root);
        rng(job.seed, 'twister');

        problem = create_problem(job.experiment_name, job.config, job.func_num);
        result.dimension = problem.dimension;
        result.maxFE = problem.maxFe;

        fprintf('Timing %s | %s | F%d | D%d | maxFE=%d\n', ...
                job.alg_name, job.experiment_name, job.func_num, ...
                problem.dimension, problem.maxFe);

        run_start = tic;
        [best_fitness, best_solution, curve, population_history, fitness_history] = ...
            feval(job.alg_name, problem);
        duration_seconds = toc(run_start);

        best_error = best_fitness - get_global_minimum(job.experiment_name, job.func_num);

        result.duration_seconds = duration_seconds;
        result.best_fitness = best_fitness;
        result.best_error = best_error;
        result.success = true;
        result.message = 'ok';

        if save_runs
            run_num = run_number_base + job.id;
            save_run(job.alg_name, job.func_num, run_num, ...
                     best_fitness, best_solution, curve, duration_seconds, ...
                     problem, job.experiment_name, population_history, ...
                     fitness_history, best_error, job.id, true);
        end

        fprintf('Done %s | %s | F%d in %.2f sec\n', ...
                job.alg_name, job.experiment_name, job.func_num, duration_seconds);
    catch ME
        result.success = false;
        result.message = get_error_text(ME);
        fprintf('FAILED %s | %s | F%d: %s\n', ...
                job.alg_name, job.experiment_name, job.func_num, ME.message);
    end
end

function problem = create_problem(experiment_name, config, func_num)
    problem = struct();
    lower_name = lower(experiment_name);

    if contains(lower_name, 'cec2021')
        problem.fhd = str2func('cec21_basic_func');
    elseif contains(lower_name, 'cec2022')
        problem.fhd = str2func('cec22_func');
    elseif contains(lower_name, 'cec2020rw')
        problem.fhd = str2func('cec20rw_func');
    elseif contains(lower_name, 'cec2020')
        problem.fhd = str2func('cec20_func');
    elseif contains(lower_name, 'cec2017')
        problem.fhd = str2func('cec17_func');
    elseif contains(lower_name, 'cec2014')
        problem.fhd = str2func('cec14_func');
    else
        error('Unknown experiment name: %s', experiment_name);
    end

    if isfield(config, 'use_cal_par') && config.use_cal_par
        par = Cal_par(func_num);
        problem.dimension = par.n;
        problem.lb = par.xmin;
        problem.ub = par.xmax;
        problem.maxFe = max_fe_for_real_world(problem.dimension);
    else
        problem.dimension = config.dimensions;
        problem.lb = config.bounds(1) * ones(1, config.dimensions);
        problem.ub = config.bounds(2) * ones(1, config.dimensions);
        problem.maxFe = config.maxFE;
    end

    problem.number = func_num;
end

function maxFE = max_fe_for_real_world(D)
    if D <= 10
        maxFE = 1e5;
    elseif D <= 30
        maxFE = 2e5;
    elseif D <= 50
        maxFE = 4e5;
    elseif D <= 150
        maxFE = 8e5;
    else
        maxFE = 1e6;
    end
end

function T = timing_results_to_table(results)
    T = table([results.id]', {results.experiment_name}', {results.alg_name}', ...
              [results.func_num]', [results.dimension]', [results.maxFE]', ...
              [results.seed]', [results.duration_seconds]', ...
              [results.best_fitness]', [results.best_error]', ...
              [results.success]', {results.message}', ...
              'VariableNames', {'JobID', 'Experiment', 'Algorithm', ...
                                'Function', 'Dimension', 'MaxFE', 'Seed', ...
                                'Seconds', 'BestFitness', 'BestError', ...
                                'Success', 'Message'});
end

function print_timing_results(results)
    fprintf('\nTiming result summary\n');
    fprintf('%-14s %-16s %4s %8s %12s %10s\n', ...
            'Algorithm', 'Experiment', 'F', 'D', 'Seconds', 'Status');
    fprintf('%s\n', repmat('-', 1, 72));
    for i = 1:numel(results)
        if results(i).success
            status = 'ok';
        else
            status = 'failed';
        end
        fprintf('%-14s %-16s %4d %8g %12.2f %10s\n', ...
                results(i).alg_name, results(i).experiment_name, ...
                results(i).func_num, results(i).dimension, ...
                results(i).duration_seconds, status);
    end
end

function warn_submission_blocked(estimate_summary, fe_check_summary)
    reasons = {};

    if ~fe_check_summary.is_valid
        reasons{end+1} = sprintf(['Result validation failed (%d invalid algorithm-experiment group(s), ' ...
                                  '%d missing result run(s)).'], ...
                                 fe_check_summary.invalid_group_count, ...
                                 fe_check_summary.missing_result_runs);
    end

    if ~estimate_summary.is_valid
        if estimate_summary.failed_sample_count > 0
            reasons{end+1} = sprintf('Timing validation failed because %d timing sample(s) failed.', ...
                                     estimate_summary.failed_sample_count);
        end

        if isfinite(estimate_summary.estimated_total_hours) && ...
                estimate_summary.estimated_total_hours < estimate_summary.minimum_time_hours
            reasons{end+1} = sprintf(['Timing validation failed because the estimated experimental duration ' ...
                                      'is too short: %.2f hours < %.2f required hours.'], ...
                                     estimate_summary.estimated_total_hours, ...
                                     estimate_summary.minimum_time_hours);
        elseif ~isfinite(estimate_summary.estimated_total_hours)
            reasons{end+1} = 'Timing validation failed because the estimated experimental duration is not finite.';
        end
    end

    if isempty(reasons)
        reasons{1} = 'Overall validation failed.';
    end

    warning('Submission was not created. %s', strjoin(reasons, ' '));
end

function warn_result_validation_blocked(fe_check_summary)
    reasons = {};

    if fe_check_summary.missing_result_runs > 0
        reasons{end+1} = sprintf('Missing result run(s): %d.', ...
                                 fe_check_summary.missing_result_runs);
    end
    if fe_check_summary.mismatched_runs > 0
        reasons{end+1} = sprintf('MaxFE mismatch run(s): %d.', ...
                                 fe_check_summary.mismatched_runs);
    end
    if fe_check_summary.missing_fe_runs > 0
        reasons{end+1} = sprintf('Missing/invalid FE metadata run(s): %d.', ...
                                 fe_check_summary.missing_fe_runs);
    end
    if fe_check_summary.unknown_function_runs > 0
        reasons{end+1} = sprintf('Unknown function run(s): %d.', ...
                                 fe_check_summary.unknown_function_runs);
    end

    if isempty(reasons)
        reasons{1} = 'Result validation failed.';
    end

    warning('Result validation failed. Timing runs and packaging were stopped. %s', ...
            strjoin(reasons, ' '));
end

function [estimate_table, summary] = estimate_completion_time(results, algorithms, experiments, workers, minimum_hours)
    times = [results.duration_seconds];
    success = [results.success] & isfinite(times) & times > 0;
    failed_count = sum(~[results.success]);

    if any(success)
        average_seconds = mean(times(success));
    else
        average_seconds = NaN;
    end

    exp_col = {};
    funcs_col = [];
    runs_col = [];
    algs_col = [];
    jobs_col = [];
    avg_col = [];
    sec_col = [];
    hour_col = [];

    total_jobs = 0;
    result_experiments = {results.experiment_name};

    for ei = 1:numel(experiments)
        exp_name = experiments{ei};
        config = experiment_factory(exp_name);
        exp_jobs = numel(algorithms) * numel(config.function_numbers) * config.runs_per_experiment;
        total_jobs = total_jobs + exp_jobs;

        exp_success = success & strcmp(result_experiments, exp_name);
        if any(exp_success)
            exp_average_seconds = mean(times(exp_success));
        else
            exp_average_seconds = average_seconds;
        end

        estimated_seconds = (exp_jobs / workers) * exp_average_seconds;

        exp_col{end+1, 1} = exp_name; %#ok<AGROW>
        funcs_col(end+1, 1) = numel(config.function_numbers); %#ok<AGROW>
        runs_col(end+1, 1) = config.runs_per_experiment; %#ok<AGROW>
        algs_col(end+1, 1) = numel(algorithms); %#ok<AGROW>
        jobs_col(end+1, 1) = exp_jobs; %#ok<AGROW>
        avg_col(end+1, 1) = exp_average_seconds; %#ok<AGROW>
        sec_col(end+1, 1) = estimated_seconds; %#ok<AGROW>
        hour_col(end+1, 1) = estimated_seconds / 3600; %#ok<AGROW>
    end

    estimated_total_seconds = (total_jobs / workers) * average_seconds;

    estimate_table = table(exp_col, funcs_col, runs_col, algs_col, jobs_col, ...
                           avg_col, sec_col, hour_col, ...
                           'VariableNames', {'Experiment', 'Functions', ...
                                             'RunsPerFunction', 'Algorithms', ...
                                             'Jobs', 'AvgSampleSeconds', ...
                                             'EstimatedSeconds', 'EstimatedHours'});

    summary = struct();
    summary.workers = workers;
    summary.total_jobs = total_jobs;
    summary.average_sample_seconds = average_seconds;
    summary.estimated_total_seconds = estimated_total_seconds;
    summary.estimated_total_hours = estimated_total_seconds / 3600;
    summary.minimum_time_hours = minimum_hours;
    summary.time_limit_hours = Inf;
    summary.failed_sample_count = failed_count;
    summary.is_valid = isfinite(estimated_total_seconds) && ...
                       estimated_total_seconds >= minimum_hours * 3600 && ...
                       failed_count == 0;
end

function summary = compute_wilcoxon_score(base_dir, alg_a, alg_b, experiments, ...
                                          alpha, score_plus, score_equal, score_minus, ...
                                          project_root, score_fdb_bonus)
    exp_col = {};
    plus_col = [];
    eq_col = [];
    minus_col = [];
    problems_col = [];
    score_col = [];

    func_exp_col = {};
    func_col = [];
    runs_col = [];
    p_col = [];
    direction_col = [];
    symbol_col = {};

    grand_plus = 0;
    grand_eq = 0;
    grand_minus = 0;

    for ei = 1:numel(experiments)
        exp_name = experiments{ei};
        func_nums = list_function_numbers(fullfile(base_dir, alg_a, exp_name));
        if isempty(func_nums)
            continue;
        end

        config = experiment_factory(exp_name);
        func_nums = intersect(func_nums, config.function_numbers);

        n_plus = 0;
        n_eq = 0;
        n_minus = 0;

        for fi = 1:numel(func_nums)
            func_num = func_nums(fi);
            [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_num);
            [sym, p_value, direction] = wilcoxon_result(v_a, v_b, alpha);

            switch sym
                case '+'
                    n_plus = n_plus + 1;
                case '='
                    n_eq = n_eq + 1;
                case '-'
                    n_minus = n_minus + 1;
            end

            func_exp_col{end+1, 1} = exp_name; %#ok<AGROW>
            func_col(end+1, 1) = func_num; %#ok<AGROW>
            runs_col(end+1, 1) = numel(v_a); %#ok<AGROW>
            p_col(end+1, 1) = p_value; %#ok<AGROW>
            direction_col(end+1, 1) = direction; %#ok<AGROW>
            symbol_col{end+1, 1} = sym; %#ok<AGROW>
        end

        exp_score = n_plus * score_plus + n_eq * score_equal + n_minus * score_minus;

        exp_col{end+1, 1} = exp_name; %#ok<AGROW>
        plus_col(end+1, 1) = n_plus; %#ok<AGROW>
        eq_col(end+1, 1) = n_eq; %#ok<AGROW>
        minus_col(end+1, 1) = n_minus; %#ok<AGROW>
        problems_col(end+1, 1) = numel(func_nums); %#ok<AGROW>
        score_col(end+1, 1) = exp_score; %#ok<AGROW>

        grand_plus = grand_plus + n_plus;
        grand_eq = grand_eq + n_eq;
        grand_minus = grand_minus + n_minus;
    end

    experiment_table = table(exp_col, plus_col, eq_col, minus_col, problems_col, score_col, ...
        'VariableNames', {'Experiment', 'Plus', 'Equal', 'Minus', 'Problems', 'Score'});
    function_table = table(func_exp_col, func_col, runs_col, p_col, direction_col, symbol_col, ...
        'VariableNames', {'Experiment', 'Function', 'PairedRuns', 'PValue', ...
                          'MedianDifference', 'Symbol'});

    summary = struct();
    summary.experiment_table = experiment_table;
    summary.function_table = function_table;
    summary.grand_plus = grand_plus;
    summary.grand_equal = grand_eq;
    summary.grand_minus = grand_minus;
    summary.base_project_score = grand_plus * score_plus + ...
                                 grand_eq * score_equal + ...
                                 grand_minus * score_minus;
    summary.uses_fitness_distance_balance = algorithm_uses_fitness_distance_balance(project_root, alg_a);
    if summary.uses_fitness_distance_balance
        summary.fitness_distance_balance_bonus = score_fdb_bonus;
    else
        summary.fitness_distance_balance_bonus = 0;
    end
    summary.project_score = summary.base_project_score + summary.fitness_distance_balance_bonus;
    summary.score_plus = score_plus;
    summary.score_equal = score_equal;
    summary.score_minus = score_minus;
    summary.score_fdb_bonus = score_fdb_bonus;
end

function print_score_summary(score_summary, algorithms, alpha)
    fprintf('\nWilcoxon signed-rank score: %s vs %s (alpha = %.3f)\n', ...
            algorithms{1}, algorithms{2}, alpha);
    fprintf('%-16s %6s %6s %6s %8s %8s\n', ...
            'Experiment', '+', '=', '-', 'Problems', 'Score');
    fprintf('%s\n', repmat('-', 1, 62));

    T = score_summary.experiment_table;
    for i = 1:height(T)
        fprintf('%-16s %6d %6d %6d %8d %8d\n', ...
                T.Experiment{i}, T.Plus(i), T.Equal(i), T.Minus(i), ...
                T.Problems(i), T.Score(i));
    end

    fprintf('%s\n', repmat('-', 1, 62));
    fprintf('%-16s %6d %6d %6d %8s %8d\n', ...
            'TOTAL', score_summary.grand_plus, score_summary.grand_equal, ...
            score_summary.grand_minus, '', score_summary.base_project_score);
    fprintf('Fitness-Distance Balance bonus: %+d\n', ...
            score_summary.fitness_distance_balance_bonus);
    fprintf('Project score with bonus: %d\n', score_summary.project_score);
end

function uses_fdb = algorithm_uses_fitness_distance_balance(project_root, algorithm_name)
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

function [check_table, summary] = validate_result_max_fe(base_dir, algorithms, experiments, sample_run_number_base)
    alg_col = {};
    exp_col = {};
    func_col = [];
    run_col = [];
    dim_col = [];
    expected_col = [];
    result_col = [];
    match_col = false(0, 1);
    status_col = {};
    run_dir_col = {};

    missing_alg_col = {};
    missing_exp_col = {};
    missing_func_col = [];
    missing_run_col = [];
    missing_path_col = {};

    for ai = 1:numel(algorithms)
        alg_name = algorithms{ai};

        for ei = 1:numel(experiments)
            exp_name = experiments{ei};
            config = experiment_factory(exp_name);
            exp_dir = fullfile(base_dir, alg_name, exp_name);
            existing_func_nums = list_function_numbers(exp_dir);
            target_runs = infer_target_run_indices(exp_dir, config.function_numbers, sample_run_number_base);

            if isempty(target_runs)
                % No saved run exists for this algorithm/experiment yet.
                % Only inspect unexpected function folders, if any.
                func_nums = existing_func_nums;
            else
                func_nums = union(config.function_numbers, existing_func_nums);
            end

            for fi = 1:numel(func_nums)
                func_num = func_nums(fi);
                func_dir = fullfile(exp_dir, sprintf('F%d', func_num));
                runs = filter_full_run_indices(list_run_indices(func_dir), sample_run_number_base);
                expected_runs = [];

                if ismember(func_num, config.function_numbers) && ~isempty(target_runs)
                    expected_runs = target_runs;
                    missing_runs = setdiff(expected_runs, runs);

                    for mi = 1:numel(missing_runs)
                        missing_run = missing_runs(mi);
                        missing_run_dir = fullfile(func_dir, sprintf('run%d', missing_run));

                        missing_alg_col{end+1, 1} = alg_name; %#ok<AGROW>
                        missing_exp_col{end+1, 1} = exp_name; %#ok<AGROW>
                        missing_func_col(end+1, 1) = func_num; %#ok<AGROW>
                        missing_run_col(end+1, 1) = missing_run; %#ok<AGROW>
                        missing_path_col{end+1, 1} = fullfile(missing_run_dir, 'run_info.mat'); %#ok<AGROW>
                    end
                end

                if ~isempty(expected_runs)
                    runs_to_check = intersect(runs, expected_runs);
                else
                    runs_to_check = runs;
                end

                for ri = 1:numel(runs_to_check)
                    run_num = runs_to_check(ri);
                    run_dir = fullfile(func_dir, sprintf('run%d', run_num));
                    [result_max_fe, result_dimension, read_status, run_identity] = read_run_max_fe(run_dir);
                    [expected_max_fe, expected_status] = expected_max_fe_for_function(config, func_num, result_dimension);

                    if ~strcmp(expected_status, 'ok')
                        status = expected_status;
                        is_match = false;
                    elseif ~strcmp(read_status, 'ok')
                        status = read_status;
                        is_match = false;
                    else
                        identity_status = validate_run_identity(run_identity, alg_name, exp_name, func_num, run_num);
                        if ~strcmp(identity_status, 'ok')
                            status = identity_status;
                            is_match = false;
                        elseif isequal(result_max_fe, expected_max_fe)
                            status = 'ok';
                            is_match = true;
                        else
                            status = 'max_fe_mismatch';
                            is_match = false;
                        end
                    end

                    alg_col{end+1, 1} = alg_name; %#ok<AGROW>
                    exp_col{end+1, 1} = exp_name; %#ok<AGROW>
                    func_col(end+1, 1) = func_num; %#ok<AGROW>
                    run_col(end+1, 1) = run_num; %#ok<AGROW>
                    dim_col(end+1, 1) = result_dimension; %#ok<AGROW>
                    expected_col(end+1, 1) = expected_max_fe; %#ok<AGROW>
                    result_col(end+1, 1) = result_max_fe; %#ok<AGROW>
                    match_col(end+1, 1) = is_match; %#ok<AGROW>
                    status_col{end+1, 1} = status; %#ok<AGROW>
                    run_dir_col{end+1, 1} = run_dir; %#ok<AGROW>
                end
            end
        end
    end

    check_table = table(alg_col, exp_col, func_col, run_col, dim_col, ...
                        expected_col, result_col, match_col, status_col, run_dir_col, ...
        'VariableNames', {'Algorithm', 'Experiment', 'Function', 'Run', ...
                          'Dimension', 'ExpectedMaxFE', 'ResultMaxFE', ...
                          'Match', 'Status', 'RunDir'});

    missing_table = table(missing_alg_col, missing_exp_col, missing_func_col, ...
                          missing_run_col, missing_path_col, ...
        'VariableNames', {'Algorithm', 'Experiment', 'Function', 'Run', ...
                          'ExpectedRunInfo'});

    summary = summarize_fe_check(check_table, missing_table, algorithms, experiments);
end

function target_runs = infer_target_run_indices(exp_dir, function_numbers, sample_run_number_base)
    all_runs = [];

    for fi = 1:numel(function_numbers)
        func_dir = fullfile(exp_dir, sprintf('F%d', function_numbers(fi)));
        runs = filter_full_run_indices(list_run_indices(func_dir), sample_run_number_base);
        all_runs = union(all_runs, runs);
    end

    if isempty(all_runs)
        target_runs = [];
    else
        target_runs = 1:max(all_runs);
    end
end

function runs = filter_full_run_indices(runs, sample_run_number_base)
    runs = runs(runs > 0 & runs < sample_run_number_base);
end

function [expected_max_fe, status] = expected_max_fe_for_function(config, func_num, result_dimension)
    expected_max_fe = NaN;
    status = 'ok';

    if ~ismember(func_num, config.function_numbers)
        status = 'unknown_function';
        return;
    end

    if isfield(config, 'use_cal_par') && config.use_cal_par
        try
            par = Cal_par(func_num);
            expected_max_fe = max_fe_for_real_world(par.n);
        catch
            if isfinite(result_dimension)
                expected_max_fe = max_fe_for_real_world(result_dimension);
            else
                status = 'expected_max_fe_unavailable';
            end
        end
    else
        expected_max_fe = config.maxFE;
    end
end

function [max_fe, dimension, status, identity] = read_run_max_fe(run_dir)
    max_fe = NaN;
    dimension = NaN;
    status = 'ok';
    identity = struct('algorithm', '', ...
                      'experiment_name', '', ...
                      'function_number', NaN, ...
                      'run_number', NaN);

    info_file = fullfile(run_dir, 'run_info.mat');
    if ~isfile(info_file)
        status = 'missing_run_info';
        return;
    end

    try
        S = load(info_file, 'run_info');
    catch
        status = 'load_error';
        return;
    end

    if ~isfield(S, 'run_info')
        status = 'missing_run_info_struct';
        return;
    end

    ri = S.run_info;
    if isfield(ri, 'algorithm')
        identity.algorithm = text_scalar_or_empty(ri.algorithm);
    end
    if isfield(ri, 'experiment_name')
        identity.experiment_name = text_scalar_or_empty(ri.experiment_name);
    end
    if isfield(ri, 'function_number')
        identity.function_number = numeric_scalar_or_nan(ri.function_number);
    end
    if isfield(ri, 'run_number')
        identity.run_number = numeric_scalar_or_nan(ri.run_number);
    end

    if isfield(ri, 'dimension')
        dimension = numeric_scalar_or_nan(ri.dimension);
    end

    if isfield(ri, 'max_fe')
        max_fe = numeric_scalar_or_nan(ri.max_fe);
    elseif isfield(ri, 'maxFe')
        max_fe = numeric_scalar_or_nan(ri.maxFe);
    else
        status = 'missing_max_fe';
        return;
    end

    if ~isfinite(max_fe)
        status = 'invalid_max_fe';
    end
end

function status = validate_run_identity(identity, expected_algorithm, expected_experiment, expected_function, expected_run)
    status = 'ok';

    if isempty(identity.algorithm)
        status = 'missing_algorithm_metadata';
    elseif ~strcmp(identity.algorithm, expected_algorithm)
        status = 'algorithm_mismatch';
    elseif isempty(identity.experiment_name)
        status = 'missing_experiment_metadata';
    elseif ~strcmp(identity.experiment_name, expected_experiment)
        status = 'experiment_mismatch';
    elseif ~isfinite(identity.function_number)
        status = 'missing_function_metadata';
    elseif identity.function_number ~= expected_function
        status = 'function_mismatch';
    elseif ~isfinite(identity.run_number)
        status = 'missing_run_metadata';
    elseif identity.run_number ~= expected_run
        status = 'run_mismatch';
    end
end

function value = numeric_scalar_or_nan(value_in)
    value = NaN;
    if isnumeric(value_in) && isscalar(value_in)
        value = double(value_in);
    elseif ischar(value_in) || (isstring(value_in) && isscalar(value_in))
        value = str2double(value_in);
    end
end

function value = text_scalar_or_empty(value_in)
    value = '';
    if ischar(value_in)
        value = value_in;
    elseif isstring(value_in) && isscalar(value_in)
        value = char(value_in);
    end
end

function summary = summarize_fe_check(check_table, missing_table, algorithms, experiments)
    alg_col = {};
    exp_col = {};
    expected_col = [];
    checked_col = [];
    match_col = [];
    missing_run_col = [];
    mismatch_col = [];
    missing_col = [];
    unknown_col = [];
    valid_col = false(0, 1);

    missing_statuses = {'missing_run_info', 'missing_run_info_struct', ...
                        'missing_max_fe', 'invalid_max_fe', 'load_error', ...
                        'expected_max_fe_unavailable'};

    for ai = 1:numel(algorithms)
        for ei = 1:numel(experiments)
            alg_name = algorithms{ai};
            exp_name = experiments{ei};

            mask = strcmp(check_table.Algorithm, alg_name) & ...
                   strcmp(check_table.Experiment, exp_name);
            missing_mask = strcmp(missing_table.Algorithm, alg_name) & ...
                           strcmp(missing_table.Experiment, exp_name);

            checked = sum(mask);
            matched = sum(mask & check_table.Match);
            missing_runs = sum(missing_mask);
            expected = checked + missing_runs;
            mismatched = sum(mask & strcmp(check_table.Status, 'max_fe_mismatch'));
            missing = sum(mask & ismember(check_table.Status, missing_statuses));
            unknown = sum(mask & strcmp(check_table.Status, 'unknown_function'));
            is_valid = expected == matched && ...
                       missing_runs == 0 && missing == 0 && unknown == 0;

            alg_col{end+1, 1} = alg_name; %#ok<AGROW>
            exp_col{end+1, 1} = exp_name; %#ok<AGROW>
            expected_col(end+1, 1) = expected; %#ok<AGROW>
            checked_col(end+1, 1) = checked; %#ok<AGROW>
            match_col(end+1, 1) = matched; %#ok<AGROW>
            missing_run_col(end+1, 1) = missing_runs; %#ok<AGROW>
            mismatch_col(end+1, 1) = mismatched; %#ok<AGROW>
            missing_col(end+1, 1) = missing; %#ok<AGROW>
            unknown_col(end+1, 1) = unknown; %#ok<AGROW>
            valid_col(end+1, 1) = is_valid; %#ok<AGROW>
        end
    end

    summary_table = table(alg_col, exp_col, expected_col, checked_col, ...
                          match_col, missing_run_col, mismatch_col, ...
                          missing_col, unknown_col, valid_col, ...
        'VariableNames', {'Algorithm', 'Experiment', 'ExpectedRuns', 'CheckedRuns', ...
                          'MatchingRuns', 'MissingResultRuns', ...
                          'MismatchedRuns', 'MissingOrInvalidRuns', ...
                          'UnknownFunctionRuns', 'Valid'});

    summary = struct();
    summary.summary_table = summary_table;
    summary.missing_table = missing_table;
    summary.total_expected_runs = sum(summary_table.ExpectedRuns);
    summary.total_runs = height(check_table);
    summary.matching_runs = sum(check_table.Match);
    summary.missing_result_runs = sum(summary_table.MissingResultRuns);
    summary.mismatched_runs = sum(strcmp(check_table.Status, 'max_fe_mismatch'));
    summary.missing_fe_runs = sum(ismember(check_table.Status, missing_statuses));
    summary.unknown_function_runs = sum(strcmp(check_table.Status, 'unknown_function'));
    summary.is_valid = all(summary_table.Valid);
    summary.invalid_group_count = sum(~summary_table.Valid);
end

function print_fe_check_summary(summary, check_table)
    fprintf('\nMaxFE validation: results/run_info.mat vs experiment_factory\n');
    fprintf('%-14s %-16s %10s %10s %10s %10s %10s %12s %10s %10s\n', ...
            'Algorithm', 'Experiment', 'Expected', 'Checked', 'Matched', ...
            'Missing', 'Mismatch', 'MissingFE', 'Unknown', 'Valid');
    fprintf('%s\n', repmat('-', 1, 122));

    T = summary.summary_table;
    for i = 1:height(T)
        fprintf('%-14s %-16s %10d %10d %10d %10d %10d %12d %10d %10d\n', ...
                T.Algorithm{i}, T.Experiment{i}, T.ExpectedRuns(i), ...
                T.CheckedRuns(i), T.MatchingRuns(i), T.MissingResultRuns(i), ...
                T.MismatchedRuns(i), T.MissingOrInvalidRuns(i), ...
                T.UnknownFunctionRuns(i), T.Valid(i));
    end

    fprintf('%s\n', repmat('-', 1, 122));
    fprintf('%-14s %-16s %10d %10d %10d %10d %10d %12d %10d %10d\n', ...
            'TOTAL', '', summary.total_expected_runs, summary.total_runs, ...
            summary.matching_runs, summary.missing_result_runs, ...
            summary.mismatched_runs, summary.missing_fe_runs, ...
            summary.unknown_function_runs, summary.is_valid);

    if summary.is_valid
        fprintf('MaxFE validation: VALID\n');
    else
        fprintf('MaxFE validation: NOT VALID\n');
        missing_rows = summary.missing_table;
        max_missing_print = min(height(missing_rows), 20);
        if max_missing_print > 0
            fprintf('\nFirst missing result runs:\n');
            fprintf('%-14s %-16s %4s %6s %-40s\n', ...
                    'Algorithm', 'Experiment', 'F', 'Run', 'Expected run_info');
            fprintf('%s\n', repmat('-', 1, 88));
            for i = 1:max_missing_print
                fprintf('%-14s %-16s %4d %6d %-40s\n', ...
                        missing_rows.Algorithm{i}, missing_rows.Experiment{i}, ...
                        missing_rows.Function(i), missing_rows.Run(i), ...
                        missing_rows.ExpectedRunInfo{i});
            end
        end

        invalid_idx = find(~check_table.Match);
        max_print = min(numel(invalid_idx), 20);
        if max_print > 0
            fprintf('\nFirst invalid MaxFE rows:\n');
            fprintf('%-14s %-16s %4s %6s %12s %12s %-24s\n', ...
                    'Algorithm', 'Experiment', 'F', 'Run', 'Expected', 'Result', 'Status');
            fprintf('%s\n', repmat('-', 1, 96));
            for i = 1:max_print
                row = invalid_idx(i);
                fprintf('%-14s %-16s %4d %6d %12g %12g %-24s\n', ...
                        check_table.Algorithm{row}, check_table.Experiment{row}, ...
                        check_table.Function(row), check_table.Run(row), ...
                        check_table.ExpectedMaxFE(row), check_table.ResultMaxFE(row), ...
                        check_table.Status{row});
            end
        end
    end
end

function package_info = create_submission_package(project_root, base_dir, algorithms, ...
        developed_algorithms, experiments, timing_table, estimate_table, ...
        estimate_summary, score_summary, fe_check_table, fe_check_summary, ...
        overall_validation)

    package_root = fullfile(project_root, base_dir);
    ensure_dir(package_root);

    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    staging_name = ['submission_' timestamp];
    staging_dir = fullfile(package_root, staging_name);
    ensure_dir(staging_dir);

    alg_stage = fullfile(staging_dir, 'algorithms');
    report_stage = fullfile(staging_dir, 'reports');
    ensure_dir(alg_stage);
    ensure_dir(report_stage);

    copied_algorithm_files = copy_algorithm_files(project_root, alg_stage, developed_algorithms);

    all_run_fitness_table = collect_all_run_best_fitness(base_dir, developed_algorithms, experiments);

    writetable(timing_table, fullfile(report_stage, 'time_samples.csv'));
    writetable(estimate_table, fullfile(report_stage, 'time_estimate.csv'));
    writetable(score_summary.experiment_table, fullfile(report_stage, 'wilcoxon_score_by_experiment.csv'));
    writetable(score_summary.function_table, fullfile(report_stage, 'wilcoxon_score_by_function.csv'));
    writetable(fe_check_table, fullfile(report_stage, 'max_fe_check_all_runs.csv'));
    writetable(fe_check_summary.summary_table, fullfile(report_stage, 'max_fe_check_summary.csv'));
    writetable(all_run_fitness_table, fullfile(report_stage, 'all_run_best_fitness.csv'));

    save(fullfile(report_stage, 'submission_summary.mat'), ...
         'timing_table', 'estimate_table', 'estimate_summary', ...
         'score_summary', 'fe_check_table', 'fe_check_summary', ...
         'overall_validation', 'all_run_fitness_table', 'copied_algorithm_files');

    write_submission_report(fullfile(report_stage, 'submission_report.txt'), ...
        algorithms, developed_algorithms, copied_algorithm_files, ...
        estimate_summary, score_summary, fe_check_summary, overall_validation, ...
        all_run_fitness_table);

    zip_file = fullfile(package_root, [staging_name '.zip']);
    original_dir = pwd;
    cleanup = onCleanup(@() cd(original_dir));
    cd(staging_dir);
    zip(zip_file, {'algorithms', 'reports'});
    cd(original_dir);
    remove_staging_dir(staging_dir);

    package_info = struct();
    package_info.zip_file = zip_file;
    package_info.all_run_fitness_table = all_run_fitness_table;
end

function remove_staging_dir(staging_dir)
    if ~isfolder(staging_dir)
        return;
    end

    last_message = '';
    for attempt = 1:4
        if attempt > 1
            pause(0.25 * attempt);
        end

        [removed, cleanup_message] = rmdir(staging_dir, 's');
        if removed
            return;
        end
        last_message = cleanup_message;
    end

    warning('validate_and_package:StagingCleanupFailed', ...
            ['Submission zip was created, but temporary folder could not be removed: %s. ' ...
             'Last cleanup message: %s'], ...
            staging_dir, last_message);
end

function copied_files = copy_algorithm_files(project_root, alg_stage, algorithms)
    copied_files = {};

    for i = 1:numel(algorithms)
        src = resolve_algorithm_file(project_root, algorithms{i});
        if isempty(src)
            warning('Algorithm file could not be found for %s', algorithms{i});
            continue;
        end
        dst = fullfile(alg_stage, [algorithms{i} '.m']);
        copyfile(src, dst);
        copied_files{end+1, 1} = fullfile('algorithms', [algorithms{i} '.m']); %#ok<AGROW>
    end
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

function fitness_table = collect_all_run_best_fitness(base_dir, algorithms, experiments)
    alg_col = {};
    exp_col = {};
    func_col = [];
    run_col = [];
    dim_col = [];
    max_fe_col = [];
    best_fitness_col = [];
    best_error_col = [];
    exec_time_col = [];
    feasible_col = [];
    status_col = {};
    run_dir_col = {};

    for ai = 1:numel(algorithms)
        alg_name = algorithms{ai};
        for ei = 1:numel(experiments)
            exp_name = experiments{ei};
            func_nums = list_function_numbers(fullfile(base_dir, alg_name, exp_name));
            if isempty(func_nums)
                continue;
            end

            config = experiment_factory(exp_name);
            func_nums = intersect(func_nums, config.function_numbers);

            for fi = 1:numel(func_nums)
                func_num = func_nums(fi);
                func_dir = fullfile(base_dir, alg_name, exp_name, sprintf('F%d', func_num));
                runs = list_run_indices(func_dir);

                for ri = 1:numel(runs)
                    run_dir = fullfile(func_dir, sprintf('run%d', runs(ri)));
                    values = read_run_best_fitness_values(run_dir);

                    alg_col{end+1, 1} = alg_name; %#ok<AGROW>
                    exp_col{end+1, 1} = exp_name; %#ok<AGROW>
                    func_col(end+1, 1) = func_num; %#ok<AGROW>
                    run_col(end+1, 1) = runs(ri); %#ok<AGROW>
                    dim_col(end+1, 1) = values.dimension; %#ok<AGROW>
                    max_fe_col(end+1, 1) = values.max_fe; %#ok<AGROW>
                    best_fitness_col(end+1, 1) = values.best_fitness; %#ok<AGROW>
                    best_error_col(end+1, 1) = values.best_error; %#ok<AGROW>
                    exec_time_col(end+1, 1) = values.execution_time; %#ok<AGROW>
                    feasible_col(end+1, 1) = values.is_feasible; %#ok<AGROW>
                    status_col{end+1, 1} = values.status; %#ok<AGROW>
                    run_dir_col{end+1, 1} = run_dir; %#ok<AGROW>
                end
            end
        end
    end

    fitness_table = table(alg_col, exp_col, func_col, run_col, dim_col, ...
        max_fe_col, best_fitness_col, best_error_col, exec_time_col, ...
        feasible_col, status_col, run_dir_col, ...
        'VariableNames', {'Algorithm', 'Experiment', 'Function', 'Run', ...
                          'Dimension', 'MaxFE', 'BestFitness', 'BestError', ...
                          'ExecutionTimeSeconds', 'IsFeasible', 'Status', 'RunDir'});
end

function values = read_run_best_fitness_values(run_dir)
    values = struct('dimension', NaN, ...
                    'max_fe', NaN, ...
                    'best_fitness', NaN, ...
                    'best_error', NaN, ...
                    'execution_time', NaN, ...
                    'is_feasible', NaN, ...
                    'status', 'ok');

    info_file = fullfile(run_dir, 'run_info.mat');
    if ~isfile(info_file)
        values.status = 'missing_run_info';
        return;
    end

    try
        S = load(info_file, 'run_info');
    catch
        values.status = 'load_error';
        return;
    end

    if ~isfield(S, 'run_info')
        values.status = 'missing_run_info_struct';
        return;
    end

    ri = S.run_info;
    if isfield(ri, 'dimension')
        values.dimension = numeric_scalar_or_nan(ri.dimension);
    end
    if isfield(ri, 'max_fe')
        values.max_fe = numeric_scalar_or_nan(ri.max_fe);
    elseif isfield(ri, 'maxFe')
        values.max_fe = numeric_scalar_or_nan(ri.maxFe);
    end
    if isfield(ri, 'best_fitness')
        values.best_fitness = numeric_scalar_or_nan(ri.best_fitness);
    else
        values.status = 'missing_best_fitness';
    end
    if isfield(ri, 'best_error')
        values.best_error = numeric_scalar_or_nan(ri.best_error);
    end
    if isfield(ri, 'execution_time')
        values.execution_time = numeric_scalar_or_nan(ri.execution_time);
    end
    if isfield(ri, 'is_feasible')
        values.is_feasible = numeric_scalar_or_nan(ri.is_feasible);
    end
end

function write_submission_report(report_file, algorithms, developed_algorithms, copied_files, ...
                                 estimate_summary, score_summary, fe_check_summary, ...
                                 overall_validation, all_run_fitness_table)
    fid = fopen(report_file, 'w');
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, 'Two-algorithm submission report\n');
    generated_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    fprintf(fid, 'Generated: %s\n\n', generated_at);
    fprintf(fid, 'Comparison: %s vs %s\n', algorithms{1}, algorithms{2});
    fprintf(fid, 'Algorithm files packaged: %s\n\n', strjoin(developed_algorithms, ', '));

    fprintf(fid, 'Copied algorithm files:\n');
    for i = 1:numel(copied_files)
        fprintf(fid, '  %s\n', copied_files{i});
    end

    fprintf(fid, '\nTiming estimate:\n');
    fprintf(fid, '  Workers/cores: %d\n', estimate_summary.workers);
    fprintf(fid, '  Total jobs: %d\n', estimate_summary.total_jobs);
    fprintf(fid, '  Average sample seconds: %.4f\n', estimate_summary.average_sample_seconds);
    fprintf(fid, '  Estimated total hours: %.4f\n', estimate_summary.estimated_total_hours);
    fprintf(fid, '  Minimum required hours: %.4f\n', estimate_summary.minimum_time_hours);
    fprintf(fid, '  Maximum allowed hours: none\n');
    fprintf(fid, '  Valid by time: %d\n', estimate_summary.is_valid);

    fprintf(fid, '\nWilcoxon score:\n');
    fprintf(fid, '  +: %d\n', score_summary.grand_plus);
    fprintf(fid, '  =: %d\n', score_summary.grand_equal);
    fprintf(fid, '  -: %d\n', score_summary.grand_minus);
    fprintf(fid, '  Base project score: %d\n', score_summary.base_project_score);
    fprintf(fid, '  Uses Fitness-Distance Balance: %d\n', score_summary.uses_fitness_distance_balance);
    fprintf(fid, '  Fitness-Distance Balance bonus: %d\n', score_summary.fitness_distance_balance_bonus);
    fprintf(fid, '  Project score: %d\n', score_summary.project_score);

    fprintf(fid, '\nMaxFE validation:\n');
    fprintf(fid, '  Expected runs: %d\n', fe_check_summary.total_expected_runs);
    fprintf(fid, '  Checked runs: %d\n', fe_check_summary.total_runs);
    fprintf(fid, '  Matching runs: %d\n', fe_check_summary.matching_runs);
    fprintf(fid, '  Missing result runs: %d\n', fe_check_summary.missing_result_runs);
    fprintf(fid, '  Mismatched runs: %d\n', fe_check_summary.mismatched_runs);
    fprintf(fid, '  Missing/invalid FE runs: %d\n', fe_check_summary.missing_fe_runs);
    fprintf(fid, '  Unknown function runs: %d\n', fe_check_summary.unknown_function_runs);
    fprintf(fid, '  Invalid algorithm-experiment groups: %d\n', fe_check_summary.invalid_group_count);
    fprintf(fid, '  Valid by MaxFE: %d\n', fe_check_summary.is_valid);

    fprintf(fid, '\nBest fitness file:\n');
    fprintf(fid, '  File: reports/all_run_best_fitness.csv\n');
    fprintf(fid, '  Rows: %d\n', height(all_run_fitness_table));

    fprintf(fid, '\nOverall validation: %d\n', overall_validation);
end

function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end

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
    runs = [];
    if ~isfolder(function_dir)
        return;
    end
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

function [sym, p_value, direction] = wilcoxon_result(v_a, v_b, alpha)
    sym = '=';
    p_value = NaN;
    direction = NaN;

    if numel(v_a) < 2 || numel(v_b) < 2 || numel(v_a) ~= numel(v_b)
        return;
    end

    diffs = v_a - v_b;
    if all(diffs == 0)
        direction = 0;
        return;
    end

    try
        p_value = signrank(v_a, v_b);
    catch
        return;
    end

    direction = median(diffs);
    if direction == 0
        direction = mean(diffs);
    end

    if isnan(p_value) || p_value > alpha
        return;
    end

    if direction < 0
        sym = '+';
    elseif direction > 0
        sym = '-';
    end
end

function text = get_error_text(ME)
    try
        text = getReport(ME, 'basic', 'hyperlinks', 'off');
    catch
        text = ME.message;
    end
end

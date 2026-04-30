clear; clc; close all;
addpath('algorithm');
addpath('proposed');
addpath('problem/CEC2021');
addpath('problem/CEC2022');

% Algorithm configuration
algorithms = {'sfs', 'fdb_sfs'};

% CPU çekirdek sayısını tespit et
num_cores = feature('numcores');
fprintf('Detected %d CPU cores\n', num_cores);

% Optimal worker sayısı
optimal_workers = max(4, num_cores);
fprintf('Using %d parallel workers\n', optimal_workers);

% Parallel pool başlat
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

% List of all available experiments
all_experiments = {
    'cec2021_10', 'cec2021_20', 'cec2022_10', 'cec2022_20'
};

% Calculate total number of jobs for preallocation
% Need to calculate based on each experiment's function count
total_jobs = 0;
for exp_idx = 1:length(all_experiments)
    experiment_name = all_experiments{exp_idx};
    config = experiment_factory(experiment_name);
    config.runs_per_experiment = 21;
    config.maxFE = 1;
    
    jobs_for_this_exp = length(algorithms) * length(config.function_numbers) * config.runs_per_experiment;
    fprintf('  %s: %d algorithms x %d functions x %d runs = %d jobs\n', ...
            experiment_name, length(algorithms), length(config.function_numbers), config.runs_per_experiment, jobs_for_this_exp);
    total_jobs = total_jobs + jobs_for_this_exp;
end
fprintf('Preallocating for %d total jobs...\n', total_jobs);

% Preallocate jobs array for better performance
all_jobs = repmat(struct('id', 0, 'exp_idx', 0, 'experiment_name', '', 'alg_name', '', ...
                        'func_num', 0, 'func_idx', 0, 'run', 0, 'config', struct()), total_jobs, 1);

job_id = 0;

for exp_idx = 1:length(all_experiments)
    experiment_name = all_experiments{exp_idx};
    config = experiment_factory(experiment_name);
    config.runs_per_experiment = 21;
    config.maxFE = 1;
    
    for alg_idx = 1:length(algorithms)
        alg_name = algorithms{alg_idx};
        
        for func_idx = 1:length(config.function_numbers)
            func_num = config.function_numbers(func_idx);
            
            for run_idx = 1:config.runs_per_experiment
                job_id = job_id + 1;
                
                % Fill preallocated structure
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

% Check which jobs are already completed
fprintf('\nChecking for completed jobs...\n');
jobs_to_run = false(total_jobs, 1);  % Logical array for jobs to run
completed_count = 0;

for job_idx = 1:total_jobs
    job = all_jobs(job_idx);
    
    % Check if this job is already completed
    base_dir = 'results';
    run_dir = fullfile(base_dir, job.alg_name, job.experiment_name, ...
                      sprintf('F%d', job.func_num), sprintf('run%d', job.run));
    info_file = fullfile(run_dir, 'run_info.mat');
    
    if exist(info_file, 'file')
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

% Filter to get only jobs that need to be run
jobs_array = all_jobs(jobs_to_run);

fprintf('\nStarting parallel execution of %d remaining jobs...\n', remaining_jobs);

% Parallel execution
parfor job_idx = 1:remaining_jobs
    job = jobs_array(job_idx);
    
    try
        % Set random seed for reproducibility (same seed for same run number)
        % This ensures each run_idx gets the same random sequence across all experiments
        rng(job.run, 'twister');
        
        % Setup problem for this job
        problem = struct();
        
        % Determine which CEC function to use
        if contains(job.experiment_name, 'cec2014')
            problem.fhd = str2func('cec14_func');
        elseif contains(job.experiment_name, 'cec2017')
            problem.fhd = str2func('cec17_func');
        elseif contains(job.experiment_name, 'cec2020rw')
            problem.fhd = str2func('cec20rw_func');
        elseif contains(job.experiment_name, 'cec2020')
            problem.fhd = str2func('cec20_func');
        elseif contains(job.experiment_name, 'cec2021')
            problem.fhd = str2func('cec21_basic_func');
        elseif contains(job.experiment_name, 'cec2022')
            problem.fhd = str2func('cec22_func');
        end
        
        % Handle bounds and dimensions
        if isfield(job.config, 'use_cal_par') && job.config.use_cal_par
            par = Cal_par(job.func_num);
            problem.dimension = par.n;
            problem.lb = par.xmin;
            problem.ub = par.xmax;
            
            % Calculate MaxFE based on dimension
            D = problem.dimension;
            if D <= 10
                problem.maxFe = 1 * 10^5;
            elseif D <= 30
                problem.maxFe = 2 * 10^5;
            elseif D <= 50
                problem.maxFe = 4 * 10^5;
            elseif D <= 150
                problem.maxFe = 8 * 10^5;
            else
                problem.maxFe = 10^6;
            end
        else
            problem.dimension = job.config.dimensions;
            problem.lb = job.config.bounds(1) * ones(1, job.config.dimensions);
            problem.ub = job.config.bounds(2) * ones(1, job.config.dimensions);
            problem.maxFe = job.config.maxFE;
        end
        
        problem.number = job.func_num;
        
        % Print progress (show actual job number and total remaining)
        fprintf('Running Job %d (of %d remaining): %s | %s | F%d | Run %d (seed=%d)\n', ...
                job_idx, remaining_jobs, job.experiment_name, job.alg_name, ...
                job.func_num, job.run, job.run);
        
        % Run algorithm with individual timing
        run_start = tic;
        [best_fitness, best_solution, curve, population_history, fitness_history] = feval(job.alg_name, problem);
        exec_time = toc(run_start);
        
        fprintf('Job %d (Exp:%s, Alg:%s, F%d, Run%d) completed in %.2f sec (fitness: %.6e)\n', ...
                job.id, job.experiment_name, job.alg_name, job.func_num, job.run, exec_time, best_fitness);
        
        % Calculate error (skip for CEC2020RW)
        if ~contains(job.experiment_name, 'cec2020rw')
            global_min = get_global_minimum(job.experiment_name, job.func_num);
            best_error = best_fitness - global_min;
        else
            best_error = NaN; % No error calculation for CEC2020RW
        end
        
        % Check feasibility for the best solution
        if contains(job.experiment_name, 'cec2020rw')
            [~, ~, is_feasible] = calculate_fitness(best_solution', problem, 0);
        else
            is_feasible = true;  % Other problems are unconstrained
        end
        
        % SAVE TO DISK
        if contains(job.experiment_name, 'cec2020rw')
            save_run(job.alg_name, job.func_num, job.run, ...
                     best_fitness, best_solution, curve, ...
                     exec_time, problem, job.experiment_name, ...
                     population_history, fitness_history, ...
                     job.id, is_feasible);
        else
            save_run(job.alg_name, job.func_num, job.run, ...
                     best_fitness, best_solution, curve, ...
                     exec_time, problem, job.experiment_name, ...
                     population_history, fitness_history, ...
                     best_error, job.id, is_feasible);
        end
        
    catch ME
        % Log error for reporting (but don't stop execution)
        fprintf('Job %d (Exp:%s, Alg:%s, F%d, Run%d) failed: %s\n', ...
                 job.id, job.experiment_name, job.alg_name, ...
                 job.func_num, job.run, ME.message);
        
        % Print stack trace for debugging
        for k = 1:length(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
end

fprintf('\n=== Execution Complete ===\n');
fprintf('Completed %d new jobs\n', remaining_jobs);
fprintf('Total completed jobs: %d / %d\n', completed_count + remaining_jobs, total_jobs);

function save_run(algorithm_name, func_num, run_num, best_fitness, best_solution, curve, exec_time, problem, experiment_name, population_history, fitness_history, varargin)
    % varargin can contain best_error, job_id, is_feasible OR job_id, is_feasible for CEC2020RW
    % Parallel-safe version of save_run with unique file naming including population and fitness histories
    % Added job_id parameter to ensure unique file/folder names
    
    base_dir = 'results';
    alg_dir = fullfile(base_dir, algorithm_name);
    benchmark_dir = fullfile(alg_dir, experiment_name);
    func_dir = fullfile(benchmark_dir, sprintf('F%d', func_num));
    
    % Create run directory (using only run number)
    run_dir = fullfile(func_dir, sprintf('run%d', run_num));
    
    % Create directories if they don't exist (parallel-safe)
    try
        if ~exist(run_dir, 'dir')
            mkdir(run_dir);
        end
    catch ME
        % If mkdir fails due to parallel conflict, wait and retry
        pause(rand() * 0.1); % Random small delay
        if ~exist(run_dir, 'dir')
            mkdir(run_dir);
        end
    end
    
    % Save run metadata
    run_info = struct();
    run_info.algorithm = algorithm_name;
    run_info.experiment_name = experiment_name;
    run_info.function_number = func_num;
    run_info.run_number = run_num;
    run_info.random_seed = run_num; % Random seed used (same as run number for reproducibility)
    run_info.dimension = problem.dimension;
    run_info.max_fe = problem.maxFe;
    run_info.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    run_info.best_fitness = best_fitness;
    run_info.execution_time = exec_time;
    
    % Handle variable arguments based on experiment type
    if contains(experiment_name, 'cec2020rw')
        % For CEC2020RW: varargin = {job_id, is_feasible}
        if length(varargin) >= 2
            run_info.is_feasible = varargin{2};
        else
            run_info.is_feasible = true; % default
        end
        % No best_error for CEC2020RW
    else
        % For other experiments: varargin = {best_error, job_id, is_feasible}
        if length(varargin) >= 1
            run_info.best_error = varargin{1};
        end
        if length(varargin) >= 3
            run_info.is_feasible = varargin{3};
        else
            run_info.is_feasible = true; % default for unconstrained problems
        end
    end
    
    % Create simple file names
    info_file = fullfile(run_dir, 'run_info.mat');
    solution_file = fullfile(run_dir, 'best_solution.mat');
    curve_file = fullfile(run_dir, 'convergence_curve.mat');
    population_history_file = fullfile(run_dir, 'population_history.mat');
    fitness_history_file = fullfile(run_dir, 'fitness_history.mat');
    
    % Parallel-safe file saving with error handling
    % Check if population_history exceeds 2GB limit (use -v7.3 only when needed)
    pop_info = whos('population_history');
    fit_info = whos('fitness_history');
    size_limit = 2 * 1024^3;  % 2 GB in bytes
    
    try
        save(info_file, 'run_info');
        save(solution_file, 'best_solution');
        save(curve_file, 'curve');
        
        % Use -v7.3 only for large files (>2GB), otherwise use default (faster)
        if pop_info.bytes > size_limit
            save(population_history_file, 'population_history', '-v7.3');
        else
            save(population_history_file, 'population_history');
        end
        
        if fit_info.bytes > size_limit
            save(fitness_history_file, 'fitness_history', '-v7.3');
        else
            save(fitness_history_file, 'fitness_history');
        end
    catch ME
        % If save fails due to parallel conflict or size issue, wait and retry with -v7.3
        pause(rand() * 0.2); % Random delay 0-200ms
        save(info_file, 'run_info');
        save(solution_file, 'best_solution');
        save(curve_file, 'curve');
        save(population_history_file, 'population_history', '-v7.3');
        save(fitness_history_file, 'fitness_history', '-v7.3');
    end
    
end

clear; clc; close all;
addpath('algorithm');
addpath('problem/CEC2014');
addpath('problem/CEC2017');
addpath('problem/CEC2020');
addpath('problem/CEC2020RW');
addpath('problem/CEC2021');
addpath('problem/CEC2022');

fprintf('=== Quick Algorithm Integration Test ===\n\n');

% Test configuration - minimal settings for quick validation
algorithms = {'lshade_spacma'};  % Test Weighted Differential Evolution algorithm
% Her CEC benchmark'ından 3 problem: CEC2014, CEC2017, CEC2020, CEC2020RW, CEC2021, CEC2022
% test_experiments = {'cec2014_10', 'cec2017_10', 'cec2020_10', 'cec2020rw', 'cec2021_10', 'cec2022_10'};
test_experiments = {'cec2020rw'};
test_functions = [1, 2, 3, 26];      % Her benchmark'tan F1, F2, F3
test_runs = 3;                   % 3 runs for validation

fprintf('Test Configuration:\n');
fprintf('  Algorithms: %s\n', strjoin(algorithms, ', '));
fprintf('  Experiments: %s\n', strjoin(test_experiments, ', '));
fprintf('  Functions: F%s\n', sprintf('%d ', test_functions));
fprintf('  Runs: %d\n\n', test_runs);

% Calculate total jobs
total_jobs = length(algorithms) * length(test_experiments) * length(test_functions) * test_runs;
fprintf('Total test jobs: %d\n\n', total_jobs);

% Create results table
results = cell(total_jobs + 1, 8);
results(1, :) = {'Algorithm', 'Experiment', 'Function', 'Run', 'Best Fitness', 'Error', 'Time (s)', 'Status'};
row_idx = 2;

% Sequential execution (no parallel for easier debugging)
for alg_idx = 1:length(algorithms)
    alg_name = algorithms{alg_idx};
    
    fprintf('--- Testing Algorithm: %s ---\n', upper(alg_name));
    
    for exp_idx = 1:length(test_experiments)
        test_experiment = test_experiments{exp_idx};
        
        fprintf('\n=== Experiment: %s ===\n', test_experiment);
        
        % Get experiment configuration
        config = experiment_factory(test_experiment);
        
        for func_idx = 1:length(test_functions)
            func_num = test_functions(func_idx);
            
            for run_idx = 1:test_runs
            fprintf('  Running F%d, Run %d... ', func_num, run_idx);
            
            try
                % Set random seed for reproducibility
                rng(run_idx, 'twister');
                
                % Setup problem based on experiment type
                problem = struct();
                problem.number = func_num;
                problem.dimension = config.dimensions;
                problem.lb = config.bounds(1) * ones(1, config.dimensions);
                problem.ub = config.bounds(2) * ones(1, config.dimensions);
                problem.maxFe = config.maxFE;
                
                % Set appropriate function handle based on benchmark
                if contains(test_experiment, 'cec2014')
                    problem.fhd = str2func('cec14_func');
                elseif contains(test_experiment, 'cec2017')
                    problem.fhd = str2func('cec17_func');
                elseif contains(test_experiment, 'cec2020') && ~contains(test_experiment, 'rw')
                    problem.fhd = str2func('cec20_func');
                elseif contains(test_experiment, 'cec2021')
                    problem.fhd = str2func('cec21_bias_shift_rot_func');
                elseif contains(test_experiment, 'cec2022')
                    problem.fhd = str2func('cec22_func');
                elseif contains(test_experiment, 'cec2020rw')
                    problem.fhd = str2func('cec20rw_func');
                    % Get problem-specific parameters using Cal_par (returns struct)
                    par = Cal_par(func_num);
                    problem.lb = par.xmin;
                    problem.ub = par.xmax;
                    problem.dimension = par.n;
                    problem.maxFe = 20000;  % Standard maxFE for CEC2020RW
                end
                
                % Run algorithm with timing
                run_start = tic;
                [best_fitness, best_solution, curve, population_history, fitness_history] = feval(alg_name, problem);
                exec_time = toc(run_start);
                
                % Calculate error
                global_min = get_global_minimum(test_experiment, func_num);
                best_error = best_fitness - global_min;
                
                % Store results
                results{row_idx, 1} = alg_name;
                results{row_idx, 2} = test_experiment;
                results{row_idx, 3} = sprintf('F%d', func_num);
                results{row_idx, 4} = run_idx;
                results{row_idx, 5} = best_fitness;
                results{row_idx, 6} = best_error;
                results{row_idx, 7} = exec_time;
                results{row_idx, 8} = 'OK';
                
                fprintf('✓ (Time: %.2fs, Fitness: %.6e, Error: %.6e)\n', ...
                        exec_time, best_fitness, best_error);
                
                % Validate outputs
                assert(~isnan(best_fitness), 'Best fitness is NaN');
                assert(~isinf(best_fitness), 'Best fitness is Inf');
                assert(length(best_solution) == problem.dimension, 'Solution dimension mismatch');
                assert(length(curve) == problem.maxFe, 'Curve length mismatch');
                assert(size(population_history, 3) == problem.dimension, 'Population history dimension mismatch');
                assert(size(fitness_history, 1) == size(population_history, 1), 'History size mismatch');
                
            catch ME
                % Store error
                results{row_idx, 1} = alg_name;
                results{row_idx, 2} = test_experiment;
                results{row_idx, 3} = sprintf('F%d', func_num);
                results{row_idx, 4} = run_idx;
                results{row_idx, 5} = NaN;
                results{row_idx, 6} = NaN;
                results{row_idx, 7} = NaN;
                results{row_idx, 8} = 'FAILED';
                
                fprintf('✗ FAILED: %s\n', ME.message);
                
                % Print stack trace for debugging
                for k = 1:length(ME.stack)
                    fprintf('    at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
                end
            end
            
            row_idx = row_idx + 1;
            end
        end
    end
    fprintf('\n');
end

% Display results summary
fprintf('\n=== Test Results Summary ===\n');
fprintf('%-10s %-12s %-8s %-5s %-15s %-15s %-10s %-10s\n', ...
        results{1, :});
fprintf('%s\n', repmat('-', 1, 100));
for i = 2:size(results, 1)
    fprintf('%-10s %-12s %-8s %-5d %-15.6e %-15.6e %-10.2f %-10s\n', ...
            results{i, :});
end
fprintf('\n');

% Check for failures
failed_jobs = sum(strcmp(results(2:end, 8), 'FAILED'));
if failed_jobs == 0
    fprintf('✓✓✓ ALL TESTS PASSED! ✓✓✓\n');
    fprintf('The algorithm is properly integrated and working correctly.\n');
else
    fprintf('✗✗✗ %d/%d TESTS FAILED ✗✗✗\n', failed_jobs, total_jobs);
    fprintf('Please check the error messages above.\n');
end

fprintf('\n=== Test Complete ===\n');


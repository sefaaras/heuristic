function save_run(algorithm_name, func_num, run_num, best_fitness, best_solution, curve, exec_time, problem, experiment_name, population_history, fitness_history, varargin)
% Writes one finished run to results/<alg>/<experiment>/F<n>/run<k>/.
% varargin is {job_id, is_feasible, violation, objective, max_violation} on a
% constrained suite (CEC2020RW, CEDP) and {best_error, job_id, is_feasible}
% elsewhere. Those two suites are the ones whose problems carry constraints,
% so they are the ones that persist is_feasible / objective /
% constraint_violation and store best_error as NaN.

    % Pushed once per parfor worker by main.m:
    %   save_run('set_provenance', struct('pipeline_id', .., 'pipeline_commit', ..))
    persistent PROVENANCE

    if nargin == 2 && (ischar(algorithm_name) || isstring(algorithm_name)) ...
            && strcmpi(algorithm_name, 'set_provenance')
        PROVENANCE = func_num;
        return;
    end

    % problem.results_root aims a campaign at another drive; main.m does not set
    % it, so it writes into the repo tree.
    if isstruct(problem) && isfield(problem, 'results_root') && ~isempty(problem.results_root)
        base_dir = problem.results_root;
    else
        base_dir = 'results';
    end
    alg_dir = fullfile(base_dir, algorithm_name);
    benchmark_dir = fullfile(alg_dir, experiment_name);
    func_dir = fullfile(benchmark_dir, sprintf('F%d', func_num));

    run_dir = fullfile(func_dir, sprintf('run%d', run_num));

    try
        if ~exist(run_dir, 'dir')
            mkdir(run_dir);
        end
    catch ME
        pause(rand() * 0.1);   % lost a race against another worker; retry
        if ~exist(run_dir, 'dir')
            mkdir(run_dir);
        end
    end

    run_info = struct();
    run_info.algorithm = algorithm_name;
    run_info.experiment_name = experiment_name;
    run_info.function_number = func_num;
    run_info.run_number = run_num;
    run_info.random_seed = run_num;
    run_info.dimension = problem.dimension;
    run_info.max_fe = problem.maxFe;
    run_info.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    run_info.best_fitness = best_fitness;
    run_info.execution_time = exec_time;

    if ~isempty(curve)
        run_info.bsf_at_maxfe = curve(end);
    else
        run_info.bsf_at_maxfe = NaN;
    end

    if contains(experiment_name, 'cec2020rw') || strcmpi(experiment_name, 'cedp')
        run_info.targets       = [];
        run_info.fe_to_target  = [];
        run_info.err_at_target = [];
        run_info.success_tol   = NaN;
        run_info.fe_to_tol     = NaN;
    else
        global_min = get_global_minimum(experiment_name, func_num);
        [fe_to_target, err_at_target, targets] = target_hits(curve, global_min);
        run_info.targets       = targets;
        run_info.fe_to_target  = fe_to_target;
        run_info.err_at_target = err_at_target;
        run_info.success_tol   = targets(end);
        run_info.fe_to_tol     = fe_to_target(end);
        run_info.success       = ~isnan(fe_to_target(end));
    end

    if contains(experiment_name, 'cec2020rw') || strcmpi(experiment_name, 'cedp')
        if length(varargin) >= 2
            run_info.is_feasible = varargin{2};
        else
            run_info.is_feasible = true;
        end
        if length(varargin) >= 3
            run_info.constraint_violation = varargin{3};
        end
        if length(varargin) >= 4
            run_info.objective = varargin{4};
        end
        if length(varargin) >= 5
            run_info.max_violation = varargin{5};
        end
    else
        if length(varargin) >= 1
            run_info.best_error = varargin{1};
        end
        if length(varargin) >= 3
            run_info.is_feasible = varargin{3};
        else
            run_info.is_feasible = true;   % unconstrained problem
        end
    end

    % Read straight from the evaluator, so this is what the run really spent and
    % not what the algorithm believes it spent. The padded tail of curve
    % ("curve(FE:end) = bsf") cannot tell an early stop from stagnation.
    stats = eval_stats();
    run_info.fe_used = stats.n_eval;
    run_info.fe_of_best = stats.fe_best_obj;
    run_info.best_objective_seen = stats.best_obj;

    % Where the reported best first appears in the curve. Disagreeing with
    % fe_of_best means the port's best-so-far bookkeeping is off.
    run_info.fe_of_best_curve = NaN;
    if ~isempty(curve)
        idx = find(double(curve(:).') <= double(best_fitness), 1, 'first');
        if ~isempty(idx)
            run_info.fe_of_best_curve = idx;
        end
    end

    if contains(experiment_name, 'cec2020rw') || strcmpi(experiment_name, 'cedp')
        run_info.n_ineq = stats.n_ineq;
        run_info.n_eq   = stats.n_eq;
        run_info.first_fe_feasible = stats.first_fe_feasible;
        run_info.feas_rate = stats.n_feasible / max(1, stats.n_eval);
        run_info.viol_mean_all = stats.viol_sum / max(1, stats.n_eval);
        run_info.best_feasible_objective = stats.best_feas_obj;
        run_info.fe_best_feasible = stats.fe_best_feas_obj;
    end

    if isstruct(PROVENANCE)
        prov_fields = fieldnames(PROVENANCE);
        for pf = 1:numel(prov_fields)
            run_info.(prov_fields{pf}) = PROVENANCE.(prov_fields{pf});
        end
    else
        run_info.pipeline_id = 'unset';
        run_info.pipeline_commit = 'unset';
    end
    run_info.matlab_version = version();
    run_info.worker_id = worker_id();
    run_info.host = host_name();

    info_file        = fullfile(run_dir, 'run_info.mat');
    solution_file    = fullfile(run_dir, 'best_solution.mat');
    pop_metrics_file = fullfile(run_dir, 'pop_metrics.mat');
    dim_metrics_file = fullfile(run_dir, 'dim_metrics.mat');

    pop_metrics = population_history;
    dim_metrics = fitness_history;

    LB = metric_labels();
    nlab = numel(LB.pop);
    labels = LB.pop;   % 'bsf' is appended below, together with its column

    % Storage split by precision, worth 30 % of the biggest file a run writes.
    % min/max are compared against lb/ub by an out-of-bounds audit -- single(ub)
    % rounds ABOVE ub -- and x_best has to re-evaluate to the same fitness, so
    % those three keep double; the other four are descriptive statistics.
    % tools/load_dim_metrics.m reverses the split from the stored name lists.
    dim_labels = LB.dim;
    exact_layers = ismember(dim_labels, {'min', 'max', 'x_best'});
    dim_labels_exact  = dim_labels(exact_layers);
    dim_labels_approx = dim_labels(~exact_layers);
    split_ok = ~isempty(dim_metrics) && ndims(dim_metrics) == 3 ...
               && size(dim_metrics, 3) == numel(dim_labels);

    % record_history writes each sample into the slot its FE belongs to, so a
    % measured row is already on the canonical grid; only the gaps are filled, by
    % holding the last measured row. No value is invented -- a held row is a
    % verbatim copy whose fe column still carries the FE at which it was MEASURED.
    n_genuine = 0;
    if ~isempty(pop_metrics)
        n_genuine = nnz(isfinite(pop_metrics(:, 1)));
    end
    [pop_metrics, dim_metrics] = spread_to_grid(pop_metrics, dim_metrics);
    if split_ok
        dim_metrics_exact  = double(dim_metrics(:, :, exact_layers));
        dim_metrics_approx = single(dim_metrics(:, :, ~exact_layers));
    end

    if ismatrix(pop_metrics) && size(pop_metrics, 2) == nlab
        pop_metrics = double(pop_metrics);
        % bsf is read at the row's OWN point on the FE grid, h*step, not at the fe
        % the population was sampled at: curve has every FE, so the convergence
        % curve stays exact at all T rows even where the population metrics had to
        % be held. Row 1 keeps its own fe so it stays the t=0 pair with f_init,
        % and the last row is pinned to the end of curve, which also covers a
        % budget that T does not divide.
        T_rows = size(pop_metrics, 1);
        step = max(1, floor(max(1, round(double(problem.maxFe))) / T_rows));
        grid_fe = (1:T_rows).' * step;
        grid_fe(1) = round(pop_metrics(1, 1));
        if ~isempty(curve)
            grid_fe(T_rows) = numel(curve);
        end
        fe_idx = min(round(grid_fe), numel(curve));
        % bsf stays double: single resolves 2.4e-4 at f = 2100 and would flatten
        % the converged tail of the curve.
        bsf = nan(size(fe_idx));
        ok = isfinite(fe_idx) & fe_idx >= 1;
        bsf(ok) = double(curve(fe_idx(ok)));
        pop_metrics = [pop_metrics, bsf];
        labels = [LB.pop, {'bsf'}];
    end

    % Baseline f(x0) for data profiles (More & Wild 2009) and normalised
    % convergence. record_history fires unconditionally on its first call, so this
    % is the population right after initialisation.
    run_info.fe_init = NaN;
    run_info.f_init_best = NaN;
    run_info.f_init_mean = NaN;
    if ~isempty(pop_metrics) && size(pop_metrics, 1) >= 1 && isfinite(pop_metrics(1, 1))
        run_info.fe_init     = double(pop_metrics(1, 1));
        run_info.f_init_best = double(pop_metrics(1, strcmp(LB.pop, 'f_best')));
        run_info.f_init_mean = double(pop_metrics(1, strcmp(LB.pop, 'f_mean')));
    end
    % Counted before the resampling above: it is the only thing separating a run
    % sampled at full resolution from one whose rows are largely held copies.
    run_info.n_history_samples = n_genuine;
    run_info.n_history_rows = 0;
    if ~isempty(pop_metrics)
        run_info.n_history_rows = size(pop_metrics, 1);
    end

    % run_info.mat is written LAST because main.m treats it as the marker that the
    % job finished. Writing it first makes a run that died midway look complete,
    % and the resume scan then never repairs it.
    try
        save(solution_file, 'best_solution');
        save(pop_metrics_file, 'pop_metrics', 'labels');
        if split_ok
            save(dim_metrics_file, 'dim_metrics_exact', 'dim_metrics_approx', ...
                 'dim_labels_exact', 'dim_labels_approx', 'dim_labels');
        else
            save(dim_metrics_file, 'dim_metrics', 'dim_labels');
        end
        save(info_file, 'run_info');
    catch ME
        pause(rand() * 0.2);
        save(solution_file, 'best_solution');
        save(pop_metrics_file, 'pop_metrics', 'labels', '-v7.3');
        if split_ok
            save(dim_metrics_file, 'dim_metrics_exact', 'dim_metrics_approx', ...
                 'dim_labels_exact', 'dim_labels_approx', 'dim_labels', '-v7.3');
        else
            save(dim_metrics_file, 'dim_metrics', 'dim_labels', '-v7.3');
        end
        save(info_file, 'run_info');
    end

end

function [P, D] = spread_to_grid(P, D)
% Fill the slots no evaluation reached by holding the last measured row. A
% measured row must stay at its own index: re-deriving the index from the FE
% shifts every row by one whenever a sample sits strictly inside its cell, which
% silently halves the number of distinct rows.
    if isempty(P) || ~ismatrix(P)
        return;
    end
    ok = isfinite(P(:, 1));
    if all(ok) || ~any(ok)
        return;                      % already full, or nothing measured
    end

    T = size(P, 1);
    idx = zeros(T, 1);
    last = 0;
    for h = 1:T
        if ok(h)
            last = h;
        end
        idx(h) = last;
    end
    idx(idx == 0) = find(ok, 1);     % leading gap: row 1 is normally written

    P = P(idx, :);
    if ~isempty(D) && ndims(D) == 3 && size(D, 1) == T
        D = D(idx, :, :);
    end
end

function s = eval_stats()
% Counters of the run that just finished. The try guards against a stale
% calculate_fitness copy shadowing the one that keeps them.
    try
        s = calculate_fitness('stats');
    catch
        s = struct('n_eval', NaN, 'n_feasible', NaN, 'viol_sum', NaN, ...
                   'best_obj', NaN, 'fe_best_obj', NaN, ...
                   'best_feas_obj', NaN, 'fe_best_feas_obj', NaN, ...
                   'first_fe_feasible', NaN, 'n_ineq', [], 'n_eq', [], 'is_rw', false);
    end
end

function id = worker_id()
% parfor worker that produced this run; 0 when running serially.
    id = 0;
    try
        t = getCurrentTask();
        if ~isempty(t)
            id = t.ID;
        end
    catch
    end
end

function h = host_name()
    h = getenv('COMPUTERNAME');
    if isempty(h)
        h = getenv('HOSTNAME');
    end
    if isempty(h)
        h = 'unknown';
    end
end

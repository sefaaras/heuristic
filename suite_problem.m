function problem = suite_problem(experiment_name, func_num)
%SUITE_PROBLEM The evaluator contract for one (experiment, function).
%
%   problem.fhd        handle to the REAL CEC routine. calculate_fitness reads
%                      the suite out of func2str(fhd), so this must never be a
%                      wrapper -- an unrecognised name takes the MEX suites down
%                      with an uncatchable access violation
%   problem.dimension  \
%   problem.lb         |  the box, per problem on the suites that define one
%   problem.ub         /
%   problem.maxFe      FE budget
%   problem.number     function index within the suite
%
% main.m, tools/redo_runs.m and tools/verify_campaign.m all built this by hand
% from experiment_factory. Three copies of one rule drift; a suite added to one
% of them and not the others produces runs nothing can verify or reproduce. They
% read it from here instead.
%
% Where the box and the budget come from is decided by experiment_factory:
%
%   cfg.par_func empty   fixed dimension and a symmetric box (the numeric CEC
%                        suites); budget is cfg.maxFE
%   cfg.par_func set     dimension and box are per problem, read from that
%                        function (Cal_par for CEC2020RW, cedp_par for
%                        CEDP); budget is cfg.maxFE when the suite fixes one,
%                        otherwise the CEC2020RW dimension ladder

    cfg = experiment_factory(experiment_name);

    problem = struct('fhd', str2func(suite_fhd_name(experiment_name)), ...
                     'dimension', 0, 'lb', [], 'ub', [], 'maxFe', 0, ...
                     'number', func_num);

    if isfield(cfg, 'par_func') && ~isempty(cfg.par_func)
        par = feval(cfg.par_func, func_num);
        problem.dimension = par.n;
        problem.lb = par.xmin;
        problem.ub = par.xmax;
        if cfg.maxFE > 0
            problem.maxFe = cfg.maxFE;
        else
            problem.maxFe = rw_budget(par.n);
        end
    else
        problem.dimension = cfg.dimensions;
        problem.lb = cfg.bounds(1) * ones(1, cfg.dimensions);
        problem.ub = cfg.bounds(2) * ones(1, cfg.dimensions);
        problem.maxFe = cfg.maxFE;
    end
end

function name = suite_fhd_name(experiment_name)
% Order matters: cec2020rw has to be tested before cec2020.
    if contains(experiment_name, 'cec2014')
        name = 'cec14_func';
    elseif contains(experiment_name, 'cec2017')
        name = 'cec17_func';
    elseif contains(experiment_name, 'cec2020rw')
        name = 'cec20rw_func';
    elseif contains(experiment_name, 'cec2020')
        name = 'cec20_func';
    elseif contains(experiment_name, 'cec2021')
        name = 'cec21_bias_shift_rot_func';
    elseif contains(experiment_name, 'cec2022')
        name = 'cec22_func';
    elseif strcmpi(experiment_name, 'cedp')
        name = 'cedp_func';
    else
        error('suite_problem:unknownExperiment', ...
              'No evaluator is mapped for experiment "%s".', experiment_name);
    end
end

function fe = rw_budget(D)
% CEC2020RW gives no budget per problem; the guidelines set it by dimension.
    if D <= 10
        fe = 1 * 10^5;
    elseif D <= 30
        fe = 2 * 10^5;
    elseif D <= 50
        fe = 4 * 10^5;
    elseif D <= 150
        fe = 8 * 10^5;
    else
        fe = 10^6;
    end
end

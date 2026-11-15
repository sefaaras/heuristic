function [results, studies, problems] = hold_literature(varargin)
%HOLD_LITERATURE Published results for the HOLD pool, as tables.
%
%   R = hold_literature()                         every result row
%   R = hold_literature('problem', 'wdn_balerma') one problem
%   R = hold_literature('problem', 'protein_ab3d', 'instance', '1CRN', ...
%                       'max_fe', 200000, 'verdict', {'ok'})
%   [R, S, P] = hold_literature(...)              plus the studies and problems
%
% Filters (all optional, combined with AND):
%   'problem'   problem id, as in literature/problems.csv
%   'instance'  instance id; '' selects the base instance
%   'max_fe'    keep rows whose budget equals this value (the protocol to match)
%   'runs'      keep rows over exactly this many runs
%   'verdict'   cell of verdicts to keep, e.g. {'ok'} or {'ok','caution'}
%   'from_year' keep rows whose producing study appeared in or after this year
%
% The rows are what other studies published, recorded with the conditions they
% were produced under. A row is comparable with a run of ours only when problem,
% instance, max_fe and runs all match; problem/HOLD/README.md gives the rules and
% the meaning of every column.

    p = inputParser;
    p.addParameter('problem', '', @(x) ischar(x) || isstring(x));
    p.addParameter('instance', [], @(x) ischar(x) || isstring(x));
    p.addParameter('max_fe', [], @(x) isempty(x) || isscalar(x));
    p.addParameter('runs', [], @(x) isempty(x) || isscalar(x));
    p.addParameter('verdict', {}, @(x) iscell(x) || ischar(x) || isstring(x));
    p.addParameter('from_year', [], @(x) isempty(x) || isscalar(x));
    p.parse(varargin{:});
    o = p.Results;

    here = fileparts(mfilename('fullpath'));
    results  = read_csv(fullfile(here, 'literature', 'results.csv'));
    studies  = read_csv(fullfile(here, 'literature', 'studies.csv'));
    problems = read_csv(fullfile(here, 'literature', 'problems.csv'));

    keep = true(height(results), 1);
    if strlength(o.problem) > 0
        keep = keep & results.problem == string(o.problem);
    end
    if ~isempty(o.instance)
        keep = keep & results.instance == string(o.instance);
    end
    if ~isempty(o.max_fe)
        keep = keep & results.max_fe == o.max_fe;
    end
    if ~isempty(o.runs)
        keep = keep & results.runs == o.runs;
    end
    if ~isempty(o.verdict)
        keep = keep & ismember(results.verdict, string(o.verdict));
    end
    if ~isempty(o.from_year)
        [tf, k] = ismember(results.study, studies.study);
        yr = nan(height(results), 1);
        yr(tf) = studies.year(k(tf));
        keep = keep & yr >= o.from_year;
    end
    results = results(keep, :);
end

function T = read_csv(fn)
% Text columns stay text (an empty instance is "", not missing); the numeric
% columns are read as double so they can be filtered and compared.
    numeric = {'D', 'best_known', 'year', 'max_fe', 'runs', 'pop', 'best', 'mean', ...
               'worst', 'std', 'median', 'success_rate', 'fe_to_target'};
    opts = detectImportOptions(fn, 'Delimiter', ',', 'Encoding', 'UTF-8', ...
                               'TextType', 'string');
    names = opts.VariableNames;
    opts = setvartype(opts, names, 'string');
    opts = setvartype(opts, intersect(names, numeric), 'double');
    opts = setvaropts(opts, names(~ismember(names, numeric)), 'FillValue', "");
    T = readtable(fn, opts);
end

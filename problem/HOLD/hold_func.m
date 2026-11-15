function [f, g, h] = hold_func(x, k)
%HOLD_FUNC The HOLD suite: one entry point for every case.
%
%   x -----> ps x D, one candidate per ROW (the CEC2020RW convention)
%   k -----> case number, see hold_par
%   f -----> ps x 1 objective, minimisation throughout
%   g -----> ng x ps inequality constraints, feasible where g <= 0
%   h -----> nh x ps equality constraints
%
% Constraint blocks are returned constraint-per-ROW, the orientation
% calculate_fitness reads first; an unconstrained problem returns 0 x ps
% blocks, which the evaluator scores as feasible with fitness = objective.
%
% Each problem lives in its own hold_<problem>.m and knows nothing about
% cases; the case only decides which problem is called (budget and run count
% are the harness's business).
%
% The finite-element problems (200-bar truss, domes) cost 8-15 ms per design,
% and the history recorder re-reads the constraint state of the whole population
% at each of its 1000 samples -- points the algorithm has just evaluated. Their
% results are therefore remembered: a row seen before is answered from memory,
% bit-identical because the evaluators are deterministic (eigs draws its start
% vector from a private stream, not the run's). Without this the recorder alone
% costs several times the run. The memory never crosses a case.

    persistent kk par MX MF MG pos
    CAP = 20000;
    if isempty(kk)
        % third-party problem code lives in subfolders, unmodified
        here = fileparts(mfilename('fullpath'));
        addpath(fullfile(here, 'truss200'));
        addpath(fullfile(here, 'gtopx'));
        addpath(fullfile(here, 'epanet'));
    end
    if isempty(kk) || kk ~= k
        par = hold_par(k);
        kk = k;
        MX = zeros(0, par.n);  MF = zeros(0, 1);  MG = zeros(0, par.g);  pos = 0;
    end
    ps = size(x, 1);
    h = zeros(0, ps);

    switch par.problem
        case 'pv_ddm_rtc'
            f = hold_pv_ddm(x);
            g = zeros(0, ps);
        case 'protein_ab3d'
            f = hold_protein_ab3d(x, par.seq);
            g = zeros(0, ps);
        case {'chped48', 'chped96'}
            [f, g] = hold_chped(x, par.nblk);
        case 'gtop_cassini1'
            [f, g] = hold_cassini1(x);
        case {'wdn_balerma', 'wdn_hanoi'}
            [f, g] = hold_wdn(x, par.net);
        case {'truss200_freq', 'dome1180_freq', 'dome1410_freq'}
            [hit, loc] = ismember(x, MX, 'rows');
            f = zeros(ps, 1);  G = zeros(ps, par.g);
            f(hit) = MF(loc(hit));  G(hit, :) = MG(loc(hit), :);
            miss = find(~hit);
            if ~isempty(miss)
                if strcmp(par.problem, 'truss200_freq')
                    [fm, gm] = hold_truss200(x(miss, :));
                else
                    [fm, gm] = hold_dome(x(miss, :), par.dome);
                    gm = gm.';
                end
                f(miss) = fm;  G(miss, :) = gm;
                % ring buffer of the last CAP distinct designs
                for r = 1:numel(miss)
                    pos = mod(pos, CAP) + 1;
                    MX(pos, :) = x(miss(r), :);
                    MF(pos, 1) = fm(r);
                    MG(pos, :) = gm(r, :);
                end
            end
            g = G.';
        otherwise
            error('hold_func:noProblem', 'No evaluator for problem "%s".', par.problem);
    end
end

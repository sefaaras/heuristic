function [cost, g, pmin] = hold_wdn(X, net)
%HOLD_WDN Least-cost design of a water distribution network (Hanoi, Balerma).
%
%   X -----> ps x npipe, one design per row; each entry is a diameter INDEX in
%            [0.5, nd + 0.5], rounded to the nearest commercial size
%   net ---> 'hanoi' or 'balerma'
%   cost --> ps x 1 network cost, million $ (Hanoi) or million EUR (Balerma)
%   g -----> 1 x ps total pressure deficit over all junctions, m; <= 0 is
%            feasible (every junction at or above the minimum head)
%   pmin --> ps x 1 lowest junction pressure, m
%
% Hydraulics are EPANET's own (OWA EPANET 2.2 through epanet/hold_epanet.mexw64,
% a full steady-state solution per design) on the Exeter Centre for Water
% Systems network files in wdn/ (HAN.inp, BIN.inp; not redistributed here, see
% the README), so pressures are those of the tool every reference used:
%
%   Hanoi (Fujiwara & Khang 1990): 34 pipes, 31 junctions, Hazen-Williams with
%     C = 130 in EPANET's SI form, minimum head 30 m, 6 diameters 12-40 in at
%     1.1 D^1.5 $/m (D in inches).
%   Balerma (Reca & Martinez 2006): 454 pipes, 443 junctions, 4 reservoirs,
%     Darcy-Weisbach, minimum head 20 m, 10 PVC diameters 113-581.8 mm at
%     7.22-215.85 EUR/m.
%
% The diameters are discrete; the search sees a continuous index box and the
% evaluator rounds, as the DE references encode them. A design EPANET cannot
% solve (return code above 100) is scored infeasible by a large deficit.

    persistent N
    if isempty(N) || ~strcmp(N.net, net)
        N = load_network(net);
    end
    ps = size(X, 1);
    idx = min(max(round(X), 1), numel(N.diam));
    cost = (N.ucost(idx) * N.length) / 1e6;
    if ps == 1
        cost = cost(:);
    end
    g = zeros(1, ps);
    pmin = zeros(ps, 1);
    for r = 1:ps
        [p, err] = hold_epanet('solve', N.diam(idx(r, :)));
        if err > 100 || any(~isfinite(p))
            g(r) = 1e6;
            pmin(r) = -Inf;
        else
            g(r) = sum(max(0, N.hmin - p));
            pmin(r) = min(p);
        end
    end
end

function N = load_network(net)
    here = fileparts(mfilename('fullpath'));
    switch lower(net)
        case 'hanoi'
            file = 'HAN.inp';
            din = [12 16 20 24 30 40];
            N.diam = din * 25.4;                    % mm, the file's units
            N.ucost = 1.1 * din .^ 1.5;             % $/m
            N.hmin = 30;
        case 'balerma'
            file = 'BIN.inp';
            N.diam = [113.0 126.6 144.6 162.8 180.8 226.2 285.0 361.8 452.2 581.8];
            N.ucost = [7.22 9.10 11.92 14.84 18.38 28.60 45.39 76.32 124.64 215.85];
            N.hmin = 20;
        otherwise
            error('hold_wdn:net', 'Unknown network "%s".', net);
    end
    inp = fullfile(here, 'wdn', file);
    if ~exist(inp, 'file')
        error('hold_wdn:missing', ['%s is not in problem/HOLD/wdn/. Download it from the ' ...
              'Exeter CWS benchmark page (see problem/HOLD/README.md).'], file);
    end
    info = hold_epanet('open', inp);
    N.length = info.length;
    N.net = net;
end

function [cost, g, dispatch] = hold_chped(X, nblk)
%HOLD_CHPED Combined heat and power economic dispatch, 24-unit blocks.
%
%   X -------> ps x 30*nblk, one dispatch per row, grouped as the literature
%              numbers the units: [P of the POUs, P of the CHPs, H of the CHPs,
%              H of the HOUs]; nblk = 2 is the 48-unit system, 4 the 96-unit
%   cost ----> ps x 1 fuel cost of the REPAIRED dispatch, $/h
%   g -------> 2 x ps: unabsorbed power and heat residual (MW, MWth) minus a
%              1e-6 tolerance; <= 0 is feasible
%   dispatch > ps x 30*nblk, the repaired dispatch the cost belongs to
%
% The systems are the 24-unit set (13 power-only units, 6 CHP units, 5
% heat-only units; 2350 MW / 1250 MWth) replicated nblk times, lossless, with
% the valve-point effect on every power-only unit: 4700/2500 for 48 units,
% 9400/5000 for 96. Unit data are those of IGA-NCM (Zou et al. 2019, Tables 4-5)
% and the ICHHO appendix; every published dispatch this was checked on
% reproduces its cost. Valve-point term: |e sin(f (Pmin - P))|, which puts 55 MW
% (not 40) in the sine of units 12-13 -- the only convention under which the
% published costs reproduce.
%
% Constraints are handled HERE, by repair, as the best-known results (HECS,
% Huang et al. 2019) do, so the search sees a plain box:
%   1. a CHP point outside its feasible operating region is moved to the
%      nearest point of the region (Euclidean projection);
%   2. the power residual is absorbed by the power-only units, largest
%      capacity first, each within its limits;
%   3. the heat residual likewise by the heat-only units.
% CHP units are never touched by 2-3, so step 1 stays valid. The repair is
% deterministic, so re-evaluating a reported solution gives the same cost.
% Whatever residual the units cannot absorb is returned in g and makes the
% point infeasible; with these capacities that only happens far from any
% sensible dispatch.

    persistent D
    if isempty(D) || D.nblk ~= nblk
        D = hold_chped_data(nblk);
    end
    ps = size(X, 1);
    X = min(max(X, D.lb), D.ub);

    iP = 1:D.np;                        % POU power
    iO = D.np + (1:D.nc);               % CHP power
    iH = D.np + D.nc + (1:D.nc);        % CHP heat
    iT = D.np + 2*D.nc + (1:D.nh);      % HOU heat

    % 1. CHP points into their feasible operating regions
    for j = 1:D.nc
        poly = D.for{D.chp(j)};         % [H, P] vertices
        [X(:, iH(j)), X(:, iO(j))] = into_polygon(X(:, iH(j)), X(:, iO(j)), poly);
    end

    % 2-3. balance residuals onto the slack units
    rp = D.PD - sum(X(:, iP), 2) - sum(X(:, iO), 2);
    [X(:, iP), rp] = absorb(X(:, iP), rp, D.lb(iP), D.ub(iP), D.order_p);
    rh = D.HD - sum(X(:, iH), 2) - sum(X(:, iT), 2);
    [X(:, iT), rh] = absorb(X(:, iT), rh, D.lb(iT), D.ub(iT), D.order_h);

    % fuel cost
    P = X(:, iP);
    cost = sum(D.a .* P.^2 + D.b .* P + D.c + abs(D.e .* sin(D.f .* (D.pmin - P))), 2);
    O = X(:, iO);  H = X(:, iH);
    cost = cost + sum(D.ca .* O.^2 + D.cb .* O + D.cc + D.cd .* H.^2 + D.ce .* H + D.cf .* O .* H, 2);
    T = X(:, iT);
    cost = cost + sum(D.ha .* T.^2 + D.hb .* T + D.hc, 2);

    g = [abs(rp).'; abs(rh).'] - 1e-6;
    dispatch = X;
    if ps == 0
        g = zeros(2, 0);
    end
end

function [U, r] = absorb(U, r, lo, hi, order)
% Hand the residual r to the units in ORDER, each up to its limit.
    for u = order
        up = min(max(r, 0), hi(u) - U(:, u));      % raise where r > 0
        dn = min(max(-r, 0), U(:, u) - lo(u));     % lower where r < 0
        U(:, u) = U(:, u) + up - dn;
        r = r - up + dn;
    end
end

function [h, p] = into_polygon(h, p, poly)
% Nearest point of the polygon for every (h, p) outside it; points inside stay.
    out = ~inpolygon(h, p, poly(:, 1), poly(:, 2));
    if ~any(out)
        return;
    end
    ho = h(out);  po = p(out);
    best = inf(size(ho));  bh = ho;  bp = po;
    nv = size(poly, 1);
    for i = 1:nv
        a = poly(i, :);  b = poly(mod(i, nv) + 1, :);
        d = b - a;
        t = ((ho - a(1)) * d(1) + (po - a(2)) * d(2)) / (d * d.');
        t = min(max(t, 0), 1);
        qh = a(1) + t * d(1);  qp = a(2) + t * d(2);
        dist = (ho - qh).^2 + (po - qp).^2;
        k = dist < best;
        best(k) = dist(k);  bh(k) = qh(k);  bp(k) = qp(k);
    end
    h(out) = bh;  p(out) = bp;
end

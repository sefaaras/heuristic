function D = hold_chped_data(nblk)
%HOLD_CHPED_DATA Unit data, box and slack order of the CHPED systems.
%
%   D = hold_chped_data(nblk) for the 24-unit building block replicated nblk
%   times (2 = 48 units, 4 = 96 units). hold_chped evaluates with it and
%   hold_par reads the box (D.lb, D.ub) from it, so the two can never disagree.

    % power-only units: a, b, c, e, f, Pmin, Pmax
    pou = [0.00028 8.10 550 300 0.035   0 680
           0.00056 8.10 309 200 0.042   0 360
           0.00056 8.10 309 200 0.042   0 360
           0.00324 7.74 240 150 0.063  60 180
           0.00324 7.74 240 150 0.063  60 180
           0.00324 7.74 240 150 0.063  60 180
           0.00324 7.74 240 150 0.063  60 180
           0.00324 7.74 240 150 0.063  60 180
           0.00324 7.74 240 150 0.063  60 180
           0.00284 8.60 126 100 0.084  40 120
           0.00284 8.60 126 100 0.084  40 120
           0.00284 8.60 126 100 0.084  55 120
           0.00284 8.60 126 100 0.084  55 120];
    % CHP cost a P^2 + b P + c + d H^2 + e H + f P H, by type
    chpc = [0.0345 14.5 2650 0.030 4.200 0.031
            0.0435 36.0 1250 0.027 0.600 0.011
            0.1035 34.5 2650 0.025 2.203 0.051
            0.0720 20.0 1565 0.020 2.340 0.040];
    % feasible operating regions as [H, P] vertices, by type
    fors = {[0 98.8; 104.8 81; 180 215; 0 247], ...
            [0 44; 15.9 44; 75 40; 135.6 110.2; 32.4 125.8; 0 125.8], ...
            [0 20; 40 10; 55 45; 0 60], ...
            [0 35; 20 35; 45 90; 25 90; 0 105]};
    chp6 = [1 2 1 2 3 4];
    % heat-only units: a, b, c, Hmin, Hmax
    hou = [0.038 2.0109 950 0 2695.2
           0.038 2.0109 950 0   60
           0.038 2.0109 950 0   60
           0.052 3.0651 480 0  120
           0.052 3.0651 480 0  120];

    D.nblk = nblk;
    pou = repmat(pou, nblk, 1);
    chp = repmat(chp6, 1, nblk);
    hou = repmat(hou, nblk, 1);
    D.np = size(pou, 1);  D.nc = numel(chp);  D.nh = size(hou, 1);
    D.PD = 2350 * nblk;  D.HD = 1250 * nblk;

    D.a = pou(:, 1).';  D.b = pou(:, 2).';  D.c = pou(:, 3).';
    D.e = pou(:, 4).';  D.f = pou(:, 5).';  D.pmin = pou(:, 6).';
    D.ca = chpc(chp, 1).';  D.cb = chpc(chp, 2).';  D.cc = chpc(chp, 3).';
    D.cd = chpc(chp, 4).';  D.ce = chpc(chp, 5).';  D.cf = chpc(chp, 6).';
    D.ha = hou(:, 1).';  D.hb = hou(:, 2).';  D.hc = hou(:, 3).';
    D.chp = chp;  D.for = fors;

    % box: unit limits; a CHP unit's box is the bounding box of its region
    cplo = zeros(1, D.nc);  cphi = zeros(1, D.nc);  chhi = zeros(1, D.nc);
    for j = 1:D.nc
        v = fors{chp(j)};
        cplo(j) = min(v(:, 2));  cphi(j) = max(v(:, 2));  chhi(j) = max(v(:, 1));
    end
    D.lb = [pou(:, 6).', cplo, zeros(1, D.nc), hou(:, 4).'];
    D.ub = [pou(:, 7).', cphi, chhi, hou(:, 5).'];

    % slack order: largest capacity first, ties by unit number
    [~, D.order_p] = sort(pou(:, 7) - pou(:, 6), 'descend');
    D.order_p = D.order_p.';
    [~, D.order_h] = sort(hou(:, 5) - hou(:, 4), 'descend');
    D.order_h = D.order_h.';
end

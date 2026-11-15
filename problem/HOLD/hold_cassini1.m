function [dv, g] = hold_cassini1(X)
%HOLD_CASSINI1 ESA GTOP Cassini1: multiple gravity assist to Saturn.
%
%   X ---> ps x 6: [t0 (MJD2000), T1..T5 (days)], legs E-V-V-E-J-S
%   dv --> ps x 1 total Delta-V including launch and Saturn insertion, km/s
%   g ---> 4 x ps pericentre constraints, feasible where <= 0:
%          6351.8 - rp(Venus), 6351.8 - rp(Venus), 6778.1 - rp(Earth),
%          671492 - rp(Jupiter), in km
%
% Evaluated by GTOPX (Schlueter et al., gtopx/gtopxmex.cpp, GPL-3, built on the
% ESA ACT Space Mechanics Toolbox), compiled with `mex gtopxmex.cpp`. Its
% objective is the ESA/pagmo MGA function, which also folds a soft penalty into
% Delta-V below 6351.8 / 6778.1 / 600000 km; the four HARD constraints above are
% the ESA definition ("no constraint violation is allowed", Jupiter at 671492 km).
% Wherever they hold, the soft penalty is zero, so a feasible point is scored on
% its pure Delta-V. Best known 4.9307 km/s at
% x = [-789.8117 158.3020 449.3859 54.7490 1024.3621 4552.3080].

    ps = size(X, 1);
    dv = zeros(ps, 1);
    g = zeros(4, ps);
    for i = 1:ps
        [f, c] = gtopxmex(1, X(i, :));
        dv(i) = f(1);
        g(:, i) = -c(1:4).';
    end
end

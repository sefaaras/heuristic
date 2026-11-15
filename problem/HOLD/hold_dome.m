function [mass, g, freq] = hold_dome(X, which)
%HOLD_DOME Frequency-constrained dome truss: mass and frequency constraints.
%
%   X -----> ps x m member areas of one sector in cm^2 (m = 59 for the 1180-bar,
%            47 for the 1410-bar dome); every sector repeats them
%   which -> 1180 or 1410
%   mass --> ps x 1 structural mass, kg (added nodal masses excluded)
%   g -----> 2 x ps: [7 - f1; 9 - f3] in Hz, feasible where <= 0
%   freq --> ps x 6 lowest natural frequencies, Hz
%
% Full 3D pin-jointed model, every sector explicit (400 / 390 nodes), with a
% CONSISTENT mass matrix, rho A L / 6 [2I I; I 2I] per member, plus 100 kg on
% every free node in each direction -- the model of every result this problem is
% compared with (SHADE 2023, IDE 2026, ISMA 2022, Rao-2 2022). A lumped mass
% matrix gives a different problem (PFJA 2021): the optimum moves by several
% percent, so the two are never mixed.
%
% Symmetry makes eigenvalues come in pairs, so f1 = f2 at most designs and the
% constraint on f3 is a constraint on the second distinct pair; frequencies are
% the sorted spectrum, as in the sources.

    persistent D which_
    if isempty(D) || which_ ~= which
        D = assemble_pattern(hold_dome_data(which));
        which_ = which;
    end

    ps = size(X, 1);
    mass = zeros(ps, 1);
    freq = zeros(ps, 6);
    opts = struct('tol', 1e-12, 'maxit', 1000, 'disp', 0);
    for r = 1:ps
        A = X(r, D.group) * 1e-4;                       % 1 x members, m^2
        kv = D.kt .* (D.E * A ./ D.L);                  % 36 x members
        mv = D.mt .* (D.rho * A .* D.L / 6);
        K = sparse(D.I, D.J, kv(:), D.ndof, D.ndof);
        M = sparse(D.I, D.J, mv(:), D.ndof, D.ndof) + D.Madd;
        K = K(D.free, D.free);
        M = M(D.free, D.free);
        K = (K + K.') / 2;
        M = (M + M.') / 2;
        lam = sort(eigs(K, M, 8, 'smallestabs', opts));
        freq(r, :) = sqrt(abs(lam(1:6))).' / (2 * pi);
        mass(r) = D.rho * sum(A .* D.L);
    end
    g = [D.flim(1) - freq(:, 1), D.flim(2) - freq(:, 3)].';
end

function D = assemble_pattern(S)
% Node coordinates, member list and the sparse pattern of the full dome, built
% once; only the member scalars change from one design to the next.
    nper = size(S.coords, 1);
    m = size(S.conn, 1);
    nn = S.nsec * nper;
    phi = 2 * pi / S.nsec;

    P = zeros(nn, 3);
    for s = 0:S.nsec-1
        R = [cos(s*phi) -sin(s*phi) 0; sin(s*phi) cos(s*phi) 0; 0 0 1];
        P(s*nper + (1:nper), :) = S.coords * R.';
    end

    ne = S.nsec * m;
    a = zeros(ne, 1);  b = zeros(ne, 1);  grp = zeros(ne, 1);
    for s = 0:S.nsec-1
        for i = 1:m
            e = s*m + i;
            a(e) = glob(S.conn(i, 1), s);
            b(e) = glob(S.conn(i, 2), s);
            grp(e) = i;
        end
    end

    d = P(b, :) - P(a, :);
    L = sqrt(sum(d.^2, 2));
    c = d ./ L;                                       % direction cosines

    % 6x6 element templates, column-major, one column per member
    kt = zeros(36, ne);  mt = zeros(36, ne);
    I = zeros(36, ne);   J = zeros(36, ne);
    Mc = [2*eye(3) eye(3); eye(3) 2*eye(3)];
    for e = 1:ne
        cc = c(e, :).' * c(e, :);
        ke = [cc -cc; -cc cc];
        dof = [3*a(e)-2:3*a(e), 3*b(e)-2:3*b(e)];
        [jj, ii] = meshgrid(dof, dof);
        I(:, e) = ii(:);  J(:, e) = jj(:);
        kt(:, e) = ke(:);  mt(:, e) = Mc(:);
    end

    supp = (0:S.nsec-1) * nper + S.support;
    fixed = [3*supp-2, 3*supp-1, 3*supp];
    free = setdiff(1:3*nn, fixed);
    madd = S.madd * ones(3*nn, 1);
    madd(fixed) = 0;

    D = struct('I', I(:), 'J', J(:), 'kt', kt, 'mt', mt, 'L', L.', 'group', grp.', ...
               'ndof', 3*nn, 'free', free, 'Madd', spdiags(madd, 0, 3*nn, 3*nn), ...
               'E', S.E, 'rho', S.rho, 'flim', S.flim);

    function k = glob(n, s)
        if n <= nper
            k = s*nper + n;
        else
            k = mod(s + 1, S.nsec)*nper + (n - nper);
        end
    end
end

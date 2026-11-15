function E = hold_protein_ab3d(X, seq)
%HOLD_PROTEIN_AB3D Energy of the 3D AB off-lattice protein model.
%
%   X ---> ps x (2L-5), one conformation per row, angles in DEGREES in
%          [-180, 180]: the L-2 bond angles theta, then the L-3 torsions beta
%   seq -> 1 x L char of 'A' (hydrophobic) and 'B' (hydrophilic)
%   E ---> ps x 1 energy, lower is better
%
% Stillinger's model in the form the HOLD references use (DE2P, DElscr, SGDE,
% cuHjDE-3D; Li et al. 2015 code):
%
%   E = 1/4 sum_{i=1}^{L-2} (1 - cos theta_i)
%       + 4 sum_{i=1}^{L-2} sum_{j=i+2}^{L} (r_ij^-12 - C_ij r_ij^-6),
%
% with C = 1 for A-A, 0.5 for B-B and -0.5 for A-B, unit bonds, and the chain
% built as p1 = (0,0,0), p2 = (0,1,0), p3 = p2 + (cos th1, sin th1, 0),
% p_k = p_{k-1} + (cos th_{k-2} cos b_{k-3}, sin th_{k-2} cos b_{k-3},
% sin b_{k-3}). The bending term is written with the bond ANGLE theta; the
% bond-vector form 1/4 sum (1 - b_i . b_{i+1}) is a different model and gives
% different energies on the same angles, so results under it are not comparable.

    L = numel(seq);
    ps = size(X, 1);
    th = X(:, 1:L-2);
    be = X(:, L-1:2*L-5);

    % chain coordinates, ps x L for each axis
    px = zeros(ps, L);  py = zeros(ps, L);  pz = zeros(ps, L);
    py(:, 2) = 1;
    px(:, 3) = cosd(th(:, 1));
    py(:, 3) = 1 + sind(th(:, 1));
    for k = 4:L
        t = th(:, k-2);  b = be(:, k-3);
        cb = cosd(b);
        px(:, k) = px(:, k-1) + cosd(t) .* cb;
        py(:, k) = py(:, k-1) + sind(t) .* cb;
        pz(:, k) = pz(:, k-1) + sind(b);
    end

    E = sum(1 - cosd(th), 2) / 4;

    isA = (seq == 'A');
    for i = 1:L-2
        j = i+2:L;
        % pair coefficients for residue i against i+2..L
        C = 0.5 * ones(1, numel(j));
        C(isA(i) & isA(j)) = 1;
        C(isA(i) ~= isA(j)) = -0.5;
        r2 = (px(:, j) - px(:, i)).^2 + (py(:, j) - py(:, i)).^2 + (pz(:, j) - pz(:, i)).^2;
        r6 = r2 .^ 3;
        E = E + 4 * sum(1 ./ r6.^2 - C ./ r6, 2);
    end
end

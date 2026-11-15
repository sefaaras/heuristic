function rmse = hold_pv_ddm(X)
%HOLD_PV_DDM Double-diode model of the R.T.C. France solar cell, RMSE.
%
%   X ----> ps x 7, one candidate per row: [Iph, I01, I02, n1, n2, Rs, Rp]
%           Iph in A, I01 and I02 in microampere, Rs and Rp in ohm
%   rmse -> ps x 1
%
% The 26 I-V points measured at 33 C (Easwarakhanthan et al. 1986) and the
% residual form of the error: the model current is evaluated at the MEASURED
% current on the right-hand side,
%
%   I_m = Iph - I01 (exp((V + I Rs)/(n1 Vt)) - 1)
%             - I02 (exp((V + I Rs)/(n2 Vt)) - 1) - (V + I Rs)/Rp,
%
% and RMSE = sqrt(mean((I_m - I)^2)) over N = 26. This is the form the
% best-known 9.8248485e-4 belongs to (SEDE, Liang et al. 2020). Solving the
% implicit equation for the current instead ("exact" form) gives a different
% and smaller number for the same parameters, so the two are never compared.
%
% Data, constants (k = 1.3806503e-23, q = 1.60217646e-19, T = 306.15 K) and the
% parameter order are those of the rePVest reference code (Gao et al., MIT
% licence), the same ones the ecilab codes of SEDE, IJAYA and PGJAYA use. The
% saturation currents are searched in microampere, as every source states its
% box, so all seven variables live on comparable scales.

    persistent V I Vt
    if isempty(V)
        V = [-0.2057 -0.1291 -0.0588  0.0057  0.0646  0.1185  0.1678  0.2132 ...
              0.2545  0.2924  0.3269  0.3585  0.3873  0.4137  0.4373  0.4590 ...
              0.4784  0.4960  0.5119  0.5265  0.5398  0.5521  0.5633  0.5736 ...
              0.5833  0.5900];
        I = [ 0.7640  0.7620  0.7605  0.7605  0.7600  0.7590  0.7570  0.7570 ...
              0.7555  0.7540  0.7505  0.7465  0.7385  0.7280  0.7065  0.6755 ...
              0.6320  0.5730  0.4990  0.4130  0.3165  0.2120  0.1035 -0.0100 ...
             -0.1230 -0.2100];
        Vt = 1.3806503e-23 * 306.15 / 1.60217646e-19;
    end

    Iph = X(:, 1);
    I01 = X(:, 2) * 1e-6;
    I02 = X(:, 3) * 1e-6;
    n1  = X(:, 4);
    n2  = X(:, 5);
    Rs  = X(:, 6);
    Rp  = X(:, 7);

    U = V + I .* Rs;                       % ps x 26, V + I*Rs at every point
    Im = Iph - I01 .* (exp(U ./ (n1 * Vt)) - 1) ...
             - I02 .* (exp(U ./ (n2 * Vt)) - 1) - U ./ Rp;
    rmse = sqrt(mean((Im - I) .^ 2, 2));
end

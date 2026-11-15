function [mass, viol] = hold_truss200(X)
%HOLD_TRUSS200 200-bar planar truss with natural-frequency constraints.
%
%   X ----> ps x 29 cross-sectional areas of the 29 member groups, cm^2
%   mass -> ps x 1 structural mass, kg
%   viol -> ps x 1 total frequency violation, 0 = feasible
%
% Evaluated with the authors' own code, unmodified (truss200/, File Exchange
% 180060, Ozturk & Kahraman 2025, BSD licence in LICENSE_fx180060.txt), because
% the optimum depends on modelling choices the papers do not all share: this code
% uses a LUMPED mass matrix and the rounded 6.10 m / 3.66 m geometry, and the
% published 2174.333 kg belongs to exactly that model. Two of its details decide
% feasibility and are kept as they are: frequencies are rounded to 3 decimals
% before the check (so f1 = 4.9995 Hz passes f1 >= 5), and the constraint is
% f1 >= 5, f2 >= 10, f3 >= 15 Hz. The violation measure is the code's own sum of
% relative shortfalls, which uses f1 in all three terms; that only scales the
% violation of an infeasible design and never changes which designs are feasible.

    ps = size(X, 1);
    mass = zeros(ps, 1);
    viol = zeros(ps, 1);
    for i = 1:ps
        out = Truss_modal_analysis_200bar(X(i, :));
        mass(i) = out.OBJVAL;
        viol(i) = out.VIO;
    end
end

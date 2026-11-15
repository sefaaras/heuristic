function D = hold_dome_data(which)
%HOLD_DOME_DATA One sector of the 1180-bar and 1410-bar frequency-constrained domes.
%
%   D.nsec      sectors (rotational copies about the z axis)
%   D.coords    nper x 3 node coordinates of sector 1, m
%   D.conn      m x 2 members of sector 1; a node number above nper is node
%               (number - nper) of the NEXT sector
%   D.support   the node of every sector that is pinned
%   D.flim      [f1, f3] lower limits, Hz
%
% Geometry and members as tabulated by Kaveh et al. (SHADE, PPCE 2023, CC-BY) and
% the later papers of the group, with ONE correction: 1180-bar node 18 is at
% x = 14.4917 m; the tables print 14.9179, which the published weights do not
% reproduce. Supports (never stated in the papers: node 10 of every sector for
% 1180, node 7 for 1410) and the full models built from these sectors were
% checked against independent SAP2000 models of both domes: same 400/390 nodes
% (coordinates within 7e-5 m / 2e-3 m, the latter from 3-decimal rounding in the
% tables), the same 1180/1410 members and the same supports.
%
% Every member of a sector is its own design variable (59 and 47 groups); the
% sectors repeat it. Common data: E = 200 GPa, rho = 7850 kg/m^3, areas 1-100
% cm^2, 100 kg added at every free node.

    switch which
        case 1180
            D.nsec = 20;
            D.support = 10;
            D.flim = [7 9];
            D.coords = [3.1181 0.0000 14.6723
            6.1013 0.0000 13.7031
            8.8166 0.0000 12.1354
            11.1476 0.0000 10.0365
            12.9904 0.0000 7.5000
            14.2657 0.0000 4.6358
            14.9179 0.0000 1.5676
            14.9179 0.0000 -1.5677
            14.2656 0.0000 -4.6359
            12.9903 0.0000 -7.5001
            4.5788 0.7252 14.2657
            7.4077 1.1733 12.9904
            9.9130 1.5701 11.1476
            11.9860 1.8984 8.8165
            13.5344 2.1436 6.1013
            14.4917 2.2953 3.1180
            14.8153 2.3465 0.0000
            14.4917 2.2953 -3.1181
            13.5343 2.1436 -6.1014
            3.1181 0.0000 13.7031];
            D.conn = [1 2
            1 11
            1 20
            1 21
            1 40
            2 3
            2 11
            2 12
            2 20
            2 22
            3 4
            3 12
            3 13
            3 23
            4 5
            4 13
            4 14
            4 24
            5 6
            5 14
            5 15
            5 25
            6 7
            6 15
            6 16
            6 26
            7 8
            7 16
            7 17
            7 27
            8 9
            8 17
            8 18
            8 28
            9 10
            9 18
            9 19
            9 29
            10 19
            10 30
            11 21
            11 22
            12 22
            12 23
            13 23
            13 24
            14 24
            14 25
            15 25
            15 26
            16 26
            16 27
            17 27
            17 28
            18 28
            18 29
            19 29
            19 30
            20 40];
        case 1410
            D.nsec = 30;
            D.support = 7;
            D.flim = [7 9];
            D.coords = [1.0000 0.0000 4.0000
            3.0000 0.0000 3.7500
            5.0000 0.0000 3.2500
            7.0000 0.0000 2.7500
            9.0000 0.0000 2.0000
            11.0000 0.0000 1.2500
            13.0000 0.0000 0.0000
            1.9890 0.2090 3.0000
            3.9780 0.4180 2.7500
            5.9670 0.6270 2.2500
            7.9560 0.8360 1.7500
            9.9450 1.0453 1.0000
            11.9340 1.2543 -0.5000];
            D.conn = [1 2
            1 8
            1 14
            2 3
            2 8
            2 9
            2 15
            3 4
            3 9
            3 10
            3 16
            4 5
            4 10
            4 11
            4 17
            5 6
            5 11
            5 12
            5 18
            6 7
            6 12
            6 13
            6 19
            7 13
            8 9
            8 14
            8 15
            8 21
            9 10
            9 15
            9 16
            9 22
            10 11
            10 16
            10 17
            10 23
            11 12
            11 17
            11 18
            11 24
            12 13
            12 18
            12 19
            12 25
            13 19
            13 20
            13 26];
        otherwise
            error('hold_dome_data:which', 'Unknown dome %s.', mat2str(which));
    end
    D.E = 2e11;
    D.rho = 7850;
    D.madd = 100;
end

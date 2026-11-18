function par = hold_par(k)
%HOLD_PAR Grid data for one HOLD case.
%
%   n = hold_par() returns the number of cases.
%
%   par.n           dimension
%   par.g, par.h    numbers of inequality and equality constraints
%   par.xmin/xmax   1 x n box
%   par.name        problem name as the literature writes it
%   par.problem     problem id in literature/problems.csv
%   par.instance    instance id in literature/problems.csv ('' = base)
%   par.best_known  best feasible objective known for this instance, for
%                   reference only -- nothing in the pipeline scores against it
%   par.maxFE       FE budget of this case
%   par.runs        independent runs of this case
%   par.matches     the studies whose protocol this case reproduces: each has a
%                   result at exactly this instance, budget and run count in
%                   literature/results.csv (documentation only; '' = none)
%
% Same shape and field names as Cal_par, so the harness reads both through one
% code path, plus the budget and the run count: HOLD compares
% with published numbers, and a published number is only comparable with a run
% made under its own protocol. A CASE is therefore one problem instance under one
% protocol, and a problem the literature ran at two budgets is two cases.
%
% Numbering is APPEND-ONLY. Results are stored by case number, so a case keeps
% its number for good; a new problem or protocol takes the next free one, and a
% retired case is left in place and marked, never reused.

    CASES = { ...
        % problem       instance  maxFE   runs  matches
        'pv_ddm_rtc',   '',       50000,  30,   'liang2020_sede; yu2019_pgjaya; chen2025_decm; yu2017_ijaya'
        'pv_ddm_rtc',   '',       30000,  30,   'hu2021_rlde'
        'protein_ab3d', '2KGU',   200000, 30,   'boskovic2020_de2p; rakhshani2019_sgde; boskovic2016_depfo; boiani2020_cuhjde3d; li2015_iabc'
        'protein_ab3d', '1AGT',   200000, 30,   'boskovic2020_de2p; rakhshani2019_sgde; boskovic2016_depfo; boiani2020_cuhjde3d; li2015_iabc'
        'protein_ab3d', '1CRN',   200000, 30,   'boskovic2020_de2p; rakhshani2019_sgde; boskovic2016_depfo; boiani2020_cuhjde3d; li2015_iabc'
        'protein_ab3d', '1GK4',   200000, 30,   'boskovic2020_de2p; rakhshani2019_sgde; boskovic2016_depfo; boiani2020_cuhjde3d; li2015_iabc'
        'protein_ab3d', '2EWH',   800000, 30,   'boskovic2020_de2p; boskovic2016_depfo; boiani2020_cuhjde3d; li2015_iabc'
        % cases 8-9: no study prints this exact protocol. 8 = the 100 x 3000 of
        % hakmi2024_iaqt / hakmi2023_koa (runs not stated) with the 50 runs of
        % huang2019_hecs; 9 = the 1.6M FEs of zou2019_igancm (30 runs) with the 50
        % runs of li2024_mga (FEs not stated). Compared with the nearest protocols.
        'chped48',      '',       300000, 50,   ''
        'chped48',      '',      1600000, 50,   ''
        'chped96',      '',       300000, 50,   'liu2022_ndide'
        'chped96',      '',      2000000, 50,   'liu2021_nde'
        'truss200_freq', 'fx180060', 290000, 21, 'ozturk2025_nsm'
        'truss200_freq', 'fx180060', 580000, 20, 'ghiaskar2026_pfa'
        'dome1180_freq', '',       20000,  20,   'kaveh2023_shade'
        'dome1180_freq', '',       30000,  20,   'kaveh2026_ide; kaveh2021_chaoticwsa'
        'dome1410_freq', '',       20000,  20,   'kaveh2023_shade'
        'dome1410_freq', '',       30000,  20,   'kaveh2026_ide; kaveh2025_ihgo; kaveh2021_chaoticwsa'
        'gtop_cassini1', '',      150000, 25,   'tang2022_hlshade'
        'gtop_cassini1', '',       50000, 30,   'turkoglu2025_caaa'
        'wdn_balerma',  '',      2500000, 10,   'du2023_aepdrsde'
        'wdn_balerma',  '',      2000000, 10,   'neshat2019_cmags; zheng2011_nlpde'
        'wdn_balerma',  '',       500000, 10,   'moosavian2019_fde'
        'wdn_balerma',  '',        45400, 10,   'moosavian2019_fde'
        'wdn_hanoi',    'w10.667', 200000, 100, 'du2023_aepdrsde'
        'wdn_hanoi',    'w10.667', 100000, 100, 'moosavian2019_fde'
        'wdn_hanoi',    'w10.667', 40000,  100, 'zheng2014_blpde'
        'wdn_hanoi',    'w10.667', 10000,  100, 'moosavian2019_fde'
        'wdn_hanoi',    'w10.667', 5000,   100, 'moosavian2019_fde'
        'wdn_hanoi',    'w10.667', 1000,   100, 'moosavian2019_fde'
        };

    if nargin == 0
        par = size(CASES, 1);
        return;
    end
    if ~isscalar(k) || k < 1 || k > size(CASES, 1) || k ~= fix(k)
        error('hold_par:badIndex', 'Case index must be an integer in 1..%d, got %s.', ...
              size(CASES, 1), mat2str(k));
    end

    par = struct('n', 0, 'g', 0, 'h', 0, 'xmin', [], 'xmax', [], 'name', '', ...
                 'problem', CASES{k, 1}, 'instance', CASES{k, 2}, 'best_known', NaN, ...
                 'maxFE', CASES{k, 3}, 'runs', CASES{k, 4}, 'matches', CASES{k, 5});

    switch par.problem
        case 'pv_ddm_rtc'
            % [Iph (A), I01 (uA), I02 (uA), n1, n2, Rs (ohm), Rp (ohm)], the box
            % every source states for the R.T.C. France cell
            par.name = 'PV double-diode model, R.T.C. France cell';
            par.n = 7;
            par.xmin = [0, 0, 0, 1, 1, 0, 0];
            par.xmax = [1, 1, 1, 2, 2, 0.5, 100];
            par.best_known = 9.8248485e-4;

        case 'protein_ab3d'
            % AB sequences of DE2P Table 2 (Boskovic & Brest 2020); D = 2L - 5
            % angles in degrees. Best known: the long-budget runs (DE2P 4-day,
            % cuDSMjDE 5e10 FEs for 2EWH), far below anything a 1e5-scale budget
            % reaches -- the gap is what the problem measures.
            SEQ = struct( ...
                'S2KGU', {{'ABAABBAABABBABAABAABABABABABAAABBB', -53.6756}}, ...
                'S1AGT', {{'AAAABABABABABAABAABBAAABBABAABBBABABAB', -66.2973}}, ...
                'S1CRN', {{'BBAAABAAABBBBBAABAAABABAAAABBBAAAAAAAABAAABBAB', -95.3159}}, ...
                'S1GK4', {{['ABABAABABBBBABBBABBABBBBAABAABBBBBAABABBBABBABBBAABBABBBBBAABAB' ...
                            'AAABABAABBBBAABABBBBA'], -112.3674}}, ...
                'S2EWH', {{['AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAAB' ...
                            'AAABAABBAABAAAAABAAABABBBABBAAABAABA'], -257.3538}});
            e = SEQ.(['S' par.instance]);
            par.seq = e{1};
            L = numel(par.seq);
            par.name = sprintf('Protein 3D AB off-lattice model, %s (L = %d)', par.instance, L);
            par.n = 2 * L - 5;
            par.xmin = -180 * ones(1, par.n);
            par.xmax =  180 * ones(1, par.n);
            par.best_known = e{2};

        case {'chped48', 'chped96'}
            % [P of 13*nblk power-only units, P and H of 6*nblk CHP units, H of
            % 5*nblk heat-only units]; the box and the data come from
            % hold_chped_data, and hold_chped repairs the balances, so the search
            % sees no constraint unless a residual is left unabsorbed
            nblk = str2double(par.problem(6:end)) / 24;
            D = hold_chped_data(nblk);
            par.name = sprintf('Combined heat and power economic dispatch, %d units (%d MW / %d MWth)', ...
                               24 * nblk, D.PD, D.HD);
            par.n = numel(D.lb);
            par.g = 2;
            par.xmin = D.lb;
            par.xmax = D.ub;
            par.nblk = nblk;
            par.best_known = 115611.7369;          % HECS, Huang et al. 2019
            if nblk == 4
                par.best_known = 231206.6228;
            end

        case 'truss200_freq'
            % 29 group areas in cm^2, the box of the authors' code
            % (problem_bounds, type 5); evaluated by their own analysis
            par.name = '200-bar planar truss with frequency constraints (FX 180060 model)';
            par.n = 29;
            par.g = 1;
            par.xmin = 0.1 * ones(1, 29);
            par.xmax = 30 * ones(1, 29);
            par.best_known = 2174.333;          % NSM-LSHADE-CnEpSin, Ozturk & Kahraman 2025

        case {'dome1180_freq', 'dome1410_freq'}
            % one area per member of a sector, cm^2; consistent-mass model
            par.dome = str2double(par.problem(5:8));
            D = hold_dome_data(par.dome);
            par.name = sprintf('%d-bar dome truss with frequency constraints', par.dome);
            par.n = size(D.conn, 1);
            par.g = 2;
            par.xmin = ones(1, par.n);
            par.xmax = 100 * ones(1, par.n);
            par.best_known = 37321.72;          % SHADE, Kaveh et al. 2023, 20k FEs
            if par.dome == 1410
                par.best_known = 10232.61;      % IDE, Kaveh et al. 2026, 30k FEs
            end

        case 'gtop_cassini1'
            % [t0 (MJD2000), T1..T5 (days)], the ESA GTOP box
            par.name = 'Cassini1 multiple gravity assist trajectory (ESA GTOP)';
            par.n = 6;
            par.g = 4;
            par.xmin = [-1000, 30, 100, 30, 400, 1000];
            par.xmax = [0, 400, 470, 400, 2000, 6000];
            par.best_known = 4.9307;

        case {'wdn_balerma', 'wdn_hanoi'}
            % one diameter index per pipe, rounded inside hold_wdn
            if strcmp(par.problem, 'wdn_hanoi')
                par.net = 'hanoi';  nd = 6;  par.n = 34;
                par.name = 'Hanoi water distribution network design (EPANET 2, H-W C = 130)';
                par.best_known = 6.081;              % M$
            else
                par.net = 'balerma';  nd = 10;  par.n = 454;
                par.name = 'Balerma irrigation network design (EPANET 2, Darcy-Weisbach)';
                par.best_known = 1.9205;             % MEUR, SA-SSDE and AEPD-RSDE
            end
            par.g = 1;
            par.xmin = 0.5 * ones(1, par.n);
            par.xmax = (nd + 0.5) * ones(1, par.n);

        otherwise
            error('hold_par:noProblem', 'No definition for problem "%s".', par.problem);
    end
end

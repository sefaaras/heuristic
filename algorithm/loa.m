% ----------------------------------------------------------------------- %
% Lion Optimization Algorithm (LOA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   nPop  = 50              % Lions: 4 prides of 8 F + 2 M, and 10 nomads (2 F + 8 M)
%   P, N  = 4, 0.2          % Prides; nomad share of the lions
%   S     = 0.8             % Sex rate: female share in prides, male share among nomads
%   R     = 0.2             % Roaming: share of the territory each resident male visits
%   Ma    = 0.3             % Mating probability of each female
%   Mu    = 0.2             % Per-gene mutation probability of one of the two cubs
%   I     = 0.4             % Immigration: share of a pride's female quota that leaves
%   theta = U(-pi/6, pi/6)  % Turn of the safe-place and roaming moves
%
% Algorithm Concept:
%   - Hunting: each pride female hunts in one of three groups or stays; hunters close
%     on a dummy prey at their centroid, the wings from the opposite point 2*prey - x
%   - A hunter that improves makes the prey flee: prey + rand*PI*(prey - hunter)
%   - Safe place: the other females move 2*D*rand toward a territory point won in a
%     tournament of size max(2, ceil(K/2)), K = lions improved, plus a sideways step
%   - Roaming: each resident male walks U(0, 2d) toward R of the territory points in
%     turn, then settles on his best visited position
%   - Nomads redraw each coordinate with probability 0.1 + min(0.5, gap to the best)
%   - Mating: cubs beta*x_f + (1 - beta)*mean(x_m) and the converse, beta ~ N(0.5, 0.1),
%     one of them mutated; resident females take 1..NR males, nomad females one
%   - Defence, migration, equilibrium: the weakest males are expelled, nomad males take
%     over prides, I of the females leave, the best nomad females refill, the worst die
%
% Reference:
% Maziar Yazdani, Fariborz Jolai,
% Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm,
% Journal of Computational Design and Engineering 3 (2016) 24-36.
% https://doi.org/10.1016/j.jcde.2015.06.003
% ----------------------------------------------------------------------- %
% Implementation Note:
% No author code was found: ported from the paper (Sec. 2.2, Pseudo-codes 1-5, Fig.
% 11, Table 2, 50 lions per Sec. 3); NiaPy/opytimizer loa.py only cross-checked.
% Open in the paper, chosen here: a pride female hunts in group 1, 2, 3 or stays,
% uniformly (as both ports); the centre is the non-empty group of largest summed
% f; PI = (f_old - f_new)/|f_old|; {R1} is a unit vector and roaming turns it by
% theta, both in box-normalised space; R*n rounds, at least 1; mating uses current
% positions of 1..NR random males and mutates one random cub; a nomad female takes
% one random male (Sec. 2.2.5; Fig. 11 says one of the best); Eq. (12) divides by
% |Best_nomad|; moves are not greedy, lions rank by current f (Eq. 2), territory by
% personal best; cubs count as improved; moves are clamped; history = all lions.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = loa(problem)

    dim = problem.dimension;
    lb = problem.lb(:)';
    ub = problem.ub(:)';
    span = ub - lb;
    maxFE = problem.maxFe;

    nPop = 50;
    nPride = 4;
    pNomad = 0.2;
    pSex = 0.8;
    pRoam = 0.2;
    pMate = 0.3;
    pMut = 0.2;
    pImm = 0.4;

    nNomad = round(pNomad * nPop);
    nRes = nPop - nNomad;
    prideSize = floor(nRes / nPride) * ones(1, nPride);
    prideSize(1:mod(nRes, nPride)) = prideSize(1:mod(nRes, nPride)) + 1;
    femQuota = round(pSex * prideSize);            % maximum females of a pride (Sec. 2.2.7)
    maleQuota = prideSize - femQuota;              % resident males the defence keeps
    nomFemQuota = round((1 - pSex) * nNomad);      % the sex rate is inverted for nomads
    nomMaleQuota = nNomad - nomFemQuota;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    X = lb + rand(nPop, dim) .* span;
    [F, FE] = calculate_fitness(X', problem, FE);
    F = F(:)';
    PB = X;   % best visited positions; a pride's territory is its members' PB
    PF = F;

    [best_fitness, bidx] = min(F);
    best_solution = X(bidx, :);
    for e = 1:min(nPop, maxFE)
        curve(e) = min(F(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, X, F, population_history, fitness_history, ...
            history_index, maxFE);
    end
    stop = FE >= maxFE;

    % grp 0 = nomad, 1..nPride = pride; fem = female
    perm = randperm(nPop);
    grp = zeros(1, nPop);
    fem = false(1, nPop);
    fem(perm(1:nomFemQuota)) = true;
    c = nNomad;
    for j = 1:nPride
        idx = perm(c + 1:c + prideSize(j));
        grp(idx) = j;
        fem(idx(1:femQuota(j))) = true;
        c = c + prideSize(j);
    end
    imp = true(1, nPop);   % S(i, t, P) of Eq. (7); a lion new to the run counts as improved

    while ~stop
        n0 = numel(F);
        PFstart = PF;

        for j = 1:nPride
            % Hunting (Sec. 2.2.2, Pseudo-code 1)
            fs = find(grp == j & fem);
            role = randi(4, 1, numel(fs)) - 1;   % 0 stays behind, 1..3 hunting group
            hunters = fs(role > 0);
            hrole = role(role > 0);
            if ~isempty(hunters)
                cum = -inf(1, 3);
                for g = unique(hrole)
                    cum(g) = sum(F(hunters(hrole == g)));
                end
                [~, centre] = max(cum);
                prey = mean(X(hunters, :), 1);
                for q = randperm(numel(hunters))
                    k = hunters(q);
                    if hrole(q) == centre
                        xn = X(k, :) + rand(1, dim) .* (prey - X(k, :));   % Eq. (5)
                    else
                        opp = 2 * prey - X(k, :);
                        xn = opp + rand(1, dim) .* (prey - opp);   % Eq. (4)
                    end
                    fold = F(k);
                    move(k, min(max(xn, lb), ub));
                    if stop, break; end
                    if F(k) < fold
                        prey = prey + rand * ((fold - F(k)) / abs(fold)) * (prey - X(k, :));   % Eq. (3)
                        prey = min(max(prey, lb), ub);
                    end
                end
                if stop, break; end
            end

            % Moving toward safe place (Sec. 2.2.3, Eqs. 6-9)
            members = find(grp == j);
            tsize = min(numel(members), max(2, ceil(sum(imp(members)) / 2)));
            for k = fs(role == 0)
                cand = members(randperm(numel(members), tsize));
                [~, w] = min(PF(cand));
                move(k, toward(X(k, :), PB(cand(w), :), lb, ub, span, false));
                if stop, break; end
            end
            if stop, break; end

            % Roaming of the resident males (Sec. 2.2.4, Eq. 10, Pseudo-code 3)
            nVisit = max(1, round(pRoam * numel(members)));
            for k = find(grp == j & ~fem)
                for p = members(randperm(numel(members), nVisit))
                    move(k, toward(X(k, :), PB(p, :), lb, ub, span, true));
                    if stop, break; end
                end
                if stop, break; end
                X(k, :) = PB(k, :);   % his best visited position becomes his current one
                F(k) = PF(k);
            end
            if stop, break; end

            % Mating (Sec. 2.2.5, Eqs. 13-14); cubs join the pride at once
            ms = find(grp == j & ~fem);
            for k = find(grp == j & fem)
                if rand < pMate && ~isempty(ms)
                    sel = ms(randperm(numel(ms), randi(numel(ms))));
                    breed(k, mean(X(sel, :), 1), j);
                    if stop, break; end
                end
            end
            if stop, break; end

            % Defence against new mature resident males (Pseudo-code 5)
            ms = find(grp == j & ~fem);
            if numel(ms) > maleQuota(j)
                [~, o] = sort(F(ms));
                grp(ms(o(maleQuota(j) + 1:end))) = 0;
            end
        end
        if stop, break; end

        % Nomads move randomly (Eqs. 11-12, Pseudo-code 4)
        nom = find(grp == 0);
        fb = min(F(nom));
        pr = 0.1 + min(0.5, (F(nom) - fb) / abs(fb));
        pr(F(nom) == fb) = 0.1;
        for q = 1:numel(nom)
            k = nom(q);
            xn = X(k, :);
            r = rand(1, dim) <= pr(q);
            xn(r) = lb(r) + rand(1, nnz(r)) .* span(r);
            move(k, xn);
            if stop, break; end
        end
        if stop, break; end

        % Nomad mating: one randomly chosen nomad male (Sec. 2.2.5)
        ms = find(grp == 0 & ~fem);
        for k = find(grp == 0 & fem)
            if rand < pMate && ~isempty(ms)
                breed(k, X(ms(randi(numel(ms))), :), 0);
                if stop, break; end
            end
        end
        if stop, break; end

        % Defence against nomad males: each attacks a random subset of prides (Sec. 2.2.6)
        for k = find(grp == 0 & ~fem)
            for j = find(rand(1, nPride) < 0.5)
                res = find(grp == j & ~fem);
                z = find(F(k) < F(res), 1);
                if ~isempty(z)
                    grp(res(z)) = 0;
                    grp(k) = j;
                    break;
                end
            end
        end

        % Migration (Sec. 2.2.7): the surplus plus I of the female quota leave each pride
        for j = 1:nPride
            fs = find(grp == j & fem);
            nMig = min(numel(fs), max(0, numel(fs) - femQuota(j)) + round(pImm * femQuota(j)));
            grp(fs(randperm(numel(fs), nMig))) = 0;
        end

        % The best nomad females refill the prides in random order (Fig. 10)
        fs = find(grp == 0 & fem);
        [~, o] = sort(F(fs));
        gap = zeros(1, nPride);
        for j = 1:nPride
            gap(j) = max(0, femQuota(j) - nnz(grp == j & fem));
        end
        nFill = min(sum(gap), numel(fs));
        chosen = fs(o(1:nFill));
        chosen = chosen(randperm(nFill));
        c = 0;
        for j = 1:nPride
            t = min(gap(j), nFill - c);
            grp(chosen(c + 1:c + t)) = j;
            c = c + t;
        end

        % Population equilibrium (Sec. 2.2.8): the worst nomads beyond each sex's quota die
        alive = true(1, numel(F));
        fs = find(grp == 0 & fem);
        [~, o] = sort(F(fs));
        alive(fs(o(nomFemQuota + 1:end))) = false;
        ms = find(grp == 0 & ~fem);
        [~, o] = sort(F(ms));
        alive(ms(o(nomMaleQuota + 1:end))) = false;

        imp = true(1, numel(F));
        imp(1:n0) = PF(1:n0) < PFstart;   % Eq. (7)
        X = X(alive, :);
        F = F(alive);
        PB = PB(alive, :);
        PF = PF(alive);
        grp = grp(alive);
        fem = fem(alive);
        imp = imp(alive);
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Lion k moves to xm: one evaluation, personal best kept, one history sample
    function move(k, xm)
        [fv, FE] = calculate_fitness(xm', problem, FE);
        X(k, :) = xm;
        F(k) = fv;
        if fv < PF(k)
            PB(k, :) = xm;
            PF(k) = fv;
        end
        track(xm, fv);
    end

    % Two cubs of female k and mate xmate, Eqs. (13)-(14): one mutated, one female
    function breed(k, xmate, g)
        bt = 0.5 + 0.1 * randn;
        off = [bt * X(k, :) + (1 - bt) * xmate; (1 - bt) * X(k, :) + bt * xmate];
        mi = randi(2);
        mr = rand(1, dim) < pMut;
        off(mi, mr) = lb(mr) + rand(1, nnz(mr)) .* span(mr);
        off = min(max(off, lb), ub);
        sx = rand < 0.5;
        for cc = 1:2
            [fv, FE] = calculate_fitness(off(cc, :)', problem, FE);
            X(end + 1, :) = off(cc, :);
            F(end + 1) = fv;
            PB(end + 1, :) = off(cc, :);
            PF(end + 1) = fv;
            grp(end + 1) = g;
            fem(end + 1) = xor(sx, cc == 2);
            imp(end + 1) = true;
            track(off(cc, :), fv);
            if stop, return; end
        end
    end

    % Best-so-far at this evaluation, then one history sample of every living lion
    function track(xt, ft)
        if ft < best_fitness
            best_fitness = ft;
            best_solution = xt;
        end
        curve(FE) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            FE, X, F, population_history, fitness_history, ...
            history_index, maxFE);
        stop = FE >= maxFE;
    end
end

% Eqs. (6) and (10) in box-normalised space: R1 unit toward the target, R2 a random unit normal
function xn = toward(x, target, lb, ub, span, roam)
    z = (x - lb) ./ span;
    r1 = (target - lb) ./ span - z;
    d = norm(r1);
    if ~(d > 0)
        xn = x;
        return;
    end
    R1 = r1 / d;
    R2 = randn(1, numel(x));
    R2 = R2 - (R2 * R1') * R1;
    nr = norm(R2);
    if nr > 0
        R2 = R2 / nr;
    end
    theta = (2 * rand - 1) * pi / 6;
    if roam
        z = z + 2 * d * rand * (cos(theta) * R1 + sin(theta) * R2);   % Eq. (10), turned by theta
    else
        z = z + 2 * d * rand * R1 + (2 * rand - 1) * tan(theta) * d * R2;   % Eq. (6)
    end
    xn = min(max(lb + z .* span, lb), ub);
end

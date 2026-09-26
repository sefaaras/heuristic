% ----------------------------------------------------------------------- %
% jSO with Progressive Archive (jSOa)
% Variant of jSO; CEC 2024 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N = round(25*log(D)*sqrt(D)) -> 4 (linear)  % Population; D < 6 is sized as D = 6
%   H = 5                        % Memory slots; the last one stays at MF = MCR = 0.9
%   MF = 0.3, MCR = 0.8          % Initial memory contents of the other slots
%   Asize_max = 2.6*N, Ap = 0.2  % Archive size; worst share of it that new entries replace
%   p = 0.25 -> 0.125 (linear)   % pbest share of the population
%   peig = 0.4, ps = 0.5         % Eigen-crossover chance per generation; best share for its basis
%   CR ~ N(MCR, sqrt(0.1)), F ~ Cauchy(MF, 0.1)  % Parameter sampling
%
% Algorithm Concept:
%   - jSO's current-to-pbest-w/1 with Fw = 0.7F, 0.8F, 1.2F over the budget,
%     F capped at 0.7 and CR floored at 0.7, then 0.6, early in the run
%   - Individuals are replaced in place one at a time, and once the generation
%     has a success the memory slot is rewritten after every further trial
%   - Progressive archive: once full, a new entry overwrites a random member of
%     the worst 20 % of the archive by fitness instead of any member
%   - With probability 0.4 a generation crosses over in the eigenbasis of the
%     covariance of the best half of the population
%   - Out-of-box coordinates are reflected at the violated bound until inside
%   - Linear population reduction; the archive shrinks with it by random removal
%
% Reference:
% Petr Bujok,
% Progressive Archive in Adaptive jSO Algorithm,
% Mathematics 12 (16) (2024) 2534.
% https://doi.org/10.3390/math12162534
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from jSOaEig_.m, the only released code (jSOaE.zip in github.com/PetBuj/jSOa,
% wired there to a spring-design problem). It carries the paper's Eigen crossover;
% the CEC 2024 report in the same repository describes jSOa without it.
% The archive update is kept as released: an appended entry is x(1:D), the first
% population column read linearly, and a replacement writes the trial that has
% just overwritten the parent, so the archive never holds the outperformed parent.
% Removed: the stop at the known optimum. Repeated reflection is computed in
% closed form (same point); non-finite coordinates are redrawn uniformly and a
% memory update that comes out non-finite keeps the old slot value.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = jsoa(problem)

    nDim  = problem.dimension;
    LB    = problem.lb(:)';
    UB    = problem.ub(:)';
    MaxEval = problem.maxFe;

    Ap   = 0.2;
    N_min = 4;
    ps   = 0.5;
    peig = 0.4;
    H    = 5;
    pmax = 0.25;
    pmin = pmax / 2;

    D_size = max(nDim, 6);
    N_init = round(25 * log(D_size) * sqrt(D_size));
    if N_init < 2 * N_min
        N_init = 2 * N_min;
    end
    nPop = N_init;
    Asize_max = round(nPop * 2.6);

    curve = zeros(1, MaxEval);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = repmat(LB, nPop, 1) + rand(nPop, nDim) .* repmat(UB - LB, nPop, 1);
    z = inf(nPop, 1);
    bsf = inf;
    bsf_solution = x(1, :);
    neval = 0;
    for i = 1:nPop
        if neval >= MaxEval
            break;
        end
        [fv, neval] = calculate_fitness(x(i, :)', problem, neval);
        z(i) = fv(1);
        if z(i) < bsf
            bsf = z(i);
            bsf_solution = x(i, :);
        end
        curve(neval) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            neval, x(1:i, :), z(1:i), population_history, fitness_history, ...
            history_index, MaxEval);
    end

    MF  = 0.3 * ones(1, H);
    MCR = 0.8 * ones(1, H);
    MF(H)  = 0.9;
    MCR(H) = 0.9;
    k = 1;
    Asize = 0;
    A = zeros(0, nDim + 1);                    % rows are [position, fitness]

    while neval < MaxEval
        Fpole  = -1 * ones(1, nPop);
        CRpole = -1 * ones(1, nPop);
        SCR = [];
        SF  = [];
        suc = 0;
        deltafce = -1 * ones(1, nPop);
        pp = pmax - ((pmax - pmin) * (neval / MaxEval));

        use_eig = rand < peig;                 % one draw decides the crossover of the whole generation
        if use_eig
            [~, ord] = sort(z);
            keep = nPop;
            if round(nPop * ps + 1) >= 3
                keep = round(nPop * ps + 1) - 1;
            end
            C = cov(x(ord(1:keep), :));
            C = (C + C') / 2;
            [EigVect, ~] = eig(C);
        end

        for i = 1:nPop
            if neval >= MaxEval
                break;
            end

            r = randi(H);
            CR = MCR(r) + sqrt(0.1) * randn;
            if CR > 1
                CR = 1;
            elseif CR < 0
                CR = 0;
            end
            if neval < 0.25 * MaxEval
                CR = max(CR, 0.7);
            elseif neval < 0.5 * MaxEval
                CR = max(CR, 0.6);
            end
            F = -1;
            while F <= 0
                F = 0.1 * tan(rand * pi - pi / 2) + MF(r);
            end
            if F > 1
                F = 1;
            end
            if neval < 0.6 * MaxEval && F > 0.7
                F = 0.7;
            end
            Fpole(i)  = F;
            CRpole(i) = CR;

            nx = x(i, :);
            p = max(2, ceil(pp * nPop));
            [~, xorder] = sort(z);
            xpbest = x(xorder(1 + fix(p * rand)), :);
            vyb = draw_except(nPop, i);
            r1 = x(vyb, :);
            vyb2 = draw_except(nPop + Asize, [i, vyb]);
            if vyb2 <= nPop
                r2 = x(vyb2, :);
            else
                r2 = A(vyb2 - nPop, 1:nDim);
            end
            if neval < 0.2 * MaxEval
                Fw = 0.7 * F;
            elseif neval < 0.4 * MaxEval
                Fw = 0.8 * F;
            else
                Fw = 1.2 * F;
            end
            v = nx + Fw * (xpbest - nx) + F * (r1 - r2);

            change = find(rand(1, nDim) < CR);
            if isempty(change)                 % only an empty mask gets a forced coordinate
                change = 1 + fix(nDim * rand);
            end
            if use_eig
                nxeig = EigVect' * nx';
                veig  = EigVect' * v';
                nxeig(change) = veig(change);
                nx = (EigVect * nxeig)';
            else
                nx(change) = v(change);
            end

            bad = ~isfinite(nx);
            if any(bad)
                nx(bad) = LB(bad) + rand(1, sum(bad)) .* (UB(bad) - LB(bad));
            end
            nx = reflect_fold(nx, LB, UB);

            [fv, neval] = calculate_fitness(nx', problem, neval);
            nz = fv(1);
            if nz < bsf
                bsf = nz;
                bsf_solution = nx;
            end

            if nz < z(i)
                deltafce(i) = z(i) - nz;
                z(i) = nz;
                x(i, :) = nx;
                suc = suc + 1;
                if Asize < Asize_max
                    % Linear x(1:D), read after the parent row was overwritten, as released
                    A = [A; [x(1:nDim) z(i)]]; %#ok<AGROW>
                    Asize = Asize + 1;
                else
                    [~, aord] = sort(A(:, nDim + 1));
                    ah = ceil(Asize * (1 - Ap));
                    ktere = aord(ah + randi(Asize - ah));
                    A(ktere, :) = [x(i, 1:nDim) z(i)];
                end
                SCR = [SCR, CRpole(i)]; %#ok<AGROW>
                SF  = [SF, Fpole(i)]; %#ok<AGROW>
            end

            if suc > 0
                MCR_old = MCR(k);
                MF_old  = MF(k);
                delty = deltafce(deltafce ~= -1);
                vahyw = delty / sum(delty);
                if (MCR(k) == -1) || (max(SCR) == 0)
                    MCR(k) = -1;
                else
                    MCR(k) = sum(vahyw .* SCR .* SCR) / sum(vahyw .* SCR);
                end
                MF(k) = sum(vahyw .* SF .* SF) / sum(vahyw .* SF);
                MCR(k) = (MCR(k) + MCR_old) / 2;
                MF(k)  = (MF(k) + MF_old) / 2;
                if ~isfinite(MCR(k))
                    MCR(k) = MCR_old;
                end
                if ~isfinite(MF(k))
                    MF(k) = MF_old;
                end
                k = k + 1;
                if k >= H                      % slot H is never written
                    k = 1;
                end
            end

            curve(neval) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                neval, x, z, population_history, fitness_history, history_index, MaxEval);
        end

        N_old = nPop;
        nPop = max(N_min, round(((N_min - N_init) / MaxEval) * neval + N_init));
        if nPop < N_old
            [z, xorder] = sort(z);
            x = x(xorder, :);
            x(nPop + 1:end, :) = [];
            z(nPop + 1:end) = [];
            Asize_max = round(nPop * 2.6);
            while Asize > Asize_max
                A(randi(Asize), :) = [];
                Asize = Asize - 1;
            end
        end
    end

    curve(min(max(neval, 1), MaxEval):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function idx = draw_except(n, expt)
% Uniform draw from 1..n without the listed indices (nahvyb_expt with k = 1)
    pool = 1:n;
    pool(expt) = [];
    idx = pool(randi(numel(pool)));
end

function x = reflect_fold(x, lb, ub)
% Closed form of reflecting at the violated bound until inside: a triangle wave of period 2*(ub - lb)
    out = x < lb | x > ub;
    if any(out)
        w = ub(out) - lb(out);
        y = mod(x(out) - lb(out), 2 * w);
        y(y > w) = 2 * w(y > w) - y(y > w);
        y(w == 0) = 0;
        x(out) = min(max(lb(out) + y, lb(out)), ub(out));
    end
end

% ----------------------------------------------------------------------- %
% Electromagnetism-like Mechanism (EM)
% Stored as emo, the acronym EMO that later literature uses for this method
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   m = 10*D                    % Number of sample points (charged particles)
%   LSITER = 10                 % Local search: up to LSITER - 1 trials per coordinate (see Note)
%   delta = 1e-3                % Local step multiplier: Length = delta*(ub - lb)
%
% Algorithm Concept:
%   - Each point carries a charge q_i = exp(-D*(f_i - f_best)/sum_k(f_k - f_best)),
%     so better points pull harder
%   - Total force F_i = sum_j +-(x_j - x_i)*q_i*q_j/||x_j - x_i||^2: attraction to
%     better points, repulsion from worse or equal ones
%   - Move along the normalised force by one U(0,1) step per point, scaled per dimension
%     by the room left to ub or lb, so points stay in the box; the best point stays
%   - Local: random line search on each coordinate of the best point in turn, step
%     U(0,1)*Length in a direction fixed per coordinate, ended by the first improvement
%
% Reference:
% S. Ilker Birbil, Shu-Cherng Fang,
% An Electromagnetism-like Mechanism for Global Optimization,
% Journal of Global Optimization 25(3) (2003) 263-282.
% https://doi.org/10.1023/A:1022452626305
% ----------------------------------------------------------------------- %
% Implementation Note:
% From the paper (Algorithms 1-5); its C++ code was only "available upon request". Sec. 4.2
% keeps Local on the current best only. No default m, LSITER or delta is given: m = 10*D as in
% 8 of 9 Table 5 rows, LSITER and delta the modal Table 1 values. Algorithm 3 sets counter once
% before both loops, so taken literally Local ends after LSITER - 1 trials on coordinate 1; it
% is reset per coordinate here, the only reading that reproduces the tables below (Table 4's
% Levy and Kearfott counts exceed the printed bound). MAXITER gives way to the budget; moved
% points are evaluated after Move. Length = delta*max(ub - lb) is taken per dimension (equal on
% a uniform box) and Local steps are clamped to the box. Guards: equal f gives equal charges
% (Eq. 2 is 0/0), NaN f ranks worst, coincident points exert no force. Avg f, paper / port /
% literal counter set once:
%      Table 5, 25 runs (Eq. 5 hits: paper all but 2 on S5; port 0-25 of 25; literal 0-5 of 25):
%      S5 -9.732 / -8.036 / -6.080   S10 -10.511 / -10.536 / -9.140   H6 -3.307 / -3.292 / -3.262
%      BR 0.3980 / 0.3979 / 0.3997   SHU -186.72 / -181.61 / -181.12
%      Table 4: Trid(5) -29.996 / -30.000 / -28.459   Trid(20) -1519.4 / -1384.8 / 52.9
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = emo(problem)

    D     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    m      = 10 * D;
    lsiter = 10;
    delta  = 1e-3;
    len    = delta * span;  % Algorithm 3 line 2, taken per dimension

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    x = lb + rand(m, D) .* span;

    n0 = min(m, maxFE);
    [fv, FE] = calculate_fitness(x(1:n0, :)', problem, FE);
    f = inf(m, 1);
    f(1:n0) = fv(:);

    bsf          = inf;
    bsf_solution = x(1, :);
    for k = 1:n0
        if f(k) < bsf || (isnan(bsf) && ~isnan(f(k)))
            bsf          = f(k);
            bsf_solution = x(k, :);
        end
        curve(k) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            k, x(1:n0, :), f(1:n0), population_history, fitness_history, history_index, maxFE);
    end
    % Mirror for the recorder: each point at the position its fitness was measured
    xrec = x;
    frec = f;

    while FE < maxFE
        % Algorithm 3 on the current best point only (the Sec. 4.1.3 version)
        [~, b] = min(f);
        for k = 1:D
            % counter starts over on every coordinate (see Note)
            counter = 1;
            lam1 = rand;
            while counter < lsiter
                y = x(b, :);
                lam2 = rand;
                if lam1 > 0.5
                    y(k) = y(k) + lam2 * len(k);
                else
                    y(k) = y(k) - lam2 * len(k);
                end
                y(k) = min(max(y(k), lb(k)), ub(k));
                [fy, FE] = calculate_fitness(y', problem, FE);
                if fy < bsf || (isnan(bsf) && ~isnan(fy))
                    bsf          = fy;
                    bsf_solution = y;
                end
                if fy < f(b) || (isnan(f(b)) && ~isnan(fy))
                    x(b, :)    = y;
                    f(b)       = fy;
                    xrec(b, :) = y;
                    frec(b)    = fy;
                    counter    = lsiter - 1;
                end
                counter = counter + 1;
                curve(FE) = bsf;
                [population_history, fitness_history, history_index] = record_history(...
                    FE, xrec, frec, population_history, fitness_history, history_index, maxFE);
                if FE >= maxFE
                    break;
                end
            end
            if FE >= maxFE
                break;
            end
        end
        if FE >= maxFE
            break;
        end
        [~, b] = min(f);

        % Eq. (2), Algorithm 4 lines 1-4; a NaN value ranks as the worst
        fc = f;
        fc(isnan(fc)) = inf;
        gap = fc - fc(b);
        gap(b) = 0;
        far = ~isfinite(gap);
        if any(far)
            frac = far / sum(far);  % limit of Eq. (2) as those gaps grow without bound
        elseif sum(gap) > 0
            frac = gap / sum(gap);
        else
            frac = zeros(m, 1);     % all values equal: Eq. (2) is 0/0, charges taken equal
        end
        q = exp(-D * frac);

        % Eq. (3): centred on x_best so the Gram identity for the distances stays accurate
        Z  = x - x(b, :);
        sq = sum(Z .^ 2, 2);
        R2 = sq + sq' - 2 * (Z * Z');
        W  = (q * q') ./ R2;
        rep = ~(fc' < fc);
        W(rep) = -W(rep);
        % A point on itself, or on a coincident point (below the Gram floor), exerts no force
        W(R2 <= 1e-14 * (sq + sq') | R2 < realmin) = 0;
        W = W ./ max(max(abs(W), [], 2), realmin);  % per-row scale: the direction is unchanged
        F = W * Z - sum(W, 2) .* Z;

        % Algorithm 5: one lambda per point; a zero force leaves the point where it is
        nrm = sqrt(sum(F .^ 2, 2));
        mv = nrm > 0 & isfinite(nrm);
        mv(b) = false;
        Fn  = F(mv, :) ./ nrm(mv);
        lam = rand(nnz(mv), 1);
        xm  = x(mv, :);
        room = (Fn > 0) .* (ub - xm) + (Fn <= 0) .* (xm - lb);
        x(mv, :) = min(max(xm + lam .* Fn .* room, lb), ub);

        % Every point but the best is evaluated at its new position
        idx = [1:b-1, b+1:m];
        r   = min(numel(idx), maxFE - FE);
        idx = idx(1:r);
        fe0 = FE;
        [fv, FE] = calculate_fitness(x(idx, :)', problem, FE);
        fv = fv(:);
        for k = 1:r
            j = idx(k);
            f(j)       = fv(k);
            xrec(j, :) = x(j, :);
            frec(j)    = fv(k);
            if fv(k) < bsf || (isnan(bsf) && ~isnan(fv(k)))
                bsf          = fv(k);
                bsf_solution = x(j, :);
            end
            curve(fe0 + k) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                fe0 + k, xrec, frec, population_history, fitness_history, history_index, maxFE);
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

% ----------------------------------------------------------------------- %
% Improved Covariance Matrix Learning and Searching Preference algorithm (ICMLSP)
% CEC 2015 learning-based track -- 12th place (organiser's draft ranking)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 40/50/60/70           % Children per generation at D = 10/30/50/100
%   archive = 700/1000/1200/2000  % Selected points the covariance is learned from
%   intake = pop/2              % Best losers of a generation added to the archive
%   trim = 0.6                  % Share of the archive, nearest its mean, kept for C
%   sigma ~ Cauchy(0.6 or 1, 0.1)  % Redrawn per generation, truncated to [0, 1]
%   cond(C) <= 1e6              % Ridge added to cap the condition number
%
% Algorithm Concept:
%   - (1+1) sampling around a single centre: each child is drawn from N(c, sigma^2 C)
%     and replaces the centre at once if better
%   - Every losing point (a worse child, or a centre just replaced) is kept; the
%     best half of a generation's losers enters a sliding archive
%   - C is the sample covariance of the 60 % of the archive nearest its mean, with a
%     ridge capping the condition number
%   - The step sigma is redrawn each generation from one of two Cauchy laws with
%     equal chance, a "searching preference" between finer and coarser steps
%   - Out-of-box coordinates are mirrored back at the violated bound
%
% Reference:
% Lei Chen, Chaoda Peng, Hai-Lin Liu, Shengli Xie,
% An improved covariance matrix leaning and searching preference algorithm for
% solving CEC 2015 benchmark problems,
% 2015 IEEE Congress on Evolutionary Computation (CEC), 2015, pp. 1041-1045.
% https://doi.org/10.1109/CEC.2015.7257004
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (15287-ICMSLP.zip in
% github.com/P-N-Suganthan/CEC2015-Learning-Based); CR_func.m, covm.m and
% multivrandn.m are the code that runs, the shipped purecmaes.m, PSO_func.m and
% other helpers are never called. Its main.m defines D = 10/30/50/100 only: the
% nearest set is used (ties go to the smaller D). Kept as released: the centre's
% fitness starts as the FIRST individual's, the initial sigma is not truncated (its
% test reads any(sigma)), and a full archive drops pop columns but adds pop/2.
% Numerical guards only: a non-finite child coordinate is redrawn in the box and a
% negative eigenvalue left by rounding is read as zero. The release's one-FE budget
% overrun is cut.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = icmlsp(problem)

    d     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Per-dimension settings of the release's main.m
    [~, kd] = min(abs([10 30 50 100] - d));
    pop_tab  = [40 50 60 70];
    apop_tab = [700 1000 1200 2000];
    popsize  = pop_tab(kd);
    apopsize = apop_tab(kd);
    P = 0.5;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;
    bsf = inf;
    bsf_solution = (lb + 0.5 * span)';

    % The release's resampling test reads any(sigma), which never fires here
    if rand < P
        sigma = 0.60 + 0.1 * tan(pi * (rand - 0.5));
    else
        sigma = 1 + 0.1 * tan(pi * (rand - 0.5));
    end

    val_x = lb + span .* rand(d, popsize);
    n0 = min(popsize, maxFE);
    val_f = inf(1, popsize);
    val_f(1:n0) = evaluate(val_x(:, 1:n0));

    % Population on record: slot k holds the k-th point of the latest generation, with its own fitness
    rec_x = val_x';
    rec_f = val_f';
    [population_history, fitness_history, history_index] = record_history( ...
        FE, rec_x(1:n0, :), rec_f(1:n0), population_history, fitness_history, history_index, maxFE);

    [~, loc] = sort(val_f);
    cent_x = val_x(:, loc(1));
    % As released: the centre's fitness is the first individual's, not the best one's
    cent_f = val_f(1);
    num    = floor(popsize / 2);
    sval_x = val_x(:, loc(1:num));
    [B, Dv] = covm(sval_x);

    while FE < maxFE
        stor_x = zeros(d, popsize);
        stor_f = zeros(1, popsize);
        for k = 1:popsize
            if FE >= maxFE
                break;
            end
            cld_x = cent_x + sigma * B * (Dv .* randn(d, 1));
            idx_1 = cld_x < lb;
            idx_2 = cld_x > ub;
            if any(idx_1)
                cld_x(idx_1) = min(ub(idx_1), max(lb(idx_1), 2*lb(idx_1) - cld_x(idx_1)));
            end
            if any(idx_2)
                cld_x(idx_2) = max(lb(idx_2), min(ub(idx_2), 2*ub(idx_2) - cld_x(idx_2)));
            end
            bad = ~isfinite(cld_x);
            cld_x(bad) = lb(bad) + rand(nnz(bad), 1) .* span(bad);
            cld_f = evaluate(cld_x);
            rec_x(k, :) = cld_x';
            rec_f(k) = cld_f;
            [population_history, fitness_history, history_index] = record_history( ...
                FE, rec_x, rec_f, population_history, fitness_history, history_index, maxFE);
            if cld_f < cent_f
                stor_x(:, k) = cent_x;
                stor_f(k)    = cent_f;
                cent_x       = cld_x;
                cent_f       = cld_f;
            else
                stor_x(:, k) = cld_x;
                stor_f(k)    = cld_f;
            end
        end
        if FE >= maxFE
            break;
        end
        % Searching preference: a fine (0.6) or coarse (1) Cauchy step, truncated to [0, 1]
        if rand < P
            sigma = 0.60 + 0.1 * tan(pi * (rand - 0.5));
            while sigma < 0 || sigma > 1
                sigma = 0.60 + 0.1 * tan(pi * (rand - 0.5));
            end
        else
            sigma = 1 + 0.1 * tan(pi * (rand - 0.5));
            while sigma < 0 || sigma > 1
                sigma = 1 + 0.1 * tan(pi * (rand - 0.5));
            end
        end
        [~, loc] = sort(stor_f);
        val_x = stor_x(:, loc(1:num));
        if size(sval_x, 2) < apopsize
            sval_x = [sval_x, val_x]; %#ok<AGROW>
        else
            % As released: pop columns leave, pop/2 enter
            sval_x = [sval_x(:, popsize+1:end), val_x];
        end
        [B, Dv] = covm(sval_x);
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;
    best_fitness  = bsf;
    best_solution = bsf_solution;

    % Evaluates the columns of X (already cut to the budget), tracking the best per evaluation
    function f = evaluate(X)
        [fv, FE_new] = calculate_fitness(X, problem, FE);
        f = fv(:)';
        for q = 1:numel(f)
            if f(q) < bsf
                bsf = f(q);
                bsf_solution = X(:, q)';
            end
            curve(FE + q) = bsf;
        end
        FE = FE_new;
    end
end

% Covariance of the 60 % of the archive nearest its mean, condition number capped at 1e6
function [B, Dv] = covm(a)
    dis   = sum((a - repmat(mean(a, 2), 1, size(a, 2))).^2);
    [~, loc] = sort(dis);
    b     = a(:, loc(1:0.6*size(a, 2)));
    artmp = (b - repmat(mean(b, 2), 1, size(b, 2)));
    C     = artmp * artmp' / (size(b, 2) - 1);
    C     = triu(C) + triu(C, 1)';
    [B, Dm] = eig(C);
    if max(diag(Dm)) > 1e6 * min(diag(Dm))
        tmp = max(diag(Dm)) / 1e6 - min(diag(Dm));
        C   = C + tmp * eye(size(C, 1));
        [B, Dm] = eig(C);
    end
    Dv = sqrt(max(diag(Dm), 0));
end

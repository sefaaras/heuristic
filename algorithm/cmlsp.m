% ----------------------------------------------------------------------- %
% Covariance Matrix Learning and Searching Preference algorithm (CMLSP)
% CEC 2014 competition -- top-5 entry (organiser code release)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   pop = 100/300/500/1000      % Samples per generation at D = 10/30/50/100
%   g2 = 50/100/100/100         % Generations per CMA phase before a switch test
%   g1 = 50/500/500/500         % Generations per CML phase before a switch test
%   archive = 1500/2500/4000/5000  % Selected points the CML covariance is learned from
%   sigma = 0.3                 % CMA initial step; reset on every CML phase
%   mu = pop/2                  % CMA parents (log weights) and archive intake
%
% Algorithm Concept:
%   - Two phases share one centre (the best point found); a phase that improves
%     nothing over its whole run of generations hands over to the other
%   - CMA phase: (mu/mu_w, pop)-CMA-ES whose samples are pulled back towards the
%     mean (step x0.9) until they fit the box
%   - CML phase: (1+1) sampling around the centre from a covariance learned from an
%     archive of tournament winners; every success moves the centre at once
%   - Entering CML restarts the population uniformly in the box and reseeds the
%     archive from its best half
%
% Reference:
% Lei Chen, Zhe Zheng, Hai-Lin Liu, Shengli Xie,
% An evolutionary algorithm based on Covariance Matrix Leaning and Searching
% Preference for solving CEC 2014 benchmark problems,
% 2014 IEEE Congress on Evolutionary Computation (CEC), 2014, pp. 2672-2677.
% https://doi.org/10.1109/CEC.2014.6900594
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' MATLAB release (V4-matlabCMLSP.zip in Top-Methods-Part-A
% of github.com/P-N-Suganthan/CEC2014); its only mex is the 32-bit CEC2014 function.
% Its main.m defines D = 10/30/50/100 only: the nearest set is used (ties go to the
% smaller D). Kept as released: CMA samples use rand, not randn, in the eigenbasis;
% hsig counts generations as FE/pop; the CML reseed excludes the best point and keeps
% the CMA eigenbasis until the first success. The CMA start C = I on its +/-100 box
% is scaled to each range ((ub-lb)/200 per axis, exact there). Guards, no-ops in the
% bit-exact CEC2014 comparison: the box shrink stops at 2000 steps and clamps, a
% non-finite covariance keeps its last factor, eigenvalues are floored at zero. The
% release's budget overrun (one FE, a reseed batch) is cut.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cmlsp(problem)

    d     = problem.dimension;
    lb    = problem.lb(:);
    ub    = problem.ub(:);
    maxFE = problem.maxFe;
    span  = ub - lb;

    % Per-dimension settings of the release's main.m
    [~, kd] = min(abs([10 30 50 100] - d));
    g1_tab   = [50 500 500 500];
    g2_tab   = [50 100 100 100];
    pop_tab  = [100 300 500 1000];
    apop_tab = [1500 2500 4000 5000];
    g1       = g1_tab(kd);
    g2       = g2_tab(kd);
    popsize  = pop_tab(kd);
    apopsize = apop_tab(kd);
    max_shrink = 2000;

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;
    bsf = inf;
    bsf_solution = (lb + 0.5 * span)';

    % Population on record: slot k holds the k-th point of the latest generation, with its own fitness
    rec_x = zeros(popsize, d);
    rec_f = inf(popsize, 1);

    val_x = lb + span .* rand(d, popsize);
    n0 = min(popsize, maxFE);
    val_f = inf(1, popsize);
    val_f(1:n0) = evaluate(val_x(:, 1:n0));
    rec_x(1:n0, :) = val_x(:, 1:n0)';
    rec_f(1:n0) = val_f(1:n0);
    [population_history, fitness_history, history_index] = record_history( ...
        FE, rec_x(1:n0, :), rec_f(1:n0), population_history, fitness_history, history_index, maxFE);

    [~, loc] = sort(val_f);
    cent_x = val_x(:, loc(1));
    cent_f = val_f(loc(1));
    xmean  = cent_x;
    num     = floor(popsize / 2);
    weights = log(num + 1/2) - log(1:num)';
    weights = weights / sum(weights);
    mueff   = sum(weights)^2 / sum(weights.^2);
    cc      = (4 + mueff/d) / (d + 4 + 2*mueff/d);
    cs      = (mueff + 2) / (d + mueff + 5);
    c1      = 2 / ((d + 1.3)^2 + mueff);
    cmu     = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)^2 + mueff));
    damps   = 1 + 2*max(0, sqrt((mueff - 1)/(d + 1)) - 1) + cs;
    B       = eye(d, d);
    sigma   = 0.3;
    % The release's ones(d,1) on its +/-100 box, scaled to each range
    Dv      = span / 200;
    pc      = zeros(d, 1);
    ps      = zeros(d, 1);
    C        = B * diag(Dv.^2) * B';
    invsqrtC = B * diag(Dv.^-1) * B';
    chiN     = d^0.5 * (1 - 1/(4*d) + 1/(21*d^2));
    flag     = 1;
    sval     = zeros(d, 0);
    x         = zeros(d, popsize);
    arfitness = zeros(1, popsize);

    while FE < maxFE
        stor_x = zeros(d, popsize);
        stor_f = zeros(1, popsize);
        if flag == -1
            f1 = 0;
            sigma = 0.3;
            for g = 1:g1
                for k = 1:popsize
                    if FE >= maxFE
                        break;
                    end
                    cld_x = cent_x + sigma * B * (Dv .* randn(d, 1));
                    out = ~(cld_x >= lb & cld_x <= ub);
                    cld_x(out) = lb(out) + rand(nnz(out), 1) .* span(out);
                    cld_f = evaluate(cld_x);
                    stamp(k, cld_x, cld_f);
                    if cld_f < cent_f
                        stor_x(:, k) = cent_x;
                        stor_f(k)    = cent_f;
                        cent_x       = cld_x;
                        cent_f       = cld_f;
                        C            = covm(sval, cent_x, sigma) + 0.000001 * eye(d);
                        [B, Dv]      = factor_cml(C, B, Dv);
                        f1           = 1;
                    else
                        stor_x(:, k) = cld_x;
                        stor_f(k)    = cld_f;
                    end
                end
                if FE >= maxFE
                    break;
                end
                [val_x, val_f] = tournament_halve([val_x, stor_x], [val_f, stor_f]);
                [~, loc] = sort(val_f);
                if size(sval, 2) < apopsize
                    sval = [sval, val_x(:, loc(1:num))]; %#ok<AGROW>
                else
                    sval = [sval(:, num+1:end), val_x(:, loc(1:num))];
                end
                C       = covm(sval, cent_x, sigma) + 0.000001 * eye(d);
                [B, Dv] = factor_cml(C, B, Dv);
            end
            if f1 == 0
                flag     = 1;
                pc       = zeros(d, 1);
                ps       = zeros(d, 1);
                xmean    = cent_x;
                invsqrtC = B * diag(Dv.^-1) * B';
            end
        else
            f1 = 0;
            for g = 1:g2
                for k = 1:popsize
                    if FE >= maxFE
                        break;
                    end
                    temp = rand(d, 1);
                    x(:, k) = xmean + sigma * B * (Dv .* temp);
                    n_shrink = 0;
                    while any(~(x(:, k) >= lb & x(:, k) <= ub)) && n_shrink < max_shrink
                        temp    = temp * 0.9;
                        x(:, k) = xmean + sigma * B * (Dv .* temp);
                        n_shrink = n_shrink + 1;
                    end
                    if n_shrink == max_shrink
                        x(:, k) = min(max(x(:, k), lb), ub);
                    end
                    arfitness(k) = evaluate(x(:, k));
                    stamp(k, x(:, k), arfitness(k));
                    if arfitness(k) < cent_f
                        cent_f = arfitness(k);
                        cent_x = x(:, k);
                        f1     = 1;
                    end
                end
                if FE >= maxFE
                    break;
                end
                [arfitness, arindex] = sort(arfitness);
                xold  = xmean;
                xmean = x(:, arindex(1:num)) * weights;
                ps    = (1 - cs) * ps + sqrt(cs*(2 - cs)*mueff) * invsqrtC * (xmean - xold) / sigma;
                % FE / pop stands in for the generation count, as released
                hsig  = norm(ps) / sqrt(1 - (1 - cs)^(2*FE/popsize)) / chiN < 1.4 + 2/(d + 1);
                pc    = (1 - cc) * pc + hsig * sqrt(cc*(2 - cc)*mueff) * (xmean - xold) / sigma;
                artmp = (1/sigma) * (x(:, arindex(1:num)) - repmat(xold, 1, num));
                C_new = (1 - c1 - cmu) * C ...
                        + c1 * (pc*pc' + (1 - hsig) * cc*(2 - cc) * C) ...
                        + cmu * artmp * diag(weights) * artmp';
                sigma = sigma * exp((cs/damps) * (norm(ps)/chiN - 1));
                C_new = triu(C_new) + triu(C_new, 1)';
                if all(isfinite(C_new(:)))
                    C = C_new;
                    [B, Dm]  = eig(C);
                    Dv       = sqrt(max(diag(Dm + 0.000001*eye(d)), realmin));
                    invsqrtC = B * diag(Dv.^-1) * B';
                end
            end
            if f1 == 0 && FE < maxFE
                flag  = -1;
                val_x = lb + span .* rand(d, popsize);
                n_new = min(popsize, maxFE - FE);
                val_f = inf(1, popsize);
                val_f(1:n_new) = evaluate(val_x(:, 1:n_new));
                rec_x(1:n_new, :) = val_x(:, 1:n_new)';
                rec_f(1:n_new) = val_f(1:n_new);
                [population_history, fitness_history, history_index] = record_history( ...
                    FE, rec_x, rec_f, population_history, fitness_history, history_index, maxFE);
                [~, loc] = sort(val_f);
                if val_f(loc(1)) < cent_f
                    cent_x = val_x(:, loc(1));
                    cent_f = val_f(loc(1));
                end
                % The archive starts from ranks 2..num+1, and B/D stay as CMA left them
                sval = val_x(:, loc(2:num+1));
                C    = covm(sval, cent_x, sigma) + 0.000001 * eye(d);
            end
        end
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

    % Slot k of the recorded population takes the point just evaluated
    function stamp(slot, xk, fk)
        rec_x(slot, :) = xk';
        rec_f(slot) = fk;
        [population_history, fitness_history, history_index] = record_history( ...
            FE, rec_x, rec_f, population_history, fitness_history, history_index, maxFE);
    end
end

% Covariance of the archive about the centre, in units of sigma
function c = covm(a, center, sigma)
    artmp = (1/sigma) * (a - repmat(center, 1, size(a, 2)));
    c = artmp * artmp' / size(a, 2);
end

% Eigen-factor of the CML covariance; a non-finite one keeps the previous factor
function [B, Dv] = factor_cml(C, B, Dv)
    if all(isfinite(C(:)))
        [B, Dm] = eig(C);
        Dv = sqrt(max(diag(Dm), 0));
    end
end

% Random pairs of the merged set, the better of each pair survives (the release's select.m)
function [val_x, val_f] = tournament_halve(total_x, total_f)
    [dim_x, num_x] = size(total_x);
    loc   = randperm(num_x);
    val_x = zeros(dim_x, num_x/2);
    val_f = zeros(1, num_x/2);
    j = 1;
    for i = 1:2:num_x-1
        if total_f(loc(i)) < total_f(loc(i+1))
            val_x(:, j) = total_x(:, loc(i));
            val_f(j)    = total_f(loc(i));
        else
            val_x(:, j) = total_x(:, loc(i+1));
            val_f(j)    = total_f(loc(i+1));
        end
        j = j + 1;
    end
end

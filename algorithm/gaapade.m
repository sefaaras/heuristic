% ----------------------------------------------------------------------- %
% Gaussian Adaptation based Parameter Adaptation for DE (GaAPADE)
% CEC 2014 bound-constrained track submission
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   NPP = 20*D                  % Sample set the population is drawn from
%   NP = 100                    % Members taken from the set each generation
%   p = 0.05                    % pbest rate of current-to-pbest/1
%   m = [0.5 0.5], Q = I        % Mean and normalised Cholesky factor of (F, CR)
%   r = rm + 0.1*N(0,1), rm = 1/e  % Per-individual step size and its adapted mean
%   c = 0.1, condMax = 2        % Adaptation rate, condition cap before Q is regularised
%   ts = 5                      % Tournament size when picking the population
%
% Algorithm Concept:
%   - A set of 20*D samples is kept; each generation 100 of them form the population,
%     by tournament while the mean fitness is below the median, else at random
%   - DE/current-to-pbest/1 with binomial crossover and no archive; the survivors
%     are written back into the sample set at their old places
%   - Each individual's (F, CR) pair is drawn from one bivariate Gaussian, so the
%     two parameters are adapted jointly rather than one at a time
%   - Gaussian adaptation: the pair behind the largest improvement moves the mean
%     and reshapes the Cholesky factor, whose determinant is renormalised to 1
%   - The step-size mean follows the improvement-weighted successful step sizes
%
% Reference:
% Rammohan Mallipeddi, Guohua Wu, Minho Lee, Ponnuthurai N. Suganthan,
% Gaussian adaptation based parameter adaptation for differential evolution,
% 2014 IEEE Congress on Evolutionary Computation (CEC), 2014, pp. 1760-1767.
% https://doi.org/10.1109/CEC.2014.6900601
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from JADE.m in CEC2014-GA(Rev).rar of Suganthan's CEC2014
% Top-Methods-Part-A package, Mallipeddi's run of the CEC2014 entry. The
% conference abstract describes only the (F, CR) adaptation; the 20*D sample set
% is described in the authors' 2015 extension (doi 10.1155/2015/287607), but the
% release reproduces the submitted GaAPADE results only with it (checked on six
% functions at D = 10 and 30). At D < 5 the set is smaller than 100, so the
% population is the whole set. The last generation evaluates only the trials the
% budget still allows, where the release overshoots by up to 100.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = gaapade(problem)

    dim   = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;
    span  = ub - lb;

    NPP     = min(20 * dim, maxFE);
    NP      = min(100, NPP);
    p       = 0.05;
    ND      = 2;
    rm      = 1 / exp(1);
    condMax = 2;
    c       = 0.1;
    ts      = 5;
    Q       = eye(ND);
    condC   = 1;
    mu      = [0.5; 0.5];

    FE    = 0;
    curve = zeros(1, maxFE);
    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    popl = repmat(lb, NPP, 1) + rand(NPP, dim) .* repmat(span, NPP, 1);
    [fv, FE] = calculate_fitness(popl', problem, FE);
    vall = fv(:);

    bsf          = inf;
    bsf_solution = popl(1, :);
    for i = 1:NPP
        if vall(i) < bsf
            bsf          = vall(i);
            bsf_solution = popl(i, :);
        end
        curve(i) = bsf;
        [population_history, fitness_history, history_index] = record_history(...
            i, popl(1:i, :), vall(1:i), population_history, fitness_history, history_index, maxFE);
    end

    while FE < maxFE
        r = max(1e-50, rm + 0.1 * randn(NP, 1));

        % Tournament while the set is still spread out, random picks once it sits in a basin
        if mean(vall) <= median(vall)
            mem = randi(NPP, NPP, ts);
            wins = sum(vall(mem) > repmat(vall, 1, ts), 2);
            [~, app] = sort(wins, 'descend');
        else
            app = randperm(NPP)';
        end
        sel = app(1:NP);
        pop = popl(sel, :);
        valParents = vall(sel);
        [~, indBest] = sort(valParents, 'ascend');

        arz = randn(ND, NP);
        arx = repmat(mu, 1, NP) + repmat(r', ND, 1) .* (Q * arz);
        arx(1, arx(1, :) <= 0) = 0.001;
        arx(1, arx(1, :) > 1)  = 1;
        arx(2, arx(2, :) <= 0) = 0;
        arx(2, arx(2, :) > 1)  = 1;
        F  = arx(1, :)';
        CR = arx(2, :)';

        [r1, r2] = gnR1R2(NP, NP, 1:NP);
        pNP = max(round(p * NP), 2);
        randindex = max(1, ceil(rand(1, NP) * pNP));
        pbest = pop(indBest(randindex), :);

        vi = pop + F(:, ones(1, dim)) .* (pbest - pop + pop(r1, :) - pop(r2, :));
        vi = min(max(vi, repmat(lb, NP, 1)), repmat(ub, NP, 1));
        bad = ~isfinite(vi);
        if any(bad(:))
            redraw = repmat(lb, NP, 1) + rand(NP, dim) .* repmat(span, NP, 1);
            vi(bad) = redraw(bad);
        end

        mask = rand(NP, dim) > CR(:, ones(1, dim));
        jrand = sub2ind([NP dim], (1:NP)', floor(rand(NP, 1) * dim) + 1);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);

        k = min(NP, maxFE - FE);
        FE0 = FE;
        [fv, FE] = calculate_fitness(ui(1:k, :)', problem, FE);
        valOffspring = fv(:);

        % Only the evaluated trials take part in the selection
        imp = valParents(1:k) - valOffspring;
        [newVal, I] = min([valParents(1:k), valOffspring], [], 2);
        valParents(1:k) = newVal;
        won = find(I == 2);
        pop(won, :) = ui(won, :);

        popl(sel, :) = pop;
        vall(sel) = valParents;

        for e = 1:k
            if valOffspring(e) < bsf
                bsf          = valOffspring(e);
                bsf_solution = ui(e, :);
            end
            curve(FE0 + e) = bsf;
            [population_history, fitness_history, history_index] = record_history(...
                FE0 + e, popl, vall, population_history, fitness_history, history_index, maxFE);
        end

        if ~isempty(won)
            gimp = imp(won) / sum(imp(won));
            rm_new = (1 - c) * rm + c * sum(gimp .* r(won), 1);
            % A non-finite improvement (Inf parent on CEC2020RW) would freeze rm at NaN
            if isfinite(rm_new)
                rm = rm_new;
            end

            [~, ibest] = max(imp(won));
            l = won(ibest);
            mu = (1 - c) * mu + arx(:, l) * c;

            if condC < condMax
                deltaC = (1 - c) * eye(ND) + arz(:, l) * arz(:, l)' * c;
                deltaC = triu(deltaC) + triu(deltaC, 1)';
                [B, eigVals] = eig(deltaC);
                deltaQ = B * diag(sqrt(diag(eigVals))) * B';
                Q = Q * deltaQ;
            else
                Q = Q + 1 / ND * eye(ND);
            end

            detQ = det(Q);
            % A non-positive determinant would make Q complex; never reached from Q = I
            if isfinite(detQ) && detQ > 0
                Q = Q ./ (detQ .^ (1 / ND));
                eigVals = eig(Q * Q');
                condC = max(eigVals) / min(eigVals);
            else
                Q = eye(ND);
                condC = 1;
            end
        end
    end

    curve(min(max(FE, 1), maxFE):end) = bsf;

    best_fitness  = bsf;
    best_solution = bsf_solution;
end

function [r1, r2] = gnR1R2(NP1, NP2, r0)
    NP0 = numel(r0);
    r1 = floor(rand(1, NP0) * NP1) + 1;
    pos = (r1 == r0);
    while any(pos)
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        pos = (r1 == r0);
    end
    r2 = floor(rand(1, NP0) * NP2) + 1;
    pos = (r2 == r1) | (r2 == r0);
    while any(pos)
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        pos = (r2 == r1) | (r2 == r0);
    end
    r1 = r1(:);
    r2 = r2(:);
end

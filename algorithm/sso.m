% ----------------------------------------------------------------------- %
% Social Spider Optimization (SSO) for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   spidn = 50            % Colony size (number of spiders)
%   fpl = 0.65            % Lower female percentage
%   fpu = 0.9             % Upper female percentage
%
% Algorithm Concept:
%   - Inspired by the cooperative behaviour of social spiders on a communal
%     web; the colony is split into female and male spiders
%   - Spiders communicate through vibrations whose intensity depends on the
%     weight (fitness) and distance of the emitter
%   - Female/male movement operators plus a mating operator produce offspring
%     that may replace the worst spiders
%
% Reference:
% Erik Cuevas, Miguel Cienfuegos, Daniel Zaldivar, Marco Perez-Cisneros,
% A swarm optimization algorithm inspired in the behavior of the
% social-spider,
% Expert Systems with Applications 40(16) (2013) 6374-6384
% https://doi.org/10.1016/j.eswa.2013.05.041
% ----------------------------------------------------------------------- %
% Input: problem structure with fields:
%   - dimension: problem dimension
%   - lb: lower bounds
%   - ub: upper bounds
%   - maxFe: maximum function evaluations
%   - fhd: function handle
%   - number: function number
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = sso(problem)

    % Extract problem parameters
    dims = problem.dimension;
    lb = problem.lb;
    ub = problem.ub;
    maxFE = problem.maxFe;

    spidn = 50;                         % Colony size
    itern = maxFE / (1.1 * spidn);      % Analogue of the iteration budget
    if itern < 2
        itern = 2;
    end
    fpl = 0.65;                         % Lower female percent
    fpu = 0.9;                          % Upper female percent
    fp = fpl + (fpu - fpl) * rand;      % Aleatory percent
    fn = round(spidn * fp);             % Number of females
    mn = spidn - fn;                    % Number of males

    % Probabilities of attraction or repulsion
    pm = exp(-(0.1:(3 - 0.1) / (itern - 1):3));

    FE = 0;
    curve = zeros(1, maxFE);

    % History storage with 1/10000 sampling
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, spidn, dims);
    fitness_history = zeros(history_size, spidn);
    history_index = 1;

    % ---- Population initialisation ----
    fsp = zeros(fn, dims);
    msp = zeros(mn, dims);
    for i = 1:fn
        fsp(i, 1:dims) = lb(1) + rand(1, dims) .* (ub(1) - lb(1));
    end
    for i = 1:mn
        msp(i, 1:dims) = lb(1) + rand(1, dims) .* (ub(1) - lb(1));
    end

    % ---- Evaluations ----
    [fefit, FE] = calculate_fitness(fsp', problem, FE);  fefit = fefit(:);
    [mafit, FE] = calculate_fitness(msp', problem, FE);  mafit = mafit(:);

    % ---- Assign weights ----
    spfit = [fefit; mafit];
    bfitw = min(spfit);
    wfit = max(spfit);
    spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit));
    fewei = spwei(1:fn);
    mawei = spwei(fn + 1:spidn);

    % ---- Memory of the best ----
    [~, Ibe] = max(spwei);
    if Ibe > fn
        spbest = msp(Ibe - fn, :);
        bfit = mafit(Ibe - fn);
    else
        spbest = fsp(Ibe, :);
        bfit = fefit(Ibe);
    end

    % Record the initial evaluations
    for eval_count = 1:(fn + mn)
        if eval_count <= maxFE
            curve(eval_count) = bfit;
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, [fsp; msp], spfit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end

    % ---- Iterations ----
    i = 0;
    while FE < maxFE
        i = i + 1;
        pmi = pm(min(i, numel(pm)));

        FE_before = FE;

        % Movement of spiders
        fsp = FeMove(spidn, fn, fsp, msp, spbest, Ibe, spwei, dims, lb, ub, pmi);
        msp = MaMove(fn, mn, fsp, msp, fewei, mawei, dims, lb, ub, pmi);

        % Evaluations
        [fefit, FE] = calculate_fitness(fsp', problem, FE);  fefit = fefit(:);
        [mafit, FE] = calculate_fitness(msp', problem, FE);  mafit = mafit(:);

        % Assign weights
        spfit = [fefit; mafit];
        bfitw = min(spfit);
        wfit = max(spfit);
        spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit));
        fewei = spwei(1:fn);
        mawei = spwei(fn + 1:spidn);

        % Mating operator
        ofspr = Mating(fewei, mawei, fsp, msp, dims);

        % Selection of the mating
        if ~isempty(ofspr)
            [fsp, msp, fefit, mafit, FE] = Survive(fsp, msp, ofspr, fefit, mafit, spfit, fn, problem, FE);
            % Recalculate the weights
            spfit = [fefit; mafit];
            bfitw = min(spfit);
            wfit = max(spfit);
            spwei = 0.001 + ((spfit - wfit) / (bfitw - wfit));
            fewei = spwei(1:fn);
            mawei = spwei(fn + 1:spidn);
        end

        % Memory of the best
        [~, Ibe2] = max(spwei);
        if Ibe2 > fn
            spbest2 = msp(Ibe2 - fn, :);
            bfit2 = mafit(Ibe2 - fn);
        else
            spbest2 = fsp(Ibe2, :);
            bfit2 = fefit(Ibe2);
        end

        % Global memory
        if bfit > bfit2
            bfit = bfit2;
            spbest = spbest2;
        end
        Ibe = Ibe2;

        % Record convergence curve and history over this iteration's FEs
        for eval_count = (FE_before + 1):FE
            if eval_count <= maxFE
                curve(eval_count) = bfit;
                [population_history, fitness_history, history_index] = record_history(...
                    eval_count, [fsp; msp], spfit, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end
        end
    end

    best_solution = spbest;
    best_fitness = bfit;

end

%% --- Female movement ---
function [fsp] = FeMove(spidn, fn, fsp, msp, spbest, Ibe, spmass, d, lb, ub, pm)
    dt1 = zeros(1, fn);
    dt2 = zeros(1, spidn - fn);
    scale = (-lb(1) + ub(1));
    for i = 1:fn
        for j = 1:fn
            if spmass(j) > spmass(i)
                dt1(j) = norm(fsp(i, :) - fsp(j, :));
            else
                dt1(j) = 0;
            end
        end
        for j = 1:spidn - fn
            if spmass(fn + j) > spmass(i)
                dt2(j) = norm(fsp(i, :) - msp(j, :));
            else
                dt2(j) = 0;
            end
        end
        dt = [dt1 dt2] ./ scale;
        [~, Ind, val] = find(dt);
        [~, Imin] = min(val);
        Ish = Ind(Imin);
        if Ish > fn
            spaux = msp(Ish - fn, :);
        else
            spaux = fsp(Ish, :);
        end
        if isempty(val)
            Vibs = 0;
            spaux = zeros(1, d);
        else
            Vibs = 2 * (spmass(Ish) * exp(-(rand * dt(Ish).^2)));
        end
        if Ibe > fn
            dt2b = norm(fsp(i, :) - msp(Ibe - fn, :));
        else
            dt2b = norm(fsp(i, :) - fsp(Ibe, :));
        end
        dtb = dt2b ./ scale;
        Vibb = 2 * (spmass(Ibe) * exp(-(rand * dtb.^2)));
        if rand >= pm
            betha = rand(1, d);
            gamma = rand(1, d);
            tmpf = 2 * pm .* (rand(1, d) - 0.5);
            fsp(i, :) = fsp(i, :) + (Vibs * (spaux - fsp(i, :)) .* betha) + (Vibb * (spbest - fsp(i, :)) .* gamma) + tmpf;
        else
            betha = rand(1, d);
            gamma = rand(1, d);
            tmpf = 2 * pm .* (rand(1, d) - 0.5);
            fsp(i, :) = fsp(i, :) - (Vibs * (spaux - fsp(i, :)) .* betha) - (Vibb * (spbest - fsp(i, :)) .* gamma) + tmpf;
        end
    end
    % Check limits
    for i = 1:d
        for j = 1:fn
            if fsp(j, i) < lb(i), fsp(j, i) = lb(i) + (ub(i) - lb(i)) .* rand(1, 1); end
            if fsp(j, i) == lb(i), fsp(j, i) = lb(i); end
            if fsp(j, i) > ub(i), fsp(j, i) = lb(i) + (ub(i) - lb(i)) .* rand(1, 1); end
            if fsp(j, i) == ub(i), fsp(j, i) = ub(i); end
        end
    end
end

%% --- Male movement ---
function [msp] = MaMove(fn, mn, fsp, msp, femass, mamass, d, lb, ub, pm)
    dt = zeros(1, mn);
    scale = (-lb(1) + ub(1));
    [Indb, ~] = find(mamass >= median(mamass));
    for i = 1:mn
        if ismember(i, Indb)
            for j = 1:fn
                if femass(j) > mamass(i)
                    dt(j) = norm(msp(i, :) - fsp(j, :));
                else
                    dt(j) = 0;
                end
            end
            [~, Ind, val] = find(dt);
            [~, Imin] = min(val);
            Ish = Ind(Imin);
            if isempty(val)
                Vib = 0;
                spaux = zeros(1, d);
            else
                dt = dt ./ scale;
                Vib = 2 * femass(Ish) * exp(-(rand * dt(Ish).^2));
                spaux = fsp(Ish, :);
            end
            delta = 2 * rand(1, d) - .5;
            tmpf = 2 * pm .* (rand(1, d) - 0.5);
            msp(i, :) = msp(i, :) + Vib * (spaux - msp(i, :)) .* delta + tmpf;
        else
            % Spider below median, go to weighted mean
            spdpos = [fsp' msp']';
            spdwei = [femass' mamass']';
            weigth = repmat(spdwei, 1, d);
            dim = find(size(spdpos) ~= 1, 1);
            wmean = sum(weigth .* spdpos, dim) ./ sum(weigth, dim);
            delta = 2 * rand(1, d) - .5;
            tmpf = 2 * pm .* (rand(1, d) - 0.5);
            msp(i, :) = msp(i, :) + (wmean - msp(i, :)) .* delta + tmpf;
        end
    end
    % Check limits
    for i = 1:d
        for j = 1:mn
            if msp(j, i) < lb(i), msp(j, i) = lb(i) + (ub(i) - lb(i)) .* rand(1, 1); end
            if msp(j, i) == lb(i), msp(j, i) = lb(i); end
            if msp(j, i) > ub(i), msp(j, i) = lb(i) + (ub(i) - lb(i)) .* rand(1, 1); end
            if msp(j, i) == ub(i), msp(j, i) = ub(i); end
        end
    end
end

%% --- Mating operator ---
function [ofsp] = Mating(femass, mamass, fsp, msp, dims)
    ofsp = [];
    cont = 1;
    [Indf, ~] = find(femass);
    [Indm, ~] = find(mamass > median(mamass));
    fespid = fsp(Indf, :);
    maspid = msp(Indm, :);
    sp2mate = [];
    rad = zeros(1, dims);
    spid = [fsp' msp']';
    for i = 1:dims
        rad(i) = max(spid(:, i)) - min(spid(:, i));
    end
    r = (sum(rad) / 2) / (dims);
    [sz, ~] = size(Indf);
    dist = zeros(1, sz);
    for i = 1:size(Indm, 1)
        iaux = 1;
        for j = 1:size(Indf, 1)
            dist(j) = norm(msp(Indm(i), :) - fsp(Indf(j), :));
        end
        for kk = 1:size(Indf, 1)
            if dist(kk) < r
                mate(iaux, :) = fsp(Indf(kk), :);
                mass(iaux) = femass(Indf(kk));
                iaux = iaux + 1;
                sp2mate = [msp(Indm(i), :)' mate']';
                masmate = [mamass(Indm(i)) mass];
            end
        end
        if isempty(sp2mate)
            % do nothing
        else
            [num2, n] = size(sp2mate);
            for kk = 1:num2
                for j = 1:n
                    accumulation = cumsum(masmate);
                    p = rand() * accumulation(end);
                    chosen_index = -1;
                    for index = 1:length(accumulation)
                        if (accumulation(index) > p)
                            chosen_index = index;
                            break;
                        end
                    end
                    choice = chosen_index;
                    ofsp(kk, j) = sp2mate(choice, j);
                end
            end
            cont = cont + 1;
        end
    end
end

%% --- Survival / offspring replacement ---
function [fsp, msp, fefit, mafit, FE] = Survive(fsp, msp, ofspr, fefit, mafit, spfit, fn, problem, FE)
    [n1, ~] = size(ofspr);
    offit = zeros(1, n1);
    for j = 1:n1
        [offit(j), FE] = calculate_fitness(ofspr(j, :)', problem, FE);
    end
    for i = 1:n1
        [w1, w2] = max(spfit);
        if offit(i) < w1
            if w2 > fn
                msp(w2 - fn, :) = ofspr(i, :);
                mafit(w2 - fn) = offit(i);
            else
                fsp(w2, :) = ofspr(i, :);
                fefit(w2) = offit(i);
            end
            spfit(w2) = offit(i);
        end
    end
end

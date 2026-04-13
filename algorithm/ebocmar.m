% ----------------------------------------------------------------------- %
% EBO with CMAR (Ensemble Butterfly Optimization with CMA-ES Restart)
% for unconstrained benchmark problems
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PS1 = 18*n              % Population size for EBO phase
%   PS2 = 4+floor(3*log(n)) % Population size for CMA-ES phase
%   
% Algorithm Concept:
%   - Hybrid algorithm combining two phases:
%     1. EBO (Ensemble Butterfly Optimization): DE-based with adaptive operators
%     2. Scout/CMAR: CMA-ES based exploration
%   - Adaptive probability control between phases based on quality and diversity
%   - Linear population reduction for EBO
%   - Local search (fmincon) in late stages
%   - Archive mechanism for diversity
%
% Reference:
% Kumar, A., Misra, R. K., & Singh, D. (2017).
% Improving the local search capability of effective butterfly optimizer 
% using covariance matrix adapted retreat phase.
% IEEE Congress on Evolutionary Computation (CEC), 1622-1629.
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
function [best_fitness, best_solution, curve, population_history, fitness_history] = ebocmar(problem)

    % Extract problem parameters
    dim = problem.dimension;       % Problem dimension
    lb = problem.lb;              % Lower bounds
    ub = problem.ub;              % Upper bounds
    maxFE = problem.maxFe;        % Maximum function evaluations
    
    FE = 0;                           % Function Evaluation Counter
    curve = zeros(1, maxFE);          % Convergence curve
    
    % Initialize parameters
    Par = Introd_Par(dim, lb, ub, maxFE);
    
    %% Define variables
    PS1 = Par.PopSize;            % Population size for EBO
    PS2 = 4 + floor(3*log(Par.n));  % Population size for CMA-ES
    Par.PopSize = PS1 + PS2;        % Total population size
    
    % For history recording (store only top 100 individuals to save disk space)
    history_pop_size = 100;
    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, history_pop_size, dim);
    fitness_history = zeros(history_size, history_pop_size);
    history_index = 1;
    
    %% Initialize population
    x = repmat(Par.xmin, Par.PopSize, 1) + repmat((Par.xmax - Par.xmin), Par.PopSize, 1) .* rand(Par.PopSize, Par.n);
    
    % Evaluate initial population
    [fitx, FE] = calculate_fitness(x', problem, FE);
    [mS, ~] = size(fitx);
    if mS > 1
        fitx = fitx';
    end
    
    % Record initial evaluations
    [bestold, bes_l] = min(fitx);
    bestx = x(bes_l, :);
    
    for eval_count = 1:Par.PopSize
        curve(eval_count) = bestold;
        if eval_count <= PS1
            [sorted_fit, sorted_idx] = sort(fitx(1:PS1));
            top_k = min(history_pop_size, PS1);
            rec_pop = NaN(history_pop_size, dim);
            rec_fit = NaN(1, history_pop_size);
            rec_pop(1:top_k, :) = x(sorted_idx(1:top_k), :);
            rec_fit(1:top_k) = sorted_fit(1:top_k);
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, rec_pop, rec_fit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    %% Split population for each phase
    EA_1 = x(1:PS1, :);    
    EA_obj1 = fitx(1:PS1);   
    EA_1old = x(randperm(PS1), :);
    
    EA_2 = x(PS1+1:size(x,1), :);    
    EA_obj2 = fitx(PS1+1:size(x,1));
    
    %% Initialize CMA-ES parameters
    setting = [];
    [setting] = init_cma_par(setting, EA_2, Par.n, PS2);
    
    %% Probability of each operator
    probDE1 = 1./Par.n_opr .* ones(1, Par.n_opr);
    probSC = 1./Par.n_opr .* ones(1, Par.n_opr);
    
    %% Archive data
    arch_rate = 2.6;
    archive.NP = arch_rate * PS1;
    archive.pop = zeros(0, Par.n);
    archive.funvalues = zeros(0, 1);
    
    %% Memory for adaptive CR and F
    hist_pos = 1;
    memory_size = 6;
    archive_f = ones(1, memory_size) .* 0.7;
    archive_Cr = ones(1, memory_size) .* 0.5;
    archive_T = ones(1, memory_size) .* 0.1;
    archive_freq = ones(1, memory_size) .* 0.5;
    
    stop_con = 0; 
    InitPop = PS1;
    cy = 0; 
    indx = 0; 
    Probs = ones(1, 2);
    iter = 0;
    bnd = []; 
    fitness = [];
    
    %% Main loop
    while stop_con == 0
        iter = iter + 1;
        cy = cy + 1;
        
        % Determine the best phase
        if cy == ceil(Par.CS + 1)
            qual(1) = EA_obj1(1); 
            qual(2) = EA_obj2(1);
            norm_qual = qual ./ sum(qual);
            norm_qual = 1 - norm_qual;
            
            D(1) = mean(pdist2(EA_1(2:PS1,:), EA_1(1,:)));
            if isnan(EA_2)
                [mSet, nSet] = size(EA_2);
                EA_2 = ones(mSet, nSet) * -10;
            end
            D(2) = mean(pdist2(EA_2(2:PS2,:), EA_2(1,:)));
            norm_div = D ./ sum(D);
            
            Probs = norm_qual + norm_div;
            Probs = max(0.1, min(0.9, Probs ./ sum(Probs)));
            
            [~, indx] = max(Probs);
            if Probs(1) == Probs(2)
                indx = 0;
            end
            
        elseif cy == 2*ceil(Par.CS)
            if indx == 1
                list_ind = randperm(PS1);
                list_ind = list_ind(1:(min(PS2, PS1)));
                EA_2(1:size(list_ind,2), :) = EA_1(list_ind, :);
                EA_obj2(1:size(list_ind,2)) = EA_obj1(list_ind);
                [setting] = init_cma_par(setting, EA_2, Par.n, PS2);
                setting.sigma = setting.sigma * (1 - (FE / Par.Max_FES));
            else
                if (min(EA_2(1,:))) > -100 && (max(EA_2(1,:))) < 100
                    EA_1(PS1, :) = EA_2(1, :);
                    EA_obj1(PS1) = EA_obj2(1);
                    [EA_obj1, ind] = sort(EA_obj1);
                    EA_1 = EA_1(ind, :);
                end
            end
            cy = 1;   
            Probs = ones(1, 2);
        end
        
        %% EBO Phase
        if FE < Par.Max_FES
            if rand < Probs(1)
                % Linear reduction of population size
                UpdPopSize = round((((Par.MinPopSize - InitPop) / Par.Max_FES) * FE) + InitPop);
                if PS1 > UpdPopSize
                    reduction_ind_num = PS1 - UpdPopSize;
                    if PS1 - reduction_ind_num < Par.MinPopSize
                        reduction_ind_num = PS1 - Par.MinPopSize;
                    end
                    for r = 1:reduction_ind_num
                        vv = PS1;
                        EA_1(vv, :) = [];
                        EA_1old(vv, :) = [];
                        EA_obj1(vv) = [];
                        PS1 = PS1 - 1;
                    end
                    archive.NP = round(arch_rate * PS1);
                    if size(archive.pop, 1) > archive.NP
                        rndpos = randperm(size(archive.pop, 1));
                        rndpos = rndpos(1:archive.NP);
                        archive.pop = archive.pop(rndpos, :);
                    end
                end
                
                % Apply EBO
                [EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, ...
                    archive_f, archive_Cr, archive_T, archive_freq, FE, curve, population_history, ...
                    fitness_history, history_index] = ...
                    EBO(EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, archive, hist_pos, memory_size, ...
                    archive_f, archive_Cr, archive_T, archive_freq, Par.xmin, Par.xmax, Par.n, PS1, ...
                    FE, problem, curve, Par.Max_FES, Par.Gmax, iter, ...
                    population_history, fitness_history, history_index, sampling_interval, history_size, history_pop_size);
            end
        end
        
        %% Scout/CMAR Phase
        if FE < Par.Max_FES
            if rand < Probs(2)
                [EA_2, EA_obj2, setting, bestold, bestx, bnd, fitness, FE, curve, ...
                    population_history, fitness_history, history_index] = ...
                    Scout(EA_2, EA_obj2, probSC, setting, iter, bestold, bestx, fitness, bnd, ...
                    Par.xmin, Par.xmax, Par.n, PS2, FE, problem, curve, Par.Max_FES, ...
                    EA_1, EA_obj1, PS1, population_history, fitness_history, history_index, sampling_interval, history_size, history_pop_size);
            end
        end
        
        %% Local Search (LS2)
        if FE > 0.75 * Par.Max_FES
            if rand < Par.prob_ls
                old_fit_eva = FE;
                [bestx, bestold, FE, succ] = LS2(bestx, bestold, Par, FE, problem, Par.Max_FES, Par.xmin, Par.xmax);
                
                % Record curve for LS evaluations
                for ls_eval = old_fit_eva+1:min(FE, maxFE)
                    curve(ls_eval) = bestold;
                end
                
                if succ == 1
                    EA_1(PS1, :) = bestx';
                    EA_obj1(PS1) = bestold;
                    [EA_obj1, sort_indx] = sort(EA_obj1);
                    EA_1 = EA_1(sort_indx, :);
                    
                    EA_2 = repmat(EA_1(1,:), PS2, 1);
                    [setting] = init_cma_par(setting, EA_2, Par.n, PS2);
                    setting.sigma = 1e-05;
                    EA_obj2(1:PS2) = EA_obj1(1);
                    Par.prob_ls = 0.1;
                else
                    Par.prob_ls = 0.01;
                end
            end
        end
        
        %% Stopping criterion
        if FE >= Par.Max_FES
            stop_con = 1;
        end
    end
    
    % Return best solution
    best_solution = bestx;
    best_fitness = bestold;
end

%% ==================== EBO Function ====================
function [x, xold, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, ...
    archive_f, archive_Cr, archive_T, archive_freq, FE, curve, ...
    population_history, fitness_history, history_index] = ...
    EBO(x, xold, fitx, prob, bestold, bestx, archive, hist_pos, memory_size, ...
    archive_f, archive_Cr, archive_T, archive_freq, xmin, xmax, n, PopSize, ...
    FE, problem, curve, Max_FES, G_Max, gg, ...
    population_history, fitness_history, history_index, sampling_interval, history_size, history_pop_size)

    vi = zeros(PopSize, n);
    
    % Calculate CR and F
    mem_rand_index = ceil(memory_size * rand(PopSize, 1));
    mu_sf = archive_f(mem_rand_index);
    mu_cr = archive_Cr(mem_rand_index);
    mu_T = archive_T(mem_rand_index);
    mu_freq = archive_freq(mem_rand_index);
    
    % Generate CR
    cr = (mu_cr + 0.1*sqrt(pi)*(asin(-rand(1,PopSize))+asin(rand(1,PopSize))))';
    cr(mu_cr == -1) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);
    
    % Generate F
    F = mu_sf + 0.1 * tan(pi * (rand(1,PopSize) - 0.5));
    pos = find(F <= 0);
    while ~isempty(pos)
        F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1,length(pos)) - 0.5));
        pos = find(F <= 0);
    end
    F = min(F, 1)';
    
    % Generate T
    T = mu_T + 0.05*(sqrt(pi)*(asin(-rand(1, PopSize))+asin(rand(1, PopSize))));
    T = max(T, 0)'; 
    T = min(T, 0.5)';
    l = floor(n*rand(1,PopSize))+1;
    CR = zeros(PopSize, n);
    for i = 1:PopSize
        % Handle both even and odd dimensions
        half_n = floor(n/2);
        if half_n < 1
            half_n = 1;  % Minimum 1 element
        end
        mm = exp(-T(i)/n*(0:half_n-1));
        ll_temp = [mm fliplr(mm)];
        % Adjust ll to have exactly n elements
        if length(ll_temp) < n
            ll = [ll_temp, ll_temp(1:n-length(ll_temp))];
        elseif length(ll_temp) > n
            ll = ll_temp(1:n);
        else
            ll = ll_temp;
        end
        ll = cr(i) .* ll;
        CR(i,[l(i):n (1:l(i)-1)]) = ll;
    end
    
    % Generate freq
    freq = mu_freq + 0.1 * tan(pi*(rand(1, PopSize) - 0.5));
    pos_f = find(freq <= 0);
    while ~isempty(pos_f)
        freq(pos_f) = mu_freq(pos_f) + 0.1 * tan(pi * (rand(1,length(pos_f)) - 0.5));
        pos_f = find(freq <= 0);
    end
    freq = min(freq, 1)';
    
    if FE <= Max_FES/2
        c = rand;
        if c < 0.5
            F = 0.5.*(tan(2.*pi.*0.5.*gg+pi) .* ((G_Max-gg)/G_Max) + 1) .* ones(PopSize,1);
        else
            F = 0.5 * (tan(2*pi .* freq(:, ones(1, 1)) .* gg) .* (gg/G_Max) + 1) .* ones(PopSize, 1);
        end
    end
    
    % Generate new x
    popAll = [x; archive.pop];
    r0 = 1:PopSize;
    [r1, r2, r3] = gnR1R2(PopSize, size(popAll, 1), r0);
    
    % Mutation
    bb = rand(PopSize, 1);
    probiter = prob(1,:);
    l2 = sum(prob(1:2));
    op_1 = bb <= probiter(1)*ones(PopSize, 1);
    op_2 = bb > probiter(1)*ones(PopSize, 1) & bb <= (l2*ones(PopSize, 1));
    
    randindex = bestt(PopSize, n);
    phix = x(randindex, :);
    
    vi(op_1==1,:) = x(op_1==1,:) + F(op_1==1, ones(1, n)) .* (x(r1(op_1==1),:) - x(op_1==1,:) + x(r3(op_1==1), :) - popAll(r2(op_1==1), :));
    vi(op_2==1,:) = x(op_2==1,:) + F(op_2==1, ones(1, n)) .* (phix(op_2==1,:) - x(op_2==1,:) + x(r1(op_2==1), :) - x(r3(op_2==1), :));
    
    % Handle boundaries
    vi = han_boun(vi, xmax, xmin, x, PopSize, 1);
    
    % Crossover
    mask = rand(PopSize, n) > CR;
    rows = (1:PopSize)'; 
    cols = floor(rand(PopSize, 1) * n) + 1;
    jrand = sub2ind([PopSize n], rows, cols); 
    mask(jrand) = false;
    ui = vi; 
    ui(mask) = x(mask);
    
    % Evaluate
    [fitx_new, FE] = calculate_fitness(ui', problem, FE);
    fitx_new = fitx_new(:)';  % Ensure row vector (1 x PopSize)
    
    % Record curve with top individuals for history
    for eval_idx = 1:PopSize
        eval_count = FE - PopSize + eval_idx;
        if eval_count <= Max_FES
            curve(eval_count) = bestold;
            [sorted_fit, sorted_idx] = sort(fitx);
            top_k = min(history_pop_size, PopSize);
            rec_pop = NaN(history_pop_size, n);
            rec_fit = NaN(1, history_pop_size);
            rec_pop(1:top_k, :) = x(sorted_idx(1:top_k), :);
            rec_fit(1:top_k) = sorted_fit(1:top_k);
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, rec_pop, rec_fit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    % Calculate improvement for Cr and F
    % Ensure fitx and fitx_new are same orientation (row vectors)
    fitx = fitx(:)';
    fitx_new = fitx_new(:)';
    diff = abs(fitx - fitx_new);
    I = (fitx_new < fitx);
    I = I(:);  % Ensure column vector for indexing
    goodCR = cr(I);
    goodF = F(I);
    goodT = T(I)';
    goodFreq = freq(I);
    
    % Update archive
    archive = updateArchive(archive, x(I, :), fitx(I)');
    
    % Update probability of each DE
    diff2 = max(0, (fitx - fitx_new)) ./ abs(fitx);
    op_1_col = op_1(:);  % Ensure column vector
    op_2_col = op_2(:);
    diff2_col = diff2(:);
    count_S(1) = max(0, mean(diff2_col(op_1_col)));
    count_S(2) = max(0, mean(diff2_col(op_2_col)));
    
    if count_S ~= 0 
        prob = max(0.1, min(0.9, count_S ./ (sum(count_S))));
    else
        prob = 1/2 * ones(1, 2);
    end
    
    % Update x and fitx
    fitx(I) = fitx_new(I); 
    xold(I, :) = x(I, :);
    x(I, :) = ui(I, :);
    
    % Update memory cr and F
    num_success_params = numel(goodCR);
    if num_success_params > 0
        diff_col = diff(:);
        weightsDE = (diff_col(I) ./ sum(diff_col(I)))';  % Row vector for matrix multiplication
        goodF_col = goodF(:);  % Ensure column vector
        goodCR_col = goodCR(:);
        goodT_col = goodT(:);
        goodFreq_col = goodFreq(:);
        archive_f(hist_pos) = (weightsDE * (goodF_col .^ 2)) ./ (weightsDE * goodF_col);
        
        if max(goodCR_col) == 0 || archive_Cr(hist_pos) == -1
            archive_Cr(hist_pos) = -1;
        else
            archive_Cr(hist_pos) = (weightsDE * (goodCR_col .^ 2)) / (weightsDE * goodCR_col);
        end
        
        hist_pos = hist_pos + 1;
        if hist_pos > memory_size
            hist_pos = 1; 
        end
        
        archive_T(hist_pos) = (weightsDE * (goodT_col .^ 2)) ./ (weightsDE * goodT_col);
        
        if max(goodFreq_col) == 0 || archive_freq(hist_pos) == -1
            archive_freq(hist_pos) = -1;
        else
            archive_freq(hist_pos) = (weightsDE * (goodFreq_col .^ 2)) / (weightsDE * goodFreq_col);
        end
    end
    
    % Sort new x, fitness
    [fitx, ind] = sort(fitx);
    x = x(ind, :);
    xold = xold(ind, :);
    
    % Update best
    if fitx(1) < bestold && min(x(ind(1),:)) >= -100 && max(x(ind(1),:)) <= 100
        bestold = fitx(1);
        bestx = x(1, :);
    end
    
    % Update curve with new best
    for eval_idx = 1:PopSize
        eval_count = FE - PopSize + eval_idx;
        if eval_count <= Max_FES
            curve(eval_count) = bestold;
        end
    end
end

%% ==================== Scout Function ====================
function [x, fitx, setting, bestold, bestx, bnd, fitness, FE, curve, ...
    population_history, fitness_history, history_index] = ...
    Scout(x, ~, prob, setting, iter, bestold, bestx, fitness, bnd, ...
    xmin, xmax, n, PopSize, FE, problem, curve, Max_FES, ...
    EA_1, EA_obj1, PS1, population_history, fitness_history, history_index, sampling_interval, history_size, history_pop_size)

    fitness.raw = NaN(1, PopSize);
    
    if rand < 1*prob(1)
        arz = sqrt(pi)*(asin(rand(n,PopSize))+asin(-rand(n,PopSize)));
    else
        arz = sqrt(pi)*asin(2*rand(n,PopSize)-1);
    end
    arx = repmat(setting.xmean, 1, PopSize) + setting.sigma * (setting.BD * arz);
    
    handle_limit = 0.5;
    if FE >= handle_limit*Max_FES
        arxvalid = han_boun(arx', xmax, xmin, x, PopSize, 2);
        arxvalid = arxvalid';
    else
        arxvalid = arx;
    end
    
    % Evaluate
    [fitness.raw, FE] = calculate_fitness(arxvalid, problem, FE);
    
    % Record curve with top individuals from EA_1 for history
    for eval_idx = 1:PopSize
        eval_count = FE - PopSize + eval_idx;
        if eval_count <= Max_FES
            curve(eval_count) = bestold;
            currentPS1 = min(PS1, size(EA_1, 1));
            [sorted_fit, sorted_idx] = sort(EA_obj1(1:currentPS1));
            top_k = min(history_pop_size, currentPS1);
            rec_pop = NaN(history_pop_size, n);
            rec_fit = NaN(1, history_pop_size);
            rec_pop(1:top_k, :) = EA_1(sorted_idx(1:top_k), :);
            rec_fit(1:top_k) = sorted_fit(1:top_k);
            [population_history, fitness_history, history_index] = record_history(...
                eval_count, rec_pop, rec_fit, population_history, fitness_history, ...
                history_index, sampling_interval, history_size);
        end
    end
    
    fitness.sel = fitness.raw;
    [fitness.sel, fitness.idxsel] = sort(fitness.sel);
    
    fitness.raw = fitness.raw(fitness.idxsel);
    arxvalid = arxvalid(:, fitness.idxsel);
    arx = arx(:, fitness.idxsel);
    arz = arz(:, fitness.idxsel);
    [~, pos_ro] = min(fitness.raw);
    
    % Update best
    if fitness.raw(pos_ro) < bestold && (min(arxvalid(:,pos_ro))) >= -100 && (max(arxvalid(:,pos_ro))) <= 100
        bestold = fitness.raw(pos_ro);
        bestx = arxvalid(:, pos_ro)';
    end
    
    % Update curve with new best
    for eval_idx = 1:PopSize
        eval_count = FE - PopSize + eval_idx;
        if eval_count <= Max_FES
            curve(eval_count) = bestold;
        end
    end
    
    % Update CMA-ES parameters
    setting.weights = fitness.raw(1:setting.mu)';
    if sum(setting.weights) > 1e25
        setting.weights = 1/setting.mu*ones(setting.mu,1);
    end
    setting.weights = setting.weights/sum(setting.weights);
    setting.weights = fliplr(setting.weights);
    
    setting.xold = setting.xmean;
    cmean = 1;
    setting.xmean = (1-cmean) * setting.xold + cmean * arx(:,(1:setting.mu))*setting.weights;
    
    if FE >= handle_limit*Max_FES
        setting.xmean = han_boun(setting.xmean', xmax, xmin, x(1,:), 1, 2);
        setting.xmean = setting.xmean';
    end
    
    zmean = arz(:,(1:setting.mu))*setting.weights;
    setting.ps = (1-setting.cs)*setting.ps + sqrt(setting.cs*(2-setting.cs)*setting.mueff) * (setting.B*zmean);
    hsig = norm(setting.ps)/sqrt(1-(1-setting.cs)^(2*iter))/setting.chiN < 1.4 + 2/(n+1);
    
    setting.pc = (1-setting.cc)*setting.pc ...
        + hsig*(sqrt(setting.cc*(2-setting.cc)*setting.mueff)/setting.sigma/cmean) * (setting.xmean-setting.xold);
    
    neg.ccov = 0;
    if setting.ccov1 + setting.ccovmu > 0
        arpos = (arx(:,(1:setting.mu))-repmat(setting.xold,1,setting.mu)) / setting.sigma;
        setting.C = (1-setting.ccov1-setting.ccovmu) * setting.C ...
            + setting.ccov1 * setting.pc*setting.pc' ...
            + setting.ccovmu * arpos * (repmat(setting.weights,1,n) .* arpos');
        setting.diagC = diag(setting.C);
    end
    
    setting.sigma = setting.sigma * exp(min(1, (sqrt(sum(setting.ps.^2))/setting.chiN - 1) * setting.cs/setting.damps));
    
    if (setting.ccov1+setting.ccovmu+neg.ccov) > 0 && mod(iter, 1/(setting.ccov1+setting.ccovmu+neg.ccov)/n/10) < 1
        setting.C = triu(setting.C)+triu(setting.C,1)';
        if isnan(setting.C)
            [mSet, nSet] = size(setting.C);
            setting.C = rand(mSet, nSet);
        end
        
        [setting.B, tmp] = eig(setting.C);
        setting.diagD = diag(tmp);
        
        if min(setting.diagD) <= 0
            setting.diagD(setting.diagD<0) = 0;
            tmp = max(setting.diagD)/1e14;
            setting.C = setting.C + tmp*eye(n,n); 
            setting.diagD = setting.diagD + tmp*ones(n,1);
        end
        if max(setting.diagD) > 1e14*min(setting.diagD)
            tmp = max(setting.diagD)/1e14 - min(setting.diagD);
            setting.C = setting.C + tmp*eye(n,n); 
            setting.diagD = setting.diagD + tmp*ones(n,1);
        end
        
        setting.diagC = diag(setting.C);
        setting.diagD = sqrt(setting.diagD);
        setting.BD = setting.B.*repmat(setting.diagD',n,1);
    end
    
    x = arxvalid';
    fitx = fitness.raw;
end

%% ==================== Helper Functions ====================

function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)
    NP0 = length(r0);
    
    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000
            error('Cannot generate r1 in 1000 iterations');
        end
    end
    
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1:99999999
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else
            r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000
            error('Cannot generate r2 in 1000 iterations');
        end
    end
    
    r3 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1:99999999
        pos = ((r3 == r0) | (r3 == r1) | (r3 == r2));
        if sum(pos) == 0
            break;
        else
            r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000
            error('Cannot generate r3 in 1000 iterations');
        end
    end
end

function archive = updateArchive(archive, pop, funvalue)
    if archive.NP == 0
        return; 
    end
    
    if size(pop, 1) ~= size(funvalue, 1)
        error('check it'); 
    end
    
    popAll = [archive.pop; pop];
    funvalues = [archive.funvalues; funvalue];
    [~, IX] = unique(popAll, 'rows');
    if length(IX) < size(popAll, 1)
        popAll = popAll(IX, :);
        funvalues = funvalues(IX, :);
    end
    
    if size(popAll, 1) <= archive.NP
        archive.pop = popAll;
        archive.funvalues = funvalues;
    else
        rndpos = randperm(size(popAll, 1));
        rndpos = rndpos(1:archive.NP);
        archive.pop = popAll(rndpos, :);
        archive.funvalues = funvalues(rndpos, :);
    end
end

function i = bestt(n, D)
    i = zeros(1, n);
    k = n;
    if 2*D > n
        D = 1;
        n = max(round(0.1*n), 2);
    end
    for j = 1:k
        i(j) = min(randperm(n, D));
    end
end

function x = han_boun(x, xmax, xmin, x2, PopSize, hb)
    switch hb
        case 1 % for DE
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x(pos) = (x2(pos) + x_L(pos)) / 2;
            
            x_U = repmat(xmax, PopSize, 1);
            pos = x > x_U;
            x(pos) = (x2(pos) + x_U(pos)) / 2;
            
        case 2 % for CMA-ES
            x_L = repmat(xmin, PopSize, 1);
            pos = x < x_L;
            x_U = repmat(xmax, PopSize, 1);
            x(pos) = min(x_U(pos), max(x_L(pos), 2*x_L(pos)-x2(pos)));
            pos = x > x_U;
            x(pos) = max(x_L(pos), min(x_U(pos), 2*x_L(pos)-x2(pos)));
    end
end

function [setting] = init_cma_par(setting, EA_2, n, n2)
    setting.xmean = mean(EA_2);
    setting.xmean = setting.xmean';
    setting.insigma = 0.3;
    setting.sigma = setting.insigma;
    
    setting.sigma = max(setting.insigma);
    setting.pc = zeros(n, 1); 
    setting.ps = zeros(n, 1);
    
    if length(setting.insigma) == 1
        setting.insigma = setting.insigma * ones(n, 1);
    end
    setting.diagD = setting.insigma / max(setting.insigma);
    setting.diagC = setting.diagD.^2;
    setting.B = eye(n, n);
    setting.BD = setting.B .* repmat(setting.diagD', n, 1);
    setting.C = diag(setting.diagC);
    setting.D = ones(n, 1);
    setting.chiN = n^0.5 * (1 - 1/(4*n) + 1/(21*n^2));
    setting.mu = ceil(n2/2);
    setting.weights = log(max(setting.mu, n/2) + 1/2) - log(1:setting.mu)';
    setting.mueff = sum(setting.weights)^2 / sum(setting.weights.^2);
    setting.weights = setting.weights / sum(setting.weights);
    
    setting.cc = (4 + setting.mueff/n) / (n+4 + 2*setting.mueff/n);
    setting.cs = (setting.mueff+2) / (n+setting.mueff+3);
    setting.ccov1 = 2 / ((n+1.3)^2+setting.mueff);
    setting.ccovmu = 2 * (setting.mueff-2+1/setting.mueff) / ((n+2)^2+setting.mueff);
    setting.damps = 0.5 + 0.5*min(1, (0.27*n2/setting.mueff-1)^2) + 2*max(0,sqrt((setting.mueff-1)/(n+1))-1) + setting.cs;
    
    setting.xold = setting.xmean;
end

function [Par] = Introd_Par(dimension, lbArray, ubArray, maxIteration)
    Par.n_opr = 2;
    Par.n = dimension;
    
    if Par.n == 10
        Par.CS = 50;
        Par.Gmax = 2163;
    elseif Par.n == 30
        Par.CS = 100;
        Par.Gmax = 2745;
    elseif Par.n == 50
        Par.CS = 150;
        Par.Gmax = 3022;
    else
        Par.CS = 150;
        Par.Gmax = 3401;
    end
    Par.xmin = lbArray;
    Par.xmax = ubArray;
    Par.Max_FES = maxIteration;
    Par.PopSize = 18*Par.n;
    Par.MinPopSize = 4;
    Par.prob_ls = 0.1;
end

function [x, f, FE, succ] = LS2(bestx, f, Par, FE, problem, Max_FES, xmin, xmax)
    Par.LS_FE = ceil(20.0000e-003 * Max_FES);
    options = optimset('Display', 'off', 'algorithm', 'sqp', ...
        'UseParallel', 'never', 'MaxFunEvals', Par.LS_FE);
    
    % Create wrapper function for fmincon
    objfun = @(x_in) evaluate_for_ls(x_in, problem);
    
    [Xsqp, FUN, ~, details] = fmincon(objfun, bestx(1,:)', [], [], [], [], xmin, xmax, [], options);
    
    if (f - FUN) > 0
        succ = 1;
        f = FUN;
        x(1,:) = Xsqp;
    else
        succ = 0;
        x = bestx;
    end
    
    FE = FE + details.funcCount;
end

function f = evaluate_for_ls(x, problem)
    % Wrapper function for local search
    f = feval(problem.fhd, x, problem.number);
end

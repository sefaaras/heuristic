% ----------------------------------------------------------------------- %
% Coral Reefs Optimization (CRO)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   N1 x N2 = 10 x 10          % Reef cells, each empty or holding one coral
%   r0      = 0.6              % Occupied share of the reef at initialisation
%   Fb      = 0.9              % Share of the corals drawn as broadcast spawners
%   Fa      = 0.01             % Share of the best corals budding each generation
%   k       = 3                % Settlement attempts per larva
%   K       = 0.2*N1*N2 = 20   % Identical corals extreme depredation leaves
%   eta     = 2                % Exponent index of the released SBX, B = (2U)^(1/3)
%   sigma   = (ub - lb)/100    % Gaussian brooding step, paper Section 4.1
%
% Algorithm Concept:
%   - Broadcast spawning: round(Fb*n) corals, made even, pair up; one cut point
%     per pair, genes scaled by (1 -/+ B)/2, two larvae per pair
%   - Brooding: round((1 - Fb)*n) corals, drawn independently of the spawners,
%     each release the larva x + N(0, sigma^2)
%   - Larvae setting: larvae whose random cell is empty take it; the others try
%     k random occupied cells and replace the first coral they beat
%   - Budding: copies of the ceil(Fa*n) best corals settle the same way,
%     carrying their fitness (no evaluation)
%   - Extreme depredation: past K identical corals, later copies are removed
%
% Reference:
% S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-Lopez, J. A. Portilla-Figueras,
% The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently
% Solving Optimization Problems,
% The Scientific World Journal 2014 (2014) 739768, 1-15.
% https://doi.org/10.1155/2014/739768
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release CRO_Original_14_07_2014.zip (max-ones, on
% agamenon.tsc.uah.es/Personales/sancho/CRO.html); seeded runs are bit-identical to its
% functions. Kept as released: its parameters (paper: rho0 = 0.7, Fa ~ 0.1), two larvae per
% pair, spawners and brooders drawn independently, intersect's sorted cell order, the 'cont'
% SBX scaling every gene toward x = 0 (centre bias), and extreme depredation in place of the
% Pd/Fd depredation mainCRO.m comments out. From the paper: Gaussian brooding, CRO(G) of
% Section 4.1, not the release's polynomial step. Harness: larvae clamped to the box
% (larvaecorrection.m is empty); initialisation scores occupied cells only, not the empty
% zero rows; the best is updated at every evaluation. Paper Table 2 is not reproduced:
%     30 runs, 1e5 FE (paper: not stated)  Rosenbrock 2-D  Schwefel 10-D  Rastrigin  Griewank
%     paper, CRO(G)                        7.1e-6          1.3e-4         7.2e-3     0.22
%     this port                            6.9e-6          1.5e3          0          0
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = cro(problem)

    L     = problem.dimension;
    lb    = problem.lb(:)';
    ub    = problem.ub(:)';
    maxFE = problem.maxFe;

    N1    = 10;
    N2    = 10;
    P     = N1 * N2;
    r0    = 0.6;
    Fb    = 0.9;
    Fa    = 0.01;
    k0    = 3;
    K     = round(0.2 * P);
    eta   = 2;
    sigma = (ub - lb) / 100;

    FE    = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history    = [];
    history_index      = 1;

    % Empty cells hold zero rows, the release's marker; extreme depredation reads them
    REEF        = zeros(P, 1);
    REEFpob     = zeros(P, L);
    REEFfitness = zeros(P, 1);
    O    = round(P * r0);
    a    = randperm(P);
    ocup = a(1:O);
    REEF(ocup) = 1;
    REEFpob(ocup, :) = lb + rand(O, L) .* (ub - lb);

    occ = find(REEF);
    m   = min(numel(occ), maxFE);
    [f, FE] = calculate_fitness(REEFpob(occ(1:m), :)', problem, FE);
    REEFfitness(occ(1:m)) = f(:);

    best_fitness  = inf;
    best_solution = REEFpob(occ(1), :);
    Xrec = REEFpob(occ, :);
    frec = REEFfitness(occ)';
    for g = 1:m
        stamp(g, REEFpob(occ(g), :), REEFfitness(occ(g)));
    end

    while FE < maxFE
        nOcc = sum(REEF);

        % Broadcast spawning, broadcastspawning.m with type 'cont' and cruce 'SBX'
        nspawners = round(Fb * nOcc);
        if mod(nspawners, 2) ~= 0
            nspawners = nspawners - 1;
        end
        p = find(REEF ~= 0);
        a = randperm(length(p));
        spawners  = p(a(1:nspawners));
        spawners1 = REEFpob(spawners(1:nspawners/2), :);
        spawners2 = REEFpob(spawners(nspawners/2+1:end), :);
        np   = size(spawners1, 1);
        cut  = ceil(L * rand(1, np));
        mask = double((1:L) >= cut');   % genes from the cut on come from the second parent
        U = rand(size(spawners1));
        B = (2 * U) .^ (1 / (eta + 1));
        ESlarvae1 = spawners1.*(1-mask).*(1-B)/2+spawners2.*mask.*(1+B)/2;
        ESlarvae2 = spawners1.*(1-mask).*(1+B)/2+spawners2.*mask.*(1-B)/2;

        % Brooding: the release's selection, the paper's Gaussian mutation
        nbrooders = round((1 - Fb) * nOcc);
        p = find(REEF ~= 0);
        a = randperm(length(p));
        brooders = REEFpob(p(a(1:nbrooders)), :);
        ISlarvae = brooders + sigma .* randn(size(brooders));

        larvae = min(max([ESlarvae1; ESlarvae2; ISlarvae], lb), ub);
        nl = size(larvae, 1);
        m  = min(nl, maxFE - FE);
        [lf, FE] = calculate_fitness(larvae(1:m, :)', problem, FE);
        lf = lf(:);

        % The reef is the population; larvae join it only once they settle
        Xrec = REEFpob(REEF == 1, :);
        frec = REEFfitness(REEF == 1)';
        for g = 1:m
            stamp(FE - m + g, larvae(g, :), lf(g));
        end
        if m < nl, break; end

        [REEF, REEFpob, REEFfitness] = larvae_setting(REEF, REEFpob, REEFfitness, ...
            larvae, lf, k0);

        % Budding, budding.m: the best corals as copies with their fitness
        pob = REEFpob(REEF == 1, :);
        fit = REEFfitness(REEF == 1);
        NA  = ceil(Fa * size(pob, 1));
        [fit, ind] = sort(fit, 'ascend');
        [REEF, REEFpob, REEFfitness] = larvae_setting(REEF, REEFpob, REEFfitness, ...
            pob(ind(1:NA), :), fit(1:NA), k0);

        [REEF, REEFpob, REEFfitness] = extreme_depredation(REEF, REEFpob, REEFfitness, K);
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Best-so-far at evaluation e, then one history sample of the reef (Xrec, frec)
    function stamp(e, x, fx)
        if fx < best_fitness
            best_fitness  = fx;
            best_solution = x;
        end
        curve(e) = best_fitness;
        [population_history, fitness_history, history_index] = record_history(...
            e, Xrec, frec, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

% larvaesetting.m: free cells first, then up to k0 fights per larva over occupied cells
function [REEF, REEFpob, REEFfitness] = larvae_setting(REEF, REEFpob, REEFfitness, larvae, lfit, k0)
    Nlarvae = size(larvae, 1);
    P = size(REEFpob, 1);
    a = randperm(Nlarvae);
    larvae = larvae(a, :);
    lfit   = lfit(a);

    nreef = randperm(P);
    nreef = nreef(1:Nlarvae);
    free  = intersect(nreef, find(REEF == 0));   % sorted cell order, as released
    nf    = length(free);
    REEF(free) = 1;
    REEFpob(free, :)  = larvae(1:nf, :);
    REEFfitness(free) = lfit(1:nf);

    nreef = randperm(P);   % drawn although intersect sorts it away, as released
    ocup  = intersect(nreef, find(REEF == 1));
    for l = nf + 1:Nlarvae
        for t = 1:k0
            ind = ceil(length(ocup) * rand());
            if lfit(l) < REEFfitness(ocup(ind))
                REEFpob(ocup(ind), :) = larvae(l, :);
                REEFfitness(ocup(ind)) = lfit(l);
                ocup(ind) = [];
                break;
            end
        end
    end
end

% extremedepredation.m: rows equal to an earlier K are emptied, unique row by unique row
function [REEF, REEFpob, REEFfitness] = extreme_depredation(REEF, REEFpob, REEFfitness, K)
    [U, ~, ic] = unique(REEFpob, 'rows');
    z = find(~any(U, 2));
    % Only the zero group changes before its turn: it gains the rows emptied ahead of it
    for l = find(accumarray(ic, 1)' > K | ismember(1:size(U, 1), z))
        if ismember(l, z)
            rows = find(~any(REEFpob, 2));
        else
            rows = find(ic == l);
        end
        if numel(rows) > K
            kill = rows(K + 1:end);
            REEFpob(kill, :) = 0;
            REEFfitness(kill) = 0;
            REEF(kill) = 0;
        end
    end
end

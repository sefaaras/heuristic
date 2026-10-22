% ----------------------------------------------------------------------- %
% Komodo Mlipir Algorithm (KMA)
% ----------------------------------------------------------------------- %
% Algorithm Parameters:
%   PopSize     = 5                  % Stage-1 herd: 2 big males, 1 female, 2 small males
%   pop (st. 2) = 200 -> [20, 200]   % Adaptive, -5 / +5 per improving / stagnant spell
%   MlipirRate  = (D-1)/D, then 0.5  % Share of dimensions a small male moves, per stage
%   MutRate     = 0.5                % Per-dimension mutation probability
%   MutRadius   = 0.5                % Female mutation step, fraction of ub - lb
%   MaxGenExam1 = 100                % Stage-1 exam: stage 2 if the improve rate < 0.5
%   MaxGenExam2 = 1000               % Stage-1 generation cap
%
% Algorithm Concept:
%   - Herd sorted: best half big males, then one female, the rest small males
%   - Big males: attraction to (or random distraction from) 1-3 other big
%     males, then the best of old + new survive
%   - Female: whole arithmetic crossover with the winning big male, or a
%     uniform mutation of each dimension with probability 0.5
%   - Small males "mlipir": follow a big male on a random share of the
%     dimensions, replaced unconditionally
%   - Stage 1 runs a 5-bird herd; stage 2 (after a weak 100-generation exam)
%     splits 200 komodos into swarms of 5 that share all big males
%   - Stage 2 size: -5 after 3 improving generations, +5 Levy offspring of the
%     best after 3 stagnant ones (at 200: greedy reposition of every komodo)
%
% Reference:
% Suyanto Suyanto, Alifya Aisyah Ariyanto, Alifya Fatimah Ariyanto,
% Komodo Mlipir Algorithm,
% Applied Soft Computing 114 (2022) 108043.
% https://doi.org/10.1016/j.asoc.2021.108043
% ----------------------------------------------------------------------- %
% Implementation Note:
% Ported from the authors' release, File Exchange 102514 v1.0.1 (KMA2D.m,
% levy.m, 23 Nov 2021). The release stops on reaching the known optimum,
% which no algorithm sees here: removed, so stage 2 runs to maxFe. Its own
% counter misses the 195 stage-2 seeds and the second crossover child; FEs
% are counted exactly here. Reference defects kept: the mlipir step is coded
% r.*(q.*B) - w.*B, so a moved dimension lands on r*q, between 0 and the big
% male q; and stage 2 writes the big males back to AllHQ at their population
% row, not their slot, padding the leader pool with zero rows of fitness 0
% (66 of 194 for the last swarm of 200), so later swarms are pulled to x = 0.
% Reposition applies MutRadius twice, as released; best updated per FE.
% ----------------------------------------------------------------------- %
% Input:  problem struct (dimension, lb, ub, maxFe, fhd, number)
% Output: [best_fitness, best_solution, curve, population_history, fitness_history]
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = kma(problem)

    Nvar = problem.dimension;
    Rb = problem.lb;   % the release's Rb is the lower bound, Ra the upper
    Ra = problem.ub;
    maxFE = problem.maxFe;

    PopSize = 5;
    MinAdaPopSize = PopSize * 4;
    MaxAdaPopSize = PopSize * 40;
    MaxGenExam1 = 100;
    MaxGenExam2 = 1000;
    MutRate = 0.5;
    MutRadius = 0.5;

    FE = 0;
    curve = zeros(1, maxFE);

    population_history = [];  % record_history allocates the metric buffers on its first sample
    fitness_history = [];
    history_index = 1;

    % Operator state shared with the nested functions, as the release shares it through globals
    BigMales = []; BigMalesFX = [];
    Female = []; FemaleFX = [];
    SmallMales = []; SmallMalesFX = [];
    AllHQ = []; AllHQFX = [];
    MlipirRate = 0;
    NumBM = 0;
    base = 0;   % row offset in Pop of the herd or micro-swarm being moved

    Pop = PopConsInitialization(PopSize, Nvar, Ra, Rb);
    [FX, FE] = calculate_fitness(Pop', problem, FE);
    FX = FX(:)';

    [best_fitness, bidx] = min(FX);
    best_solution = Pop(bidx, :);
    for e = 1:min(PopSize, maxFE)
        curve(e) = min(FX(1:e));
        [population_history, fitness_history, history_index] = record_history(...
            e, Pop, FX, population_history, fitness_history, ...
            history_index, maxFE);
    end

    try
        search();
    catch ME
        if ~strcmp(ME.identifier, 'kma:budgetSpent')
            rethrow(ME);
        end
    end

    curve(min(FE, maxFE):end) = best_fitness;

    % Both stages; ends only through the budget error raised by Evaluation
    function search()
        [FX, IndFX] = sort(FX, 'ascend');
        Pop = Pop(IndFX, :);
        OneElitFX = FX(1);
        NumBM = floor(PopSize / 2);
        MlipirRate = (Nvar - 1) / Nvar;

        ImproveRate = 0;
        Gen = 0;
        GenImprove = 0;
        while Gen < MaxGenExam2
            Gen = Gen + 1;
            base = 0;
            BigMales = Pop(1:NumBM, :);
            BigMalesFX = FX(1:NumBM);
            Female = Pop(NumBM + 1, :);
            FemaleFX = FX(NumBM + 1);
            SmallMales = Pop(NumBM + 2:end, :);
            SmallMalesFX = FX(NumBM + 2:end);

            moveBigMalesFemale(1);
            moveSmallMales(1);

            Pop = [BigMales; Female; SmallMales];
            FX = [BigMalesFX FemaleFX SmallMalesFX];
            [FX, IndFX] = sort(FX, 'ascend');
            Pop = Pop(IndFX, :);
            OptVal = FX(1);

            if OptVal < OneElitFX
                GenImprove = GenImprove + 1;
                ImproveRate = GenImprove / Gen;   % refreshed only on improvement, as released
                OneElitFX = OptVal;
            end
            if Gen == MaxGenExam1 && ImproveRate < 0.5
                break;
            end
        end

        FirstStagePop = Pop;
        FirstStageFX = FX;
        SwarmSize = size(FirstStagePop, 1);
        NumBM = floor(SwarmSize / 2);
        IncAdaPopSize = SwarmSize;
        DecAdaPopSize = SwarmSize;
        MlipirRate = 0.5;
        MaxGenImprove = 2;
        MaxGenStagnan = 2;
        GenImprove = 0;
        GenStagnan = 0;

        % Evaluated one by one and appended as evaluated, so every record is a real population
        ConsPop = PopConsInitialization(MaxAdaPopSize - SwarmSize, Nvar, Ra, Rb);
        Pop = FirstStagePop;
        FX = FirstStageFX;
        for nn = 1:size(ConsPop, 1)
            fx = Evaluation(ConsPop(nn, :));
            Pop(end + 1, :) = ConsPop(nn, :);
            FX(end + 1) = fx;
            stamp();
        end
        PopSize = size(Pop, 1);
        OneElitFX = min(FX);

        while true
            AdaPopSize = size(Pop, 1);
            AllHQ = [];
            AllHQFX = [];
            for ind = 1:SwarmSize:AdaPopSize
                MicroSwarm = Pop(ind:ind + SwarmSize - 1, :);
                MicroSwarmFX = FX(ind:ind + SwarmSize - 1);
                [~, IndFX] = sort(MicroSwarmFX, 'ascend');
                MicroSwarm = MicroSwarm(IndFX, :);
                MicroSwarmFX = MicroSwarmFX(IndFX);
                AllHQ = [AllHQ; MicroSwarm(1:NumBM, :)]; %#ok<AGROW>
                AllHQFX = [AllHQFX MicroSwarmFX(1:NumBM)]; %#ok<AGROW>
            end
            for ind = 1:SwarmSize:AdaPopSize
                rows = ind:ind + SwarmSize - 1;
                % Sorted in place: the release overwrites these rows with the same order afterwards
                [~, IndFX] = sort(FX(rows), 'ascend');
                Pop(rows, :) = Pop(rows(IndFX), :);
                FX(rows) = FX(rows(IndFX));
                base = ind - 1;
                BigMales = Pop(rows(1:NumBM), :);
                BigMalesFX = FX(rows(1:NumBM));
                Female = Pop(rows(NumBM + 1), :);
                FemaleFX = FX(rows(NumBM + 1));
                SmallMales = Pop(rows(NumBM + 2:end), :);
                SmallMalesFX = FX(rows(NumBM + 2:end));

                moveBigMalesFemale(2);
                moveSmallMales(2);

                AllHQ(ind:ind + NumBM - 1, :) = BigMales;   % population row, not HQ slot: release defect kept
                AllHQFX(ind:ind + NumBM - 1) = BigMalesFX;
                Pop(rows, :) = [BigMales; Female; SmallMales];
                FX(rows) = [BigMalesFX FemaleFX SmallMalesFX];
            end

            Ind = randperm(length(FX));
            Pop = Pop(Ind, :);
            FX = FX(Ind);
            [~, IndMin] = min(FX);
            BestIndiv = Pop(IndMin, :);
            OptVal = FX(IndMin);

            if OptVal < OneElitFX
                GenImprove = GenImprove + 1;
                GenStagnan = 0;
                OneElitFX = OptVal;
            else
                GenStagnan = GenStagnan + 1;
                GenImprove = 0;
            end

            if GenImprove > MaxGenImprove
                AdaPopSize = AdaPopSize - DecAdaPopSize;
                if AdaPopSize < MinAdaPopSize
                    AdaPopSize = MinAdaPopSize;
                end
                [SortedFX, IndFX] = sort(FX, 'ascend');
                SortedPop = Pop(IndFX, :);
                Pop = SortedPop(1:AdaPopSize, :);
                FX = SortedFX(1:AdaPopSize);
                GenImprove = 0;
            end

            if GenStagnan > MaxGenStagnan
                AdaPopSizeOld = size(Pop, 1);
                AdaPopSize = AdaPopSize + IncAdaPopSize;
                NumAddPop = AdaPopSize - AdaPopSizeOld;
                if AdaPopSize > MaxAdaPopSize
                    AdaPopSize = AdaPopSizeOld;
                    NumAddPop = 0;
                end
                if AdaPopSize > AdaPopSizeOld
                    % Levy offspring of the best, appended as evaluated (the release appends the batch)
                    for nn = 1:NumAddPop
                        NewX = BestIndiv + (0.05 * levy(1, Nvar, 1.5)) .* abs(Ra - Rb);
                        NewX = trimr(NewX);
                        fx = Evaluation(NewX);
                        Pop(end + 1, :) = NewX;
                        FX(end + 1) = fx;
                        stamp();
                    end
                else
                    for nn = 1:size(Pop, 1)
                        reposition(nn);
                    end
                end
                GenStagnan = 0;
            end

            RandInd = randperm(size(Pop, 1));
            FX = FX(RandInd);
            Pop = Pop(RandInd, :);
        end
    end

    % Big males move, survivors are the best of old + new; then the winner mates the female
    function moveBigMalesFemale(stage)
        if stage == 1
            HQ = BigMales;
            HQFX = BigMalesFX;
        else
            HQ = [BigMales; AllHQ];
            HQFX = [BigMalesFX AllHQFX];
        end
        TempSM = BigMales;
        TempSMFX = BigMalesFX;

        for ss = 1:size(TempSM, 1)
            % The two stages draw in a different order, as released
            if stage == 1
                MaxFolHQ = randi(2);
                VM = zeros(1, Nvar);
                RHQ = randperm(size(HQ, 1));
            else
                VM = zeros(1, Nvar);
                RHQ = randperm(size(HQ, 1));
                MaxFolHQ = randi(3);
            end
            FolHQ = 0;
            for fs = 1:length(RHQ)
                ind = RHQ(fs);
                if ind ~= ss
                    if HQFX(ind) < TempSMFX(ss) || rand < 0.5   % attraction
                        VM = VM + rand * (HQ(ind, :) - TempSM(ss, :));
                    else   % distraction
                        VM = VM + rand * (TempSM(ss, :) - HQ(ind, :));
                    end
                end
                FolHQ = FolHQ + 1;   % counts the male itself too, as released
                if FolHQ >= MaxFolHQ
                    break
                end
            end
            NewBM = trimr(TempSM(ss, :) + VM);
            TempSM(ss, :) = NewBM;
            TempSMFX(ss) = Evaluation(NewBM);
            stamp();
        end
        [BigMales, BigMalesFX] = Replacement(BigMales, BigMalesFX, TempSM, TempSMFX);
        Pop(base + (1:NumBM), :) = BigMales;
        FX(base + (1:NumBM)) = BigMalesFX;

        WinnerBM = BigMales(1, :);
        WinnerFX = BigMalesFX(1);
        frow = base + NumBM + 1;
        if WinnerFX < FemaleFX || rand < 0.5   % sexual reproduction
            OffSprings = CrossOver(WinnerBM, Female);
            fx1 = Evaluation(OffSprings(1, :));
            stamp();
            fx2 = Evaluation(OffSprings(2, :));
            if fx1 < fx2
                if fx1 < FemaleFX
                    Female = OffSprings(1, :);
                    FemaleFX = fx1;
                end
            else
                if fx2 < FemaleFX
                    Female = OffSprings(2, :);
                    FemaleFX = fx2;
                end
            end
        else   % asexual reproduction
            NewFemale = Female;
            MaxStep = MutRadius * (Ra - Rb);
            for ii = 1:Nvar
                if rand < MutRate
                    NewFemale(:, ii) = Female(:, ii) + (2 * rand - 1) * MaxStep(ii);
                end
            end
            NewFemale = trimr(NewFemale);
            fx = Evaluation(NewFemale);
            if fx < FemaleFX
                Female = NewFemale;
                FemaleFX = fx;
            end
        end
        Pop(frow, :) = Female;
        FX(frow) = FemaleFX;
        stamp();
    end

    % Small males "mlipir" behind the big males on a random subset of dimensions
    function moveSmallMales(stage)
        if stage == 1
            HQ = BigMales;
        else
            HQ = [BigMales; AllHQ];
        end
        TempWM = SmallMales;
        TempWMFX = SmallMalesFX;

        for ww = 1:size(SmallMales, 1)
            if stage == 1
                MaxFolHQ = 1;
            else
                MaxFolHQ = randi(3);
            end
            VMlipir = zeros(1, Nvar);
            RHQ = randperm(size(HQ, 1));
            FolHQ = 0;
            for fs = 1:length(RHQ)
                ind = RHQ(fs);
                A = randperm(Nvar);
                D = round(MlipirRate * Nvar);
                if D >= Nvar
                    D = Nvar - 1;
                end
                if D < 1
                    D = 1;
                end
                B = zeros(1, Nvar);
                B(A(1:D)) = 1;
                % Release precedence: r.*(q.*B) - w.*B, not r.*(q - w).*B
                VMlipir = VMlipir + rand(1, Nvar) .* (HQ(ind, :) .* B) - (SmallMales(ww, :) .* B);
                FolHQ = FolHQ + 1;
                if FolHQ >= MaxFolHQ
                    break
                end
            end
            NewSM = trimr(SmallMales(ww, :) + VMlipir);
            TempWM(ww, :) = NewSM;
            TempWMFX(ww) = Evaluation(NewSM);
            Pop(base + NumBM + 1 + ww, :) = NewSM;   % small males are replaced unconditionally
            FX(base + NumBM + 1 + ww) = TempWMFX(ww);
            stamp();
        end
        SmallMales = TempWM;
        SmallMalesFX = TempWMFX;
    end

    % Greedy mutation of one komodo with step MutRadius*MutRadius*(ub - lb)
    function reposition(nn)
        X = Pop(nn, :);
        TempX = X;
        MaxStep = MutRadius * (Ra - Rb);
        for ii = 1:Nvar
            if rand < MutRate
                TempX(:, ii) = X(:, ii) + (2 * rand - 1) * MutRadius * MaxStep(ii);
            end
        end
        TempX = trimr(TempX);
        TempFX = Evaluation(TempX);
        if TempFX < FX(nn)
            Pop(nn, :) = TempX;
            FX(nn) = TempFX;
        end
        stamp();
    end

    % Whole arithmetic crossover, one weight per dimension
    function Offsprings = CrossOver(Parent1, Parent2)
        Offsprings = zeros(2, Nvar);
        for ii = 1:Nvar
            rval = rand;
            Offsprings(1, ii) = rval * Parent1(ii) + (1 - rval) * Parent2(ii);
            Offsprings(2, ii) = rval * Parent2(ii) + (1 - rval) * Parent1(ii);
        end
        Offsprings(1, :) = trimr(Offsprings(1, :));
        Offsprings(2, :) = trimr(Offsprings(2, :));
    end

    % The release clamps per dimension; max/min also maps a NaN to the bound
    function Z = trimr(Z)
        Z = min(max(Z, Rb), Ra);
    end

    % The one evaluation point: refuses to exceed maxFe, tracks the best and the curve
    function fx = Evaluation(xe)
        if FE >= maxFE
            error('kma:budgetSpent', 'The FE budget is spent.');
        end
        [fx, FE] = calculate_fitness(xe', problem, FE);
        if fx < best_fitness
            best_fitness = fx;
            best_solution = xe;
        end
        curve(FE) = best_fitness;
    end

    % Records (Pop, FX), kept coherent: each row is where its fitness was measured
    function stamp()
        [population_history, fitness_history, history_index] = record_history(...
            FE, Pop, FX, population_history, fitness_history, ...
            history_index, maxFE);
    end
end

% Elitist survival: the best size(X, 1) of old + new
function [Z, FZ] = Replacement(X, FX, Y, FY)
    LX = size(X, 1);
    XY = [X; Y];
    FXFY = [FX FY];
    [SortedVal, SortedInd] = sort(FXFY, 'ascend');
    Z = XY(SortedInd(1:LX), :);
    FZ = SortedVal(1:LX);
end

% Constrained initialization near the four "corners": halves of x at 1 % or 99 % of the box
function X = PopConsInitialization(PS, Nvar, Ra, Rb)
    F1 = [0.01 0.01 0.99 0.99];
    F2 = [0.01 0.99 0.01 0.99];
    X = zeros(PS, Nvar);
    IndX = 1;
    for nn = 1:4:PS
        if PS - nn >= 4
            NL = 4;
        else
            NL = PS - nn + 1;
        end
        ss = 1;
        while ss <= NL
            Temp = zeros(1, Nvar);
            for i = 1:floor(Nvar / 2)
                Temp(:, i) = Rb(i) + (Ra(i) - Rb(i)) * (F1(ss) + ((rand * 2) - 1) * 0.01);
            end
            for i = floor(Nvar / 2) + 1:Nvar
                Temp(:, i) = Rb(i) + (Ra(i) - Rb(i)) * (F2(ss) + ((rand * 2) - 1) * 0.01);
            end
            X(IndX, :) = Temp;
            IndX = IndX + 1;
            ss = ss + 1;
        end
    end
end

% Mantegna Levy steps; randn*sigma is the release's random('Normal', 0, sigma, n, m)
function z = levy(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
    sigma_u = (num / den)^(1 / beta);
    u = randn(n, m) .* sigma_u;
    v = randn(n, m);
    z = u ./ (abs(v).^(1 / beta));
end

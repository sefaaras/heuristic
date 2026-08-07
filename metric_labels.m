function L = metric_labels()
% Names and column order of the metrics record_history stores.
%
% Every entry is a function of THIS sample's population X (N x dim) and its
% fitness f (N x 1) alone, so a row can be read without its neighbours.
%
% Scale. The aggregate distance/spread scalars (9-15, 19-23, 26, 38-41, 44) are
% computed on the box-normalised copy z = (x - lb) ./ (ub - lb); in raw
% coordinates one wide dimension dictates every distance on CEC2020RW, whose
% per-dimension bounds span 1e-2..1e7. Column 32 (box_norm) is 0 when no bounds
% were available: those columns are then in raw units and 38-39 are NaN. The
% per-dimension layers in L.dim stay in RAW units, so they compare against lb/ub
% directly. Undefined metrics are NaN -- fdc and f_skew/f_kurt on a converged
% population, for instance.
%
% Fitness columns on CEC2020RW. f is NOT the objective there: the evaluator folds
% the constraints in through Eq.(6) of the competition guidelines, whose
% reference f_worst is a RUNNING MAXIMUM over the run. Columns 3-8, 16-18, 25 and
% 30 are therefore statistics of a run-private quantity -- readable along one
% run's own time axis, meaningless against another run or another algorithm.
% Columns 33-34 carry the raw objective for exactly that comparison, read in the
% same inert audit call that produces 27-29 and 35. Off CEC2020RW f IS the
% objective, so 33 and 34 both equal 3 and every column keeps one meaning across
% the whole grid.
%
% fdc (16, 30) correlates fitness with the distance to the population's CURRENT
% best individual, not to the global optimum. It describes the region the search
% is in rather than the landscape, and its reference point moves with the
% incumbent.
%
% Rows. Every run holds exactly T rows on one canonical FE grid; save_run fills
% the slots no call reached by holding the last MEASURED row forward. A held row
% is a verbatim copy whose fe column still carries the FE it was measured at, so
% repeated fe marks the repeats. run_info.n_history_samples counts the measured
% rows and n_history_rows those written; equal means full resolution.
%
% save_run appends one further column on disk, 'bsf', the best-so-far value of
% the convergence curve -- the only entry that is not a function of the sampled
% population, read at the row's own point on the FE grid. Address columns by name
% through the stored 'labels', never by index: appending metrics moves bsf.
%
% Distance floor. Pairwise distances come from a Gram identity on the centred
% cloud, which leaves about sqrt(eps)*||z-zbar|| for a coincident pair -- 1e-9
% box units for the unit box in 20-D, falling with the spread. It is relative, so
% div_mpd and diameter stay accurate to 1e-11, but exact duplicates read ~1e-9
% and not 0: test phenotypic collapse against div_dap (div_nnd/div_dap < 1e-7),
% never against zero. nnd_median (40) carries the same floor; nnd_cv (41) applies
% that test and is NaN when it fires; x_uniq_frac (36) compares coordinates and
% has no floor. Same cause: on a collapsed population radius (12) can read ~1e-16
% where diameter (21) is exactly 0, so compare the two with a tolerance.
%
% Boundary columns 38-39 need the bounds and are NaN when box_norm is 0. A
% coordinate is ON a bound within 1e-9 box units, which absorbs the rounding a
% reflection or repair leaves without reaching where a search would place a point
% deliberately, and the same tolerance keeps a clamped coordinate out of
% oob_frac. The two are disjoint -- an escaped coordinate is only in oob_frac --
% so a port that clamps drives 38 up and 39 to 0, and one that snapshots before
% repairing does the reverse; this is what the per-dimension min/max layers
% cannot say, since they report only whether SOME individual reached a bound. A
% NaN coordinate is in neither and is left to x_nonfinite_frac (37); an infinite
% one is outside. A degenerate axis (lb == ub) collapses to z = 0 and so reads as
% fully on the bound.
%
% Elite columns 43-44 describe the best ceil(0.2*N) individuals, at least 2. The
% cut is taken by VALUE, so everyone tied with the cut individual joins the elite
% and the set never depends on how sort breaks ties -- which matters, because a
% converging population is nearly all ties. Ranking is by fitness off CEC2020RW
% and feasibility-first on it (feasible by raw objective, infeasible by
% violation), with a non-finite key sorting last.
%
% Derive rather than store. eff_dim (26) is capped at min(dim, N-1) by the rank
% of the covariance, so under LPSR it falls as the population shrinks even at
% constant shape: read eff_dim / min(dim, pop_size - 1) to ask how much spread
% capacity is in use. Every rate-of-change quantity is a difference of stored
% rows and belongs in the analysis layer -- differentiate only across rows that
% were MEASURED (distinct fe), never across held copies.

    % --- Scalar population metrics: row pop_metrics(t, :) ---------------- %
    L.pop = { ...
        'fe',              ...  1  FE at which this sample was taken (x-axis)
        'pop_size',        ...  2  current population size N (varies under LPSR)
        'f_best',          ...  3  best fitness min(f); -Inf kept, NaN ignored
        'f_worst',         ...  4  worst fitness max(f); +Inf kept, NaN ignored
        'f_mean',          ...  5  mean fitness over the FINITE entries of f
        'f_median',        ...  6  median fitness over the non-NaN entries
        'f_std',           ...  7  fitness std over the finite entries (convergence indicator)
        'f_iqr',           ...  8  fitness IQR over the non-NaN entries (robust spread)
        'div_dap',         ...  9  mean distance to centroid (Morrison-DeJong diversity)
        'div_dim',         ... 10  mean of per-dimension stds (Hussain dimensional diversity)
        'div_var_sum',     ... 11  sum of per-dimension variances (= half the mean pairwise squared distance)
        'radius',          ... 12  max distance to centroid (population radius)
        'bbox_diag',       ... 13  bounding-box diagonal (extent of the spread)
        'centroid_to_best',... 14  distance between centroid and best individual
        'mean_to_best',    ... 15  mean distance of individuals to the best one
        'fdc',             ... 16  fitness-distance correlation, Pearson, best individual EXCLUDED
        'f_skew',          ... 17  skewness of the fitness distribution
        'f_kurt',          ... 18  excess kurtosis of the fitness distribution (normal = 0)
        'median_to_centroid', ... 19  median distance to centroid (robust diversity)
        'div_mpd',         ... 20  mean pairwise Euclidean distance (Barker-Martin/Olorunda distance-based diversity)
        'diameter',        ... 21  max pairwise distance (population diameter; captures split clusters)
        'div_nnd',         ... 22  mean nearest-neighbour distance (clustering/niche structure; see the floor below)
        'div_mad',         ... 23  median-based dimensional diversity (Morales-Castaneda 2020 Div; basis of XPL%/XPT%)
        'div_ent',         ... 24  mean per-dimension Shannon entropy (= mean of the dim 'ent' layer)
        'uniq_frac',       ... 25  fraction of unique FITNESS values (phenotypic collapse; genotypic is 36)
        'eff_dim',         ... 26  PCA participation ratio (sum(lam))^2/sum(lam.^2) (effective spread dimension)
        'feas_frac',       ... 27  feasible fraction OF THE CURRENT POPULATION (1 off CEC2020RW)
        'viol_mean',       ... 28  mean violation v(x) over the current population (0 off CEC2020RW)
        'viol_max',        ... 29  largest single-constraint violation in the population (0 off CEC2020RW)
        'fdc_rank',        ... 30  fitness-distance correlation, Spearman; robust to CEC's heavy-tailed f
        'nonfinite_frac',  ... 31  fraction of individuals whose FITNESS is NaN or +-Inf
        'box_norm',        ... 32  1 = the box-unit columns really are in box units, 0 = raw units
        'obj_best',        ... 33  best RAW objective in the population; equals f_best off CEC2020RW
        'obj_best_feas',   ... 34  best raw objective among the FEASIBLE individuals; NaN when none are
        'viol_median',     ... 35  median violation v(x) over the population (0 off CEC2020RW)
        'x_uniq_frac',     ... 36  fraction of distinct ROWS of X (genotypic duplication; cf. 25)
        'x_nonfinite_frac',... 37  fraction of individuals carrying a NaN/Inf COORDINATE
        'bound_frac',      ... 38  fraction of COORDINATES sitting on a bound (NaN without bounds)
        'oob_frac',        ... 39  fraction of INDIVIDUALS outside the box (NaN without bounds)
        'nnd_median',      ... 40  median nearest-neighbour distance (same floor as 22)
        'nnd_cv',          ... 41  std/mean of the nearest-neighbour distances (NaN on collapse)
        'pc1_frac',        ... 42  share of the variance on the first principal component (1 = one direction)
        'elite_div_ratio', ... 43  div_dap of the elite about ITS OWN centroid, over div_dap
        'elite_centroid_shift' ... 44  distance from the elite centroid to the population centroid
    };

    % --- Per-dimension metrics: 3rd-dimension layers of dim_metrics(t,j,k) %
    % All layers are in RAW problem units so they line up with lb/ub, and are
    % returned as double. ON DISK save_run splits them by precision (min, max and
    % x_best double, the other four single, worth 30 % of the largest file a run
    % writes); tools/load_dim_metrics.m reassembles the canonical double array.
    %
    % This is where storage goes: a scalar column is T values per run, a
    % per-dimension layer is T*dim. Anything reducible to a scalar belongs in
    % L.pop -- boundary occupancy, for one, is 38-39 and not a pair of layers.
    %
    % 'ent' is scale-free by construction (bins span the population's own range),
    % so it describes the SHAPE of the cloud and never its spread: a population
    % collapsed to width 1e-9 still scores ~1. Read it next to div_dap / div_dim.
    L.dim = { ...
        'centroid', ...  1  per-dimension mean mean(X,1) (population centre; shift/drift)
        'std',      ...  2  per-dimension standard deviation std(X,0,1) (axis-wise collapse/anisotropy)
        'min',      ...  3  per-dimension population minimum (compare with lb -> sticking to lower bound)
        'max',      ...  4  per-dimension population maximum (compare with ub -> sticking to upper bound)
        'x_best',   ...  5  position of the BEST individual of the current population
        'mad_best', ...  6  mean_i |x_ij - xbest_j| : per-dimension mean distance of individuals to the best
        'ent'       ...  7  per-dimension Shannon entropy, k=10 bins over the population's own range
    };

end

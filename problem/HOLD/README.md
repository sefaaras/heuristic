# HOLD: Held-Out Literature-referenced Design problems

HOLD is a pool of hard real-world optimisation problems, each with published
results from several independent studies. It exists to test an algorithm
**out of sample**: on problems it was never designed or tuned on, against numbers
other authors published, under each reference's own experimental protocol.

The pool has no fixed size; problems are added when they meet the criteria below.

## Rules

1. **Never tune on HOLD.** An algorithm evaluated here must not have had any
   parameter, operator or design choice made after looking at its HOLD results.
   Otherwise the out-of-sample claim is void.
2. **No CEC problems.** A problem that is part of any CEC suite (including
   CEC2011 and CEC2020RW) does not enter HOLD.
3. **Compare by protocol.** A literature number is compared only with a run made
   under the same protocol: the same problem instance and model conventions, the
   same maximum number of function evaluations (FEs), and the same number of runs.
   A problem may therefore carry several protocols, one per group of references.
4. **Entry criterion.** At least three independent 2016-2026 studies that report
   numbers on the problem, with a public or reproducible definition.
5. **Feasibility first.** For constrained problems, only results whose
   feasibility is stated or verified are compared on the objective.

## Layout

| Path | Content |
|---|---|
| `literature/problems.csv` | One row per problem instance: definition, conventions, best-known value |
| `literature/studies.csv` | One row per study: bibliographic data and DOI |
| `literature/results.csv` | One row per published result: who ran which algorithm, under which conditions, and what they found |
| `hold_literature.m` | Reads the three tables, with filters (problem, instance, budget, runs, verdict, year) |
| `hold_par.m` | The case table: one case = one problem instance under one published protocol (budget, runs) |
| `hold_func.m` | Entry point `[f, g, h] = hold_func(x, k)`, the CEC2020RW contract |
| `hold_<problem>.m` | One evaluator per problem |
| `hold_run.m` | Runs algorithms on the cases, keeping only what a literature comparison needs |
| `hold_summary.m` | Per-case statistics of those runs, next to the published results under the same protocol |
| `truss200/` | Ozturk & Kahraman's 200-bar code (File Exchange 180060), unmodified, BSD licence |
| `gtopx/` | GTOPX (Schlueter et al.), GPL-3; build with `mex gtopxmex.cpp` |
| `epanet/` | OWA EPANET 2.2 (`epanet2.dll`, MIT licence) and `hold_epanet.c`, a MEX bridge to it; build with `mex hold_epanet.c` |
| `wdn/` | Network files `HAN.inp` and `BIN.inp`, NOT in the repository (see below) |

Every evaluator was checked against published solutions before use: the
published best designs reproduce their reported objective and feasibility.

### Water network files

The Hanoi and Balerma networks are the Exeter Centre for Water Systems benchmark
files, which carry no licence for redistribution. `hold_run` downloads them into
`wdn/` when they are missing; without internet access, copy them there by hand:

    https://www.exeter.ac.uk/v8media/universityofexeter/emps/research/cws/downloads/data/3-epanet/HAN.inp
    https://www.exeter.ac.uk/v8media/universityofexeter/emps/research/cws/downloads/data/3-epanet/BIN.inp

The evaluator was validated on these exact files, and `hold_run` skips a
network whose file has a different SHA-256:

    HAN.inp  2d3ba8ada3cd566a7a7fa515ee55931f9bd7576fb52fad6094c70cc7bc6b77fe
    BIN.inp  4fcd41b5da07b4e4b77cb90b86e97e1a6dea307a864269030d9e32172054d3af

### Running

The literature reports the final result of each run (best, mean, worst, std,
success rate, FEs to a target), so that is all HOLD keeps:

    hold_run('pacede')                    % every case, its own budget and run count
    hold_run('pacede', 'cases', 20:29)    % a subset; finished runs are skipped
    hold_summary('pacede')                % per-case statistics
    [T, L] = hold_summary('pacede');      % L: published results, same protocols

Each run is one small file, `results/<alg>/hold_best/F<case>/run<k>.mat`: the
reported solution, its objective and feasibility re-evaluated by the problem,
the evaluations spent, and the FEs at which the best feasible objective improved.
The seed is the run number. See `hold_run.m` for every field.

HOLD is also an experiment of the main harness, `hold`, outside the routine
grid: with `all_experiments = {'hold'}` in `main.m` the runs carry the full
population history under `results/<alg>/hold/F<case>/`, and are verified with
`verify_campaign(alg, root, 'experiments', {'hold'})`. That history is not needed
for a literature comparison, and recording it re-evaluates the constraints of
the whole population at each of its 1000 samples.

Case numbers are append-only: a case keeps its number for good, so results
stored under it stay meaningful when the pool grows.

| Case | Problem | Budget x runs | Protocol of |
|---|---|---|---|
| 1-2 | PV double-diode model, R.T.C. France | 50k / 30k x 30 | SEDE, PGJAYA, DECM, IJAYA / RLDE |
| 3-7 | Protein 3D AB: 2KGU, 1AGT, 1CRN, 1GK4 (200k), 2EWH (800k) | x 30 | DE2P, SGDE, DEpfo, cuHjDE-3D, I-ABC |
| 8-9 | CHPED, 48 units | 300k / 1.6M x 50 | none exactly: the budget of IAQT, KOA / IGA-NCM with the runs of HECS / MGA |
| 10-11 | CHPED, 96 units | 300k / 2M x 50 | NDIDE / NDE |
| 12-13 | 200-bar truss, frequency constraints | 290k x 21 / 580k x 20 | Ozturk & Kahraman 2025 / Ghiaskar & Taheri 2026 |
| 14-15 | 1180-bar dome | 20k / 30k x 20 | SHADE / IDE, Chaotic WSA |
| 16-17 | 1410-bar dome | 20k / 30k x 20 | SHADE / IDE, IHGO, Chaotic WSA |
| 18-19 | Cassini1 (ESA GTOP) | 150k x 25 / 50k x 30 | Tang et al. 2022 / Turkoglu et al. 2025 |
| 20-23 | Balerma irrigation network | 2.5M / 2M / 500k / 45.4k x 10 | AEPD-RSDE / CMA-ES-GS, NLP-DE / FDE / FDE |
| 24-29 | Hanoi network | 200k / 100k / 40k / 10k / 5k / 1k x 100 | AEPD-RSDE / FDE / BLP-DE / FDE / FDE / FDE |

## Schema

All files are UTF-8 CSV with a header row and comma separators. Text containing
commas is double-quoted. Numbers are written in ASCII exactly as printed in the
source (no rounding, no padding, no thousands separators, `-` as the minus sign).
An unknown or unreported value is an empty field.

### problems.csv

| Column | Meaning |
|---|---|
| `problem` | Stable identifier, lower case (e.g. `wdn_balerma`) |
| `instance` | Sub-instance where a problem has several (e.g. a protein sequence `1CRN`); empty otherwise |
| `name` | Name as the literature writes it |
| `domain` | structural, water, power, energy, biology, space |
| `D` | Number of decision variables |
| `objective` | What is minimised, with its unit |
| `unit` | Unit of the objective (kg, MEUR, MUSD, USD/h, km/s, RMSE, energy) |
| `best_known` | Best known objective value under the conventions in `conventions` |
| `best_known_study` | Study key that holds it |
| `conventions` | Model choices that change the optimum (mass matrix, Hazen-Williams coefficient, demand, energy form, ...) |
| `status` | `active` (in the pool), `reserve` (numbers collected, not in the pool) |
| `note` | Anything else a user must know |

### studies.csv

| Column | Meaning |
|---|---|
| `study` | Key: first author's surname in lower-case ASCII, year, underscore, the study's own algorithm or a keyword (e.g. `du2023_aepdrsde`) |
| `authors` | Author list as printed (surnames and initials) |
| `year` | Publication year |
| `title` | Title |
| `venue` | Journal or conference, with volume and pages or article number |
| `doi` | DOI without the `https://doi.org/` prefix; empty if none |
| `group` | Research group or affiliation, used to judge independence |
| `note` | Anything else (preprint, erratum, known problems) |

### results.csv

One row per (problem, instance, producing study, algorithm, protocol).

| Column | Meaning |
|---|---|
| `problem`, `instance` | As in `problems.csv` |
| `study` | Key of the study that PRODUCED the result, i.e. ran the algorithm |
| `algorithm` | Algorithm name as printed |
| `role` | `proposed` (the study's own algorithm) or `rerun` (a competitor the study ran itself) |
| `max_fe` | Maximum number of FEs, as an integer |
| `fe_note` | How the budget is stated (population x iterations, convergence stop, not stated, ...) |
| `runs` | Number of independent runs the statistics are over |
| `runs_note` | Caveats (e.g. statistics over feasible runs only, 20 of 25 attempted) |
| `pop` | Population size, if stated |
| `best`, `mean`, `worst`, `std`, `median` | Statistics as printed |
| `success_rate` | Fraction of runs in [0, 1] that reached the target in `success_def` |
| `success_def` | The target that defines success |
| `fe_to_target` | Average FEs to reach the target or the final solution |
| `conditions` | Instance and model conventions this result was produced under |
| `tag` | `R` read in the producing study's own table; `S` read in another study's table; `D` computed by us from published data; `L` read in the authors' run logs |
| `read_in` | Study key(s) where the number was read, `;`-separated (same as `study` when `tag` is `R`) |
| `location` | Table and page in `read_in` |
| `verdict` | `ok` (read in the producing study's own table, no known problem), `caution` (seen only in copies, or a stated doubt), or `do_not_cite` (a measured defect: infeasible, impossible statistics, does not reproduce) |
| `note` | Required when `verdict` is not `ok`: the factual reason |

A competitor result that a study copied from earlier work is recorded under the
study that originally produced it, with `tag` = `S` and the copying study in
`read_in`. When copies of the same result disagree, each version gets its own
row with `verdict` = `caution`.

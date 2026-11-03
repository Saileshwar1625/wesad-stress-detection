# Wrist-only stress detection on WESAD: who does the average hide?

Leave-one-subject-out (LOSO) evaluation of wrist-only stress detection on the
[WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
dataset. Results are reported per person and for three model families, with
feature-group diagnostics for the people the models fail on.

The full write-up is in [REPORT.md](REPORT.md), also available as a paper-style PDF: [docs/technical-report.pdf](docs/technical-report.pdf) (LaTeX source: [docs/technical-report.tex](docs/technical-report.tex)).

## What I found

- At the default threshold the average hides people. Mean balanced accuracy is
  0.81 to 0.85, but every model has at least one subject at or near chance (random
  forest and gradient boosting: S14 at 0.50 and 0.52; logistic regression: S9 at
  0.52 and S15 at 0.56).
- Most of that is the threshold, not the ranking. Per-subject AUROC is high for
  nearly everyone (mean 0.972 random forest, 0.949 gradient boosting, 0.935
  logistic regression; lowest 0.93, 0.74, 0.73). Random forest ranks S14's windows
  almost perfectly (AUROC 0.98) but gives them low stress probabilities, so its
  balanced accuracy at the default threshold is 0.50. This is consistent with
  Akkaya (2026), who argues that calibration, not discrimination, is the bottleneck.
- Some model dependence remains. Logistic regression ranks S9 and S15 worse
  (AUROC 0.73, 0.78) than the tree models, and gradient boosting ranks S14 worse (0.74).
  Cross-model correlations over 15 subjects are fragile (the 0.79 between the tree
  models falls to 0.45 without S14).
- Per-subject normalization is not a reliable fix. Using the whole recording
  raises mean balanced accuracy to about 0.89, but that is not available in deployment
  and it breaks S17. Using only the first 10 to 50% of each person's windows helps
  logistic regression sometimes, does not help the tree models, and creates new
  threshold failures in other subjects.
- Explaining the failures turned out harder than finding them. Several explanations
  I formed from descriptive statistics (for example "S5 failed because they sat
  still") did not survive testing, and others (such as "S14's EDA carries no
  signal") are no longer supported once ranking and threshold are separated. The
  report lists them as retracted.

| Balanced accuracy, LOSO (15 subjects) | Random forest | Gradient boosting | Logistic regression |
|---|---|---|---|
| Mean | 0.846 | 0.833 | 0.806 |
| Std across subjects | 0.121 | 0.109 | 0.147 |
| Worst subject | S14 0.500 | S14 0.524 | S9 0.521 |
| Subjects below 0.7 | S14, S5 | S14 | S15, S2, S9 |

All numbers come from running the code in this repository on the real WESAD
data. Fifteen subjects with about 35 independent windows each is small, so
read everything as descriptive.

## Task

- Binary: stress vs. non-stress (baseline and amusement are non-stress; meditation is excluded).
- Wrist only: Empatica E4 BVP, EDA, temperature and accelerometer. Chest sensors are not used.
- Windows: 60 s, 30 s stride, kept only if at least 90% of the window has one label. 1032 windows (307 stress, 725 non-stress, about 70 per subject).
- Evaluation: leave-one-subject-out, class-balanced models, and balanced accuracy as the main metric. Always guessing non-stress scores about 0.70 plain accuracy, so I report plain accuracy but don't use it to judge anyone.
- Features: 9 hand-engineered features per window: mean heart rate and a crude inter-beat-interval variability proxy from BVP; EDA mean, std, slope; temperature mean and slope; accelerometer magnitude mean and std. The variability proxy is not validated HRV (see Limitations).

## Repository layout

```
wesad-stress-detection/
├── README.md
├── REPORT.md                    # full write-up
├── requirements.txt
├── data/
│   ├── raw/                     # put the unzipped WESAD folder here (not included)
│   └── processed/               # windows.pkl, features.parquet (generated)
├── pipeline/
│   ├── load_wesad.py            # parse S*.pkl, window, label (binary), write windows.pkl
│   ├── features.py              # per-window features -> features.parquet
│   ├── explain.py               # computed per-subject feature comparison (no API call)
│   ├── explain_llm.py           # optional: model-written description of that comparison, with a check that every number appears in the inputs
│   ├── review_llm.py            # prints a saved description next to the data it was given, for hand review
│   ├── segment_order.py         # protocol segment order per subject
│   ├── order_check.py           # stress-minus-non-stress change, grouped by protocol order
│   └── test_*.py                # tests on synthetic fixtures
├── evaluation/
│   ├── train_eval.py            # LOSO for RF / GB / logistic regression / ensemble
│   ├── ablation.py              # feature-group ablations under LOSO
│   ├── auroc_norm.py            # per-subject AUROC, best-threshold gap, per-subject normalization
│   └── test_*.py
└── reports/                     # generated text/CSV outputs
```

## Reproduce

Python 3.10+ and scikit-learn 1.2+ (needed for `class_weight` in `HistGradientBoostingClassifier`).

```bash
pip install -r requirements.txt

# 1. Download WESAD (UCI link above), unzip, and place the WESAD folder inside data/raw/
python pipeline/load_wesad.py            # -> data/processed/windows.pkl, reports/load_report.txt
python pipeline/features.py              # -> data/processed/features.parquet

# 2. Main result: per-subject LOSO for all three families
python evaluation/train_eval.py --models random_forest gradient_boosting logistic_regression

# 3. Diagnostics
python evaluation/ablation.py --models random_forest gradient_boosting logistic_regression
python evaluation/auroc_norm.py                  # ranking vs threshold, normalization variants (a few minutes)
python pipeline/segment_order.py
python pipeline/order_check.py
python pipeline/explain.py --subject S14             # computed statistics only, no API call
python pipeline/explain_llm.py --dry-run              # shows the exact prompt and inputs, no API call
# needs ANTHROPIC_API_KEY and `pip install anthropic`; writes reports/llm_explanations/
python pipeline/explain_llm.py --model claude-sonnet-5-5
python pipeline/review_llm.py > reports/llm_review_printout.txt   # hand-review aid, no API call

# Tests (synthetic data, no download needed)
python pipeline/test_load_wesad.py && python pipeline/test_segment_order.py && python pipeline/test_order_check.py
python pipeline/test_explain_llm.py   # fake client, no API call
python evaluation/test_train_eval.py && python evaluation/test_ablation.py && python evaluation/test_auroc_norm.py
```

On Windows, run the same commands from a terminal in the repository folder.
The `joblib` "physical cores" warning is harmless (`set LOKY_MAX_CPU_COUNT=4` silences it).

## Design decisions worth knowing about

- LOSO, not random splits. Random splits put the same person in train and test and inflate scores.
- Balanced accuracy for "who struggles". A threshold of 0.7 on plain accuracy would have flagged nobody who always guessed the majority class.
- Three model families, not three tree models. Two tree ensembles agreed on S14 and I first read that as "the data is the problem". Adding logistic regression overturned that, so the final claim is about model dependence.
- Rules written before the ablations: headline results use all features; a difference under about 0.10 for one subject is treated as noise. The second rule turned out to be too lenient. It covers sampling noise only, while a subject's score moves by up to about 0.4 when the model or feature set changes (see Limitations and REPORT.md 4.1).
- No tuning to rescue the failing subjects. Fixing features until S14 passes would overfit an evaluation of 15 people.
- The language model never classifies and no result depends on it. `explain.py` computes the statistics; `explain_llm.py` has a model describe them for three subjects (S14, S5, S9). The model name must be given on the command line, the served model, prompt, inputs and hashes are saved, and every number in the output is checked against the inputs. That check covers numbers only, so I read directions and comparisons by hand and found five problems (REPORT.md 5.8). The outputs are labeled as descriptions, not evidence. It was one run, with one sample per subject.

## Limitations (short version; details in REPORT.md)

- 15 subjects, laboratory protocol, about 35 independent windows per person (windows overlap 50%).
- Hand-engineered features; the HRV proxy has implausible magnitudes and is a known weak feature.
- The feature-group analysis for S14 was done after seeing S14 fail (post hoc), and mechanisms are untested.
- The model-written descriptions come from a single run at default sampling, so wording will differ on a rerun, and one reviewer, who also helped write the prompt, did the hand review.
- Single-subject scores move by up to about 0.4 when the model or feature set changes, so individual-subject claims are fragile.
- The ablations and S14 diagnostics use balanced accuracy at the default threshold and were not repeated with AUROC, so some feature-set differences may be threshold effects.
- No calibration metrics or per-person threshold calibration were computed: the data show that scores are shifted, not why.
- One prior public project (see REPORT.md) already does a per-participant LOSO analysis on WESAD wrist data and also finds S14 failing. I don't claim that finding as new.

## Related work

See REPORT.md for the full list and for what was and was not read. The key
comparison points are the original WESAD paper (Schmidt et al., ICMI 2018,
which reports LOSO results with five classifiers) and a public repository,
`LucasWard22/wesad-wrist-stress-classification`, which analyzes WESAD wrist
data per participant. Akkaya (2026, BMC Medical Informatics and Decision
Making, article in press) evaluates WESAD wrist data with per-subject AUROC and
per-subject normalization and reports high discrimination for every subject;
in line with that, most near-chance subjects here turn out to be threshold
failures rather than ranking failures (REPORT.md, section 5.7). The setups
differ (features, windowing, normalization), so this is consistent with her
argument and not a replication.

## Process note

I wrote this project with help from an AI assistant (Claude), which I used for
code, analysis and drafting the text. The results come from running the code
on the real WESAD data. Several of my early conclusions were revised or
retracted when later runs contradicted them, and REPORT.md documents that.

## Data

WESAD is not redistributed here. Download it from the UCI repository and
follow its terms. Cite: Schmidt, P., Reiss, A., Duerichen, R., Marberger, C.,
Van Laerhoven, K. *Introducing WESAD, a Multimodal Dataset for Wearable
Stress and Affect Detection.* ICMI 2018.

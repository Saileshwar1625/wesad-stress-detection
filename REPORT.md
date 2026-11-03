# Wrist-only stress detection on WESAD: does the average hide who the detector fails on?

This is a descriptive study of 15 subjects. Every number comes from running the code in this repository on the real WESAD data. Choices I made after seeing results are marked post hoc, and section 4.1 lists them all. Anything I did not run is marked not run.

## 1. Summary

Wrist-worn stress detectors are usually judged by an average across people. I wanted to know whether that average hides individuals the detector fails on, and whether those failures come from the person's data or from the model.

- With the default decision threshold, leave-one-subject-out (LOSO) mean balanced accuracy is 0.81 to 0.85 across three model families, and every model has at least one subject at or near chance.
- Most of that failure is the threshold, not the ranking. Per-subject AUROC is high for nearly everyone (means 0.972 for random forest, 0.949 for gradient boosting and 0.935 for logistic regression; lowest values 0.93, 0.74 and 0.73). Random forest ranks S14's windows almost perfectly (AUROC 0.98) but gives them low stress probabilities (mean 0.07, against a true stress share of 0.30), so its balanced accuracy at the default threshold is 0.50. A best per-person threshold would give 0.95, but that uses the subject's own labels, so it is only an upper bound. This is consistent with Akkaya (2026), who reports high WESAD discrimination alongside poor calibration. Her features, windows and normalization differ from mine, so it is not a replication.
- Some model dependence is left at the ranking level. Logistic regression ranks S9 and S15 worse (AUROC 0.73 and 0.78) than the tree models (0.95 to 0.99), and gradient boosting ranks S14 worse (0.74) than random forest and logistic regression (0.98 and 0.99). Cross-model correlations over 15 subjects are fragile and depend heavily on S14 (5.2).
- Per-subject normalization is not a reliable fix. Normalizing by each person's whole recording raises mean balanced accuracy to about 0.89 for all three models, but it can't be done in deployment and it breaks S17 (AUROC from about 0.94 to about 0.56). Using only each person's first 10 to 50% of windows helps logistic regression at 50% and 20% but not at 10%, does not help the tree models, and creates new threshold failures in other subjects.
- Some of my earlier explanations for S14 (no usable EDA signal, reversed temperature) came from fixed-threshold scores and have to be read in that light (5.4, 6). Several other explanations I formed from descriptive statistics did not survive testing and are listed as retracted in section 6.
- I also had a language model write short descriptions of three subjects (S14, S5, S9) from the computed statistics, and checked that every number it cites appears in them. The result is a readable restatement, not a finding. It contained a few overstatements and one speculation that my own prompt induced; these are listed in 5.8.

## 2. Background and related work

- WESAD (Schmidt et al., ICMI 2018) has wrist (Empatica E4) and chest (RespiBAN) recordings from 15 participants across baseline, amusement, stress and meditation conditions. The paper uses leave-one-subject-out cross-validation with five classifiers (decision tree, random forest, AdaBoost, LDA, k-NN). For the wrist-only binary stress task its best accuracy is 88.33% (random forest). It reports aggregate means and standard deviations, and notes that subject-specific accuracies range from 82% to 100% for the binary task when LDA is run on all physiological features. [PDF](https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf)
- A public repository, LucasWard22/wesad-wrist-stress-classification, runs a participant-independent (LOSO) comparison of wrist sensor combinations with a majority-class baseline, logistic regression and histogram gradient boosting. It reports mean balanced accuracy of 0.786 for all wrist sensors and 0.801 for EDA plus temperature. Its README describes a participant-level analysis in which S14 had all of its stress windows missed while S13 reached 0.959 balanced accuracy. That is close prior work: same dataset, same evaluation scheme, same subject. I only found it after my analysis was done. [Repository](https://github.com/LucasWard22/wesad-wrist-stress-classification)
- arXiv:2203.09663 proposes a neural network over EDA, BVP and temperature embeddings and reports 94.50% (±5.64%) LOSO accuracy on WESAD with 60 s windows. It mentions the variation in accuracy between subjects but does not analyze which subjects fail. [arXiv](https://arxiv.org/pdf/2203.09663) A second paper (arXiv:2203.09669) reports per-subject balanced accuracies on WESAD-Wrist (mostly above 94% with SVM, two subjects at 83.3% and 87.5%) and argues that personalized models beat population-level ones. From my reading I could not tell whether those per-subject figures are for personalized or population models. [arXiv](https://arxiv.org/pdf/2203.09669)
- CalmSense (public repository urmeo/CalmSense) reports a LOSO benchmark on WESAD (random forest accuracy 0.913 overall, wrist about 0.89 to 0.91). It stresses that subject leakage, motion, dataset shift and calibration inflate typical numbers. It does not break results down per subject. [Repository](https://github.com/urmeo/CalmSense)
- Akkaya (2026), "Calibration, not architecture, limits cross-subject wearable stress detection" (BMC Medical Informatics and Decision Making, article in press; received 16 June 2026, accepted 8 September 2026; DOI 10.1186/s12911-026-03842-1). I read the full PDF. It evaluates three public Empatica E4 datasets (WESAD, Nurse, Exercise-Stress) with leakage-controlled LOSO, 60 s non-overlapping windows, and per-subject median/IQR normalization computed over each held-out subject's whole recording. The main model is a gradient-boosting classifier on handcrafted features (a 1D-CNN, with and without self-supervised pretraining, is run on the Nurse dataset only). On WESAD (stress as the positive class, baseline plus amusement as negative) it reports a per-subject mean AUROC of 0.992 (95% CI 0.979 to 1.000), a per-fold range of 0.927 to 1.000, and a calibration slope of 0.54 despite a low per-subject expected calibration error (0.036). On the two harder datasets some subjects are near chance. Her argument is that calibration, not architecture, limits decision-useful performance, and she studies few-shot per-subject recalibration and decision-curve net benefit. Where her setup differs from mine: she reports AUROC rather than balanced accuracy at a fixed threshold, normalizes per subject, uses 18 WESAD features including BVP-derived HRV, and does not compare model families per subject on WESAD. The text I read does not identify individual WESAD subjects. Her WESAD windows per subject (median 37) and stress prevalence (0.30) are consistent with the roughly 35 independent windows per person and 0.30 prevalence here.

On novelty: a per-participant LOSO analysis of WESAD wrist data, with S14 as a failing subject, already exists in the sources above. Few-shot per-subject recalibration and label efficiency are covered by Akkaya (2026) on other datasets. Numbers across papers are not comparable, because windows, labels, features, normalization and metrics differ. My literature check was limited (four searches, five pages read through automated summaries, and the BMC paper read in full from the PDF), so it cannot show that something has not been done. What this repository offers is the combination: one transparent pipeline, three model families compared per person, a ranking-versus-threshold breakdown of per-person failures with a per-subject normalization check, and feature-group diagnostics for the failing subject, including the conclusions I had to revise.

Akkaya's paper also suggested an alternative explanation for my results, which I test in 5.7. She reports per-subject WESAD AUROC of at least 0.927 for every subject, while I find subjects at chance on balanced accuracy at a fixed threshold. AUROC does not change under any monotone rescaling of the scores, so a subject can rank their own windows well and still be classified at chance if their scores are shifted. Section 5.7 computes per-subject AUROC for the same LOSO predictions. With raw features, random forest's lowest per-subject AUROC is 0.928 (hers is 0.927) and both of its flagged subjects are score-shift failures. Gradient boosting and logistic regression have lower floors (0.74 and 0.73). Her features, windowing, normalization and model all differ from mine, so this is consistent with her argument, not a replication of it.

## 3. Data and task

- Data: 15 subjects (IDs S2 to S17; S12 is absent from the release). Wrist signals only: BVP (64 Hz), accelerometer (32 Hz, 3 axes), EDA (4 Hz) and temperature (4 Hz). I inspected the real files before the analysis. BVP, EDA and temperature are stored as `(n, 1)` arrays, which the loader flattens. Labels are sampled at 700 Hz and all signals span the same duration.
- Task: binary stress vs. non-stress. Baseline and amusement count as non-stress. Meditation is excluded because it is a distinct state, not a resting baseline.
- Windows: 60 s with a 30 s stride (50% overlap), kept only if one allowed label covers at least 90% of the window. This gives 1032 windows: 307 stress and 725 non-stress (about 30% stress), 66 to 72 per subject. The class mix is almost identical across subjects (stress 19 to 23, non-stress 47 to 49) because every subject followed the same timed protocol.
- Protocol order: there are two orders. Eight subjects (S10, S13, S15, S17, S4, S5, S7, S8) did baseline, amusement, meditation, stress, meditation. Seven (S11, S14, S16, S2, S3, S6, S9) did baseline, stress, meditation, amusement, meditation.

## 4. Methods

- Features (9 per window): mean heart rate from BVP (smoothed peak detector); a crude inter-beat-interval standard deviation, which I call the "HRV proxy"; EDA mean, standard deviation and slope; temperature mean and slope; accelerometer magnitude mean and standard deviation.
- Models, all class-balanced: random forest (200 trees); histogram gradient boosting (200 iterations, learning rate 0.05, depth 3); logistic regression with standardization. No hyperparameter tuning. I added an average-probability ensemble of the three after the fact (post hoc, see 5.6).
- Evaluation: LOSO. For each subject I train on the other 14 and test on the held-out one. Per held-out subject I record plain accuracy, balanced accuracy (the mean of recall on each class, so chance is 0.5 whatever the class balance), stress F1 and the majority-class share. A subject "struggles" when balanced accuracy is below 0.7. That cutoff is a judgment call, and I checked it at 0.6 and 0.8.
- Why not plain accuracy: about 70% of windows are non-stress, so always predicting non-stress scores about 0.70 plain accuracy while detecting nothing. An early version of the struggling rule used plain accuracy, and I changed it for this reason.
- Diagnostics: `ablation.py` reruns LOSO with feature groups removed or isolated. `explain.py` compares a subject's class means to the other subjects' per-subject means (z-scores, plus the direction and size of the stress response). `segment_order.py` and `order_check.py` control for protocol order.
- Ranking versus threshold (added after reading Akkaya 2026): `auroc_norm.py` reruns the same LOSO and records, per held-out subject, the AUROC; the balanced accuracy at the default threshold; the best balanced accuracy over all thresholds (an oracle, since it uses that subject's labels); and the mean predicted stress probability against the true stress share. It also tries per-subject median/IQR normalization that uses no labels: `full` (the subject's whole recording, which would not be available in deployment) and `earlyN` (the subject's first N% of windows, at least 5, which is usable here because every WESAD recording starts at rest). The status rules were fixed before the run, along with the normalization variants. A subject is labeled ranking when AUROC is below 0.70, threshold when AUROC is at least 0.85 and the gap to the best threshold is at least 0.15, and mixed otherwise. I wrote the interpretation in 5.7 after the run.
- LLM description step (5.8): `explain_llm.py` sends a model one JSON object per subject (the `explain.py` comparison plus that subject's rows from the `auroc_norm.py` output) and asks for two short paragraphs. The model never classifies and never sees raw signals. The subjects (S14, S5, S9), the model name and the number check were fixed before the run. Each output is labeled as a description, not evidence.
- Rules set before the ablations: headline results use all features; a single-subject difference under about 0.10 is noise (roughly 35 independent windows per person); an effect should appear across models.

### 4.1 Which choices were made before and after seeing results

Fixed in advance, from the original project plan: the dataset, wrist-only signals, binary labels, leave-one-subject-out testing, 60 s windows with a 30 s stride, the nine features, and fixed hyperparameters for random forest and gradient boosting. Declared before the ablations: headline results use all features, a single-subject difference under about 0.10 is noise, and an effect should appear across models. Declared before the AUROC run (5.7): the status cutoffs and the normalization variants. Those cutoffs were still chosen after I had seen the balanced-accuracy results, so they are not blind.

Decided after seeing results (post hoc):

| Choice | What prompted it |
|---|---|
| Balanced accuracy, not plain accuracy, for "struggling" (cutoff 0.7) | Always predicting non-stress scores about 0.70 plain accuracy, so the early rule could not flag a majority-class guesser. The 0.7 level is a judgment call, checked at 0.6 and 0.8 |
| Logistic regression as a third model family | Random forest and gradient boosting both failed S14 and I had read that as a data problem; adding a linear model overturned it (Section 6) |
| Feature-group ablations, the S14 and S5 diagnostics, the protocol-order check | Examined after those subjects failed; mechanisms untested (Section 7) |
| Response ratio in `explain.py` (weak-response cutoff 0.25, a judgment call) | The direction-only check counted S14's EDA as matching the population at about 2% of the typical response |
| Soft-voting ensemble | Added after the complementary failure pattern across families was seen (5.6) |
| AUROC, best-threshold and normalization analysis (5.7) | Prompted by reading Akkaya (2026) after the first results |
| Spearman correlations | Added after the first "rank" correlations turned out to be Pearson (5.2) |
| Subjects for the LLM description step (S14, S5, S9) | Chosen before the run, but because they were the flagged subjects |

One of the three rules set before the ablations turned out too lenient: the second, that a single-subject difference under about 0.10 is noise. It only covers sampling noise (about 35 independent windows per person). A subject's balanced accuracy moves by up to 0.43 when the model changes (S14) and by up to about 0.4 with the feature set, so a difference above 0.10 is not evidence of a real effect. The third rule, that an effect should appear across models, did the work the second one couldn't (5.2, Section 7).

## 5. Results

### 5.1 Averages at the default threshold hide individuals

| Balanced accuracy, LOSO | Random forest | Gradient boosting | Logistic regression |
|---|---|---|---|
| Mean plain accuracy | 0.887 | 0.847 | 0.808 |
| Mean balanced accuracy | 0.846 | 0.833 | 0.806 |
| Std across subjects | 0.121 | 0.109 | 0.147 |
| Mean stress F1 | 0.764 | 0.742 | 0.716 |
| Worst subject | S14 0.500 | S14 0.524 | S9 0.521 |

Per-subject balanced accuracy:

| Subject | RF | GB | LR | Spread (max - min) |
|---|---|---|---|---|
| S10 | 0.911 | 0.776 | 0.875 | 0.136 |
| S11 | 0.915 | 0.878 | 0.918 | 0.040 |
| S13 | 0.857 | 0.990 | 0.714 | 0.275 |
| **S14** | **0.500** | **0.524** | 0.925 | **0.425** |
| **S15** | 0.966 | 0.854 | **0.561** | **0.405** |
| S16 | 0.900 | 0.825 | 0.885 | 0.075 |
| S17 | 0.874 | 0.857 | 0.840 | 0.034 |
| **S2** | 0.790 | 0.816 | **0.605** | 0.211 |
| S3 | 0.936 | 0.854 | 0.929 | 0.082 |
| S4 | 0.974 | 1.000 | 0.921 | 0.079 |
| **S5** | **0.690** | 0.740 | 0.744 | 0.054 |
| S6 | 0.873 | 0.823 | 1.000 | 0.177 |
| S7 | 0.888 | 0.827 | 0.847 | 0.061 |
| S8 | 0.810 | 0.857 | 0.810 | 0.047 |
| **S9** | 0.800 | 0.875 | **0.521** | 0.354 |

### 5.2 Who fails depends on the model

- Below 0.7: random forest {S14, S5}; gradient boosting {S14}; logistic regression {S15, S2, S9}. No subject is below 0.7 under all three models, and the same holds at 0.6. At 0.8, S5 is the only subject flagged under all three (0.69, 0.74, 0.74), so S5 is mildly hard everywhere rather than severely failing.
- Cross-model correlation of per-subject balanced accuracy (n = 15, so imprecise). Pearson: random forest vs gradient boosting 0.79; random forest vs logistic regression 0.05; gradient boosting vs logistic regression -0.17. My earlier drafts called these "rank" correlations, which was wrong, because Pearson is not a rank statistic. The Spearman values are 0.45, 0.30 and -0.13. The Pearson 0.79 depends on S14, which both tree models fail: without S14 it is 0.45, and random forest vs logistic regression becomes 0.38. What the data support is that particular subjects (S14, S15, S9) change a lot with the model, not that the models' orderings of people are unrelated.
- A per-person table from a single model would mislead. The sampling-noise rule of about 0.10 doesn't cover this, since changing the model moves a single subject by up to 0.43 (S14). Section 5.7 shows that much of this movement is in the threshold, not the ranking.

### 5.3 Feature-group ablations (balanced accuracy, mean over subjects)

| Feature set | GB | RF | LR |
|---|---|---|---|
| all | 0.833 | 0.846 | 0.806 |
| no accelerometer | 0.836 | 0.841 | not run |
| no HRV proxy | 0.843 | 0.851 | not run |
| no EDA | 0.799 | 0.778 | 0.756 |
| no temperature | 0.769 | 0.765 | 0.778 |
| no EDA, no temperature | 0.737 | 0.693 | 0.694 |
| EDA only | 0.745 | 0.711 | 0.692 |
| temperature only | 0.691 | 0.672 | 0.748 |
| accelerometer only | 0.696 | 0.648 | 0.639 |
| heart rate only | 0.604 | 0.574 | 0.648 |
| heart rate + HRV proxy | 0.641 | 0.599 | not run |

- All numbers in 5.3 and 5.4 are balanced accuracy at the default threshold. Section 5.7 shows that this metric can be driven by score shifts, so differences between feature sets may be threshold effects. I did not rerun the ablations with AUROC.
- Removing movement costs nothing on average, so the models don't rely on it overall. Movement alone is above chance, partly as a protocol effect, since the stress task involves speaking and arithmetic.
- EDA and temperature each contribute, and combining signal groups beats any single group.
- Dropping the noisy HRV proxy slightly improves the means, within noise. I had flagged that feature as invalid before seeing its score.

### 5.4 Subject S14

| Feature set | GB | RF | LR |
|---|---|---|---|
| all | 0.524 | 0.500 | 0.925 |
| no EDA | 0.473 | 0.595 | 0.888 |
| no temperature | 0.548 | 0.667 | 1.000 |
| no EDA, no temperature | 0.718 | 0.758 | 0.918 |
| EDA only | 0.500 | 0.500 | 0.500 |
| temperature only | 0.143 | 0.126 | 0.248 |
| accelerometer only | 0.776 | 0.745 | 0.765 |
| heart rate only | 0.823 | 0.697 | 0.949 |

- These scores are balanced accuracy at the default threshold, so read them with 5.7. With all features, random forest ranks S14's windows almost perfectly (AUROC 0.98) but assigns them low stress probabilities (mean 0.07 against a true share of 0.30); a best per-person threshold would reach 0.95. Gradient boosting ranks them worse (AUROC 0.74, best threshold 0.79) and logistic regression ranks them well (AUROC 0.99, balanced accuracy 0.93). So S14's failure under random forest is a score shift, not an inability to separate the classes. Under gradient boosting it is partly both.
- I interpret the fixed-threshold diagnostics cautiously. EDA-only scores exactly 0.5 for S14 in all three families, which is also what constant predictions give, so it does not show that EDA lacks rank information (I did not compute AUROC for the ablations). Temperature-only scores far below chance (0.13 to 0.25), which takes systematically inverted predictions, so the temperature relationship does look reversed for S14. Heart rate alone and movement alone score above chance (0.70 to 0.95 and 0.75 to 0.78).
- The failure is model-dependent. With all features the tree models score S14 at chance and logistic regression does not. Dropping both EDA and temperature lifts both tree models by 0.19 to 0.26 at the default threshold. One reading is that S14's EDA and temperature values push the tree models' scores for all of S14's windows down while heart rate and movement still order them. That fits the random forest AUROC, but I did not test it and it rests on one subject.
- Descriptive context from `explain.py`: S14's heart rate rises 28 bpm under stress versus 6 bpm on average; EDA rises 0.04 versus 2.29; temperature rises 0.78 °C where the average falls 0.63 °C; movement is normal.
- Order check: S14 is in the stress-early protocol group with six others. S16 shows the same temperature rise (+0.77), the same order, a similar heart-rate rise and a normal EDA rise, and scores 0.90, 0.83 and 0.89. S17 has a flat EDA response with typical temperature and also scores well. So the tree models tolerate each deviation alone, and S14 has both. This is an observation about three subjects, not a tested explanation, and it is less well motivated now that random forest's failure on S14 is a score shift rather than a ranking failure (5.7).
- Another public project reports S14 with all stress windows missed under gradient boosting (Section 2), using different features and windowing. That suggests the tree-model failure isn't an artifact of my feature code.

### 5.5 Subject S5

S5 scores 0.69, 0.74 and 0.74 under the three models. S5's movement went down during the stress task while the group's went slightly up (-0.61 versus +0.28 between-person SDs of the class mean, 5.8). That made me guess that the models use movement as a shortcut for S5. The ablations did not support that. Removing movement changed S5 by +0.09 (gradient boosting) and +0.03 (random forest), movement alone detected S5 with gradient boosting (0.82) but not with random forest (0.63), and S5's score swings between 0.52 and 0.93 across small feature-set changes. I found no stable mechanism, and I retract the movement hypothesis.

### 5.6 Post hoc ensemble

I implemented a soft-voting ensemble (the average of the three families' predicted probabilities) after seeing the complementary failure pattern, so it is post hoc and not the headline. Run it with `python evaluation/train_eval.py --models random_forest gradient_boosting logistic_regression ensemble`. The three base models reproduce the earlier numbers.

| LOSO, default threshold | Ensemble | RF | GB | LR |
|---|---|---|---|---|
| Mean balanced accuracy | 0.8575 | 0.8455 | 0.8329 | 0.8064 |
| Mean plain accuracy | 0.8757 | 0.8869 | 0.8468 | 0.8079 |
| Std across subjects (balanced accuracy) | 0.1117 | 0.1210 | 0.1090 | 0.1465 |
| Mean stress F1 | 0.7777 | 0.7635 | 0.7417 | 0.7160 |
| Worst subject | S14 0.571 | S14 0.500 | S14 0.524 | S9 0.521 |

- There is no evidence that the ensemble is better. Its mean balanced accuracy is the highest (+0.012 over random forest, +0.025 over gradient boosting, +0.051 over logistic regression), but paired by subject it beats random forest in 7 subjects, loses in 7 and ties in 1 (Wilcoxon p = 0.53). Against gradient boosting the split is 8 / 4 / 3 (p = 0.17), and against logistic regression 8 / 7 / 0 (p = 0.29). With 15 subjects and the 0.10 noise rule, these differences can't be told apart from zero.
- It rescues the logistic-regression failures but not S14. S15 (0.905), S9 (0.800) and S2 (0.763) are above 0.7, and S5 is 0.740. S14 stays at 0.571 (flagged by random forest, gradient boosting and the ensemble) although logistic regression alone scores 0.925 on S14. The averaged probabilities for S14 follow the tree models. I did not investigate further; it fits the score shift seen in 5.7 but does not test it.
- The ensemble's correlation with its own components (Pearson 0.89 with random forest, 0.84 with gradient boosting, 0.15 with logistic regression) is mechanical and is not independent evidence.
- I did not compute AUROC for the ensemble, though `auroc_norm.py` supports it.

### 5.7 Ranking versus threshold, and per-subject normalization

I added this after reading Akkaya (2026). `evaluation/auroc_norm.py` reruns the same LOSO (same models, features and seeds; the mean balanced accuracy reproduces the earlier run: 0.846, 0.833, 0.806) and adds the quantities described in Section 4.

AUROC / balanced accuracy at the default threshold, raw features (bold: balanced accuracy below 0.7):

| Subject | RF | GB | LR |
|---|---|---|---|
| S10 | 0.98 / 0.91 | 0.96 / 0.78 | 0.95 / 0.88 |
| S11 | 0.96 / 0.92 | 0.96 / 0.88 | 1.00 / 0.92 |
| S13 | 0.97 / 0.86 | 1.00 / 0.99 | 0.94 / 0.71 |
| S14 | **0.98 / 0.50** | **0.74 / 0.52** | 0.99 / 0.93 |
| S15 | 0.98 / 0.97 | 0.95 / 0.85 | **0.78 / 0.56** |
| S16 | 1.00 / 0.90 | 1.00 / 0.82 | 1.00 / 0.89 |
| S17 | 0.94 / 0.87 | 0.95 / 0.86 | 0.93 / 0.84 |
| S2 | 0.97 / 0.79 | 0.96 / 0.82 | **0.94 / 0.61** |
| S3 | 0.98 / 0.94 | 0.95 / 0.85 | 0.98 / 0.93 |
| S4 | 1.00 / 0.97 | 1.00 / 1.00 | 1.00 / 0.92 |
| S5 | **0.93 / 0.69** | 0.92 / 0.74 | 0.92 / 0.74 |
| S6 | 0.93 / 0.87 | 0.89 / 0.82 | 1.00 / 1.00 |
| S7 | 0.98 / 0.89 | 0.99 / 0.83 | 0.92 / 0.85 |
| S8 | 0.99 / 0.81 | 1.00 / 0.86 | 0.94 / 0.81 |
| S9 | 0.99 / 0.80 | 0.97 / 0.88 | **0.73 / 0.52** |

Mean AUROC / mean balanced accuracy / number of subjects below 0.7, by normalization:

| Normalization | RF | GB | LR |
|---|---|---|---|
| none | 0.972 / 0.846 / 2 | 0.949 / 0.833 / 1 | 0.935 / 0.806 / 3 |
| full (whole recording) | 0.955 / 0.887 / 2 | 0.957 / 0.893 / 2 | 0.933 / 0.894 / 1 |
| early50 | 0.977 / 0.838 / 3 | 0.967 / 0.832 / 3 | 0.949 / 0.884 / 1 |
| early20 | 0.960 / 0.817 / 4 | 0.954 / 0.855 / 3 | 0.953 / 0.867 / 1 |
| early10 | 0.949 / 0.801 / 4 | 0.945 / 0.805 / 4 | 0.918 / 0.804 / 4 |

- With raw features, ranking is good for nearly everyone. AUROC is at least 0.9 for all 15 subjects under random forest and for 13 under gradient boosting and logistic regression. The cases below 0.8 are S14 (gradient boosting 0.74), S15 (logistic regression 0.78) and S9 (logistic regression 0.73).
- Flagged subjects, raw features. Random forest: S14 and S5 are threshold failures (AUROC 0.98 and 0.93; a best per-person threshold would give 0.95 and 0.88; mean predicted stress probability 0.07 and 0.21 against true shares of 0.30 and 0.29). Gradient boosting: S14 is mixed (AUROC 0.74). Logistic regression: S2 is a threshold failure (AUROC 0.94, mean probability 0.13), and S15 and S9 are mixed (AUROC 0.78 and 0.73). S15's scores are shifted up (mean probability 0.78) rather than down. No flagged subject has an AUROC below 0.7 with raw features.
- Size of the threshold effect. The mean gap between best-per-person-threshold and default-threshold balanced accuracy is 0.098 (RF), 0.085 (GB) and 0.107 (LR). It is an upper bound, because the threshold is chosen on the test labels with about 35 independent windows. For gradient boosting and logistic regression a subject's balanced accuracy tracks their AUROC (Pearson 0.86 for both). For random forest it doesn't (0.19), because all its AUROCs lie between 0.93 and 1.0, so its balanced accuracy differences come from the threshold. (I computed these two correlations from the 2-decimal table above, n = 15.)
- Cross-model agreement at the ranking level is also weak. The Spearman correlation of per-subject AUROC is 0.63 (RF vs GB), 0.17 (RF vs LR) and 0.05 (GB vs LR); the Pearson values are 0.30, -0.11 and -0.15 (n = 15). Most AUROCs sit between 0.9 and 1.0, so these are fragile.
- Whole-recording normalization (`full`) raises mean balanced accuracy by 0.04 (RF), 0.06 (GB) and 0.09 (LR) with mean AUROC roughly unchanged. It makes S17 a failure under all three models (AUROC 0.94, 0.95, 0.93 raw; 0.56, 0.60, 0.55 normalized) and leaves S14 flagged for random forest and gradient boosting (0.62 and 0.66). I did not investigate why S17 breaks.
- Early-window normalization. Logistic regression improves at 50% and 20% (0.884, 0.867) but not at 10% (0.804). The tree models do not improve (RF 0.838, 0.817, 0.801; GB 0.832, 0.855, 0.805; raw 0.846 and 0.833). The number of flagged subjects rises as the window shrinks (RF 2, 3, 4, 4; GB 1, 3, 3, 4), mostly subjects whose AUROC stays at 0.93 or higher. For example S3 under random forest goes from a balanced accuracy of 0.94 raw to 0.53, 0.50 and 0.50 at 50%, 20% and 10%, with AUROC between 0.93 and 0.99. S14's AUROC is unstable under early normalization (RF 0.98 raw, then 0.96, 0.76, 0.70; GB 0.74 raw, then 0.91, 0.61, 0.56; LR 0.99 raw, then 0.53, 0.92, 0.58).
- Akkaya (2026) reports that statistics from only the first 50%, 20% and 10% of windows keep discrimination (per-subject AUROC 0.986, 0.947, 0.928) but worsen calibration. Here, mean AUROC under early normalization stays between 0.92 and 0.98 while the number of fixed-threshold failures grows. The setups differ, so this is a similar pattern, not a replication.

Caveats. With about 35 independent windows per subject, single-subject differences under about 0.1 are noise, and per-subject AUROC is noisy too (windows overlap 50%). The status cutoffs are judgment calls. I computed no calibration metrics (expected calibration error, calibration slope) and no per-person threshold calibration, so the data show that scores are shifted but not why. I did not test the cause of S14's shift under random forest or of S17's failure under `full`.

### 5.8 Model-written descriptions of S14, S5 and S9 (one run; description, not evidence)

What was run. One call per subject to `claude-sonnet-5-5` (the requested and served names were the same). The installed SDK rejected a `temperature` argument, so the call used the API's default sampling. Wording will differ on a rerun, and I did not measure sampling variance. The input was the computed statistics only. The prompt told the model to use only numbers present in the input, to compute no new numbers, to state no causes, and to say what the statistics cannot show. The verbatim outputs are in Appendix A.

Machine checks. I extracted 58 numbers from the three outputs (20, 19 and 19) and every one was found in the input statistics, exactly or as a correct rounding (0, 0.5 and 1.0 are allowed as reference points). The count is of distinct number tokens per output: a number repeated within one description is counted once, subject IDs such as S14 are ignored, and the study-context values 15 and 35 are included. This only checks that the numbers exist, not that the sentence around them is right. A cause-wording flagger fired twice per output (six hits). On reading, all six are harmless: three are the required disclaimer ("did not test whether they cause"), two belong to the HRV-proxy caveat ("unreliable because it comes from a crude peak detector"), and one is the hedged motion-artifact remark for S5. So the flagger tells us nothing here.

Manual review. The review had two passes. The first used only the text of the descriptions. The second compared each description with a printout of the saved input statistics (`pipeline/review_llm.py`): the model table, each feature's subject change, population change and response ratio, and every sentence listed under the row it refers to. The second pass confirmed every model-level statement (AUROC, balanced accuracy, mean predicted probability against true share, status label) for all three subjects, and every "reversed", "weak" and "larger than typical" label against the stated rules. The quoted ratios match (S14: heart rate 28.081 / 6.061 = 4.63, EDA 0.043 / 2.293 = 0.02, temperature +0.781 versus -0.632; S9: -2.978 / 8.279 = -0.36; the "weak" ratios of 0.17 to 0.21 are below the 0.25 cutoff). It also corrected the first pass (items 1, 3 and 4 below). The printout expresses each change in between-person standard deviations (SDs) of the feature's class mean, recovered from the z-scores in the inputs, so the figures are approximate. A first version of that column divided by the subject's own mean and was discarded. Problems found:

1. A flat response called "reversed". For S14 `eda_slope`, the subject's change is about zero (ratio -0.00) while the population's is 1.66 SDs, so S14 is flat, not reversed. That is still a real difference from the others, but not the one described. S5 `acc_mag_std` is similar (ratio -0.10: -0.13 SDs for the subject against +1.28 for the population, so close to flat). The first pass also listed S14 `temp_slope` here, which was wrong. Its ratio is -0.5 against a population change of 2.18 SDs, so S14 moves +1.09 SDs the other way, a real reversal that only looks like zero in a three-decimal printout.
2. S5, first sentence. "All three models ... are less accurate at the default threshold" states no comparison. It is true relative to each model's best-threshold balanced accuracy (gaps of 0.19 for random forest, 0.12 for gradient boosting, 0.14 for logistic regression). But it reads as contradicting the "ok" labels for gradient boosting and logistic regression in the same paragraph, and "a cutoff that is slightly off" fits those two better than random forest. Ambiguous rather than wrong.
3. The motion-artifact remarks (all three subjects) came from a rule in the prompt ("if accelerometer features are unusual, raise motion artifact"), not from the model. Nothing in this project tested motion artifact, and 5.4 says S14's movement is normal. The population change in `acc_mag_mean` is the smallest of the features whose SD can be recovered in each subject: 0.34 SDs (S14), 0.28 (S5) and 0.25 (S9). The subject-side changes are +0.57 SDs (S14), -0.61 (S5) and -0.24 (S9). None of this is evidence of artifact.
4. Response ratios are unstable when the population change is small. Only S9's `acc_mag_mean` falls below my 0.25-SD flag (printed as 0.25 and flagged). S14's (0.34) and S5's (0.28) sit just above it, so the cutoff doesn't cleanly separate them. S9 "also reversed (-0.98)" is the clearest case: both changes are about a quarter of an SD, and the description gives the ratio without that caveat. The same caution applies to S14's "somewhat larger than typical (1.67)" and S5's "-2.19". The first pass listed the S14 HRV-proxy ratio (-1.63) here and said S5's movement result was consistent with the "barely moved" wording in 5.5. Both were wrong or unsupported. The HRV proxy's population change is 1.15 SDs and S14's is -1.87 SDs, a large opposite change in a feature already known to be unreliable. S5's movement change is -0.61 SDs against a population change of +0.28, so S5's movement went down while the group's went up. An earlier draft of 5.5 said S5 "barely moved", which was wrong, and I corrected it.
5. This is not an independent check. The status labels ("threshold", "mixed", "ok") came from my own rules and were given to the model, which restates them, and the prompt was written in the same session as the review, with AI assistance. A second reader should check the outputs against the JSON files.

What the descriptions do and don't show. The atypical features they list do not predict which model fails. S9's heart-rate response is reversed (-2.978 bpm against +8.279 for the population, ratio -0.36), yet random forest ranks S9 at AUROC 0.988 and gradient boosting at 0.971; only logistic regression struggles (0.726). S14's temperature response is reversed (+0.781 °C against -0.632 °C) and its EDA response is weak (ratio 0.02), yet logistic regression ranks S14 at AUROC 0.987 with balanced accuracy 0.925. This fits the model dependence in 5.2 and tests no mechanism. One untested hypothesis: S14's heart-rate response is 4.63 times typical (3.45 between-person SDs, the largest subject change in the S14 table), and tree models can't extrapolate beyond the training range while a linear model can. The S14 AUROC ablation mentioned under limitations would test it. The step gives a readable restatement and an auditable procedure (pinned model name, saved prompts, inputs, hashes and served model, number check). It adds no result and shouldn't be cited as one. Three subjects, one model and one sample each can't support more.

## 6. Retracted and revised claims

- "The average hides people at chance." True at the default threshold, mostly false at the ranking level. With raw features no subject has an AUROC below 0.7, and random forest's lowest is 0.93 (5.7).
- "S14 is detected well or not at all depending on the model." Overstated. Random forest ranks S14 at AUROC 0.98 and fails only at the threshold (5.4, 5.7).
- "EDA carries no usable signal for S14." I inferred this from a balanced accuracy of exactly 0.5, which constant predictions also produce. I did not test it at the ranking level.
- "Rank correlation" of per-subject scores. The 0.79, 0.05 and -0.17 are Pearson correlations, not rank correlations. The Spearman values differ, and the 0.79 depends on S14 (5.2).
- "S14's failure comes from the data, not the model." Retracted. It rested on two tree models agreeing, and logistic regression handles S14 well.
- "Movement is a shortcut that explains S5." Not supported (5.5).
- "Heart rate alone does not detect S14." This was first indicated by a feature set that also contained the HRV proxy. With heart rate alone S14 is detected (0.70 to 0.95).
- Reporting bug: the first three-model report mislabeled results and hid S14 (flagged by two of three models). Fixed.
- Struggling rule: switched from plain to balanced accuracy.
- Direction check: the early "direction matches the population" test ignored effect size (S14's EDA counted as a match at about 2% of the typical response). I added a response ratio.

## 7. Limitations

- Small and homogeneous: 15 subjects, one laboratory protocol, about 70 windows per subject with 50% overlap (roughly 35 independent). Each stress window is worth about 2.5 points of stress recall. Correlations across 15 subjects are imprecise.
- Single-subject scores are fragile. They move by up to about 0.4 with model or feature-set changes. The 0.10 noise rule I set in advance covers sampling noise only.
- Weak features. The HRV proxy has population means of 350 to 800 ms against real values in the tens of milliseconds. It comes from a simple peak detector and should be read as an artifact-sensitive interval statistic, not HRV. Heart rates above about 150 bpm can't be detected with the smoothing and refractory settings I used.
- Post hoc diagnostics. I examined the feature groups that rescue or fail S14 after S14 failed. The mechanisms (threshold learning vs. linear extrapolation; session drift vs. physiology for temperature) are untested.
- Order and drift. Protocol order interacts with slow signals such as temperature. Group means of the stress-minus-non-stress temperature change differ (-0.79 for the stress-late group, -0.26 for the stress-early group) but not significantly (t about 1.4).
- Thresholds. The 0.7 cutoff for "struggling" is a judgment call; I checked conclusions at 0.6 and 0.8.
- No tuning and no personalization. Results use fixed, modest hyperparameters. I ran per-subject median/IQR normalization as a diagnostic (5.7). I ran no per-person threshold calibration and no calibration metrics.
- Ablations and S14 diagnostics are at the default threshold. I did not repeat them with AUROC, so feature-set differences may be threshold effects.
- The LLM descriptions are not evidence (5.8). I ran them once on three subjects with one model, at the API's default sampling, so the wording is not reproducible. The numbers were machine-checked against the computed statistics. Directions and comparisons were checked by hand by one reviewer, and five problems were found. The cause-wording flagger fires on the prompt's own required disclaimer. No result in this report depends on the LLM output; the descriptive statistics come from `explain.py`.
- Comparisons to the literature are not like-for-like, because windows, label handling, features and metrics differ.

## 8. Discussion

The main finding is about measurement more than physiology, and it has two parts. First, a fixed-threshold per-person table exaggerates failure. With raw features random forest ranks every subject well (AUROC at least 0.93), and its near-chance balanced accuracies are mostly shifted scores. That is the pattern Akkaya (2026) describes when she says calibration, not discrimination, limits use. Second, which subjects fail, and by how much, still moves with the model and with the normalization. S14, S15 and S9 change by 0.35 to 0.43 in balanced accuracy and by 0.20 to 0.26 in AUROC across models, so a single-model per-person table is not a stable description of a person. Per-subject normalization from a short window did not fix this reliably. The natural next step is per-person threshold calibration from a few labeled windows, which Akkaya studies on other datasets. I did not test it here, and since I computed no calibration metrics, the data show score shifts but not their cause.

## 9. Reproducibility

See `README.md`. The data are not redistributed. The tests use synthetic fixtures that mirror the real file schema, including the `(n, 1)` array shapes, and a deliberately atypical synthetic subject whose class signatures are swapped.

## References

- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018. https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf
- LucasWard22. *wesad-wrist-stress-classification* (GitHub repository). https://github.com/LucasWard22/wesad-wrist-stress-classification
- *An Improved Subject-Independent Stress Detection Model* (arXiv:2203.09663). https://arxiv.org/pdf/2203.09663
- *Analysing the Performance of Stress Detection Models on ...* (arXiv:2203.09669). https://arxiv.org/pdf/2203.09669
- urmeo. *CalmSense* (GitHub repository). https://github.com/urmeo/CalmSense
- Akkaya, A. *Calibration, not architecture, limits cross-subject wearable stress detection: a multi-dataset, decision-focused evaluation with label-efficient recalibration.* BMC Medical Informatics and Decision Making (article in press, 2026). https://doi.org/10.1186/s12911-026-03842-1 (read in full from the PDF; supplementary files and code repository not read)


## Appendix A. Model-written descriptions (verbatim)

> DESCRIPTION, NOT EVIDENCE. Generated by a language model from the computed statistics in the JSON file saved next to each description (`reports/llm_explanations/`). No explanation below was tested. Numbers were machine-checked against those statistics; the sentences around them were not. Known problems are listed in 5.8.

**S14**

> For S14, the random_forest ranked stress windows very well (auroc 0.9815) but had a balanced_accuracy of 0.5. Its mean predicted probability was 0.0661 against a true stress share of 0.3, which is consistent with a shifted threshold rather than poor ranking. The gradient_boosting model was weaker and mixed: auroc 0.7376, balanced_accuracy 0.5238, mean predicted probability 0.0785 against the same 0.3 share. This points partly to weaker ranking and partly to low-shifted scores. The logistic_regression model showed no problem, with auroc 0.9874, balanced_accuracy 0.9252 and a mean predicted probability of 0.4395.
>
> Several of S14's feature responses differ from the other participants. Reversed responses were seen for hrv_proxy_std_ms (ratio -1.63, though this feature is unreliable because of the crude peak detector), temp_mean (-1.23), temp_slope (-0.5) and eda_slope (-0.0). Weak responses were seen for eda_mean (0.02) and eda_std (0.05). Heart rate rose far more than typical (hr_mean_bpm response_ratio 4.63), and acc_mag_mean was somewhat larger than typical (1.67), which could reflect motion artifact as a data-quality possibility. These are observations only, and this analysis did not test whether they cause the models' behavior. With 15 participants and about 35 independent windows for S14, these statistics cannot tell us why the models behaved this way or whether the pattern would hold for other people.

**S5**

> For S5, all three models rank stress windows well but are less accurate at the default threshold. random_forest has auroc 0.9276 and balanced_accuracy 0.6898, labelled "threshold". Its mean predicted probability of 0.2129 sits below the true stress share of 0.2899, and its best-threshold balanced accuracy is 0.8842. gradient_boosting has auroc 0.9173 and balanced_accuracy 0.7398, and logistic_regression has auroc 0.9214 and balanced_accuracy 0.7444. Both are labelled "ok", though their mean predicted probabilities (0.1995 and 0.2088) are also below 0.2899. Together this is consistent with somewhat low scores and a cutoff that is slightly off, not with poor ranking.
>
> Compared with the other participants, S5 shows reversed stress responses in three features. hrv_proxy_std_ms has a response_ratio of -0.68, but that feature comes from a crude peak detector and is unreliable. acc_mag_mean has a response_ratio of -2.19 and acc_mag_std has -0.1. Because these are accelerometer features, motion artifact is a possible data-quality explanation. Several responses are much larger than typical: eda_slope (2.75), temp_mean (2.25) and hr_mean_bpm (1.89). No feature is flagged as weak. These are observations only, and this analysis did not test whether they cause the models' behavior. With 15 participants and about 35 independent windows for S5, these statistics cannot establish why the thresholds were off or whether the pattern would repeat.

**S9**

> For S9, random_forest had an auroc of 0.988 and a balanced_accuracy of 0.8, and gradient_boosting had an auroc of 0.9708 and a balanced_accuracy of 0.875. Both rank this participant's windows well and are labeled "ok". Their mean predicted stress probabilities (0.1832 and 0.2165) sit somewhat below the true stress share of 0.2941, which may reflect mildly low scores. logistic_regression had an auroc of 0.726 and a balanced_accuracy of 0.5208, labeled "mixed". Its ranking is weaker and its default-threshold accuracy is near chance, so this is not a pure threshold shift. Its best-threshold balanced_accuracy was 0.7667.
>
> Compared with the other participants, S9's heart rate response is reversed: hr_mean_bpm changed by -2.978 versus a population change of 8.2792 (response_ratio -0.36). acc_mag_mean is also reversed (response_ratio -0.98). Three features show weak responses: eda_std (0.21), eda_slope (0.17) and acc_mag_std (0.21). hrv_proxy_std_ms has a response_ratio of 1.48, but that feature is unreliable because it comes from a crude peak detector. The accelerometer differences may reflect motion artifact as a data-quality issue. These are observations only, and this analysis did not test whether they cause the models' behavior. With 15 participants and about 35 independent windows for this person, these statistics cannot tell us why the models behaved as they did.

# EDA: Parkinson's Disease UPDRS Prediction

Generated from [`data/eda.py`](eda.py) · Interactive reports: [train](eda_train.html) · [test](eda_test.html)

---

## Dataset at a glance

| Split | Rows | Columns |
|---|---|---|
| Training (`X_train` + `y_train`) | 44 590 | 13 (12 features + `target`) |
| Test (`X_test`) | 11 013 | 12 features, no target |

The dataset captures repeated clinical visits for Parkinson's patients from two cohorts (A and B). Each row is one visit for one patient. The task is **regression**: predicting a continuous UPDRS-like composite motor score (`target`).

---

## Per-column findings

| Column | dtype | Missing % (train) | Missing % (test) | n_unique (train) | Notes |
|---|---|---|---|---|---|
| `patient_id` | string | 0 % | 0 % | 5 576 | Repeat-measures ID |
| `cohort` | string | 0 % | 0 % | 2 | A / B |
| `sexM` | int | 0 % | 0 % | 2 | Binary |
| `gene` | string | **32.4 %** | **32.0 %** | 4 | Not all patients genotyped |
| `age_at_diagnosis` | float | **5.2 %** | **4.6 %** | 563 | Quasi-constant per patient |
| `age` | float | 0 % | 0 % | 1 082 | Increases within patient |
| `ledd` | float | **36.6 %** | **38.6 %** | 1 320 | See missingness section |
| `time_since_intake_on` | float | **46.4 %** | **47.8 %** | 64 | Only present for on-state visits |
| `time_since_intake_off` | float | **78.8 %** | **79.1 %** | 178 | Only present for off-state visits |
| `rater_id` | string | 0 % | 0 % | 50 | 50 distinct raters |
| `on` | float | **29.6 %** | **31.2 %** | 83 | UPDRS on-state score; see below |
| `off` | float | **42.4 %** | **40.8 %** | 101 | UPDRS off-state score; see below |
| `target` | float | 0 % (train) | — | 867 | Regression target |

---

## Missingness deep-dive: `on`, `off`, `ledd`

### `on` and `off` — mutually exclusive by visit type

The critical finding: **`on` and `off` are never simultaneously missing**.

| Pattern | Count | % of rows |
|---|---|---|
| Both `on` and `off` present | 12 457 | 27.9 % |
| `on` only (off missing) | 18 913 | 42.4 % |
| `off` only (on missing) | 13 220 | 29.7 % |
| Both missing | **0** | 0 % |

This is structural, not random missingness. Each visit is typed:
- **On-state visit** → `on` is recorded, `off` is absent, `time_since_intake_on` is filled.
- **Off-state visit** → `off` is recorded, `on` is absent, `time_since_intake_off` is filled.
- **~28 % of visits** have both scores, likely at baseline or dual-assessment appointments.

The `time_since_intake_on` / `time_since_intake_off` columns confirm this interpretation — their missingness rates mirror those of `on` / `off` exactly.

### `ledd` — medication dose

`ledd` is missing in **36.6 %** of training rows. Crucially, when both `on` and `off` are simultaneously absent (0 rows), `ledd` is also never absent — the missingness is correlated with visit type rather than random data loss. `ledd` tracks medication load and is a moderately strong predictor of `target` (Pearson r ≈ 0.30). Imputation or encoding as a separate `ledd_missing` indicator is warranted.

---

## Target distribution

- **Mean:** 37.5 · **Median:** 37.3 · **SD:** 16.5
- **IQR:** 25.6 → 49.3 (range 0 – 109.5)
- Distribution is approximately **bell-shaped** (roughly symmetric around 37–38); no severe skew or multimodality.

Histogram bin counts (bins of ~11 UPDRS points):
`[2 477, 5 641, 9 764, 10 703, 8 844, 5 203, 1 757, 180, 13, 8]` — peaks in the 33–44 range.

The target has **no missing values** in the training set. Standard regression metrics (RMSE, MAE) apply without class-imbalance correction.

---

## Visits per patient

| Statistic | Value |
|---|---|
| Unique patients | 5 576 |
| Min visits | 4 |
| Max visits | 12 |
| Mean visits | 8.0 |
| Median visits | 7.0 |

Visit count distribution:

| Visits | Patients |
|---|---|
| 4 | 1 601 |
| 5 | 672 |
| 6 | 322 |
| 7 | 216 |
| 8 | 203 |
| 9 | 145 |
| 10 | 172 |
| 11 | 207 |
| 12 | 2 038 |

Two modes at **4** and **12** visits dominate (~65 % of patients). Most patients have between 4 and 12 longitudinal records. The same patient appears across rows — this is a **repeated-measures / grouped** dataset.

---

## Structure signals

- **No datetime columns** detected by skrub (dates are not stored as date types in the raw CSV).
- **`patient_id`** has the highest unique ratio (0.125), confirming it is a group key, not a predictive feature. It repeats ~8 times on average.
- **`age_at_diagnosis`** is near-constant within a patient (high Pearson correlation with `age`: r = 0.94 — they encode the same underlying time dimension).
- No explicit time index, but `age` increases within patient visits and could serve as a proxy for visit ordering.

---

## Associations

Top associations with `target` (Pearson r, Cramér V):

| Feature | Pearson r | Cramér V | Notes |
|---|---|---|---|
| `off` | **0.87** | 0.41 | Strongest predictor — high linear correlation |
| `on` | **0.67** | 0.37 | Strong predictor — both measure disease severity |
| `ledd` | 0.30 | 0.24 | Moderate — medication dosage tracks severity |
| `age` | 0.31 | 0.11 | Moderate — disease progresses with age |
| `age_at_diagnosis` | 0.13 | 0.05 | Weaker after conditioning on age |
| `time_since_intake_on` | ~0 | 0.18 | Near-zero linear, non-linear structure possible |
| `cohort` | — | 0.09 | Small but non-negligible cohort effect |
| `gene` | — | 0.05 | Minor |
| `sexM` | ~0 | 0.02 | Negligible |
| `rater_id` | — | 0.02 | Negligible |

**Notable inter-feature links:**
- `age_at_diagnosis` ↔ `age`: Pearson r = 0.94 — near-collinear; one may be dropped or the difference used as disease duration.
- `on` ↔ `off`: Pearson r = 0.87 — both are partial measurements of the same construct; their ~28 % co-occurrence rows carry both, the rest carry only one.
- `on` ↔ `off` ↔ `target` triangle: `off` is a slightly stronger predictor of `target` than `on`, consistent with `off` (unmedicated severity) being closer to the composite UPDRS rating.

**Leakage flag**: `off` (r = 0.87) and `on` (r = 0.67) have very high correlations with `target`. This is likely genuine (they measure the same disease dimension at different assessment moments), but worth flagging to the domain expert: if `target` is itself derived from `on` / `off`, there could be information leakage.

---

## Modelling implications

| Finding | Implication |
|---|---|
| `patient_id` repeats 4–12 times | **Use `GroupKFold`** (groups = `patient_id`) to avoid a patient appearing in both train and val folds. Standard `KFold` would leak within-patient information. |
| `on`, `off`, `ledd` have 30–43 % structured missingness | Treat as **visit-type indicators**, not random gaps. Pipeline must handle NaN without leaking visit type. Encoding `ledd_missing`, `on_missing`, `off_missing` as binary features, or using missingness-aware estimators (e.g. `HistGradientBoosting`), is advisable. |
| `off` has Pearson r = 0.87 with `target` | Strong predictor — ensure it is in the feature set. Investigate whether the `target` definition includes `off` to rule out leakage. |
| `age` ↔ `age_at_diagnosis` collinearity (r = 0.94) | Consider engineering `disease_duration = age − age_at_diagnosis` and dropping one. |
| `gene` is missing for 32 % of rows | Encode as a 5-level categorical (`LRRK2+`, `LRRK2-`, `GBA+`, `SNCA`, + "missing"); do not impute a most-frequent value. |
| Regression target, near-symmetric distribution | RMSE is a reasonable primary metric; no need for log-transform or stratification. |
| No detected datetime column | `TimeSeriesSplit` is not obviously warranted unless visit ordering (via `age`) is enforced explicitly. |

---

## Open questions

1. **Is `target` derived from `on` / `off`?** If `target` is a linear combination of `on` and `off`, using them as features would leak the label. Clarification is needed before finalising the feature set.
2. **What does the `cohort` column encode?** The two cohorts (A / B) have a small but detectable effect (Cramér V ≈ 0.09). Understanding their clinical difference could justify separate models or a cohort interaction term.
3. **Are multiple visits per patient independent?** The current data structure treats each visit as a row. If the evaluation metric is per-patient, grouping strategy matters even more.
4. **Visit ordering**: No explicit visit number or timestamp. Can visits be ordered by `age` within a patient reliably, or are ties possible?

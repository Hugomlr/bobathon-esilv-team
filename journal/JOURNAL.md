# JOURNAL: Parkinson's Disease UPDRS Prediction

## Status

- **Goal:** Predict a continuous UPDRS-like composite motor score (`target`) from repeated clinical visit data for Parkinson's patients.
- **Phase:** EDA complete — baseline not yet designed.

## Workspace decisions

| Decision | Value | Date |
|---|---|---|
| Tabular library | pandas | 2025-10-11 |
| Target column | `target` | 2025-10-11 |

---

## Data understanding (EDA)

- **Status:** done - 2025-10-11
- **Summary:** 44 590 training rows across 5 576 patients (4–12 visits each), plus 11 013 test rows. `on`, `off`, and `ledd` carry 30–43 % structured missingness tied to visit type (on-state vs. off-state), never both absent simultaneously. `off` (r = 0.87) and `on` (r = 0.67) are the strongest predictors of the regression target. `patient_id` repeats → `GroupKFold` required.
- **Report:** [data/eda.md](../data/eda.md)

---

## Experiments

*(No experiments yet — baseline design note pending.)*

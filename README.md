# Healthcare Appointment No-Shows: Who Shows Up and Who Doesn't?

An analysis of about 107k medical appointments that compares patients who showed up with those who didn't, ending in a practical **no-show risk-tier recommendation**.

The full analysis, with charts and commentary, is in [`noshow_analysis.ipynb`](noshow_analysis.ipynb).

## Data source & credit

This dataset is courtesy of **[Kaggle](https://www.kaggle.com/)**. It comes from the
*Medical Appointment No Shows* dataset (public appointment records from Vitória, Espírito Santo, Brazil, 2016),
originally published by Joni Hoppen / Aquarela Analytics:
<https://www.kaggle.com/datasets/joniarroba/noshowappointments>.

The file used here, `healthcare_noshows.csv`, is a cleaned version of that data (outcome recoded as `Showed_up`,
with a precomputed `Date.diff` lead-time column). **The CSV isn't included in this repo.** Download it from Kaggle
and place it in the project root as `healthcare_noshows.csv`.

All credit for data collection and publication goes to the original authors and Kaggle; the analysis and conclusions here are our own.

## Key findings

- **Lead time is the strongest signal.** Same-day bookings have about a 5% no-show rate; any gap raises it to about 24–38%.
- **Past behaviour predicts future no-shows.** A prior no-show nearly doubles the odds (adjusted OR about 1.8).
- **Age matters:** teens and young adults miss most; patients over 55 are the most reliable.
- **SMS reminders work** (adjusted OR about 0.78). The raw data suggests the opposite only because reminders go to long-lead appointments, a case of Simpson's paradox.
  Reminders help kids and seniors most, especially on routine, booked-ahead visits.
- **Clustering and PCA underperform** supervised models (AUC about 0.67–0.70 vs 0.72–0.74). The roughly 0.75 AUC ceiling comes from missing data (appointment type, cancellations), not from the method.

## Recommendation: 3-level decision-tree risk tiers

| Tier | Rule (at booking) | No-show rate | Share of no-shows | Action |
|---|---|---|---|---|
| 🟢 Low | Same-day booking (unless 24 or under with 2+ prior no-shows) | ~4% | ~7% | No reminder needed |
| 🟡 Medium | Booked ahead, age 47+, no prior no-shows | ~21% | ~24% | Standard SMS reminder |
| 🔴 High | Booked ahead and age 46 or under; or 47+ with a prior no-show | ~33% | ~69% | SMS + confirm/reschedule or call; overbooking candidate |

## Notebook contents

1–12: data cleaning, exploratory analysis, significance tests, and baseline models.
13: routine vs. need-driven bookings for kids and seniors (SMS effects, repeat missers).
14: full logistic regression (statsmodels, patient-clustered SEs) with odds ratios and predicted rates.
15: segmentation comparison: k-means, PCA, logistic, decision tree, gradient boosting.
16: **final recommendation**: decision-tree risk tiers and action plan.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# put healthcare_noshows.csv (from Kaggle) in this folder, then:
.venv/bin/jupyter lab noshow_analysis.ipynb
```

The notebook is generated from `build_nb.py` and `section13.py`–`section16.py`. To rebuild and re-execute it:

```bash
.venv/bin/python build_nb.py noshow_analysis.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace noshow_analysis.ipynb
```

## Caveats

These are observational data from about 6 weeks in one city (2016), so the results show correlation, not causation. Re-fit the tiers on local, current data before operational use.

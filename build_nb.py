import nbformat as nbf
import sys

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Healthcare Appointment No-Shows: Who Shows Up and Who Doesn't?

> **Data source & credit:** this dataset is courtesy of **[Kaggle](https://www.kaggle.com/)**. It comes from the *Medical Appointment No Shows* dataset (public appointment records from Vitória, Espírito Santo, Brazil, 2016) originally published on Kaggle by Joni Hoppen / Aquarela Analytics: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>. The file used here, `healthcare_noshows.csv`, is a cleaned version of that data (outcome recoded as `Showed_up`, with a precomputed `Date.diff` lead-time column). All credit for data collection and publication goes to the original authors and Kaggle; the analysis and conclusions in this notebook are our own.

This notebook looks at about 107k medical appointments (Vitória, Brazil, Apr–Jun 2016) and compares patients who **showed up** with those who **didn't**.

**Libraries:** pandas, NumPy, matplotlib, seaborn, SciPy (statistical tests), scikit-learn (predictive model).

**Contents**
1. Load and clean the data
2. Overall show rate
3. Demographics: age and gender
4. Health conditions and welfare (Scholarship)
5. Lead time (days between booking and appointment)
6. SMS reminders, and why the raw numbers mislead
7. Day of week
8. Neighbourhood
9. Patient history: repeat no-shows
10. Statistical significance summary
11. Predictive model (logistic regression and random forest)
12. Key takeaways (first pass)
13. Deep dive: routine vs. need-driven bookings for kids and seniors (SMS effects, repeat missers)
14. Full logistic regression with odds ratios and predicted rates
15. Segmentation: clustering vs. PCA vs. supervised models
16. **Final recommendation: 3-level decision tree risk tiers**
""")

code("""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious matmul warnings from macOS Accelerate BLAS
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", "{:.3f}".format)
SHOW_COLORS = {"Showed up": "#2a9d8f", "No-show": "#e76f51"}
""")

md("## 1. Load and clean the data")

code("""
df = pd.read_csv("healthcare_noshows.csv", parse_dates=["ScheduledDay", "AppointmentDay"])
print(df.shape)
df.head()
""")

code("""
df.info()
print("\\nMissing values:", df.isna().sum().sum())
print("Unique patients:", df.PatientId.nunique())
print("Lead time (Date.diff) < 0:", (df["Date.diff"] < 0).sum())
print("Age < 0:", (df.Age < 0).sum(), "| Age > 100:", (df.Age > 100).sum())
""")

md("""
A few rows have a **negative lead time** (appointment dated before it was scheduled), which must be data-entry errors, so we drop them.
Then we add some helper columns: age bands, lead-time bands, day of week, and a readable outcome label.
""")

code("""
df = df[df["Date.diff"] >= 0].copy()
df = df.rename(columns={"Hipertension": "Hypertension", "Handcap": "Handicap", "Date.diff": "LeadDays"})

df["NoShow"] = (~df.Showed_up).astype(int)
df["Outcome"] = np.where(df.Showed_up, "Showed up", "No-show")
df["AgeGroup"] = pd.cut(df.Age, [-1, 12, 17, 25, 35, 45, 55, 65, 75, 120],
                        labels=["0-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"])
df["LeadBand"] = pd.cut(df.LeadDays, [-1, 0, 2, 7, 14, 30, 60, 200],
                        labels=["Same day", "1-2", "3-7", "8-14", "15-30", "31-60", "60+"])
df["Weekday"] = df.AppointmentDay.dt.day_name()
print(df.shape)
""")

md("## 2. Overall show rate")

code("""
counts = df.Outcome.value_counts()
print(counts, "\\n")
print(f"No-show rate: {df.NoShow.mean():.1%}")

fig, ax = plt.subplots(figsize=(5, 3))
counts.plot.barh(ax=ax, color=[SHOW_COLORS[k] for k in counts.index])
ax.set_xlabel("Appointments"); ax.set_ylabel("")
ax.set_title("About 1 in 5 appointments is missed")
plt.tight_layout(); plt.show()
""")

md("""
We'll compare every group to this **~20% baseline no-show rate**. A small helper builds the rate table and chart for each factor.
""")

code("""
BASE = df.NoShow.mean()

def rate_table(col, data=df):
    t = (data.groupby(col, observed=True)
             .agg(appointments=("NoShow", "size"), no_show_rate=("NoShow", "mean")))
    t["vs_baseline_pp"] = (t.no_show_rate - BASE) * 100
    return t

def plot_rate(col, ax=None, data=df, title=None, horizontal=False, order=None):
    t = rate_table(col, data)
    if order is not None:
        t = t.loc[order]
    ax = ax or plt.gca()
    kind = "barh" if horizontal else "bar"
    getattr(t.no_show_rate.plot, kind)(ax=ax, color="#e76f51")
    line = ax.axvline if horizontal else ax.axhline
    line(BASE, ls="--", color="gray", lw=1, label=f"Overall {BASE:.1%}")
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.xaxis if horizontal else ax.yaxis).set_major_formatter(fmt)
    ax.set_title(title or f"No-show rate by {col}")
    ax.set_xlabel("" if not horizontal else "No-show rate")
    ax.set_ylabel("No-show rate" if not horizontal else "")
    ax.legend(loc="best", fontsize=8)
    return t
""")

md("## 3. Demographics: age and gender")

code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.kdeplot(data=df, x="Age", hue="Outcome", common_norm=False, fill=True,
            palette=SHOW_COLORS, ax=axes[0])
axes[0].set_title("Age distribution by outcome")

plot_rate("AgeGroup", ax=axes[1], title="No-show rate by age group")
axes[1].tick_params(axis="x", rotation=45)

plot_rate("Gender", ax=axes[2], title="No-show rate by gender")
plt.tight_layout(); plt.show()

print(df.groupby("Outcome").Age.describe())
display(rate_table("AgeGroup"))
display(rate_table("Gender"))
""")

md("""
**What we see:**
- Patients who show up are **older** on average. No-shows peak among **teens and young adults (13–25)** and drop steadily after 45. Patients over 65 have the lowest no-show rates.
- **Gender makes almost no difference.** Women book about twice as many appointments, but men and women miss them at nearly the same rate.
""")

md("## 4. Health conditions and welfare (Scholarship)")

code("""
flags = ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap"]
rows = []
for f in flags:
    t = df.groupby(f).NoShow.agg(["mean", "size"])
    rows.append({"factor": f,
                 "no_show_if_TRUE": t.loc[True, "mean"], "n_TRUE": t.loc[True, "size"],
                 "no_show_if_FALSE": t.loc[False, "mean"],
                 "diff_pp": (t.loc[True, "mean"] - t.loc[False, "mean"]) * 100})
flag_tbl = pd.DataFrame(rows).set_index("factor")
display(flag_tbl)

long = flag_tbl[["no_show_if_TRUE", "no_show_if_FALSE"]].rename(
    columns={"no_show_if_TRUE": "Has flag", "no_show_if_FALSE": "No flag"})
ax = long.plot.bar(figsize=(9, 4), color=["#e76f51", "#adb5bd"], rot=0)
ax.axhline(BASE, ls="--", color="gray", lw=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("No-show rate: patients with vs without each condition")
ax.set_ylabel("No-show rate"); plt.tight_layout(); plt.show()
""")

md("""
**What we see:**
- **Scholarship** (enrolment in Bolsa Família, Brazil's welfare program, so a rough proxy for low income) goes with a **higher** no-show rate.
- **Hypertension and diabetes** go with **lower** no-show rates. These are chronic conditions with regular check-ups, and the patients are older. Age is likely a big part of the story (the model in section 11 controls for it).
- **Alcoholism** and **handicap** make little difference overall.
""")

code("""
# Does the chronic-condition effect survive within age groups?
chronic = (df.assign(Chronic=df.Hypertension | df.Diabetes)
             .pivot_table(index="AgeGroup", columns="Chronic", values="NoShow", aggfunc="mean", observed=True))
chronic.columns = ["No chronic condition", "Hypertension/Diabetes"]
display(chronic.style.format("{:.1%}"))
""")

md("""
**Once age is held constant, the chronic-condition effect disappears.** Within most age bands, patients with hypertension or diabetes miss appointments at about the same rate as everyone else, and sometimes slightly more often. The apparent "chronic patients are reliable" effect is really an **age effect**: chronic patients are older, and older patients show up.
""")

md("## 5. Lead time: days between booking and appointment")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
t = plot_rate("LeadBand", ax=axes[0], title="No-show rate by lead time (days)")
axes[0].tick_params(axis="x", rotation=0)

sns.boxplot(data=df[df.LeadDays > 0], x="Outcome", y="LeadDays", hue="Outcome",
            palette=SHOW_COLORS, ax=axes[1], showfliers=False, legend=False)
axes[1].set_title("Lead time by outcome (excluding same-day bookings)")
plt.tight_layout(); plt.show()

display(t)
print(df.groupby("Outcome").LeadDays.describe())
""")

md("""
**This is the strongest single signal in the data.**
- **Same-day** appointments are almost always kept (~5% no-show).
- Once there's any gap, the no-show rate jumps to about 23%, then climbs to about 34% for waits of 1–2 months.
- The median lead time for no-shows is several times that of patients who showed up.
""")

md("## 6. SMS reminders, and why the raw numbers mislead")

code("""
display(rate_table("SMS_received"))
""")

md("""
On the raw numbers, patients who got an SMS reminder were **more likely** to miss their appointment. That sounds backwards, but it's **confounded by lead time**: reminders only go out for appointments booked several days ahead, and those appointments already have high no-show rates. (No same-day appointment gets an SMS.)

The fair test is to compare SMS vs no SMS **within the same lead-time band**:
""")

code("""
print(pd.crosstab(df.LeadBand, df.SMS_received), "\\n")

sms = df[df.LeadDays >= 3].pivot_table(index="LeadBand", columns="SMS_received",
                                       values="NoShow", aggfunc="mean", observed=True)
sms.columns = ["No SMS", "SMS received"]
sms["SMS effect (pp)"] = (sms["SMS received"] - sms["No SMS"]) * 100
display(sms.style.format({"No SMS": "{:.1%}", "SMS received": "{:.1%}", "SMS effect (pp)": "{:+.1f}"}))

ax = sms[["No SMS", "SMS received"]].plot.bar(figsize=(9, 4), color=["#adb5bd", "#2a9d8f"], rot=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Within the same lead time, SMS reminders reduce no-shows")
ax.set_ylabel("No-show rate"); ax.set_xlabel("Lead time (days)")
plt.tight_layout(); plt.show()
""")

md("""
**This is a textbook Simpson's paradox.** Within each lead-time band, patients who got a reminder have a **lower** no-show rate than those who didn't. The reminders seem to work. They just get sent to the riskiest appointments.
""")

md("## 7. Day of week")

code("""
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
order = [d for d in order if d in df.Weekday.unique()]
plt.figure(figsize=(8, 3.5))
t = plot_rate("Weekday", order=order, title="No-show rate by appointment weekday")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
display(t.loc[order])
""")

md("""
Weekday differences are **small** (within a point or two of baseline). Saturday has very few appointments, so its rate is noisy.
""")

md("## 8. Neighbourhood")

code("""
nb = rate_table("Neighbourhood")
nb = nb[nb.appointments >= 500].sort_values("no_show_rate")   # ignore tiny neighbourhoods
print(f"{len(nb)} neighbourhoods with >= 500 appointments; "
      f"no-show rate ranges {nb.no_show_rate.min():.1%} to {nb.no_show_rate.max():.1%}")

fig, ax = plt.subplots(figsize=(8, 5))
sel = pd.concat([nb.head(10), nb.tail(10)])
ax.barh(sel.index, sel.no_show_rate,
        color=["#2a9d8f"] * 10 + ["#e76f51"] * 10)
ax.axvline(BASE, ls="--", color="gray", lw=1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("10 best and 10 worst neighbourhoods (>= 500 appointments)")
plt.tight_layout(); plt.show()
""")

md("""
Neighbourhood shows a **moderate spread** of about 13 percentage points (about 16% to 29%) between the best and worst areas with a meaningful number of appointments. This probably reflects access, transport, and socioeconomic factors that the dataset doesn't capture directly.
""")

md("## 9. Patient history: repeat no-shows")

code("""
# For each appointment, count the patient's *previous* appointments and previous no-shows
d = df.sort_values(["PatientId", "AppointmentDay", "AppointmentID"]).copy()
g = d.groupby("PatientId")
d["PrevAppts"] = g.cumcount()
d["PrevNoShows"] = g.NoShow.cumsum() - d.NoShow

d["History"] = np.select(
    [d.PrevAppts == 0, d.PrevNoShows == 0, d.PrevNoShows >= 1],
    ["First visit", "Previously always showed", "Has missed before"], default="")

hist = rate_table("History", d).loc[["Previously always showed", "First visit", "Has missed before"]]
display(hist)

plt.figure(figsize=(7, 3))
plot_rate("History", data=d, horizontal=True,
          order=["Previously always showed", "First visit", "Has missed before"],
          title="Past behaviour predicts future no-shows")
plt.tight_layout(); plt.show()

df = df.join(d[["PrevAppts", "PrevNoShows"]])
""")

md("""
**Past behaviour is a strong predictor.** Patients who have **missed an appointment before** are far more likely to miss again. Patients with a record of showing up are the most reliable group.
(The dataset covers only about 6 weeks of appointments, so "history" here is short, but the signal is still clear.)
""")

md("""
## 10. Statistical significance summary

A chi-square test of independence for each categorical factor against the outcome, plus **Cramér's V** for effect size (0 = no association; ~0.1 is small).
With about 107k rows almost everything is "significant", so **effect size is the number that matters**. (SMS_received ranks high here only because of its confounding with lead time; see section 6.)
""")

code("""
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

factors = ["LeadBand", "AgeGroup", "SMS_received", "Scholarship", "Hypertension",
           "Diabetes", "Alcoholism", "Handicap", "Gender", "Weekday", "Neighbourhood"]
res = []
for f in factors:
    chi2, p, v = cramers_v(df[f], df.NoShow)
    res.append({"factor": f, "chi2": chi2, "p_value": p, "cramers_v": v})
sig = pd.DataFrame(res).set_index("factor").sort_values("cramers_v", ascending=False)
display(sig.style.format({"chi2": "{:,.0f}", "p_value": "{:.2e}", "cramers_v": "{:.3f}"})
          .bar(subset="cramers_v", color="#e76f51"))

# Continuous variables: Mann-Whitney U
for col in ["Age", "LeadDays"]:
    a, b = df.loc[df.Showed_up, col], df.loc[~df.Showed_up, col]
    u, p = stats.mannwhitneyu(a, b)
    print(f"{col:9s} median showed={a.median():.0f}  no-show={b.median():.0f}  Mann-Whitney p={p:.1e}")
""")

md("""
## 11. Predictive model

Can we predict no-shows from these features? We fit an interpretable **logistic regression** (odds ratios) and a **random forest** (non-linear benchmark).
`Neighbourhood` is left out to keep the model simple and interpretable.
""")

code("""
features = ["Age", "LeadDays", "SameDay", "Male", "Scholarship", "Hypertension", "Diabetes",
            "Alcoholism", "Handicap", "SMS_received", "PrevAppts", "PrevNoShows"]
X = (df.assign(SameDay=(df.LeadDays == 0), Male=(df.Gender == "M"))[features]).astype(float)
y = df.NoShow

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
logit.fit(X_tr, y_tr)

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced",
                            n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

for name, m in [("Logistic regression", logit), ("Random forest", rf)]:
    print(f"{name:20s} ROC AUC = {roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]):.3f}")

print("\\nRandom forest classification report (threshold 0.5, balanced class weights):")
print(classification_report(y_te, rf.predict(X_te), target_names=["Showed up", "No-show"]))

fig, ax = plt.subplots(figsize=(5, 4))
RocCurveDisplay.from_estimator(logit, X_te, y_te, ax=ax, name="Logistic")
RocCurveDisplay.from_estimator(rf, X_te, y_te, ax=ax, name="Random forest")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("ROC curves (test set)")
plt.tight_layout(); plt.show()
""")

code("""
# Logistic regression: odds ratio per 1 standard deviation (continuous) / for the flag (binary)
coefs = pd.Series(logit[-1].coef_[0], index=features)
odds = np.exp(coefs).sort_values()
colors = ["#2a9d8f" if v < 1 else "#e76f51" for v in odds]

# Random forest: permutation importance on the test set
pi = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=features).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(odds.index, odds.values, color=colors)
axes[0].axvline(1, color="black", lw=1)
axes[0].set_title("Logistic regression odds ratios (per 1 SD)\\n> 1 raises no-show risk, < 1 lowers it")
axes[1].barh(imp.index, imp.values, color="#264653")
axes[1].set_title("Random forest permutation importance (drop in AUC)")
plt.tight_layout(); plt.show()

display(pd.DataFrame({"odds_ratio": np.exp(coefs), "rf_importance": pd.Series(pi.importances_mean, index=features)})
          .sort_values("rf_importance", ascending=False))
""")

md("""
**Model takeaways:**
- The model ranks no-show risk **moderately well** (AUC about 0.7–0.75). That's useful for prioritising reminders, but far from perfect. Much of why people miss appointments isn't in this data.
- **Lead time / same-day booking** and **prior no-shows** dominate. Age is next.
- Once lead time is controlled for, **SMS reminders reduce** the odds of a no-show (odds ratio < 1), which confirms the stratified analysis in section 6.
- Gender, alcoholism and handicap add almost nothing.
""")

md("""
## 12. Key takeaways: showed up vs. no-show

| Factor | Patients who **showed up** tend to... | Patients who **didn't** tend to... | Strength |
|---|---|---|---|
| **Lead time** | book same-day or within a few days | have waited 1+ weeks (and longer waits are worse) | **Strongest** |
| **History** | have a record of attending | have missed an appointment before | **Strong** |
| **Age** | be older (median ~38); 55+ most reliable | be younger; teens and young adults (13–25) miss most | Moderate |
| **SMS reminder** | have received one (within the same lead time) | have *not* received one | Moderate (hidden by confounding) |
| **Welfare (Scholarship)** | not be enrolled | be enrolled in Bolsa Família (lower income) | Small |
| **Chronic conditions** | have hypertension or diabetes (raw) | not have them | None once age is controlled for |
| **Neighbourhood** | live in certain areas | live in other areas (about 13 pp spread) | Small–moderate |
| **Gender, weekday, alcoholism, handicap** | no meaningful difference | no meaningful difference | Negligible |

**Practical implications**
1. **Shorten the wait.** Every extra day between booking and appointment raises no-show risk. Same-day and next-few-day slots are the most reliable.
2. **Target reminders** at long-lead appointments, younger patients, and anyone with a prior no-show. The data suggests reminders help once confounding is removed.
3. **Use history.** A patient's past attendance is one of the cheapest and most predictive signals for overbooking or confirmation calls.

**Caveats:** this is observational data (correlation, not causation). The data covers a short window (~6 weeks) in one city, and important drivers like transport, appointment type, and weather aren't recorded.
""")

exec(open(__file__.replace("build_nb.py", "section13.py")).read())
exec(open(__file__.replace("build_nb.py", "section14.py")).read())
exec(open(__file__.replace("build_nb.py", "section15.py")).read())
exec(open(__file__.replace("build_nb.py", "section16.py")).read())

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])

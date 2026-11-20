md("""
---
# 14. Full logistic regression: what independently predicts a no-show?

Section 11 used scikit-learn for prediction. Here we fit an **inferential** logistic regression with `statsmodels` on all ~107k appointments:
- **Outcome:** `NoShow` (1 = missed). An **odds ratio (OR) > 1 raises** the odds of missing; **< 1 lowers** them.
- **Each OR is adjusted** for every other variable in the model.
- **Standard errors are clustered by patient**, because many patients have several appointments.
- **Reference levels:** same-day booking, first visit, age 36–45, female, Wednesday.
""")

code("""
m14 = d13.copy()
m14["AgeBand"] = pd.cut(m14.Age, [-1, 12, 17, 25, 35, 45, 55, 65, 200],
                        labels=["0-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
m14["Lead"] = pd.cut(m14.LeadDays, [-1, 0, 2, 7, 14, 30, 60, 200],
                     labels=["Same day", "1-2d", "3-7d", "8-14d", "15-30d", "31-60d", "60d+"])
m14["Hist"] = np.select([m14.PrevAppts == 0, m14.MissedBefore == 1],
                        ["First visit", "Missed before"], default="Always showed")
for c in ["SMS_received", "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap"]:
    m14[c] = m14[c].astype(int)
m14["pid"] = pd.factorize(m14.PatientId)[0]

formula = ("NoShow ~ C(Lead, Treatment('Same day')) + SMS_received + C(Hist, Treatment('First visit'))"
           " + C(AgeBand, Treatment('36-45')) + Scholarship + Hypertension + Diabetes + Alcoholism + Handicap"
           " + C(Gender) + C(Weekday, Treatment('Wednesday'))")
full = smf.logit(formula, data=m14).fit(disp=0, cov_type="cluster", cov_kwds={"groups": m14.pid})

print(f"n = {int(full.nobs):,}   pseudo R² = {full.prsquared:.3f}   "
      f"AUC = {roc_auc_score(m14.NoShow, full.predict(m14)):.3f}")
full.summary2().tables[1].iloc[:, :4]
""")

code("""
def clean(name):
    for pat, rep in [("C(Lead, Treatment('Same day'))[T.", "Lead time: "),
                     ("C(Hist, Treatment('First visit'))[T.", "History: "),
                     ("C(AgeBand, Treatment('36-45'))[T.", "Age: "),
                     ("C(Weekday, Treatment('Wednesday'))[T.", "Day: "),
                     ("C(Gender)[T.M", "Male"), ("_received", " received")]:
        name = name.replace(pat, rep)
    return name.rstrip("]")

ci = full.conf_int()
or_full = pd.DataFrame({"odds_ratio": np.exp(full.params), "ci_low": np.exp(ci[0]),
                        "ci_high": np.exp(ci[1]), "p_value": full.pvalues}).drop("Intercept")
or_full.index = [clean(i) for i in or_full.index]
display(or_full.style.format({"odds_ratio": "{:.2f}", "ci_low": "{:.2f}", "ci_high": "{:.2f}", "p_value": "{:.1e}"})
               .background_gradient(subset="odds_ratio", cmap="RdYlGn_r", vmin=0.5, vmax=1.5))

plot_df = or_full.drop(index=[i for i in or_full.index if i == "Day: Saturday"])  # 39 appts, CI too wide to show
fig, ax = plt.subplots(figsize=(8, 9))
y = np.arange(len(plot_df))[::-1]
colors = np.where(plot_df.ci_high < 1, "#2a9d8f", np.where(plot_df.ci_low > 1, "#e76f51", "#adb5bd"))
for yi, (_, r), c in zip(y, plot_df.iterrows(), colors):
    ax.errorbar(r.odds_ratio, yi, xerr=[[r.odds_ratio - r.ci_low], [r.ci_high - r.odds_ratio]],
                fmt="none", ecolor=c, capsize=3)
ax.scatter(plot_df.odds_ratio, y, c=colors, zorder=3)
ax.axvline(1, color="black", lw=1, ls="--")
ax.set_xscale("log"); ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 4, 8, 14]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "4", "8", "14"])
ax.set_yticks(y); ax.set_yticklabels(plot_df.index)
ax.set_xlabel("Adjusted odds ratio of no-show (log scale)")
ax.set_title("Adjusted odds ratios (95% CI)\\nred = raises no-show risk, green = lowers it, grey = not significant")
plt.tight_layout(); plt.show()
""")

md("""
Odds ratios can be hard to interpret, so the next cell converts them to **predicted no-show rates**. For each level of a factor, we set every appointment to that level, keep everything else as observed, and average the model's predictions.
""")

code("""
def predicted_rates(col, levels):
    out = {}
    for lv in levels:
        tmp = m14.copy(); tmp[col] = lv
        if col == "Lead":
            tmp["SMS_received"] = 0   # isolate lead time from reminders
        out[lv] = full.predict(tmp).mean()
    return pd.Series(out, name="predicted no-show rate")

for col, levels in [("Lead", ["Same day", "1-2d", "3-7d", "8-14d", "15-30d", "31-60d"]),
                    ("Hist", ["Always showed", "First visit", "Missed before"]),
                    ("AgeBand", ["0-12", "13-17", "18-25", "36-45", "56-65", "66+"]),
                    ("SMS_received", [0, 1]), ("Scholarship", [0, 1])]:
    display(predicted_rates(col, levels).to_frame().T.style.format("{:.1%}").set_caption(col))
""")

md("""
**What the full model says:**

| Factor | Adjusted effect | Takeaway |
|---|---|---|
| **Lead time** | OR about **7 (1–2 days) up to 13 (31–60 days)** vs. same day | By far the biggest factor. Even a 1–2 day gap multiplies the odds of missing about 7x compared with booking the same day. |
| **Missed before** | OR **1.80** vs. first visit; "always showed" 0.87 | A prior no-show nearly doubles the odds, even with lead time and age held constant. |
| **SMS reminder** | OR **0.78** | A reminder cuts the odds of missing by about 22%. This confirms the raw "SMS → more no-shows" pattern was confounding. |
| **Age** | Ages 13–17: **1.55**; 18–25: 1.36; 56–65: 0.67; 66+: **0.64** (vs. 36–45) | Teens and young adults are the riskiest; patients over 55 are the most reliable. |
| **Welfare (Scholarship)** | OR **1.23** | A real but modest effect, likely a proxy for low income or access barriers. |
| **Alcoholism** | OR **1.36** | **Hidden in the raw data** (20% vs. 20%). It shows up once age is controlled for, because people with alcoholism tend to be middle-aged, a lower-risk age group. |
| **Diabetes** | OR **1.16** | **Flips direction.** Raw, people with diabetes look more reliable (18% vs. 20%), but that's because they're older. At the same age, they're slightly *more* likely to miss. |
| **Hypertension** | OR 0.99 (not significant) | No independent effect. The raw "reliability" was entirely age. |
| **Gender** | Male OR 1.07 | Statistically detectable but tiny. |
| **Handicap, weekday** | About 1.0 (Fri/Tue about 1.06) | Negligible. |

**Model fit:** AUC about 0.73 and pseudo R² about 0.12. These factors clearly matter, but they explain only part of why people miss appointments. Unrecorded factors (transport, appointment type, work schedules, weather) carry a lot of weight.
""")

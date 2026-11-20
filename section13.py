# Appended into build_nb.py: section 13 cells (uses md/code helpers defined there)
md("""
---
# 13. Deep dive: routine vs. need-driven bookings for kids and seniors

**Questions**
1. For **kids and seniors who booked ahead**, did SMS reminders help?
2. Among kids and seniors who **missed before**, were their booked-ahead appointments missed again?
3. Does booking **within a week** (a sign of real need) mean people intend to keep the appointment?

**Working hypothesis (from the analyst):** routine, booked-ahead appointments are more fragile. Parents may only realize after a reminder that they can't make it, and seniors may forget or be unable to arrange transport. A booking made within a week reflects real need and therefore intent to attend.

**Proxy for appointment type.** The dataset has **no appointment-type field** (routine / follow-up / urgent), so we use lead time as a stand-in:

| Booking type | Lead time | Interpretation |
|---|---|---|
| Same day | 0 days | Walk-in or urgent; often booked during a visit |
| Within a week | 1–7 days | Need-driven: the patient wants to be seen soon |
| Routine | 8–30 days | Planned check-up or follow-up |
| Long-planned | 31+ days | Scheduled far ahead, e.g. periodic reviews |

**Note on SMS:** reminders were only sent for appointments booked **3+ days** ahead, so all SMS comparisons use lead times of 3 days or more.
""")

code("""
import statsmodels.formula.api as smf

SEG_LABELS = ["Kids (0-12)", "Teens (13-17)", "Adults (18-59)", "Seniors (60+)"]
BOOK_LABELS = ["Same day", "Within a week", "Routine (8-30d)", "Long-planned (31d+)"]

d13 = d.copy()   # appointment-level frame with history columns from section 9
d13 = d13.rename(columns={"Hipertension": "Hypertension", "Date.diff": "LeadDays"})
d13["Segment"] = pd.cut(d13.Age, [-1, 12, 17, 59, 200], labels=SEG_LABELS)
d13["Booking"] = pd.cut(d13.LeadDays, [-1, 0, 7, 30, 200], labels=BOOK_LABELS)
d13["MissedBefore"] = (d13.PrevNoShows > 0).astype(int)

display(pd.crosstab(d13.Segment, d13.Booking, margins=True))
""")

md("## 13.1 Does booking within a week signal intent to keep the appointment?")

code("""
rate = d13.pivot_table(index="Segment", columns="Booking", values="NoShow", aggfunc="mean", observed=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1.1, 1]})
sns.heatmap(rate, annot=True, fmt=".0%", cmap="OrRd", vmin=0, vmax=0.45, cbar=False, ax=axes[0])
axes[0].set_title("No-show rate by age segment and booking type"); axes[0].set_xlabel(""); axes[0].set_ylabel("")

rate.T.plot(marker="o", ax=axes[1])
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].set_title("Lead-time gradient: steepest for kids, flattest for seniors")
axes[1].set_xlabel(""); axes[1].legend(title="")
plt.tight_layout(); plt.show()

# Where do the no-shows come from?
share = d13.groupby("Segment", observed=True)[["LeadDays", "NoShow"]].apply(lambda x: pd.Series({
    "% of appointments booked 8+ days ahead": x.LeadDays.ge(8).mean(),
    "% of no-shows booked 8+ days ahead": x.loc[x.NoShow == 1, "LeadDays"].ge(8).mean()}))
display(share.style.format("{:.0%}"))
""")

md("""
**Reading this:**
- **The strongest intent signal is same-day booking**, not "within a week". Same-day no-show is 3–6% for kids, adults and seniors (12% for teens). Once there's even a 1–7 day gap, the rate jumps to about 19–26%.
- **Within a week is better than routine, but only partly.** Kids go from about 25% (within a week) to 34% (routine) to 40% (long-planned). For seniors the curve is flat: 19% → 23% → 23%.
- **Routine bookings are where kids' no-shows come from.** About 36% of kids' appointments are booked 8+ days ahead, but those produce about **62% of their no-shows**.
- **Seniors are the most reliable group at every lead time.** Booking ahead hurts them much less than it hurts kids.
""")

md("## 13.2 For kids and seniors who booked ahead, did reminders help?")

code("""
sms_base = d13[d13.LeadDays >= 3].copy()
sms_base["Booking3"] = pd.cut(sms_base.LeadDays, [2, 7, 30, 200],
                              labels=["Within a week (3-7d)", "Routine (8-30d)", "Long-planned (31d+)"])

def prop_diff_ci(g):
    a, b = g[g.SMS_received].NoShow, g[~g.SMS_received].NoShow
    p1, p0 = a.mean(), b.mean()
    se = np.sqrt(p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(b))
    diff = p1 - p0
    return pd.Series({"no_SMS": p0, "SMS": p1, "n_no_SMS": len(b), "n_SMS": len(a),
                      "diff_pp": diff * 100, "ci_low_pp": (diff - 1.96 * se) * 100,
                      "ci_high_pp": (diff + 1.96 * se) * 100})

sms_tbl = sms_base.groupby(["Segment", "Booking3"], observed=True).apply(prop_diff_ci, include_groups=False)
display(sms_tbl.style.format({"no_SMS": "{:.1%}", "SMS": "{:.1%}", "n_no_SMS": "{:,.0f}", "n_SMS": "{:,.0f}",
                              "diff_pp": "{:+.1f}", "ci_low_pp": "{:+.1f}", "ci_high_pp": "{:+.1f}"}))
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, seg in zip(axes, SEG_LABELS):
    t = sms_tbl.loc[seg][["no_SMS", "SMS"]]
    t.columns = ["No SMS", "SMS received"]
    t.plot.bar(ax=ax, color=["#adb5bd", "#2a9d8f"], rot=0, legend=(seg == SEG_LABELS[0]))
    ax.set_title(seg); ax.set_xlabel("")
    ax.set_xticklabels(["3-7d", "8-30d", "31d+"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_ylabel("No-show rate")
fig.suptitle("SMS reminders and no-shows, by age segment and lead time", y=1.02)
plt.tight_layout(); plt.show()
""")

md("""
These raw comparisons don't account for other differences between patients. Below, a **logistic regression** estimates the SMS effect for each age segment while holding constant lead time, prior no-shows, welfare enrolment and gender.
An **odds ratio below 1** means reminders reduce the odds of a no-show.
""")

code("""
model_df = sms_base.assign(Seg=sms_base.Segment.cat.rename_categories(["Kids", "Teens", "Adults", "Seniors"]),
                           SMS=sms_base.SMS_received.astype(int))
logit13 = smf.logit("NoShow ~ SMS * C(Seg, Treatment('Kids')) + np.log1p(LeadDays) + MissedBefore"
                    " + Scholarship + C(Gender)", data=model_df).fit(disp=0)

names = list(logit13.params.index)
rows = []
for seg in ["Kids", "Teens", "Adults", "Seniors"]:
    w = np.zeros(len(names)); w[names.index("SMS")] = 1
    if seg != "Kids":
        w[names.index(f"SMS:C(Seg, Treatment('Kids'))[T.{seg}]")] = 1
    t = logit13.t_test(w)
    lo, hi = np.exp(t.conf_int()[0])
    rows.append({"segment": seg, "SMS odds ratio": float(np.exp(t.effect[0])),
                 "95% CI low": lo, "95% CI high": hi, "p_value": float(t.pvalue)})
or_tbl = pd.DataFrame(rows).set_index("segment")
display(or_tbl.style.format({"SMS odds ratio": "{:.2f}", "95% CI low": "{:.2f}", "95% CI high": "{:.2f}", "p_value": "{:.1e}"}))

fig, ax = plt.subplots(figsize=(6, 3))
y = np.arange(len(or_tbl))[::-1]
ax.errorbar(or_tbl["SMS odds ratio"], y,
            xerr=[or_tbl["SMS odds ratio"] - or_tbl["95% CI low"], or_tbl["95% CI high"] - or_tbl["SMS odds ratio"]],
            fmt="o", color="#2a9d8f", capsize=4)
ax.axvline(1, color="black", lw=1, ls="--")
ax.set_yticks(y); ax.set_yticklabels(or_tbl.index)
ax.set_xlabel("Adjusted odds ratio of no-show with SMS (< 1 = helps)")
ax.set_title("Reminders help every age group, kids and seniors most")
plt.tight_layout(); plt.show()

# Was SMS targeted at riskier patients? SMS coverage by month suggests an operational change instead
print("Share of 3+ day appointments that got an SMS, by appointment month:")
print(sms_base.groupby(sms_base.AppointmentDay.dt.month_name(), sort=False).SMS_received.mean().round(2))
""")

md("""
**Reading this:**
- **Yes, reminders helped kids and seniors who booked ahead, and they're the two groups that benefit most.** After adjustment, an SMS cuts the odds of a no-show by about **27% for kids and 26% for seniors**, compared with about 20% for adults and 17% for teens.
- **The benefit grows with lead time.** For kids, the drop is about 10–12 points on routine and long-planned bookings (e.g. 48% → 36% at 31+ days), but **almost nothing** for within-a-week bookings (25% → 24%, not significant). Seniors get a steadier 3.5–6 point reduction at every lead time.
- **What this says about your hypothesis:**
  - *"Parents only realize after the reminder that they can't go."* If reminders mostly surfaced conflicts, kids with an SMS would miss *more* often. They miss less. So reminders mainly fix **forgetting**, not hidden conflicts. Note that **cancellations aren't recorded** in this dataset, so a reminder that led a parent to cancel early (freeing the slot) is invisible here.
  - For **within-a-week kids' bookings**, reminders don't help because the parent already knows about the appointment. That's consistent with "short lead time = intent".
  - For **seniors**, reminders help even on short bookings. That fits forgetting more than transport problems, since an SMS can't fix transport. Transport barriers would show up as the remaining 18–20% no-show rate that persists *despite* the reminder.
- SMS coverage dropped from about 80% to about 52% in May, which looks operational rather than risk-targeted. Combined with lead times being nearly identical with and without SMS inside each band, this makes the comparison reasonably fair. It is still observational, not a randomized trial.
""")

md("## 13.3 Did kids and seniors who missed before miss their booked-ahead appointments again?")

code("""
d13["History"] = np.select([d13.PrevAppts == 0, d13.MissedBefore == 1],
                           ["First visit", "Missed before"], default="Always showed before")
hist13 = d13[d13.History != "First visit"]

fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
for ax, h in zip(axes, ["Always showed before", "Missed before"]):
    p = hist13[hist13.History == h].pivot_table(index="Segment", columns="Booking", values="NoShow",
                                                aggfunc="mean", observed=True)
    sns.heatmap(p, annot=True, fmt=".0%", cmap="OrRd", vmin=0, vmax=0.55, cbar=False, ax=ax)
    ax.set_title(f"Patients who {h.lower()}"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.show()

display(hist13.pivot_table(index=["Segment", "History"], columns="Booking", values="NoShow",
                           aggfunc=["mean", "size"], observed=True)
              .style.format("{:.1%}", subset="mean").format("{:,.0f}", subset="size"))
""")

code("""
# Repeat missers on routine / long-planned bookings (8+ days): did a reminder help?
rep = d13[(d13.MissedBefore == 1) & (d13.LeadDays >= 8)]
rep_tbl = rep.groupby("Segment", observed=True).apply(prop_diff_ci, include_groups=False)
display(rep_tbl.style.format({"no_SMS": "{:.1%}", "SMS": "{:.1%}", "n_no_SMS": "{:,.0f}", "n_SMS": "{:,.0f}",
                              "diff_pp": "{:+.1f}", "ci_low_pp": "{:+.1f}", "ci_high_pp": "{:+.1f}"}))

ax = rep_tbl[["no_SMS", "SMS"]].rename(columns={"no_SMS": "No SMS", "SMS": "SMS received"}).plot.bar(
    figsize=(8, 3.5), color=["#adb5bd", "#2a9d8f"], rot=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Patients who missed before, booked 8+ days ahead"); ax.set_xlabel(""); ax.set_ylabel("No-show rate")
plt.tight_layout(); plt.show()
""")

md("""
**Reading this:**
- **Yes, past missers miss their booked-ahead appointments again, and the gap widens with lead time.** Kids with a prior miss have about a **45% no-show rate on routine bookings**, vs. about 30% for kids who always showed. For seniors it's about **38% vs. 21%**.
- Even **same-day** appointments for kids with a prior miss have a ~22% no-show rate (vs. ~5% for kids who always showed). The family's track record matters even when they booked on the spot.
- **Reminders matter most for this group.** Among patients who missed before and booked 8+ days ahead, an SMS goes with a no-show rate about **19 points lower for kids (57% → 38%)** and **16 points lower for seniors (48% → 32%)**.
""")

md("""
## 13.4 Summary for this deep dive

| Question | Answer |
|---|---|
| Does booking within a week signal intent? | **Partly.** Same-day booking is the strong intent signal (about 3–6% no-show). Within a week (about 19–25%) beats routine bookings for kids (34–40%), but for seniors the advantage is small. |
| Did reminders help kids and seniors who booked ahead? | **Yes, more than any other age group** (adjusted odds about 27% lower). For kids, the benefit is concentrated in **routine and long-planned bookings**; for short bookings it's close to zero. |
| Did prior missers miss booked-ahead appointments again? | **Yes.** About 45% (kids) and 38% (seniors) on 8–30 day bookings, but a reminder cuts that by 16–19 points. |
| Reminder surfaces "can't go" vs. "forgot"? | The data points to **forgetting**: reminders lower no-shows rather than raise them. Cancellations aren't recorded, so the "realized they can't go" path can't be measured directly. |

**Operational implications**
1. **Kids' routine and long-planned visits** are the highest-yield reminder target, especially for families with a prior miss. Consider a **second reminder** or a confirm/reschedule prompt.
2. **Seniors:** keep reminding at every lead time. For the remaining ~20% who miss despite reminders, test **transport or caregiver outreach**, which the data can't measure but which fits what's left over.
3. **Within-a-week kids' bookings** don't benefit from SMS. Reminder budget is better spent elsewhere.
4. **Add an appointment-type field** (routine / follow-up / urgent) and **record cancellations**. Both would turn this proxy analysis into a direct one.
""")

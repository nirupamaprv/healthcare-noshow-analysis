md("""
---
# 16. Final recommendation: use the 3-level decision tree to target no-show interventions

**Recommendation:** classify every booked appointment into a **Low / Medium / High** no-show risk tier using the 3-level decision tree from section 15.3, and match the intervention to the tier.

**Why this approach**
- **Accurate enough:** test AUC about 0.72, close to the best model (gradient boosting, about 0.74) and well above clustering (about 0.67) and PCA (about 0.69). See section 15.
- **Transparent:** it uses only three things known at booking time (**lead time, age, prior no-shows**), and every tier is a rule staff can apply without a model.
- **Consistent with the rest of the analysis:** these are the same three factors that dominate the logistic regression (section 14). Reminders are most effective exactly where the tree puts its high-risk tier: booked-ahead, younger patients and repeat missers (section 13).
""")

md("""
## 16.1 The tier rules

| Tier | Rule (at time of booking) | No-show rate | Share of appointments | Share of no-shows |
|---|---|---|---|---|
| 🟢 **Low** | **Same-day** booking, unless the patient is **24 or under with 2+ prior no-shows** | about 3–8% | about 34% | about 7% |
| 🟡 **Medium** | Booked **1+ days ahead**, age **47+**, **no** prior no-shows | about 21% | about 23% | about 24% |
| 🔴 **High** | Booked **1+ days ahead** and age **46 or under** (any history) | about 29% (1–9 days) to 36% (10+ days) | about 38% | about 62% |
| 🔴 **High** | Booked **1+ days ahead**, age **47+**, **1+** prior no-shows | about 32% | about 4% | about 6% |
| 🔴 **High** | **Same-day**, age **24 or under**, **2+** prior no-shows | about 43% (rare) | about 1% | about 1% |

The cell below implements these rules as a plain function and checks that it matches the fitted tree on every appointment.
""")

code("""
def assign_tier(lead_days, age, prior_no_shows):
    \"\"\"No-show risk tier from the 3-level decision tree (section 15.3), using only booking-time data.\"\"\"
    if lead_days == 0:
        return "High" if (prior_no_shows >= 2 and age <= 24) else "Low"
    if age <= 46:
        return "High"
    return "Medium" if prior_no_shows == 0 else "High"

final = seg_df.copy()
final["Tier"] = [assign_tier(l, a, p) for l, a, p in zip(final.LeadDays, final.Age, final.PrevNoShows)]

# Check the hand-written rules against the fitted tree's own leaf -> tier mapping
leaf_tier = {}
for leaf, conds in leaf_rules(tree, F15).items():
    rate = seg_tbl.loc[seg_tbl["segment rule"] == describe(conds), "tier"].iloc[0]
    leaf_tier[leaf] = rate
tree_tier = pd.Series(tree.apply(X15)).map(leaf_tier).astype(str).values
print(f"Rule function agrees with fitted tree on {(final.Tier.values == tree_tier).mean():.2%} of appointments")

tier_order = ["Low", "Medium", "High"]
summary16 = (final.groupby("Tier").agg(appointments=("NoShow", "size"), no_show_rate=("NoShow", "mean"),
                                       no_shows=("NoShow", "sum"))
                  .reindex(tier_order))
summary16["share_of_appointments"] = summary16.appointments / len(final)
summary16["share_of_no_shows"] = summary16.no_shows / final.NoShow.sum()
display(summary16.style.format({"appointments": "{:,.0f}", "no_shows": "{:,.0f}", "no_show_rate": "{:.1%}",
                                "share_of_appointments": "{:.0%}", "share_of_no_shows": "{:.0%}"}))

tier_colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
axes[0].bar(tier_order, summary16.no_show_rate, color=tier_colors)
axes[0].axhline(BASE, ls="--", color="gray", lw=1, label=f"Overall {BASE:.1%}"); axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_title("No-show rate by risk tier")
x = np.arange(2)
bottom = np.zeros(2)
for t, c in zip(tier_order, tier_colors):
    vals = summary16.loc[t, ["share_of_appointments", "share_of_no_shows"]].values.astype(float)
    axes[1].barh(x, vals, left=bottom, color=c, label=t)
    for xi, (b, v) in enumerate(zip(bottom, vals)):
        axes[1].text(b + v / 2, xi, f"{v:.0%}", ha="center", va="center", color="white", fontweight="bold")
    bottom += vals
axes[1].set_yticks(x); axes[1].set_yticklabels(["Appointments", "No-shows"])
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].set_title("High tier: a minority of appointments, most of the no-shows"); axes[1].legend(loc="lower right")
plt.tight_layout(); plt.show()
""")

md("""
## 16.2 Action plan by tier

| Tier | Action | Evidence |
|---|---|---|
| 🟢 **Low** | **No reminder needed.** Same-day patients already intend to come. | Same-day no-show rate is about 3–5% (section 5); same-day bookings never received SMS, yet attendance is near-perfect. |
| 🟡 **Medium** | **Standard SMS reminder.** | Reminders cut the odds of a no-show by about 26% for seniors at every lead time (section 13.2). |
| 🔴 **High** | **SMS plus a second touch:** a day-before reminder with a **confirm / reschedule** option, or a phone call for patients with prior no-shows. Treat these slots as **overbooking candidates**. | Reminders reduce no-shows most on routine bookings for kids (40% → 30%, 48% → 36%) and for prior missers (57% → 38%, sections 13.2–13.3). |

**Special cases within High**
- **Kids' routine and long-planned visits:** reminders work well here, so a second reminder is likely to pay off. Kids' bookings made **within a week** don't benefit from SMS (section 13.2), so effort is better spent elsewhere.
- **Repeat missers of any age:** a prior no-show raises the odds about 1.8x even after everything else is controlled for (section 14). A personal call is justified here.
- **Seniors still missing despite reminders:** test transport or caregiver outreach. The data can't measure this, but it fits the remaining no-show rate.
""")

md("""
## 16.3 Shortening lead time, and measuring the impact

- **Shorten waits where possible.** Moving an appointment from 10+ days to under a week drops a younger patient from the 36% segment to the 29% segment. Same-day or next-day slots are the most reliable lever of all (OR about 7–13 for any lead time vs. same day, section 14).
- **Roll out as a test, not a blanket change.** Apply the tier actions to a random half of High-tier bookings first and compare no-show rates. This turns the observational findings here into causal evidence.
- **Track monthly:** no-show rate by tier, the share of no-shows falling in High, and reminder delivery rate (SMS coverage dropped to 52% in May 2016 in this data).

## 16.4 Limits and when to revisit

- This is observational data from about 6 weeks in one city (Vitória, Brazil, 2016). The tier rules should be **re-fit on local, current data** before use.
- The model's ceiling (AUC about 0.75) comes from **missing information**, not the method. **Start recording appointment type (routine / follow-up / urgent) and cancellations**, then re-fit the tree. Those fields are the most likely to sharpen the tiers and to test the "routine vs. need-driven" hypothesis directly.
- Keep the logistic model (section 14) for explaining effects and for finer-grained risk scores when needed.

**Bottom line:** three facts known at booking (**how far ahead, how old, and whether they've missed before**) sort patients into tiers that separate about 4% from about 30%+ no-show rates. Focusing reminders and follow-up on the High tier reaches about two-thirds of all no-shows by acting on fewer than half of appointments.
""")

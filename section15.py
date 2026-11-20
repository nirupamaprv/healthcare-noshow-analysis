md("""
---
# 15. Segmentation: clustering vs. PCA vs. supervised models, and the recommended approach

The full logistic model has a low pseudo R² (about 0.12). Would an **unsupervised** approach (k-means clustering, or PCA) give a better handle on who shows up?
We compare all approaches on the **same held-out test set** using ROC AUC, then turn the best interpretable option into usable segments.

> **Decision recorded:** the **3-level decision tree** (section 15.3) is the recommended segmentation approach. It is nearly as accurate as the best model, far better than clustering or PCA, and every segment is a plain-English rule staff can apply.
""")

code("""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import HistGradientBoostingClassifier

seg_df = df.assign(SameDay=(df.LeadDays == 0).astype(int), Male=(df.Gender == "M").astype(int))
F15 = ["Age", "LeadDays", "SameDay", "SMS_received", "Scholarship", "Hypertension", "Diabetes",
       "Alcoholism", "Handicap", "Male", "PrevAppts", "PrevNoShows"]
X15 = seg_df[F15].astype(float)
y15 = seg_df.NoShow
Xtr, Xte, ytr, yte = train_test_split(X15, y15, test_size=0.25, stratify=y15, random_state=0)

# Scaled copy with log lead time, for the distance-based methods (k-means, PCA, logistic)
def prep(X):
    return X.assign(LeadDays=np.log1p(X.LeadDays))
scaler = StandardScaler().fit(prep(Xtr))
Ztr, Zte = scaler.transform(prep(Xtr)), scaler.transform(prep(Xte))

results = {}

# 1. K-means: clusters ignore the outcome; score each test row with its cluster's training no-show rate
for k in [4, 6, 8, 12]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Ztr)
    cl_rate = pd.Series(ytr.values).groupby(km.labels_).mean()
    results[f"K-means clustering (k={k})"] = roc_auc_score(yte, cl_rate.reindex(km.predict(Zte)).values)

# 2. PCA: keep the top components, then fit a logistic regression on them
pca_full = PCA().fit(Ztr)
for n in [2, 4, 6]:
    pipe = make_pipeline(PCA(n_components=n), LogisticRegression(max_iter=1000)).fit(Ztr, ytr)
    results[f"PCA ({n} components) + logistic"] = roc_auc_score(yte, pipe.predict_proba(Zte)[:, 1])

# 3. Supervised models
lr15 = LogisticRegression(max_iter=1000).fit(Ztr, ytr)
results["Logistic regression (all features)"] = roc_auc_score(yte, lr15.predict_proba(Zte)[:, 1])
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=0).fit(Xtr, ytr)
results["Decision tree, 3 levels (8 segments)"] = roc_auc_score(yte, tree.predict_proba(Xte)[:, 1])
hgb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
results["Gradient boosting"] = roc_auc_score(yte, hgb.predict_proba(Xte)[:, 1])

res15 = pd.Series(results, name="test ROC AUC").sort_values()
kind = ["Unsupervised" if ("K-means" in i or "PCA" in i) else "Supervised" for i in res15.index]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(res15.index, res15.values, color=["#adb5bd" if k == "Unsupervised" else "#2a9d8f" for k in kind])
ax.barh(["Decision tree, 3 levels (8 segments)"], [results["Decision tree, 3 levels (8 segments)"]], color="#e76f51")
for i, v in enumerate(res15.values):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.5, 0.8); ax.set_xlabel("Test ROC AUC (0.5 = coin flip)")
ax.set_title("Unsupervised (grey) vs supervised (teal); recommended tree in orange")
plt.tight_layout(); plt.show()
res15.to_frame()
""")

md("""
## 15.1 Why clustering and PCA underperform

Both methods are **unsupervised**: they never see whether a patient showed up. They organize the data around whatever varies most, which here is not what drives attendance.
""")

code("""
# What k-means actually found (k=6)
km6 = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Ztr)
prof = (Xtr.assign(cluster=km6.labels_, NoShow=ytr.values)
           .groupby("cluster")
           .agg(appointments=("NoShow", "size"), no_show_rate=("NoShow", "mean"), median_age=("Age", "median"),
                median_lead_days=("LeadDays", "median"), pct_SMS=("SMS_received", "mean"),
                pct_hypertension=("Hypertension", "mean"), pct_diabetes=("Diabetes", "mean"))
           .sort_values("no_show_rate"))
display(prof.style.format({"no_show_rate": "{:.1%}", "pct_SMS": "{:.0%}", "pct_hypertension": "{:.0%}",
                           "pct_diabetes": "{:.0%}", "appointments": "{:,.0f}", "median_age": "{:.0f}",
                           "median_lead_days": "{:.0f}"}))

# What PCA's components are made of
fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2]})
axes[0].bar(range(1, len(F15) + 1), pca_full.explained_variance_ratio_.cumsum(), color="#264653")
axes[0].axhline(0.8, ls="--", color="gray", lw=1)
axes[0].set_xlabel("Number of components"); axes[0].set_ylabel("Cumulative variance explained")
axes[0].set_title("Variance is spread out; no small set of components dominates")
load = pd.DataFrame(pca_full.components_[:4].T, index=F15, columns=[f"PC{i}" for i in range(1, 5)])
sns.heatmap(load.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar=False, ax=axes[1])
axes[1].set_title("PCA loadings: PC1 = age/chronic conditions, not attendance drivers")
plt.tight_layout(); plt.show()
""")

md("""
**Reading this:**
- **K-means** puts same-day bookings in one group and hypertension/diabetes patients in their own groups. But hypertension has **no independent effect** on attendance (section 14). The highest-risk cluster is basically "everyone under ~50 who booked ahead", which is too broad to act on.
- **PCA**'s first component is mostly **age plus chronic conditions**, because those variables move together. Lead time and prior no-shows, the strongest predictors, are spread across later components. With only 12 mostly yes/no features and little redundancy among them, PCA has nothing useful to compress, and it costs interpretability.
- **The ceiling is the data, not the method.** Gradient boosting, which can learn any interaction, beats plain logistic regression by only about 0.01 AUC. Every method stops around 0.75, because key drivers (appointment type, transport, work schedules, cancellations) aren't recorded.
""")

md("## 15.2 Risk tiers from the logistic model")

code("""
p_lr = lr15.predict_proba(Zte)[:, 1]
tiers = pd.qcut(pd.Series(p_lr).rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
tier_tbl = (pd.DataFrame({"tier": tiers, "NoShow": yte.values, "pred": p_lr})
              .groupby("tier", observed=True)
              .agg(appointments=("NoShow", "size"), actual_no_show=("NoShow", "mean"), predicted=("pred", "mean")))
tier_tbl["share_of_all_no_shows"] = (tier_tbl.actual_no_show * tier_tbl.appointments) / yte.sum()
display(tier_tbl.style.format({"actual_no_show": "{:.1%}", "predicted": "{:.1%}", "share_of_all_no_shows": "{:.0%}"}))

ax = tier_tbl.actual_no_show.plot.bar(figsize=(8, 3.5), color="#e76f51", rot=0)
ax.axhline(BASE, ls="--", color="gray", lw=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Actual no-show rate by predicted-risk decile (D1 = lowest risk)")
ax.set_xlabel("Risk decile"); plt.tight_layout(); plt.show()
""")

md("""
## 15.3 Recommended: 3-level decision tree segments

A depth-3 tree (each segment has at least 500 training appointments) splits patients into **8 segments**, each defined by a plain-English rule.
No-show rates below are measured on the **held-out test set**.
""")

code("""
fig, ax = plt.subplots(figsize=(20, 7))
plot_tree(tree, feature_names=F15, class_names=["Show", "No-show"], filled=True, rounded=True,
          proportion=True, impurity=False, precision=2, fontsize=9, ax=ax)
ax.set_title("3-level decision tree (value = [show share, no-show share])")
plt.tight_layout(); plt.show()
""")

code("""
def leaf_rules(model, names):
    t = model.tree_
    rules = {}
    def walk(node, conds):
        if t.children_left[node] == -1:
            rules[node] = conds
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", thr)])
        walk(t.children_right[node], conds + [(f, ">", thr)])
    walk(0, [])
    return rules

def describe(conds):
    # Merge all splits on the same feature into one range, then phrase it in plain English
    lo, hi = {}, {}
    for f, op, thr in conds:
        if op == ">":
            lo[f] = max(lo.get(f, -np.inf), np.floor(thr) + 1)   # integer-valued features
        else:
            hi[f] = min(hi.get(f, np.inf), np.floor(thr))
    parts = []
    for f in dict.fromkeys(c[0] for c in conds):
        l, h = lo.get(f, 0), hi.get(f, np.inf)
        if f in ("LeadDays", "SameDay"):
            if f == "SameDay":
                l, h = (1, np.inf) if h == 0 else (0, 0)
            txt = ("same-day booking" if h == 0 else f"booked {l:.0f}+ days ahead" if h == np.inf
                   else f"booked {l:.0f}-{h:.0f} days ahead")
        elif f == "Age":
            txt = (f"age {l:.0f}+" if h == np.inf else f"age {h:.0f} or under" if l == 0 else f"age {l:.0f}-{h:.0f}")
        elif f == "PrevNoShows":
            txt = ("no prior no-shows" if h == 0 else f"{l:.0f}+ prior no-shows" if h == np.inf
                   else f"exactly {l:.0f} prior no-show" if l == h else f"{l:.0f}-{h:.0f} prior no-shows")
        else:
            txt = f if l >= 1 else f"no {f}"
        parts.append(txt)
    return ", ".join(parts)

rules = leaf_rules(tree, F15)
leaf_te = tree.apply(Xte)
seg_tbl = (pd.DataFrame({"leaf": leaf_te, "NoShow": yte.values})
             .groupby("leaf").agg(test_appointments=("NoShow", "size"), test_no_show_rate=("NoShow", "mean")))
seg_tbl["share_of_appointments"] = seg_tbl.test_appointments / len(yte)
seg_tbl["share_of_no_shows"] = seg_tbl.test_no_show_rate * seg_tbl.test_appointments / yte.sum()
seg_tbl["segment rule"] = [describe(rules[l]) for l in seg_tbl.index]
seg_tbl = seg_tbl.sort_values("test_no_show_rate").reset_index(drop=True)
seg_tbl.index = [f"S{i}" for i in range(1, len(seg_tbl) + 1)]
seg_tbl["tier"] = pd.cut(seg_tbl.test_no_show_rate, [0, 0.10, 0.25, 1], labels=["Low", "Medium", "High"])
display(seg_tbl[["segment rule", "tier", "test_no_show_rate", "share_of_appointments", "share_of_no_shows", "test_appointments"]]
        .style.format({"test_no_show_rate": "{:.1%}", "share_of_appointments": "{:.0%}",
                       "share_of_no_shows": "{:.0%}", "test_appointments": "{:,.0f}"}))

colors = seg_tbl.tier.map({"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"})
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(seg_tbl.index[::-1], seg_tbl.test_no_show_rate[::-1], color=colors[::-1])
ax.axvline(BASE, ls="--", color="gray", lw=1, label=f"Overall {BASE:.1%}")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("No-show rate by decision-tree segment (test set)"); ax.legend()
plt.tight_layout(); plt.show()
""")

md("""
## 15.4 Conclusion and recorded decision

| Approach | Test AUC | Interpretable? | Verdict |
|---|---|---|---|
| K-means clustering | ~0.67–0.68 | Partly (clusters need interpreting) | Not recommended: ignores the outcome, groups by hypertension and age |
| PCA + logistic | below full logistic | No (components mix variables) | Not recommended: nothing redundant to compress, loses meaning |
| Logistic regression | ~0.74 | Yes (odds ratios) | Use for **explanation** (section 14) and **risk scores and deciles** (15.2) |
| Gradient boosting | ~0.75 | No | Barely better than logistic; not worth the opacity |
| **3-level decision tree** | **~0.72–0.73** | **Yes, plain-English rules** | **✅ Recommended for segmentation and operations** |

**Decision:** use the **3-level decision tree segments** as the working segmentation for no-show interventions:
| Tier | Segments (from 15.3) | Test no-show rate | Share of no-shows | Action |
|---|---|---|---|---|
| **Low** | Same-day bookings (except young repeat missers) | about 3–8% | about 6% | No reminder needed |
| **Medium** | Booked ahead, age 47+, no prior no-shows | about 21% | about 24% | Standard SMS reminder |
| **High** | Booked ahead and age 46 or under (1–9 days: about 29%; 10+ days: about 36%); booked ahead, age 47+, with a prior no-show (about 32%); same-day, age 24 or under, 2+ prior no-shows (about 43%, small) | about 29–43% | about **69%** | Second reminder, confirm/reschedule prompt or call; overbooking candidates |

The three high-risk segments hold about 43% of appointments but about **69% of all no-shows**. That's where intervention effort should go.

Use the logistic model's risk deciles when finer ranking is needed. **Revisit** the tree once appointment type and cancellations are recorded; those are the most likely ways past the ~0.75 AUC ceiling.
""")

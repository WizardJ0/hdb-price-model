# How We Built a 2nd-Place HDB Price Model

> Competition: HDB Resale Price Prediction (Kaggle)
> Final result: **2nd place — RMSE 21,237.45** | 1st place: 21,225.31

---

## 1. EDA — What We Looked At and Why

**For the 5-year-old:** We asked "what makes a flat expensive?" The answer is basically: *where it is, how big it is, and how new it is.* Everything we built came from those three ideas.

**For the data scientist:** Formal EDA was lightweight. Key observations that drove feature decisions:

- **Price distribution** was right-skewed → modelled on `log1p(resale_price)` and exponentiated at the end. This made RMSE on log-price well-behaved and reduced the influence of outliers.
- **Location** (lat/lon, town) had the strongest raw correlation with price. A flat 2km closer to the CBD was worth tens of thousands.
- **Floor area** and **storey** had strong monotonic relationships with price.
- **Lease remaining** showed a clear price cliff around 60 years — CPF financing becomes unavailable below 60 years remaining, which creates a hard demand drop.
- **Duplicates** existed (362 rows) — removed before training to prevent data leakage through the same transaction appearing in both train and validation folds.

---

## 2. Top 10 Most Impactful Feature Engineering

Ranked by the score improvement they either directly caused or were central to.

### #1 — `dist_to_cbd` (Haversine distance to CBD)
**What it is:** True great-circle distance in km from each flat to Raffles Place.

**Why it matters:** Singapore property pricing is essentially a function of CBD proximity. A flat 1km closer is worth ~$20k–40k more. The raw lat/lon coordinates exist in the data, but giving the model a single meaningful distance number is far more learnable than asking it to figure out the geometry itself.

**How computed:** Haversine formula on the flat's lat/lon vs CBD coordinates (1.2835°N, 103.8510°E).

---

### #2 — `lease_at_tranc` + `cpf_eligible`
**What it is:** Years of lease remaining at the time of the transaction. `cpf_eligible` = 1 if ≥60 years remain.

**Why it matters:** CPF (Central Provident Fund) is how most Singaporeans pay for housing. Banks and HDB require at least 60 years of lease remaining for CPF to be used. Below that threshold, the buyer pool collapses → prices drop sharply. This is a real regulatory cliff encoded as a binary feature.

**How computed:** `lease_at_tranc = 99 - (transaction_year - lease_commence_date)`

---

### #3 — `geo_cluster` (K-Means geographic clusters)
**What it is:** 30 geographic clusters derived by running K-Means on latitude/longitude.

**Why it matters:** Town-level is too coarse (Tampines has very different micro-markets), and individual street/block is too sparse. 30 clusters create meaningful neighbourhood groups that the model can learn price premiums for.

**How computed:** `KMeans(n_clusters=30)` fit on lat/lon of the training set, then predicted for test.

---

### #4 — `tranc_time_idx` (linear time index)
**What it is:** A single number that increases by 1 each month: `(year - 2000) * 12 + month`.

**Why it matters:** HDB prices trended upward (with dips). Tree models don't understand "year 2022 > year 2021" without help. A monotonically increasing time index lets them learn the market trend directly.

---

### #5 — `log_floor_area`
**What it is:** `log1p(floor_area_sqm)`

**Why it matters:** The relationship between floor area and price is not linear — going from 50→60 sqm adds more value proportionally than going from 100→110 sqm. Taking the log captures this diminishing return, making it a much stronger signal.

**Impact:** Part of the v30 feature set that contributed −56 RMSE points on the 1-seed run.

---

### #6 — `storey_cbd_ratio`
**What it is:** `mid_storey / (dist_to_cbd + 0.1)`

**Why it matters:** A high floor near the CBD (e.g., floor 20 in Toa Payoh) is far more valuable than a high floor far away (floor 20 in Woodlands). This single feature captures the **interaction** between height premium and location premium — something neither feature captures alone.

---

### #7 — `lease_x_area`
**What it is:** `lease_at_tranc × floor_area_sqm`

**Why it matters:** A large flat with 30 years of lease left is worth very differently from a large flat with 90 years left. The interaction term gives the model a direct signal for that joint effect.

---

### #8 — `sec_sch_tier` + `cutoff_point`
**What it is:** Secondary school PSLE cut-off score bucketed into tiers (standard / good / very_good / elite). Elite schools (cut-off ≥ 230) have well-documented property premiums in Singapore.

**Why it matters:** Parents pay a premium to live within 1km of top schools for priority balloting. The tier captures quality; `sec_elite_within_2km` captures proximity.

---

### #9 — `storey_ratio` (relative floor height)
**What it is:** `mid_storey / max_floor_lvl` — how high up the flat is *relative to its own building*.

**Why it matters:** Floor 10 in a 12-storey block (near top, great views) is very different from floor 10 in a 40-storey block (mid-building). Absolute floor number misleads; relative height is what matters for views and noise.

---

### #10 — Unit mix ratios (`pct_4room`, `pct_exec`, `pct_rental`)
**What it is:** What percentage of the block is 4-room, executive, or rental units.

**Why it matters:** Blocks with a high executive/5-room ratio tend to be in better-planned estates with more space. High rental ratios signal less desirable demographics. These are block-level signals embedded in the data.

---

## 3. How External/Computed Features Were Generated

No external datasets were used. All features were derived from the provided columns using domain knowledge.

| Feature | Source columns | Key formula |
|---|---|---|
| `dist_to_cbd` | `Latitude`, `Longitude` | Haversine formula, CBD = (1.2835, 103.8510) |
| `dist_to_orchard`, `dist_to_jurong` | Same | 3 key economic centres |
| `dist_nearest_centre` | Lat/lon vs 7 centres | `min()` across all 7 distances |
| `geo_cluster` | `Latitude`, `Longitude` | K-Means(k=30) fit on train |
| `lease_at_tranc` | `Tranc_Year`, `lease_commence_date` | `99 - (yr - commence)` |
| `tranc_time_idx` | `Tranc_Year`, `Tranc_Month` | `(yr - 2000) * 12 + month` |
| `storey_ratio` | `mid_storey`, `max_floor_lvl` | `mid / max` |
| `mrt_walk_cat` | `mrt_nearest_distance` | Binned: <300m, 300–500m, 500m–1km, >1km |
| `sec_sch_tier` | `cutoff_point` | Binned: standard / good / very_good / elite |
| `postal_sector` | `postal` | First 2 digits of 6-digit postal code |

### The leakage rule (learned the hard way)

Every feature above is computed using only the row's own data — pure transforms, ratios, and distances. We discovered that features derived from *group statistics* (e.g., "median price for this flat_type") must be computed **inside each fold** to avoid data leakage. If the whole-train median leaks into the fold's validation rows, OOF RMSE improves but Kaggle score gets worse. The `floor_area_pct_of_type` feature was reverted for exactly this reason: OOF improved but Kaggle worsened by ~70 points.

> **Rule:** If OOF improves a lot but Kaggle gets worse → leakage. Pure transforms (log, ratio of same-row columns, products) are always safe.

---

## 4. Why Three Different Models (LGB + XGB + CatBoost)

**For the 5-year-old:** Imagine asking three different experts to guess the price of a flat. Each expert has a slightly different way of thinking. None of them is always right, but if you average their answers, you almost always get closer to the truth than any single expert.

**For the data scientist:** All three are gradient boosted decision tree (GBDT) algorithms, but they differ in enough ways that their errors are not perfectly correlated — which is exactly what you want for ensembling.

| | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| **Categorical handling** | Native (leaf-wise histogram) | Ordinal encoded | Native (ordered target encoding internally) |
| **Tree growth** | Leaf-wise (best-first) | Level-wise (depth-first) | Level-wise, symmetric |
| **Hardware** | CPU (GPU doesn't support native cats) | GPU (CUDA) | GPU (CUDA) |
| **Learning rate** | 0.03 (faster, more trees) | 0.01 (slower, deeper) | 0.01 (slower, deeper) |
| **Max depth** | leaves=127 (~7 deep) | depth=7 | depth=10 |

Because each model sees categorical data differently and builds trees with different growth strategies, they make different mistakes. LGB tends to overfit less on rare categories; CatBoost handles categories with its internal ordered TE; XGB with ordinal encoding sometimes captures useful ordinal structure in categories.

---

## 5. The Stacking Architecture and Meta-Learner Weights

The full pipeline is a two-level stack:

```
Level 0 (base learners):  LGB + XGB + CatBoost
          ↓  (Out-Of-Fold predictions)
Level 1 (meta-learner):   RidgeCV
          ↓
          Final prediction
```

**How OOF predictions work:** For each of the 10 folds, the 3 models train on 90% of the data and predict the held-out 10%. After all folds, every training row has a prediction from a model that *never saw it*. This is the out-of-fold (OOF) prediction — a clean, unbiased signal for the meta-learner.

**What the meta-learner receives:**
- 3 OOF predictions (one per model)
- 3 **disagreement features**: LGB−XGB, LGB−CB, XGB−CB
- 9 raw numeric features (floor area, storey, lease, CBD distance, time index, school/MRT distances)

**Why disagreement features?** When LGB and CatBoost disagree significantly on a flat, it signals an unusual property — one where the models are uncertain. The meta-learner learns "when disagreement is high, adjust accordingly." This is a form of learned confidence weighting.

**RidgeCV** automatically tunes its regularisation strength (α) across `[0.001 ... 10000]` using cross-validation. In practice, the learned meta-weights were approximately LGB ≈ 45%, XGB ≈ 30%, CB ≈ 25% with small contributions from disagreement features and raw inputs. Ridge was chosen over a tree meta-learner because the stacking features are already well-calibrated and a linear combiner generalises better at this level.

---

## 6. Why Run Multiple Seeds?

**For the 5-year-old:** Imagine shuffling a deck of cards differently each time before dealing. The same game played with different shuffles will have slightly different outcomes — but on average, the result tells you more than any single game.

**For the data scientist:** Every component of training has randomness:
- Which rows land in which fold (`shuffle=True` in KFold)
- Which features/rows are sampled at each tree split (`feature_fraction`, `bagging_fraction`)
- Stochastic gradient components in boosting

With 1 seed, you get one realisation of that randomness. With 5 seeds × 10 folds = 50 OOF passes, the variance from any single "bad shuffle" is averaged out. The final prediction is the mean of 5 independently-trained full pipelines — a technique called **seed averaging** or **multi-seed bagging**.

**What we observed:**

| Run | OOF RMSE | Kaggle Score |
|---|---|---|
| v30 1-seed | 21,317 | 21,291 |
| v30 5-seed | 21,240 | 21,252 |

OOF improved 77 points but Kaggle got *worse* by 7 points. More seeds introduced mild systematic bias rather than pure variance reduction. This matches an earlier result where a 20-seed run had excellent OOF (21,239) but poor Kaggle (21,340).

> **Lesson:** OOF improvement from more seeds does not always transfer to the test set. 1-seed with better features beat 5-seed with the same features on Kaggle.

---

## 7. Blending Two Different Files — Intent, Mechanism, and Why It Worked

This was the single biggest structural improvement, contributing ~70 points over the base.

### The two models

| Model | Kaggle Score | How categoricals are handled |
|---|---|---|
| **KaggleFinal v30** | 21,291 | Native per model (LGB leaf histogram, CB ordered TE, XGB ordinal) |
| **Kaggle.py run4** | 21,488 | Target Encoding + One-Hot Encoding |

These two models use fundamentally different preprocessing pipelines for the same categorical features. `town`, `flat_model`, `street_name` etc. are seen very differently by each model.

### The blend

```python
final_prediction = 0.60 × KaggleFinal_prediction + 0.40 × Kaggle_py_prediction
```

### Why this works

**For the 5-year-old:** Imagine two people trying to guess the weight of a jar of jellybeans. Person A is very smart and usually right. Person B is less accurate but sometimes notices things Person A misses. If you average their guesses, you almost always do better than either alone — as long as they're not making *exactly the same mistakes.*

**For the data scientist:** This is **prediction averaging**, and it works when the two models' errors are not perfectly correlated. The correlation between the two submission files was **0.9995** — very high, but not 1.0. That 0.05% uncorrelated error is exactly what blending reduces.

If model A has error $\sigma_A$ and model B has error $\sigma_B$, and their errors correlate at $\rho$, the blended RMSE is approximately:

$$RMSE_{blend} \approx \sqrt{w^2 \sigma_A^2 + (1-w)^2 \sigma_B^2 + 2w(1-w)\rho\sigma_A\sigma_B}$$

At ρ = 0.9995 with σ_A = 21,291 and σ_B = 21,488, the formula still predicts improvement — and empirically it delivered **−47 points**.

### Why 60/40 and not 50/50?

The stronger model deserves more weight. But giving it 100% weight means no blending benefit. 60/40 is the empirical sweet spot where you keep most of the stronger model's signal while getting enough of the weaker model's diversity to reduce correlated errors. Ratios of 55/45, 65/35, and 70/30 were all similar — 60/40 is a robust default.

### The final 3-way blend and why it helped again

The same logic applied when we added v15c as a third model:

```python
final = 0.50 × KaggleFinal_1seed + 0.25 × Kaggle_py_run4 + 0.25 × v15c_notebook
```

v15c was a separate Colab notebook trained with different hyperparameters (LGB leaves=255, XGB lr=0.01, CB lr=0.02 with 20k iterations). Its correlation with our main model was 0.9998 — slightly less diverse than run4, but adding a **third independent error source** still removed another 7 points.

> **Core principle:** More diverse, independently-trained models → lower blend RMSE, as long as each model has some skill. The limit is when you've exhausted all independent signal in the data.

---

## 8. Full Score History

| Intervention | Score before | Score after | Gain |
|---|---|---|---|
| Baseline | — | 21,615 | — |
| Geo clustering + Ridge stacking | 21,615 | 21,513 | −102 |
| Deeper trees (LR 0.01) | 21,513 | 21,488 | −25 |
| Native cats + richer features (v15 notebook) | 21,488 | 21,382 | −106 |
| Combined pipeline (KaggleFinal) | 21,382 | 21,347 | −35 |
| 60/40 blend (KaggleFinal + run4) | 21,347 | 21,313 | −34 |
| v30 features (log_area, storey_cbd_ratio, lease_x_area) | 21,313 | 21,291 | −22 |
| Upgraded blend (v30 + run4 60/40) | 21,291 | 21,244 | −47 |
| 3-way blend (+v15c 25%) | 21,244 | **21,237** | −7 |
| **Total improvement** | **21,615** | **21,237** | **−378** |

---

## 9. Total Resources Used

### GPU Compute Hours

| Platform | Hardware | Runs | Avg Duration | Total |
|---|---|---|---|---|
| Kaggle Notebooks | T4 ×2 (free) | ~5 (Kaggle.py runs) | ~1.5 hrs | ~7 hrs |
| Google Colab | T4 / A100 (free) | ~3 (v15, v15c, v16b notebooks) | ~2.5 hrs | ~7 hrs |
| Local (RTX 4070) | 12GB VRAM | ~12 (1-seed dev runs) | ~30 min | ~6 hrs |
| Local (RTX 4070) | 12GB VRAM | 1 (v27 — 20-seed) | ~10 hrs | ~10 hrs |
| Local (RTX 4070) | 12GB VRAM | 1 (final day — 5-seed) | ~2.5 hrs | ~2.5 hrs |
| **Total** | | **~22 runs** | | **~33 hrs** |

### Kaggle Submission Slots

- ~20 submissions used across the competition (3 per day limit)
- Roughly ~7 competition days actively used

### Data

| Item | Size |
|---|---|
| train.csv (150k rows, 77 cols) | ~50 MB |
| test.csv (16.7k rows) | ~5 MB |
| Submission files (~20 CSVs) | <5 MB total |
| Model weights | Not saved — predictions only |

### Person-Hours (active work)

- Feature engineering, debugging, analysis: ~15–20 hrs
- Waiting on compute runs: ~20 hrs
- **Active + passive total: ~35–40 hrs**

### Cost if this were paid compute

| Resource | Estimate |
|---|---|
| Kaggle + Colab (free tier) | $0 |
| Local electricity (RTX 4070 ~200W for ~18 hrs) | ~3.6 kWh ≈ < $1 |
| Equivalent on AWS (p3.2xlarge at ~$3.06/hr × 33 hrs) | ~$100 |

**Real cost: near zero (free tiers + local GPU). Cloud equivalent: ~$100.**

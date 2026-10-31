# Covariance matching for Odia synthetic QA selection

This folder selects a subset of LLM-generated Odia QA pairs whose embedding covariance matches that of the real IndicQA Odia training data. The method is *covariance matching* from:

> Rezaei, Kovačević, Locatello, Mondelli. **High-dimensional Analysis of Synthetic Data Selection.** ICLR 2026.

```
covariance_matching/
├── covariance_matching.py   # the selection script
├── README.md
└── outputs/                 # created by the script
```

---

## 1. Idea from the paper

The paper trains on a mix of real data `(X_t, y_t)` and synthetic data `(X_s, y_s)`, then tests on real data. It asks which properties of the synthetic data lower the test error.

Its linear-regression analysis gives three results:

| Result | Statement | Practical meaning |
|---|---|---|
| Theorem 4.1 / 4.4 | When training on real + synthetic data, the limiting test error depends on the covariances Σ_t and Σ_s, **not on the means** μ_t and μ_s. | The synthetic data does not need to be centred where the real data is. Only its spread and shape matter. |
| Proposition 4.2 | When training on synthetic data **only**, the mean shift does matter. | Real data in the training mix is what makes the mean irrelevant. |
| Theorem 4.3 / 4.5 | For a fixed trace, the error is minimised when **Σ_s ∝ Σ_t**. | Pick synthetic samples whose covariance matches the real covariance. |

The paper turns this into a greedy rule (Section 5):

```
S = {}
while |S| < n_s:
    add the pool point x minimising  || Cov(S ∪ {x}) − Cov(real) ||_F
```

Covariances are computed on feature embeddings (CLIP or DINO for images, sentence embeddings for text). To make the computation cheap, they are taken in a low-dimensional PCA space fitted on the **real** features. The paper uses 32 dimensions with 200 real samples per class.

### How the paper uses classes, and how this project differs

The theory models each class as one mixture component, so the paper matches covariance **one class at a time**. For CIFAR-10 class *c*:

1. Take the 200 real CLIP features of class *c*.
2. Fit PCA-32 on those features.
3. Greedily select 800 images from the pool generated for class *c*, matching Σ_t^c.
4. Repeat for all 10 classes and train on the union.

Extractive QA has no class labels, and every real row has category `SHORT`. This project therefore uses **one global class**: a single Σ_t from all real training rows, one PCA, and one greedy pass over the whole pool. This is the theory's single-component setting. It also mirrors the paper's text experiment (Appendix C, Table 13), which applied the same algorithm to sentence-transformer embeddings.

---

## 2. Data

| Role | File | Rows | Notes |
|---|---|---|---|
| Synthetic pool (text + metadata) | `low_rank_method/odia_low_rank_run/final_weighted_train_draws.parquet` | 37,058 draws (33,104 unique) | Output of the low-rank sampling stage (see §2.1) |
| Synthetic pool (embeddings) | `low_rank_method/odia_low_rank_run/final_weighted_train_draws_embeddings.pt` | 37,058 × 768 | Equal to `embedding_generation/odia_embeddings.pt[_source_row_idx]` (verified) |
| Real reference (text) | `data_and_scripts/odia_real_cleaned/train.jsonl` | 886 | IndicQA Odia **train** split: 176 passages, 176 articles |
| Real reference (embeddings) | `embedding_generation/odia_real_train_embeddings.pt` | 886 × 768 | Created by the script on its first run, then cached |

**Only the train split** is used as the real reference. Validation and test are never touched, so they stay valid for model selection and final evaluation.

Note: `embedding_generation/odia_embeddings.pt` and `odia_answer_embeddings.pt` (216,394 rows each) are embeddings of the **full synthetic pool** (question+context and answers respectively). They are not real-data embeddings.

### 2.1 Where the pool and its weights come from

The pool is the output of the low-rank sampling stage in `low_rank_method/Untitled1.ipynb`:

1. Each of the ≈185k synthetic train rows gets a score: `max(predicted_loss, 0) + residual_sq`.
   - `predicted_loss` is a kernel-ridge prediction of the Qwen3-1.7B loss, fitted on leverage-sampled landmark rows.
   - `residual_sq` is the squared distance of the row's embedding from the landmark PCA subspace.
2. Sampling probabilities are `p_i = score_i / Σ score`.
3. `s = 37,058` rows are drawn **with replacement** using `p_i`. This is why 3,607 rows appear 2 to 5 times.
4. Each draw keeps importance weights `1/(s·p_i)` and `1/(N·p_i)`. With these weights, a weighted average loss over the draws is an unbiased estimate of the average loss over all N rows.

Covariance matching then makes a **deterministic** choice from this pool, so the importance weights no longer describe how the final subset was sampled. The paper uses unweighted samples. For that reason the weights are **not used during selection**, but they are carried into the output as metadata.

### 2.2 Embedding recipe

Real and synthetic rows must be embedded identically. Otherwise the two covariances would describe different spaces. The script reproduces `embedding_generation/generate_odia_embeddings.ipynb` exactly:

- model `intfloat/multilingual-e5-base`, `max_length = 512`
- input text `query: Question: {question}\nContext: {passage}` (answers are not embedded)
- attention-mask mean pooling, then L2 normalisation

Before embedding the real rows, the script re-embeds 8 pool rows and checks that their cosine similarity with the stored vectors is at least 0.99. If the recipe has drifted, it stops.

---

## 3. Algorithm as implemented

1. **Candidate pool.** Remove duplicate draws by `_source_row_idx`, leaving 33,104 candidates, and record each row's `draw_count`. Use `--keep-duplicates` to treat every draw as its own candidate.
2. **Real reference.** Load the cached real embeddings, or create them (§2.2).
3. **PCA on real data only.** Centre the real embeddings on their mean μ_t, take the top-k right singular vectors, and project both real and candidate embeddings into that basis. Default: `k = 64`.
4. **Target.** Σ_t is the mean-centred covariance of the projected real rows.
5. **Greedy selection.** Repeatedly add the candidate that minimises `‖Cov(S ∪ {x}) − Σ_t‖_F` until the largest requested size is reached. Each set is centred on **its own** mean, so only shape and spread are matched, as Theorem 4.1 recommends.
6. **Export.** Because greedy selection is nested, the first n picks are exactly the size-n solution. Each requested size is a prefix of a single run.

### 3.1 Efficient greedy step

A direct implementation would build a k × k covariance for each of the 33k candidates at each of thousands of steps. The script instead scores all candidates at once with a closed form.

Let `m = |S|`, `M = Σ_{i∈S} x_i x_iᵀ`, `s = Σ_{i∈S} x_i`, and `u = s + x`. Then

```
Cov(S ∪ {x}) = (M + x xᵀ)/(m+1) − u uᵀ/(m+1)²
```

With `D = M/(m+1) − Σ_t`, `c = 1/(m+1)` and `d = c²`:

```
‖Cov(S ∪ {x}) − Σ_t‖²_F = ‖D‖² + c²‖x‖⁴ + d²‖u‖⁴ + 2c·xᵀDx − 2d·uᵀDu − 2cd·(xᵀu)²
```

`Z·M` is maintained through rank-one updates, so each step costs **O(N·k)**. A full run takes seconds to a few minutes on CPU.

This implementation was checked against a brute-force greedy (recomputing every candidate covariance from scratch) on random data. Both produced the same selection order and the same distances.

### 3.2 Conventions and small choices

| Choice | Value | Reason |
|---|---|---|
| Covariance normalisation | `1/n` for both Σ_t and Cov(S) | Keeps the incremental formula exact. The difference from `1/(n−1)` is negligible at these sizes. |
| First pick | The candidate nearest the real mean (the PCA origin) | A single point always has zero covariance, so every candidate ties at step 1. The paper does not say how it breaks this tie. |
| Already-selected rows | Excluded from later steps | Selection is without replacement, as in the paper. |
| Numeric precision | float64 throughout | Rank-one updates accumulate over thousands of steps. |

---

## 4. Hyperparameters

### 4.1 PCA dimension `k` (default 64)

PCA serves two purposes: it keeps each step cheap, and it keeps Σ_t estimable from few real samples. With n real points in k dimensions, sample eigenvalues spread out even when the true covariance is the identity. By the Marchenko–Pastur law they fall roughly in `[(1−√(k/n))², (1+√(k/n))²]`:

| Setting | k/n | Eigenvalue spread |
|---|---|---|
| Paper: k=32, n=200 per class | 0.16 | 0.36 – 1.96 |
| No PCA: k=768, n=886 | 0.87 | 0.005 – 3.7 (unusable) |
| **This project: k=64, n=886** | 0.072 | 0.53 – 1.61 |
| k=64, effective n≈176 passages | 0.36 | 0.16 – 2.57 |

The 886 real questions come from only **176 passages**, and the passage dominates each embedding. The effective sample size therefore lies somewhere between 176 and 886.

The choice of k is a trade-off:
- a **smaller k** gives a less noisy Σ_t but leaves more directions unmatched;
- a **larger k** matches more directions but partly fits noise.

At k=64, about 53% of the pool's variance lies in the matched subspace; at k=32, about 39%.

**Comparing k values fairly:**
- Do **not** compare `cm_frobenius` values across k. A distance in a lower-dimensional space is smaller simply because it covers fewer dimensions.
- Instead, split the real rows by `passage_id`, select using one half, and measure the full 768-dim covariance distance to the other half. Held-out scoring exposes overfitting to the noise in Σ_t.
- The final arbiter is QA performance on the validation split.

**Ablation result (n_s = 3,544).** Each cell is the relative distance ‖Cov(selected) − Cov(real)‖_F / ‖Cov(real)‖_F in the full 768-dim space. Lower is better. The held-out rows select using half the passages and evaluate on the other half (three passage-grouped splits).

| Evaluation | k=16 | k=32 | k=64 | k=128 | random |
|---|---|---|---|---|---|
| Held-out split 0 | 0.891 | 0.856 | 0.832 | 0.803 | 0.963 |
| Held-out split 1 | 0.882 | 0.860 | 0.841 | 0.816 | 0.969 |
| Held-out split 2 | 0.857 | 0.836 | 0.820 | 0.800 | 0.966 |
| **Held-out mean** | 0.877 | 0.851 | 0.831 | **0.806** | 0.966 |
| In-sample (all 886) | 0.718 | 0.688 | 0.664 | **0.639** | 0.959 |

- A larger k was better in every split, even on held-out passages. In that test only about 443 real rows (88 passages) were used for selection. The approximation error of a small k outweighs the extra estimation noise of a larger one, and k=32 is worse than the default.
- Much of the absolute distance (~0.8 held out) is the noise floor of a 768-dim covariance estimated from about 443 rows, which no selection can remove. The ordering of k and the gap to random are what matter.
- k=128 is worth trying alongside k=64. Choose between them using QA validation performance.

### 4.2 Number of selected rows `n_s` (default: 2×, 4× and 8× n_t)

There is no theoretical optimum for n_s:
- The paper's 4× ratio (200 real + 800 synthetic per class) is an experimental choice. Other experiments use 2× and 3×, and Table 11 varies n_s.
- In the linear theory, more covariance-matched synthetic data keeps lowering the error (Figure 1b), with diminishing returns.
- Under **model shift** (Appendix B), where synthetic labels follow a different relationship than real ones, more synthetic data can **hurt**. For LLM-generated QA, differences in question style or answer format are this kind of shift. The paper's advice is to reduce the synthetic set until performance starts to drop.
- Selection only has an effect when n_s is much smaller than the pool. Selecting all 33k unique rows is no selection at all.

The default `--multiples 2 4 8` produces 1,772, 3,544 and 7,088 rows from a single greedy run. Train with each size (plus the 886 real rows) and choose using the **validation** split.

---

## 5. Usage

Run from `hadl_courseproject/` (the script resolves its paths relative to its own location, so any working directory works):

```bash
python covariance_matching/covariance_matching.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--pca-dim K` | `64` | PCA dimension, fitted on the real rows. Must be ≤ n_t − 1. |
| `--multiples A B …` | `2 4 8` | Output sizes as multiples of n_t (rounded, capped at the pool size). |
| `--keep-duplicates` | off | Use all 37,058 draws as candidates instead of the unique rows. |
| `--recompute-real` | off | Re-embed the real rows even if the cache exists. |
| `--seed S` | `42` | Seed for the random-subset baseline in the summary. |

Examples:

```bash
python covariance_matching/covariance_matching.py --pca-dim 32
python covariance_matching/covariance_matching.py --multiples 1 2 4 8 16
```

Progress bars shown:
1. **Recipe parity check** and **Embedding real train rows** (first run, or with `--recompute-real`)
2. **Greedy covariance matching**, showing the current Frobenius distance as `frob=…`
3. **Writing selections**, one line per output size with the covariance-matching and random-subset distances

Requirements: `numpy`, `pandas`, `torch`, `tqdm`, `pyarrow`. Embedding the real rows also needs `transformers` and `huggingface_hub` (the e5 model is about 1.1 GB and is downloaded on first use). On CPU the real embedding takes a few minutes; later runs use the cache.

---

## 6. Outputs (`outputs/`)

| File | Contents |
|---|---|
| `covariance_matched_n{N}.parquet` | The first N greedy picks, in selection order |
| `greedy_order_candidate_idx.npy` | Full greedy order, as indices into the de-duplicated candidate table |
| `summary.json` | Run settings and covariance distances |

### Parquet columns

The output keeps every column of `final_weighted_train_draws.parquet` (`paragraph`, `question`, `answer`, `wiki_id`, `title`, `url`, `_source_row_idx`, `_wiki_group`, `draw_id`, `predicted_loss`, `residual_sq`, `sampling_probability`, `draw_weight_1_over_s_p`, `minibatch_importance_weight`) and adds:

| Column | Meaning |
|---|---|
| `pool_draw_row` | Row index in `final_weighted_train_draws.parquet` (first occurrence when de-duplicated) |
| `draw_count` | How many times the low-rank sampler drew this row (always 1 with `--keep-duplicates`) |
| `cm_rank` | 0-based position in the greedy order |
| `cm_frobenius_after` | ‖Cov(S) − Σ_t‖_F, in the k-dim PCA space, after this row was added |

`_source_row_idx` points back to `data_and_scripts/odia_synth_cleaned/or_synthetic_filtered.parquet` and to the rows of `embedding_generation/odia_embeddings.pt`.

### `summary.json`

- `sigma_t_frobenius`: ‖Σ_t‖_F, a scale reference
- `full_pool_frobenius`: distance if all candidates were used
- for each size: `cm_frobenius` (covariance matching) and `random_frobenius` (mean of 5 random subsets of the same size)

---

## 7. Results (k = 64, de-duplicated pool)

All distances are measured in the 64-dim real-PCA space. ‖Σ_t‖_F = 0.01977.

| n_s | Covariance matching | Random subset | Whole pool (33,104) |
|---|---|---|---|
| 1,772 (2×) | **0.00694** (35% of ‖Σ_t‖) | 0.01338 (68%) | 0.01325 (67%) |
| 3,544 (4×) | **0.00780** (39%) | 0.01334 (67%) | |
| 7,088 (8×) | **0.00889** (45%) | 0.01328 (67%) | |

How to read the table:
- A random subset is no closer to the real covariance than the whole pool. The low-rank pool is therefore systematically shifted in covariance from the real data, which fits a sampler that prefers high-loss, off-subspace rows.
- Covariance matching removes roughly half of that shift.
- The distance grows with n_s because there are only so many well-matched rows. Later picks are increasingly forced to be compromises. This is the diversity/fit trade-off to weigh when choosing n_s (§4.2).

These are distribution-matching diagnostics, not model results. Whether the selected data improves QA has to be measured by fine-tuning on real + selected data and evaluating on the validation split.

---

## 8. Limitations

- **Linear theory, nonlinear model.** The guarantees are for min-norm linear regression on a single mixture component. The paper reports that the insight carries over to deep networks, but that is an empirical finding.
- **The covariance is only matched inside the k-dim subspace.** Directions outside the real-data PCA basis are not controlled.
- **Effective real sample size is small.** With 176 distinct passages, Σ_t is noisy. In the held-out ablation (§4.1) this did not stop a larger k from helping up to k=128, but the real rows bound how far k can go (k ≤ n_t − 1).
- **Model shift is not addressed.** Covariance matching aligns the inputs (question+context), not how answers relate to them. Synthetic answers must still be made extractive (strip the trailing `।` and locate the answer's start in the passage) before training.
- **The embeddings exclude the answer.** Selection does not look at answer quality.
- **Greedy is not globally optimal.** The paper's look-ahead and Hungarian variants (Table 12) performed on par with greedy, so they are not implemented here.

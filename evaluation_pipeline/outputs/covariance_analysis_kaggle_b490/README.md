# Covariance-shift analysis of the Kaggle 2xT4 run (budget 490 s, seed 42)

How close each method's selected synthetic subset is to the covariance of the real IndicQA
train data (886 rows). Value = ||Cov(subset) − Cov(real)||_F / ||Cov(real)||_F; **lower = closer**.
"PCA" = the 64-dim real-data PCA space that covariance matching optimises; "full" = all 768 dims.

| Subset | n | PCA | full |
|---|---|---|---|
| Covariance matching only | 1,657 | **0.348** | **0.663** |
| LR + CM (no time proxy) — `quality_only` | 1,472 | 0.429 | 0.719 |
| Proposed: LR + CM + time knapsack | 2,715 | 0.459 | 0.759 |
| Random selection | 1,850 | 0.684 | 0.988 |
| Low-rank only | 1,762 | 0.789 | 1.312 |
| *ref.* whole low-rank pool | 33,104 | 0.670 | 0.948 |
| *ref.* all 7,088 CM candidates | 7,088 | 0.450 | 0.692 |
| *ref.* CM top 1,772 / 3,544 | | 0.351 / 0.395 | 0.662 / 0.664 |

Files: `covariance_diagnostic.csv/.md`, `covariance_shift_b490.png/.pdf`.

## How these numbers were produced
* **Subsets** — rebuilt with `scripts/01_build_pool.py` + `02_select_subsets.py` from the time-proxy
  coefficients printed in the Kaggle log (`time_proxy_kaggle_t4_reconstructed.json`; batch 2,
  budget = 1.0 × real-train proxy time = 490 s). The counts, mean token lengths and total utility
  of all five subsets match the Kaggle log to 6 decimals, i.e. they are the subsets that were trained.
* **Synthetic embeddings** — `embedding_generation/odia_embeddings.pt` (216,394 × 768) from the
  team Drive; `final_weighted_train_draws_embeddings.pt` from the same Drive equals it indexed by
  `_source_row_idx` (verified).
* **Real embeddings** — computed with Vanshika's `covariance_matching.load_or_embed_real()`
  (multilingual-e5-base, same recipe); its parity check re-embedded 8 stored rows with cosine 1.000.
* **Cross-check** — the reference rows reproduce `covariance_matching/outputs/summary.json`:
  whole pool 0.013246/0.019765 = 0.670, CM top 1,772 = 0.351, top 3,544 = 0.395.
* Script: `python evaluation_pipeline/scripts/07_covariance_diagnostic.py --outputs <run outputs>`.

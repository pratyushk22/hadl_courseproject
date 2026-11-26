# Hardware-Aware Synthetic Data Selection

**Low-rank scoring + covariance matching + a training-time proxy, combined in a budgeted knapsack selector.**
Course project for *Hardware-Aware Deep Learning*, IIT Guwahati.

Synthetic data is cheap to generate, but GPU time is not. Given a large pool of synthetic training examples and a fixed GPU budget, this project picks the subset that is most useful **per second of training time on the actual hardware**:

1. **Low-rank usefulness**: a predicted-loss and residual score from a low-rank model of the data ([Cohen-Addad et al., ICML 2026](https://proceedings.mlr.press/v306/cohen-addad26a.html)).
2. **Covariance matching**: prefers examples that make the synthetic set's embedding covariance resemble the real data ([Rezaei et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/97c5b2707228e7e3fb67e4ecc2e0e607-Abstract-Conference.html)).
3. **Training-time proxy**: a per-example cost model fitted by profiling real LoRA training steps on the target GPU:
   `T_step(B, L) = β0 + β_B·B + β_BL·B·L + β_BL²·B·L²`, giving each example cost `c_i = T_step(B, L_i) / B`.
4. **0/1 knapsack**: maximise total usefulness subject to total predicted training time ≤ budget.

We evaluate on **Odia extractive question answering** by fine-tuning **Qwen3-1.7B** with LoRA on NVIDIA T4 GPUs.

## Results

All methods get the same predicted synthetic training budget. Values are on the 176-question held-out test split, averaged over two runs of the full pipeline (490 s and 527 s predicted budgets, with the time proxy re-profiled each time).

| Method | Synthetic examples | Exact match | Token F1 | Char F1 | Train time |
|---|---:|---:|---:|---:|---:|
| Real data only | 0 | 23.58 | 44.94 | 59.17 | 33.2 min |
| Random | 1,854 | 26.71 | 51.94 | 64.25 | 72.5 min |
| Low-rank + covariance (quality only) | 1,475 | 27.28 | 51.12 | 64.01 | 73.1 min |
| **Low-rank + covariance + time budget (ours)** | **2,718** | **30.12** | **52.24** | **64.28** | 74.0 min |

- **Best exact match in every run**, 3.41 points above random selection each time.
- **About 47% more synthetic examples in the same budget**, by preferring shorter examples (556 vs 815 tokens on average), at near-identical training time.
- The time proxy ranks micro-batch costs accurately (**Spearman ρ ≈ 0.97**).

`evaluation_pipeline/outputs/` contains the full outputs of the 527 s run: tables, figures, logs and LoRA adapters.

## Repository layout

| Folder | Contents |
|---|---|
| [`data_and_scripts/`](data_and_scripts) | IndicQA Odia cleaning, the article-disjoint train / validation / test split (886 / 195 / 176), and the filtered IndicRAGSuite synthetic pool |
| [`embedding_generation/`](embedding_generation) | multilingual-e5-base embeddings for real and synthetic examples |
| [`low_rank_method/`](low_rank_method) | Low-rank usefulness scoring (landmark losses, residuals, sampling probabilities) |
| [`covariance_matching/`](covariance_matching) | Greedy covariance-matching ranking of 7,088 candidates ([README](covariance_matching/README.md)) |
| [`performance_profiling/`](performance_profiling) | GPU step-time profiling and the fitted time proxy |
| [`model_training/`](model_training) | Standalone Qwen3-1.7B + LoRA training and inference scripts |
| [`evaluation_pipeline/`](evaluation_pipeline) | **End-to-end pipeline**: cost the pool, select subsets, train, evaluate, aggregate and plot ([README](evaluation_pipeline/README.md)) |

## Quick start

The whole experiment runs from one notebook on Kaggle's free 2×T4 GPUs (about 3 hours):

1. Import [`evaluation_pipeline/notebooks/run_pipeline_kaggle.ipynb`](evaluation_pipeline/notebooks/run_pipeline_kaggle.ipynb) into Kaggle.
2. Set **Accelerator: GPU T4 x2** and **Internet: On**.
3. **Save & Run All**. Tables and figures appear in the notebook's Output tab.

There is also a Colab version, [`run_pipeline_colab.ipynb`](evaluation_pipeline/notebooks/run_pipeline_colab.ipynb). On any GPU machine, run:

```bash
pip install -r evaluation_pipeline/requirements.txt
PROFILE=1 SEEDS="42" REFERENCE="real_only" bash evaluation_pipeline/run_pipeline.sh
```

`PROFILE=1` re-profiles the time proxy on your GPU, which is the hardware-aware part. The CPU-only steps (building the costed pool, selection, tables and plots) finish in about 2 minutes. All options are documented in the [pipeline README](evaluation_pipeline/README.md).

## Evaluation

- **Metrics:** exact match, token F1 (main metric) and character F1, with Odia-aware normalisation of digits, punctuation and zero-width characters.
- **Confidence intervals:** paired bootstrap clustered by passage, compared against random selection on the same questions.
- **Proxy accuracy:** predicted vs measured time, both per micro-batch and per full run.
- **Covariance shift:** ‖Cov(subset) − Cov(real)‖_F / ‖Cov(real)‖_F for every selected subset.

## Data and models

- Real data: [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA), Odia portion (Doddapaneni et al., ACL 2023)
- Synthetic pool: [IndicRAGSuite](https://arxiv.org/abs/2506.01615), Odia portion: 216,394 rows after filtering
- Embeddings: [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- Model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) with LoRA (rank 8, scaling 16), one epoch, FP16 on T4

## Team

Pratyush Kumar Sethy, Aditya Jain, Vanshika Mittal, Yatika Jena, Aditya Sahu and Rahul Yadav, IIT Guwahati.

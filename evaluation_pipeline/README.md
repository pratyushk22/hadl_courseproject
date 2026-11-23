# Evaluation pipeline: integration, metrics and plots

This folder connects everyone's work into one reproducible pipeline and adds the evaluation and reporting layer:

```
Pratyush            Aditya Jain          Vanshika              Yatika              Aditya Sahu
real/synthetic  ->  low-rank scores  ->  covariance-matching -> time proxy      ->  Qwen3-1.7B + LoRA r=8
data splits         (LR pool)            ranking               c_i per example      training recipe
        \_______________________________________|______________________|__________________/
                                                v
                           evaluation_pipeline/ (this folder)
   00 re-profile (optional) -> 01 costed pool -> 02 budgeted selection (5 methods) -> 03 train -> 04 evaluate
   -> 07 covariance-shift diagnostic -> 05 tables -> 06 figures
```

Steps 1, 2, 5 and 6 run on a laptop CPU in about 2 minutes. Steps 0, 3 and 4 need a GPU (Kaggle 2×T4, or Colab A100/L4/T4). Step 7 downloads ~780 MB from the team Drive and needs a few minutes.

---

## 1. Quick start: Kaggle (free 2×T4)

1. Open https://www.kaggle.com/code → **New Notebook** → **File → Import Notebook**, and give it this link:
   `https://github.com/pratyushk22/hadl_courseproject/blob/main/evaluation_pipeline/notebooks/run_pipeline_kaggle.ipynb`
   If the link option is not offered, download that file from GitHub and upload it instead. The notebook itself clones the code from `main` of `pratyushk22/hadl_courseproject`.
2. In the right-hand panel, open **Settings**:
   - **Accelerator:** GPU T4 x2
   - **Internet:** On (needs a phone-verified Kaggle account)
   - **Persistence:** Files only
   - **Environment:** Pin to original environment
3. **Save Version → Save & Run All (Commit)**. This runs unattended for up to 12 h. Our run took about 3 h.
4. Results appear in the notebook's **Output** tab: `outputs/`, `hadl_results.zip` and `pipeline.log`. The notebook also displays the main tables and figures at the end, including the covariance diagnostic.

The notebook detects the hardware and sets these automatically:

| Setting | Value on 16 GB GPUs | Why |
|---|---|---|
| `GPUS=2` | — | Two single-GPU processes, each training or evaluating different runs. Every run is still timed on one GPU, as the proxy assumes. |
| `BATCH=2 ACCUM=8` | effective batch 16 | Fits 16 GB. |
| `EVAL_BS=4` | — | Fits 16 GB. |
| dtype | fp16 | The T4 has no native bf16 (see §5, item 10). |

Versions are pinned to ones tested with this pipeline: `transformers 4.57.*`, `peft 0.22.*`. pip prints dependency conflicts for `gradio`, `diffusers` and `pillow`; they are harmless because those preinstalled packages are not used.

**Optional: pushing results to GitHub from Kaggle.**
- Add a Kaggle secret `GITHUB_TOKEN`: a GitHub token with *Contents: read & write* on `pratyushk22/hadl_courseproject`.
- Set `PUSH_RESULTS_TO_BRANCH = True` in the first code cell.
- The results are pushed to a **separate branch** (`RESULTS_BRANCH = "kaggle-results"`), never to `main`.

### Colab

Open `notebooks/run_pipeline_colab.ipynb` in Colab (A100 recommended) and run all cells. On a 16 GB GPU, add `BATCH=2 ACCUM=8 EVAL_BS=4` to the run command, as described in the notebook.

### The command both notebooks run

```bash
PROFILE=1 SEEDS="42" REFERENCE="real_only" bash evaluation_pipeline/run_pipeline.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE=1` | off | Re-profile the LoRA step time **on this GPU** and fit a new time proxy. Recommended, because the committed proxy was profiled on Apple MPS. |
| `BUDGET_X_REAL` | `1.0` | Synthetic training-time budget, as a multiple of the proxy time of one epoch over the 886 real training rows. |
| `BUDGET_SEC` | — | Absolute budget in proxy seconds (overrides `BUDGET_X_REAL`). |
| `SEEDS` | `42` | Seeds for training and for the random baseline. Use `"42 43 44"` for error bars. |
| `METHODS` | the three headline methods | Set `all` to add the `low_rank_only` and `cov_match_only` ablations. |
| `REFERENCE` | — | `real_only` also fine-tunes on the real data alone. |
| `GPUS` | `1` | Use `2` on Kaggle 2×T4. The job list is split across GPUs, with one single-GPU process each. |
| `BATCH`, `ACCUM` | `4`, `4` | Use `2`, `8` on 16 GB GPUs. The effective batch is the same, and the proxy cost uses `BATCH`. |
| `DTYPE` | `auto` | `auto` picks bf16 on GPUs with native bf16 (A100/L4/H100) and fp16 otherwise (T4/P100). Use `fp32` if fp16 gives NaN losses. |
| `EVAL_BS` | `8` | Evaluation batch size. Use 4 on a T4. |
| `MODEL_ID`, `TRAIN_EXTRA`, `EVAL_EXTRA` | — | Model override and extra arguments for scripts 03 and 04 (debugging; used by the smoke test). |
| `STEPS` | `0 1 2 3 4 5 6 7` | Which steps to run. They always execute in the order 0 1 2 3 4 7 5 6. For example, `STEPS="5 6"` only re-aggregates and re-plots. |

Finished runs and evaluations are skipped automatically, so re-running the command in the same session resumes where it stopped. A new Kaggle commit run starts from scratch.

### Step by step (any machine)

```bash
cd evaluation_pipeline
pip install -r requirements.txt
python scripts/01_build_pool.py               # CPU, ~90 s:  integrate LR + CM + time proxy
python scripts/02_select_subsets.py --sweep   # CPU, ~2 s: one subset per method
python scripts/03_train.py --reference_runs real_only   # GPU
python scripts/04_evaluate.py                 # GPU: base model + every adapter, val + test
python scripts/07_covariance_diagnostic.py    # covariance shift of every subset (downloads embeddings)
python scripts/05_aggregate.py                # CPU: tables + significance tests + proxy error
python scripts/06_make_plots.py               # CPU: all figures (PNG + PDF)
```

Every script has `--help`. Step 0 (re-profiling) is only run through `run_pipeline.sh`.

---

## 2. What is compared

All methods get **the same synthetic training-time budget B** (proxy-predicted seconds per epoch). They are trained with the same recipe on **real train (886 rows) + the selected synthetic subset**, and evaluated on the real IndicQA Odia **validation (195)** and **test (176)** splits.

| Run | Selection rule | Role |
|---|---|---|
| `base_zero_shot` | No fine-tuning | Reference |
| `real_only` | Real data only | Reference (optional) |
| `random` | Uniformly random over the full 185k synthetic train pool; prefix fill | Baseline 1 |
| `quality_only` | Composite usefulness `u_i` (LR + CM), highest first, **cost ignored**; prefix fill | Baseline 2: "the two methods combined" |
| `proposed_knapsack` | **max Σu_i subject to Σc_i ≤ B** (0/1 knapsack), with `c_i` from the time proxy | **Our method** |
| `low_rank_only` | Low-rank score only | Ablation |
| `cov_match_only` | Greedy covariance-matching order only | Ablation |

**How the scores are integrated** (`hadl_eval/pool.py`):

- `lr_score` is the low-rank stage's `sampling_probability`, which is proportional to `max(pred_loss,0) + λ·residual²`. `lr_pct` is its percentile rank.
- `cm_score = 1 − cm_rank / 7088`, from the greedy covariance-matching order. Rows that are not ranked get 0.
- `u_i = α·lr_pct + (1−α)·cm_score`, with α = 0.5 (`--alpha` in step 1).
- `c_i` = Yatika's proxy, `T_step(B, L_i)/B`. Here `L_i` is the **exact tokenised training length** (prompt + answer, after truncation) and `B` the micro-batch size.
- `quality_only` and `proposed_knapsack` choose from the same 7,088 covariance-matched candidates. The only difference between them is whether the time cost is used, which isolates the contribution of the time proxy.

**Knapsack solver** (`hadl_eval/knapsack.py`): an exact DP on costs discretised into 2,000 units (rounded **up**, so the solution is always feasible), plus a greedy u/c pass. The better of the two is kept, and any remaining budget is filled greedily. The tests check it against brute force.

**Budget choice.** Absolute seconds depend on the GPU the proxy was profiled on. The default budget is therefore `1.0 × (proxy time of the real train split)`. This was 490 s on the Kaggle T4s (fp16) and 8,605 s with the committed Apple-MPS proxy. In both cases the methods select roughly 1.5k–3.5k examples.

---

## 3. Metrics

### QA quality (`hadl_eval/metrics.py`), per example, taking the maximum over gold answers

| Metric | Definition |
|---|---|
| **Exact match (EM)** | Normalised strings are equal. Normalisation: NFC, remove zero-width characters, Odia/Devanagari digits → ASCII, remove punctuation including `।`, collapse spaces. |
| **Token F1** | SQuAD-style overlap of whitespace tokens. This is the main metric. |
| **Character F1** | Overlap of characters. It is robust to Odia suffixes (`ଭୁବନେଶ୍ୱର` vs `ଭୁବନେଶ୍ୱରରେ`). |
| **Answer contained** | The normalised gold answer appears inside the prediction. |

- **95% CIs:** bootstrap resampling **clustered by passage**, because questions about the same passage are correlated.
- **Significance:** a paired, passage-clustered bootstrap of each run against the random baseline on the same examples (difference, CI, two-sided p).

### Training-time proxy

| Level | What is compared |
|---|---|
| Run level | Proxy-predicted total vs measured wall-clock training time: MAPE, signed error, R², Spearman. Also a **leave-one-out calibrated MAPE**, which shows the proxy error once its hardware constant is fitted. |
| Micro-batch level | Every forward+backward is timed (CUDA-synchronised) together with its real `(B, padded L)`. This tests the functional form of the proxy, not only its total. |

### Covariance shift (step 7, `scripts/07_covariance_diagnostic.py`)

For every selected subset, and for reference sets (the whole low-rank pool and the covariance-matched candidates: all, top 1,772, top 3,544):

**‖Cov(subset) − Cov(real)‖_F / ‖Cov(real)‖_F**, where lower means closer to the real training data.

It is computed in two spaces:
- **PCA space**: the 64-dim space of the real embeddings that covariance matching optimises;
- **full space**: all 768 dims, which also covers the directions covariance matching does not control.

The columns `cov_shift_pca` and `cov_shift_full` are added to `results_table.csv`.

Where the embeddings come from (all `*.pt` files are git-ignored):

| File | Source |
|---|---|
| `embedding_generation/odia_embeddings.pt` (216,394 × 768, ~665 MB) | **Downloaded from the team Google Drive** (`gdown`) |
| `low_rank_method/odia_low_rank_run/final_weighted_train_draws_embeddings.pt` (~114 MB) | **Downloaded from the team Drive**; needed for the recipe parity check below |
| `embedding_generation/odia_real_train_embeddings.pt` (886 × 768) | Not in the Drive, so it is **created with Vanshika's own `covariance_matching.load_or_embed_real()`**. Same e5 model and recipe. It aborts unless re-embedding 8 stored rows gives cosine ≥ 0.99. |

The Drive file ids are in `hadl_eval/config.py` (`DRIVE_FILE_IDS`). Existing files are reused. If the step fails (for example, the Drive temporarily refuses automated downloads), the pipeline prints the error and carries on, because this step is optional. Re-run it alone with `STEPS="7 5 6" bash run_pipeline.sh`.

### Efficiency

- Selection time per method
- Proxy-scoring time per example
- Training wall-clock time, throughput (tokens/s), peak GPU memory, padding fraction
- Generation time per example
- One-off stage costs for embeddings, low-rank scoring and covariance matching. Enter these in `configs/overhead_times.json`; they appear in the overhead plot.

---

## 4. Outputs

```
outputs/
  pool/            candidate_pool.parquet, random_pool.parquet, real_train.parquet, pool_report.json   (git-ignored, regenerable)
  profiling/       profiling_results.csv, time_proxy_model.json   (when PROFILE=1)
  selections/      budgets.json, budget_sweep.csv, b<B>/<run_id>.parquet, b<B>/selection_summary.json
  runs/<run_id>/   adapter/, training_metrics.json, log_history.json, step_timings.csv
  eval/<run_id>/   {validation,test}_predictions.jsonl, {validation,test}_metrics.json
  results/         results_table.{csv,md}, results_by_method.{csv,md}, significance_vs_random.{csv,md},
                   proxy_runs.csv, proxy_error_summary.json, overhead_components.csv, selection_summary.csv,
                   covariance_diagnostic.{csv,md}
  figures/         *.png (200 dpi) + *.pdf (vector, for the report)
  logs/            per-GPU training / evaluation logs, covariance_diagnostic.log
```

Run ids look like `proposed_knapsack_b490_s42`, meaning method, budget in seconds, and seed.

| Figure | Question it answers |
|---|---|
| `qa_scores_{split}` | Which method answers best? EM / F1 / char-F1 with 95% CIs. |
| `quality_vs_training_time_{split}` | Quality against the training time actually spent (Pareto view). |
| `token_f1_vs_budget_{split}` | How quality scales with the budget (needs ≥ 2 budgets). |
| `token_f1_by_answer_length_{split}` | Where the methods differ (short vs long answers). |
| `training_time_predicted_vs_actual` | How accurate the time proxy is per run. |
| `proxy_step_parity` | Proxy vs measured time per micro-batch, and where it is off. |
| `proxy_profile_fit` | The proxy curves on the profiling grid. |
| `training_loss_curves` | Training dynamics. |
| `selection_overview_b<B>` | What each method selects: count, Σu, length, diversity. |
| `selection_token_lengths_b<B>` | Length distribution of the selected examples. |
| `utility_vs_cost_b<B>` | Why the knapsack picks more examples (high u/c). |
| `budget_sweep_total_utility`, `budget_sweep_num_selected` | The selection objective across budgets (no GPU needed). |
| `covariance_shift_b<B>` | Covariance distance of each method's subset to the real data (step 7). |
| `pipeline_overhead` | Selection pipeline overhead plus training cost. |

**What is committed in `outputs/`:**
- `selections/` and `figures/` come from an early **CPU dry run with the committed Apple-MPS proxy** (budget `b8605`). They illustrate the selection step only; no model was trained on these subsets.
- `covariance_analysis_kaggle_b490/` holds the covariance diagnostic of the subsets that **were** trained in the Kaggle run below. See its own README for how it was produced.

---

## 5. Results of the Kaggle 2×T4 run (seed 42, budget 1.0× = 490 s, fp16)

| Run (test split, 176 questions) | EM | Token F1 | Char F1 | Actual training time | Cov. shift (PCA) |
|---|---|---|---|---|---|
| Base model, no training | 9.1 | 30.4 | 47.6 | — | — |
| Real data only | 22.7 | 44.0 | 57.5 | 30 min | — |
| Random selection | 26.1 | 51.1 | 63.5 | 73 min | 0.684 |
| LR + CM (no time proxy) | 26.1 | 50.2 | 63.9 | 67 min | 0.429 |
| **Proposed knapsack** | **29.6** | **52.2** | **64.7** | 74 min | 0.459 |

What these results show:
- **Fine-tuning and synthetic data help significantly.** Every fine-tuned run beats the base model (+14 to +22 F1). Random synthetic data beats real-only by +7.1 F1 (p = 0.03, paired passage-clustered bootstrap).
- **The proposed method is best on every metric**, on both test and validation (validation F1: 52.4 vs 52.3 random vs 50.1 LR+CM). The differences between the three selection methods are **not statistically significant** with one seed: proposed vs random gives +1.1 F1 on test, p = 0.52. More seeds (`SEEDS="42 43 44"`) are needed to separate them.
- **Time proxy.** Actual training took 4.3× the predicted time (MAPE 76%). The proxy profiles 2 decoder layers × 14 and leaves out the 151k-vocabulary output layer, gradient-checkpointing recomputation and the optimizer. It still **ranks costs almost perfectly**: Spearman 1.0 across runs and 0.97 across 4,791 micro-batches, and the leave-one-out calibrated MAPE is about 10%. Equal *predicted* budgets were not exactly equal in *actual* time: the knapsack's many short examples took 11% longer than LR+CM.
- **Covariance shift.** Covariance matching halves the distance to the real data compared with random (0.348 for CM-only vs 0.684). Low-rank selection alone moves *away* from it (0.789). The knapsack keeps most of the matching (0.459 vs 0.429 without the time proxy). In this run, the subset with the closest covariance was not the one with the highest F1.

---

## 6. Integration issues found and fixed

These were found while connecting the stages. The teammates' files are left untouched, and the corrected logic lives in `hadl_eval/` and `run_pipeline.sh`.

1. **Placeholder scores.** `model_training/prepare_training_subsets.py` looks for `low_rank_score` / `cov_match_score` columns that no stage writes. It therefore fell back to random "synthetic_fallback" scores (see `subset_selection_summary.json`). This pipeline uses the real LR probabilities and the CM ranking.
2. **Empty passages.** The synthetic files call the passage `paragraph`, but `assign_time_costs.py` and `train_qwen_lora.py` read `context`. That silently gave an **empty passage**, so models were trained without context and costs were underestimated. All data now goes through `hadl_eval/data.py` with one schema.
3. **Truncation cut the question.** Training truncated `prompt_ids[:max_len]`, which cuts the *end* of the prompt: the question and the assistant header. Now only the passage is shortened, and for training the kept window contains the answer.
4. **512-token limit removed the answer.** Odia costs about 2 Qwen tokens per character. Real test passages have a median of ~1,300 tokens and a maximum of ~3,700. Evaluation now uses 4,096 input tokens, so no passage is cut. Training uses 2,048, which is the proxy's cap.
5. **Different train and eval prompts.** Training and evaluation used different system prompts. Both now use `hadl_eval/prompts.py`.
6. **Synthetic answer format.** Synthetic answers always end with `।`, while IndicQA answers are bare spans. The trailing punctuation is now stripped from targets.
7. **fp16 LoRA weights.** fp16 training on T4 needs the LoRA weights in fp32 (otherwise "Attempting to unscale FP16 gradients"). The weights are now cast.
8. **Dual-GPU split.** `device_map="auto"` on 2×T4 splits the model across GPUs and distorts timings. Now each process uses exactly one GPU, and two GPUs run different jobs in parallel (`GPUS=2`).
9. **Evaluation size.** Evaluation previously used 10 validation examples with batch size 1. It now runs the full splits batched, with CIs.
10. **Emulated bf16 on T4.** `torch.cuda.is_bf16_supported()` also returns True on T4/P100, where bf16 is only emulated and much slower.
    - Training (`hadl_eval/modeling.py`) uses bf16 only on GPUs with compute capability ≥ 8.0, and fp16 otherwise.
    - For profiling, `run_pipeline.sh` runs Yatika's `performance_profiling/profile_hardware.py` **unchanged** through `runpy`. It first reports bf16 as unsupported on such GPUs, so profiling uses the same fp16 as training.
    - In the Kaggle run this cut the proxy's real-data epoch time from 5,028 s (emulated bf16) to 490 s.

The candidate pool was also checked for leakage. **0** of the 33,104 candidates share an 8-word shingle with a real validation or test passage, and 1 of the 30,000 random-pool rows did and was dropped.

---

## 7. GPU time

Measured in the Kaggle 2×T4 run above (default settings, 4 training runs + 5 evaluated models): **about 3 h in total.**

| Part | Time |
|---|---|
| Each 3-method training run | 67–74 min |
| `real_only` training | 30 min |
| Evaluation per model (validation + test) | ~8 min |
| Covariance diagnostic (step 7) | a few minutes: download ~780 MB, embed 886 rows |

On a single A100, expect each training run to take noticeably less time; this is an estimate, not measured.

Troubleshooting:

| Problem | Fix |
|---|---|
| OOM during training | `--max_seq_len 1536` or `--batch_size 2 --grad_accum 8`. Keep the same proxy batch size via `01_build_pool.py --batch_size`. |
| OOM during evaluation | `EVAL_BS=2`. The OOM handler also splits batches automatically. |
| NaN loss with fp16 | `DTYPE=fp32`. |
| No internet for the tokenizer | The Qwen3 tokenizer copy in `model_training/checkpoints/*/final_adapter` is used automatically. |
| Step 7: Drive download refused | Retry later with `STEPS="7 5 6" bash run_pipeline.sh` (same session), or place the two `.pt` files at the paths in §3 manually. |

---

## 8. Code layout and tests

```
hadl_eval/   config.py (paths, hyper-parameters, colours, Drive file ids) · data.py · prompts.py · text_utils.py
             time_proxy.py · pool.py · knapsack.py · selection.py · diagnostics.py · embedding_files.py
             modeling.py · training.py · inference.py · metrics.py · results.py · plotting.py
scripts/     01_build_pool · 02_select_subsets · 03_train · 04_evaluate · 05_aggregate · 06_make_plots
             07_covariance_diagnostic
run_pipeline.sh   runs everything (incl. step 0 re-profiling and step 7)
tests/       unit tests (pytest) + smoke_end_to_end.sh (steps 1-6 on CPU with a tiny random Qwen3)
notebooks/   run_pipeline_kaggle.ipynb · run_pipeline_colab.ipynb
configs/     overhead_times.json
```

```bash
python -m pytest -q tests                      # unit tests, ~5 s
bash tests/smoke_end_to_end.sh                 # full pipeline on CPU with a tiny model, incl. the 2-GPU split, ~5 min
```

The smoke test runs training, evaluation, aggregation and plotting end to end without a GPU. Its scores are meaningless, because the tiny model is random. It exists to check that every code path executes. Step 7 is left out of the smoke test because it downloads ~780 MB. Set `HADL_OUTPUT_ROOT` to write outputs elsewhere, as the smoke test does.

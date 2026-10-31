"""Covariance matching (Rezaei et al., ICLR 2026, Sec. 5) on the Odia low-rank synthetic pool.

Greedy rule from the paper: start from S = {} and, until |S| = n_s, add the pool point x
minimising || Cov(S + {x}) - Cov(real) ||_F, with covariances computed in a k-dim PCA
space fit on the real reference embeddings only.

Setting used here:
  * real reference  = IndicQA Odia *train* split (886 rows), never validation/test
  * synthetic pool  = low_rank_method/odia_low_rank_run/final_weighted_train_draws*.{pt,parquet}
  * one global class (extractive QA has no labels to split on)
  * pool de-duplicated by _source_row_idx (draws were with replacement)

Run from anywhere:
    python covariance_matching/covariance_matching.py
    python covariance_matching/covariance_matching.py --pca-dim 64 --multiples 2 4 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_EMB_PATH = PROJECT_ROOT / "low_rank_method/odia_low_rank_run/final_weighted_train_draws_embeddings.pt"
POOL_DF_PATH = PROJECT_ROOT / "low_rank_method/odia_low_rank_run/final_weighted_train_draws.parquet"
REAL_TRAIN_PATH = PROJECT_ROOT / "data_and_scripts/odia_real_cleaned/train.jsonl"
REAL_EMB_PATH = PROJECT_ROOT / "embedding_generation/odia_real_train_embeddings.pt"
OUTPUT_DIR = PROJECT_ROOT / "covariance_matching/outputs"

# Must match embedding_generation/generate_odia_embeddings.ipynb exactly.
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LENGTH = 512


def build_text(question, context):
    return f"query: Question: {question.strip()}\nContext: {context.strip()}"


def embed_texts(texts, tokenizer, model, device, batch_size, desc):
    import torch.nn.functional as F

    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
            end = min(start + batch_size, len(texts))
            tokens = tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out[start:end] = F.normalize(pooled.float(), p=2, dim=1).cpu()
    return out


def load_or_embed_real(real_df, pool_df, pool_emb, recompute):
    if REAL_EMB_PATH.is_file() and not recompute:
        real_emb = torch.load(REAL_EMB_PATH, map_location="cpu", weights_only=True)
        if real_emb.shape[0] == len(real_df):
            print(f"Loaded cached real embeddings: {REAL_EMB_PATH} {tuple(real_emb.shape)}")
            return real_emb
        print("Cached real embeddings have the wrong row count; recomputing.")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32 if device.type == "cuda" else 8
    model_dir = snapshot_download(repo_id=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    # Recipe parity check: re-embed a few pool rows and compare with the stored vectors.
    probe = pool_df.head(8)
    probe_texts = [build_text(q, p) for q, p in zip(probe["question"], probe["paragraph"])]
    probe_emb = embed_texts(probe_texts, tokenizer, model, device, batch_size, "Recipe parity check")
    cosine = (probe_emb * pool_emb[: len(probe)]).sum(dim=1)
    print("Parity cosine (re-embedded vs stored pool rows):", np.round(cosine.numpy(), 5))
    if cosine.min() < 0.99:
        raise RuntimeError("Embedding recipe does not reproduce the pool embeddings; aborting.")

    texts = [build_text(q, p) for q, p in zip(real_df["question"], real_df["passage"])]
    real_emb = embed_texts(texts, tokenizer, model, device, batch_size, "Embedding real train rows")
    torch.save(real_emb, REAL_EMB_PATH)
    print(f"Saved real embeddings: {REAL_EMB_PATH}")
    return real_emb


def covariance(Z):
    """Mean-centred covariance with 1/n normalisation (same convention as the greedy updates)."""
    Zc = Z - Z.mean(axis=0)
    return Zc.T @ Zc / len(Z)


def greedy_covariance_matching(Z, sigma_t, n_select, first_index):
    """Greedy argmin_x ||Cov(S + {x}) - sigma_t||_F, scoring every candidate per step.

    With M = sum_{i in S} x_i x_i^T, s = sum_{i in S} x_i and m = |S|, the candidate covariance is
        Cov(S + {x}) = (M + x x^T) / (m+1) - u u^T / (m+1)^2,   u = s + x.
    Writing D = M/(m+1) - sigma_t, c = 1/(m+1), d = c^2, the squared Frobenius error is
        ||D||^2 + c^2 |x|^4 + d^2 |u|^4 + 2c x^T D x - 2d u^T D u - 2cd (x^T u)^2,
    and Z @ D is kept up to date with rank-one updates, so each step costs O(N k).
    """
    N, k = Z.shape
    sq_norm = np.einsum("ij,ij->i", Z, Z)
    Z_sigma = Z @ sigma_t                  # fixed part of Z @ D
    Z_M = np.zeros_like(Z)                 # Z @ M, updated in place
    M = np.zeros((k, k))
    s = np.zeros(k)
    available = np.ones(N, dtype=bool)
    order = np.empty(n_select, dtype=np.int64)
    frob_after = np.empty(n_select)

    progress = tqdm(range(n_select), desc="Greedy covariance matching", unit="row")
    for m in progress:
        if m == 0:
            # Cov of a single point is 0 for every candidate, so the objective ties; start
            # from the candidate nearest the real mean (the PCA origin).
            best = first_index
        else:
            c = 1.0 / (m + 1)
            d = c * c
            D = M * c - sigma_t
            ZD = Z_M * c - Z_sigma
            Ds = D @ s
            xDx = np.einsum("ij,ij->i", ZD, Z)
            sDx = Z @ Ds
            sx = Z @ s
            uDu = xDx + 2.0 * sDx + s @ Ds
            u_sq = sq_norm + 2.0 * sx + s @ s
            xu = sq_norm + sx
            score = (
                np.sum(D * D)
                + c * c * sq_norm**2
                + d * d * u_sq**2
                + 2.0 * c * xDx
                - 2.0 * d * uDu
                - 2.0 * c * d * xu**2
            )
            score[~available] = np.inf
            best = int(np.argmin(score))

        x = Z[best]
        order[m] = best
        available[best] = False
        M += np.outer(x, x)
        s += x
        Z_M += np.outer(Z @ x, x)
        cov_S = M / (m + 1) - np.outer(s, s) / (m + 1) ** 2
        frob_after[m] = np.linalg.norm(cov_S - sigma_t)
        if m % 25 == 0 or m == n_select - 1:
            progress.set_postfix(frob=f"{frob_after[m]:.5f}")
    return order, frob_after


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument(
        "--multiples", type=float, nargs="+", default=[2, 4, 8],
        help="Selection sizes as multiples of n_t; greedy is nested, so one run to the largest size gives every prefix.",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="Treat each of the 37,058 draws as a candidate.")
    parser.add_argument("--recompute-real", action="store_true", help="Re-embed the real train rows even if cached.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Synthetic pool -------------------------------------------------------------
    pool_df = pd.read_parquet(POOL_DF_PATH).reset_index(drop=True)
    pool_emb = torch.load(POOL_EMB_PATH, map_location="cpu", weights_only=True)
    if pool_emb.shape[0] != len(pool_df):
        raise ValueError(f"Pool embeddings ({pool_emb.shape[0]}) != pool rows ({len(pool_df)})")
    print(f"Pool draws: {len(pool_df):,}; unique synthetic rows: {pool_df['_source_row_idx'].nunique():,}")

    if args.keep_duplicates:
        cand_df = pool_df.copy()
        cand_df["draw_count"] = 1
    else:
        draw_count = pool_df["_source_row_idx"].map(pool_df["_source_row_idx"].value_counts())
        keep = ~pool_df["_source_row_idx"].duplicated()
        cand_df = pool_df.loc[keep].copy()
        cand_df["draw_count"] = draw_count[keep].values
    cand_emb = pool_emb[cand_df.index.values]
    cand_df = cand_df.reset_index().rename(columns={"index": "pool_draw_row"})
    print(f"Candidates for selection: {len(cand_df):,}")

    # ---- Real reference (train split only) -----------------------------------------
    real_df = pd.read_json(REAL_TRAIN_PATH, lines=True)
    real_emb = load_or_embed_real(real_df, pool_df, pool_emb, args.recompute_real)
    n_t = len(real_df)

    # ---- PCA fit on the real reference only ----------------------------------------
    R = real_emb.numpy().astype(np.float64)
    X = cand_emb.numpy().astype(np.float64)
    mu_t = R.mean(axis=0)
    _, sing, Vt = np.linalg.svd(R - mu_t, full_matrices=False)
    k = args.pca_dim
    if not 0 < k <= min(n_t - 1, R.shape[1]):
        raise ValueError(f"--pca-dim must be in [1, {min(n_t - 1, R.shape[1])}]")
    basis = Vt[:k].T
    Z_t = (R - mu_t) @ basis
    Z = (X - mu_t) @ basis

    real_var = sing**2
    pool_c = X - X.mean(axis=0)
    pool_retained = np.sum((pool_c @ basis) ** 2) / np.sum(pool_c**2)
    print(f"PCA dim k={k}: real variance retained {real_var[:k].sum() / real_var.sum():.3f}, "
          f"pool variance retained {pool_retained:.3f}; n_t/k = {n_t / k:.1f}, "
          f"unique passages/k = {real_df['passage_id'].nunique() / k:.1f}")

    sigma_t = covariance(Z_t)
    sigma_norm = np.linalg.norm(sigma_t)

    # ---- Greedy selection -----------------------------------------------------------
    sizes = sorted({min(int(round(mult * n_t)), len(cand_df)) for mult in args.multiples})
    n_select = sizes[-1]
    first_index = int(np.argmin(np.einsum("ij,ij->i", Z, Z)))
    order, frob_after = greedy_covariance_matching(Z, sigma_t, n_select, first_index)

    # ---- Reference distances (paper Table 10 "covariance shift") ---------------------
    rng = np.random.default_rng(args.seed)
    full_pool_frob = np.linalg.norm(covariance(Z) - sigma_t)
    summary = {
        "n_t": n_t,
        "n_candidates": len(cand_df),
        "pca_dim": k,
        "keep_duplicates": args.keep_duplicates,
        "sigma_t_frobenius": sigma_norm,
        "full_pool_frobenius": full_pool_frob,
        "sizes": {},
    }
    print(f"\n||Sigma_t||_F = {sigma_norm:.5f}; whole-pool distance = {full_pool_frob:.5f}")
    for n in tqdm(sizes, desc="Writing selections", unit="file"):
        random_frob = np.mean([
            np.linalg.norm(covariance(Z[rng.choice(len(Z), n, replace=False)]) - sigma_t)
            for _ in range(5)
        ])
        selected = cand_df.iloc[order[:n]].copy()
        selected["cm_rank"] = np.arange(n)
        selected["cm_frobenius_after"] = frob_after[:n]
        path = OUTPUT_DIR / f"covariance_matched_n{n}.parquet"
        selected.to_parquet(path, index=False)
        summary["sizes"][n] = {"cm_frobenius": float(frob_after[n - 1]), "random_frobenius": float(random_frob), "path": str(path)}
        tqdm.write(f"n_s={n:>6}: covariance matching {frob_after[n - 1]:.5f} | random subset {random_frob:.5f} -> {path.name}")

    np.save(OUTPUT_DIR / "greedy_order_candidate_idx.npy", order)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

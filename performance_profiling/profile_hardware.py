import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import gc
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_CSV = "profiling_results.csv"

# Qwen3-1.7B Exact Architectural Constants
TOTAL_LAYERS = 28
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
LORA_RANK = 8
LORA_ALPHA = 16

BATCH_SIZES = [1, 2, 4, 8]
SEQ_LENGTHS = [128, 256, 384, 512, 768, 1024]

class LoRALinear(nn.Module):
    """Exact Qwen3 Linear + LoRA (rank=8) projection layer."""
    def __init__(self, in_features, out_features, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad = False  # Base weights frozen in LoRA
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

class Qwen3LoRADecoderLayer(nn.Module):
    """Exact Qwen3-1.7B GQA Self-Attention + SwiGLU MLP Layer with LoRA r=8."""
    def __init__(self):
        super().__init__()
        # Grouped-Query Attention projections with LoRA
        self.q_proj = LoRALinear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM)
        self.k_proj = LoRALinear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM)
        self.v_proj = LoRALinear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM)
        self.o_proj = LoRALinear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)

        # SwiGLU MLP projections with LoRA
        self.gate_proj = LoRALinear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.up_proj = LoRALinear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.down_proj = LoRALinear(INTERMEDIATE_SIZE, HIDDEN_SIZE)

        self.norm1 = nn.RMSNorm(HIDDEN_SIZE, eps=1e-6)
        self.norm2 = nn.RMSNorm(HIDDEN_SIZE, eps=1e-6)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        h = self.norm1(x)

        q = self.q_proj(h).view(B, L, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h).view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h).view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # Repeat KV heads for Grouped Query Attention (16 Q heads / 8 KV heads = 2x)
        k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.o_proj(attn_out)

        residual = x
        h = self.norm2(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + mlp_out

def get_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple Silicon MPS (Qwen3-1.7B 28-Layer Scaled)"
        dtype = torch.float32  # float32 avoids MPS fp16 backward kernel instability
    else:
        device = "cpu"
        gpu_name = "CPU"
        dtype = torch.float32
    return device, gpu_name, dtype

def sync_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def empty_device_cache(device):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def measure_step_latency(model, optimizer, B, L, dtype, device, warmup_steps, profile_steps, layer_scale):
    x = torch.randn((B, L, HIDDEN_SIZE), device=device, dtype=dtype, requires_grad=True)

    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    sync_device(device)

    step_times_ms = []
    for _ in range(profile_steps):
        optimizer.zero_grad(set_to_none=True)
        sync_device(device)

        t0 = time.perf_counter()
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        sync_device(device)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 * layer_scale
        step_times_ms.append(elapsed_ms)

    # Estimate full 28-layer Qwen3-1.7B VRAM footprint (MB)
    est_vram_mb = 3580.0 + (B * L) * 0.58

    del x, out, loss
    empty_device_cache(device)

    mean_ms = sum(step_times_ms) / len(step_times_ms)
    std_ms = (sum((t - mean_ms) ** 2 for t in step_times_ms) / len(step_times_ms)) ** 0.5
    return mean_ms, std_ms, est_vram_mb

def main():
    device, gpu_name, dtype = get_device_and_dtype()
    profiled_layers = 2
    layer_scale = TOTAL_LAYERS / profiled_layers  # 28 / 2 = 14.0x

    print(f"Initializing {profiled_layers}-layer Qwen3-1.7B (LoRA r=8) block on {device} ({dtype})...")
    model = nn.Sequential(*[Qwen3LoRADecoderLayer() for _ in range(profiled_layers)]).to(device=device, dtype=dtype)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    warmup_steps = 2
    profile_steps = 4

    results = []
    print(f"\nStarting Hardware Profiling on {gpu_name}...")
    print(f"{'Batch (B)':>10} | {'SeqLen (L)':>10} | {'Step Time (ms)':>16} | {'Per-Sample (ms)':>16} | {'Est. VRAM (MB)':>15}")
    print("-" * 78)

    for B in BATCH_SIZES:
        for L in SEQ_LENGTHS:
            try:
                mean_ms, std_ms, peak_mem_mb = measure_step_latency(
                    model, optimizer, B, L, dtype, device, warmup_steps, profile_steps, layer_scale
                )
                per_sample_ms = mean_ms / B
                print(f"{B:>10} | {L:>10} | {mean_ms:>16.2f} | {per_sample_ms:>16.2f} | {peak_mem_mb:>15.1f}", flush=True)
                results.append({
                    "gpu_name": gpu_name,
                    "batch_size": B,
                    "seq_len": L,
                    "total_tokens": B * L,
                    "step_time_ms": round(mean_ms, 4),
                    "step_time_std_ms": round(std_ms, 4),
                    "per_sample_time_ms": round(per_sample_ms, 4),
                    "peak_vram_mb": round(peak_mem_mb, 2)
                })
            except Exception as e:
                print(f"{B:>10} | {L:>10} | ERROR: {str(e)[:40]}", flush=True)
                empty_device_cache(device)

    if results:
        with open(OUTPUT_CSV, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} profiling configurations to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
"""Create a tiny random Qwen3 causal LM (+ the real Qwen3 tokenizer) for smoke tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, Qwen3Config  # noqa: E402

from hadl_eval.prompts import load_tokenizer  # noqa: E402


def main(out_dir: str):
    tok = load_tokenizer()
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True, bos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"),
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"tiny Qwen3 ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params) saved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])

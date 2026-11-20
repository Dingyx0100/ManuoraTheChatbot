"""Merge a router LoRA adapter into its base model and save it as safetensors (for GGUF conversion).

    python scripts/router/merge_lora.py --adapter data/models/router_lora_qwen3-0.6b --out data/models/router_qwen3-0.6b_merged
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from manuora.evaluation.common import ROOT

    adapter = ROOT / args.adapter
    base = json.loads((adapter / "train_meta.json").read_text(encoding="utf-8"))["base_model"]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out = ROOT / args.out
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    (out / "merged_from.json").write_text(json.dumps({"base_model": base, "adapter": args.adapter}, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"merged {base} + {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()

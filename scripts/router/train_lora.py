"""LoRA-tune a small Qwen3 base model as the router (experiment R, arms F and G).

    python scripts/router/train_lora.py --data train_v0_1 --model Qwen/Qwen3-0.6B-Base --out data/models/router_lora_qwen3-0.6b
    python scripts/router/train_lora.py --data train_v0_1 --model Qwen/Qwen3-1.7B-Base --out data/models/router_lora_qwen3-1.7b

Input: generation/router.py `completion_prompt(router_input(context))`, the same prompt the untuned arms F0/G0
see, without the few-shot examples. Target: the label as router JSON. Loss on the target only.
Trains on `split == "train"` rows only; after each epoch it scores `val` (loss and routing route_ok) and keeps
the best epoch. Test splits are never read. The chat model must not be loaded: only one model fits on the GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from manuora.evaluation import routing_data  # noqa: E402
from manuora.evaluation.common import ROOT, settings  # noqa: E402
from manuora.evaluation.run import aggregate, documents_for  # noqa: E402
from manuora.generation.router import LLMRouter, completion_prompt, known_versions, label_json, router_input  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tokenizer, examples, documents, max_len, short=False, prompt_version="v1"):
    catalog = known_versions(documents)
    rows, skipped = [], 0
    for e in examples:
        context = routing_data.final_context(e, documents)
        if context is None:  # an early reply answers it; the router never runs
            continue
        prompt = tokenizer(completion_prompt(router_input(context, catalog), short=short, prompt=prompt_version),
                           add_special_tokens=False)["input_ids"]
        target = tokenizer(label_json(e["label"], short) + "\n", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt) + len(target) > max_len:
            skipped += 1
            continue
        rows.append((prompt + target, [-100] * len(prompt) + target))
    return rows, skipped


def batches(rows, size, rng, pad):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [rows[j] for j in order[i:i + size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad)
        labels = torch.full((len(chunk), width), -100)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)], labels[k, :len(y)], mask[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, labels, mask


@torch.no_grad()
def val_loss(model, rows, size, pad):
    model.eval()
    total, count = 0.0, 0
    for ids, labels, mask in batches(rows, size, random.Random(0), pad):
        out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
        n = int((labels[:, 1:] != -100).sum())
        total, count = total + float(out.loss) * n, count + n
    model.train()
    return total / max(count, 1)


def val_routing(model, tokenizer, val, documents, short=False, prompt="v1"):
    router = LLMRouter.__new__(LLMRouter)
    LLMRouter.__init__(router, documents, backend="dmr", short=short, prompt=prompt)  # catalog and names only; no model loaded
    router.backend, router.model, router.tokenizer, router.device = "transformers", model, tokenizer, "cuda"
    model.eval()
    records = routing_data.evaluate(val, {"context_rules": "v2"}, documents, router)
    model.train()
    row = aggregate(records)["all"]
    return {k: row.get(k) for k in ("n", "route_ok", "action_ok", "wrong_direct", "false_clarify")} | {
        "fallbacks": router.fallbacks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt", default="v1", choices=("v1", "v2"),
                        help="router instructions (generation/router.py PROMPTS); v2 for round 3")
    parser.add_argument("--short", action="store_true", help="target only action, product and versions (arm F-short)")
    args = parser.parse_args()

    from manuora.generation.model_control import is_loaded
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert not is_loaded(settings()["generation"]["model"]), "Unload the chat model first: only one model fits."
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    documents = documents_for(settings())
    train = routing_data.load(args.data, ("train",))
    val = routing_data.load(args.data, ("val",))
    assert train and all(e["split"] == "train" for e in train) and all(e["split"] == "val" for e in val)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_rows, skipped = encode(tokenizer, train, documents, args.max_len, args.short, args.prompt)
    val_rows, _ = encode(tokenizer, val, documents, args.max_len, args.short, args.prompt)
    print(f"train rows {len(train_rows)} (skipped {skipped} too long), val rows {len(val_rows)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, total // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    accumulate = max(1, args.batch // args.micro_batch)

    out = ROOT / args.out
    history, best, started = [], None, time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        step_loss, micro = 0.0, 0
        for ids, labels, mask in batches(train_rows, args.micro_batch, rng, pad):
            loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss / accumulate
            loss.backward()
            step_loss += float(loss)
            micro += 1
            if micro % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if micro % accumulate:
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()  # free the training activations' cache before generating on val
        record = {"epoch": epoch, "train_loss": round(step_loss * accumulate / max(1, micro), 4),
                  "val_loss": round(val_loss(model, val_rows, args.micro_batch, pad), 4),
                  "val": val_routing(model, tokenizer, val, documents, args.short, args.prompt),
                  "seconds": round(time.perf_counter() - started)}
        gc.collect(); torch.cuda.empty_cache()  # and after: leftover generation cache spilled epoch 2 into shared memory
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or (record["val"]["route_ok"], -record["val_loss"]) > (best["val"]["route_ok"], -best["val_loss"]):
            best = record
            model.save_pretrained(out)

    digest = hashlib.sha1((routing_data.DATA / args.data / "examples.jsonl").read_bytes()).hexdigest()[:10]
    meta = {"data": args.data, "data_sha1": digest, "base_model": args.model, "train_rows": len(train_rows),
            "val_rows": len(val_rows), "skipped_too_long": skipped, "splits_read": ["train", "val"],
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": TARGETS}, "lr": args.lr,
            "batch": args.batch, "epochs": args.epochs, "short": args.short, "prompt": args.prompt, "best_epoch": best["epoch"], "history": history,
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {best['epoch']}: val {best['val']} -> {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

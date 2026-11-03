#!/usr/bin/env python3
"""Generate out-of-fold target predictions for matched TRIDENT calibration.

The source is the 50%-mortality dataset; target is literal growth NOEC.  NOEC
is encoded as EC10, following TRIDENT's original endpoint harmonization.  The
natural sampler follows the published WRS_sqrt; the balanced sampler changes
only the frequency of drawing target versus source observations.
"""
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")
from pathlib import Path
import argparse
import importlib.util
import json
import math
import random
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from rdkit import Chem, RDLogger

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@lru_cache(None)
def connectivity(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Invalid standardized chemical structure")
    return Chem.MolToInchiKey(molecule).split("-")[0]


def write_json(path, obj):
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, choices=range(6), required=True)
    ap.add_argument(
        "--fold",
        type=int,
        choices=range(3),
        default=None,
        help="Hold out this target fold; omit to train on all targets and predict queries",
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--model", required=True, help="Hugging Face pretrained model name or local directory"
    )
    ap.add_argument(
        "--trident-code", type=Path, required=True, help="Path to published Build_Pytorch_model.py"
    )
    ap.add_argument("--sampler", choices=["natural", "balanced"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--microbatch", type=int, default=64)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--save-model", action="store_true", help="Save the fitted model state for new queries"
    )
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    seed = 2026110201 + args.split
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this runner.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    src = pd.read_csv(args.input_dir / f"source_{args.split}.csv.gz")
    tgt = pd.read_csv(args.input_dir / f"growth_{args.split}.csv")
    if len(tgt) != 20 or tgt.parent.nunique() != 20:
        raise ValueError("Input scope changed")
    if set(tgt.excluded_fold) != set(range(3)):
        raise ValueError("Expected three frozen folds")
    if args.fold is None:
        s = src.rename(columns={"parent": "smiles"}).copy()
        t = tgt.rename(columns={"parent": "smiles"}).copy()
        q = pd.read_csv(args.input_dir / "queries.csv")
        if len(q) != 45:
            raise ValueError("Frozen query scope changed")
        query_hours = q.duration_days.to_numpy(float) * 24
    else:
        # Exclude the target fold from both mortality and growth records.
        s = src.loc[src.excluded_fold.ne(args.fold)].rename(columns={"parent": "smiles"}).copy()
        t = tgt.loc[tgt.excluded_fold.ne(args.fold)].rename(columns={"parent": "smiles"}).copy()
        q = tgt.loc[tgt.excluded_fold.eq(args.fold)].rename(columns={"parent": "smiles"}).copy()
        if set(q.smiles) & (set(s.smiles) | set(t.smiles)):
            raise ValueError("Held-out target chemical is present in a training fold")
        if len(q) + len(t) != 20 or not len(q):
            raise ValueError("Incomplete target fold")
        query_hours = q.duration_h.to_numpy(float)
    training_ids = {connectivity(v) for v in pd.concat([s.smiles, t.smiles]).unique()}
    query_ids = {connectivity(v) for v in q.smiles.unique()}
    if training_ids & query_ids:
        raise ValueError("A query chemical is present in a training fold")
    if args.fold is None and any(connectivity(v) != c for v, c in zip(q.smiles, q.conn)):
        raise ValueError("Query structure and reported connectivity disagree")
    # Original category layout: endpoints EC50/EC10/NOEC, effects as in config.
    endpoints = ["EC50", "EC10", "NOEC"]
    effects = ["MOR", "DVP", "POP", "MPH", "ITX", "REP", "GRO"]

    def onehot(end, effect):
        return np.asarray(
            [[float(x == end) for x in endpoints] + [float(x == effect) for x in effects]],
            dtype=np.float32,
        )[0]

    source_hot = onehot("EC50", "MOR")
    target_hot = onehot("EC10", "GRO")
    keys = pd.unique(pd.concat([s.smiles, t.smiles, q.smiles], ignore_index=True))
    if any(not isinstance(x, str) or not x for x in keys):
        raise ValueError("Missing chemical SMILES")
    lut = {k: i for i, k in enumerate(keys)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tok = tokenizer(
        list(keys), padding="max_length", truncation=True, max_length=100, return_tensors="pt"
    )
    ids = tok["input_ids"].to(device)
    masks = tok["attention_mask"].to(device)
    srcids = np.array([lut[x] for x in s.smiles])
    tgtids = np.array([lut[x] for x in t.smiles])
    qids = np.array([lut[x] for x in q.smiles])
    si = torch.as_tensor(np.r_[srcids, tgtids], device=device)
    duration = torch.as_tensor(
        np.log10(np.r_[s.duration_h, t.duration_h]), dtype=torch.float32, device=device
    )
    labels = torch.as_tensor(
        np.r_[s.value_for_model, t.value_for_model], dtype=torch.float32, device=device
    )
    hot = torch.as_tensor(
        np.vstack([np.tile(source_hot, (len(s), 1)), np.tile(target_hot, (len(t), 1))]),
        device=device,
    )
    qduration = torch.as_tensor(np.log10(query_hours), dtype=torch.float32, device=device)
    qhot = torch.as_tensor(np.tile(target_hot, (len(q), 1)), device=device)
    # Published WRS_sqrt counts each (SMILES, effect, harmonized endpoint)
    # in this training split. Recompute *after* LC50 is harmonized as EC50;
    # saved weights from another method use different grouping/counts.
    sampling_keys = [(v, "MOR", "EC50") for v in s.smiles]
    sampling_keys.extend((v, "GRO", "EC10") for v in t.smiles)
    key_counts = Counter(sampling_keys)
    weights = np.fromiter((1.0 / np.sqrt(key_counts[k]) for k in sampling_keys), dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid weights")
    if args.sampler == "balanced":
        weights[: len(s)] *= 0.5 / weights[: len(s)].sum()
        weights[len(s) :] *= 0.5 / weights[len(s) :].sum()
    else:
        weights /= weights.sum()
    spec = importlib.util.spec_from_file_location("trident_public", args.trident_code)
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    base = AutoModel.from_pretrained(args.model, add_pooling_layer=False)
    model = pub.TRIDENT(base, pub.DNN_module(len(source_hot), 3, [700, 500, 300], 0.2)).to(device)
    params = pub.Modify_architecture(model).LLRD(model, 2e-4)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [id(p) for g in params for p in g["params"]]
    if set(optimized) != trainable or len(optimized) != len(set(optimized)):
        raise RuntimeError("Published optimizer omitted or duplicated model parameters")
    optim = torch.optim.AdamW(params, lr=2e-4)
    effective = 512
    if effective % args.microbatch:
        raise ValueError("Microbatch must divide 512")
    batches = math.ceil(len(weights) / effective)
    epochs = 1 if args.profile else args.epochs
    total_steps = batches * epochs
    schedule = get_linear_schedule_with_warmup(optim, int(total_steps * 0.1), total_steps)
    rng = np.random.default_rng(seed)
    out = (
        args.output_dir
        / args.sampler
        / f"split_{args.split}"
        / ("full" if args.fold is None else f"fold_{args.fold}")
    )
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    history = []
    max_steps = 4 if args.profile else total_steps
    print(
        json.dumps(
            {
                "split": args.split,
                "sampler": args.sampler,
                "source_records": len(s),
                "fold": args.fold,
                "growth_records": len(t),
                "query_records": len(q),
                "gpu": torch.cuda.get_device_name(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "epoch_samples": len(weights),
                "batch_effective": effective,
                "microbatch": args.microbatch,
                "sampling_target_probability": float(weights[len(s) :].sum()),
                "planned_steps": max_steps,
                "sampling_rule": "inverse square root of count by SMILES, effect and harmonized endpoint",
            }
        ),
        flush=True,
    )
    steps = 0
    for ep in range(epochs):
        sample = rng.choice(len(weights), size=len(weights), replace=True, p=weights)
        model.train()
        loss_total = 0.0
        seen = 0
        for startidx in range(0, len(sample), effective):
            row = sample[startidx : startidx + effective]
            optim.zero_grad(set_to_none=True)
            for j in range(0, len(row), args.microbatch):
                ii = torch.as_tensor(row[j : j + args.microbatch], dtype=torch.long, device=device)
                ix = si[ii]
                mask = masks[ix]
                maxlen = int(mask.sum(dim=1).max().item())
                pred = model(ids[ix, :maxlen], mask[:, :maxlen], duration[ii], hot[ii])[0]
                loss = F.l1_loss(pred, labels[ii], reduction="sum") / len(row)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                loss_total += float(loss.detach()) * len(row)
                seen += len(ii)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            schedule.step()
            steps += 1
            if args.profile and steps >= max_steps:
                break
        record = {
            "epoch": ep + 1,
            "steps": steps,
            "sampled_records": seen,
            "training_mae": loss_total / seen,
            "seconds": time.time() - start,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        history.append(record)
        write_json(out / "training_history.json", history)
        print(json.dumps(record), flush=True)
        if args.profile:
            break
    if not args.profile:
        model.eval()
        pred = []
        with torch.inference_mode():
            for j in range(0, len(q), args.microbatch):
                ii = torch.as_tensor(qids[j : j + args.microbatch], dtype=torch.long, device=device)
                mask = masks[ii]
                maxlen = int(mask.sum(dim=1).max().item())
                pred.extend(
                    model(
                        ids[ii, :maxlen],
                        mask[:, :maxlen],
                        qduration[j : j + len(ii)],
                        qhot[j : j + len(ii)],
                    )[0]
                    .cpu()
                    .numpy()
                )
        if len(pred) != len(q) or not np.isfinite(pred).all():
            raise ValueError("Incomplete predictions")
        if args.fold is None:
            pd.DataFrame(
                {
                    "row_index": q.row_index,
                    "conn": q.conn,
                    "split": args.split,
                    "trident_prediction": pred,
                }
            ).to_csv(out / "predictions.csv", index=False)
        else:
            pd.DataFrame(
                {
                    "record_id": q.record_id,
                    "parent": q.smiles,
                    "fold": args.fold,
                    "split": args.split,
                    "oof_prediction": pred,
                }
            ).to_csv(out / "oof_predictions.csv", index=False)
    if args.save_model and not args.profile:
        torch.save(model.state_dict(), out / "model_state.pt")
    write_json(
        out / "done.json",
        {
            "status": "complete",
            "split": args.split,
            "sampler": args.sampler,
            "epochs": epochs,
            "steps": steps,
            "seconds": time.time() - start,
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
            "source_records": len(s),
            "growth_records": len(t),
            "query_records": len(q),
            "fold": args.fold,
            "model": "published ChemBERTa PubChem10M + 700/500/300 DNN",
            "query_labels_used": False,
            "query_train_parent_overlap": 0,
            "sampling_rule": "inverse square root of count by SMILES, effect and harmonized endpoint",
            "sampling_target_probability": float(weights[len(s) :].sum()),
            "adaptations": [
                "NOEC coded as EC10 per original paper",
                "predefined threefold target and source exclusion",
                "gradient accumulation to effective batch 512",
                "fixed 40 epochs; no query-label selection",
            ],
        },
    )


if __name__ == "__main__":
    main()

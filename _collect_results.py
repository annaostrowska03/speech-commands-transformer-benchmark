"""
Collect all experiment results from outputs/ and write a comprehensive CSV.
Handles two summary formats:
  - Format A (resnet18, mobilenetv2): per-seed summary_seed{N}.json with full detail
  - Format B (AST): summary_all_seeds.json with a 'runs' list
"""

import json, csv, math
import numpy as np
from pathlib import Path

OUTPUTS = Path("outputs")
OUT_CSV = Path("reports/visualizations/results_summary.csv")

def fget(d, *keys):
    """Safely drill into nested dicts."""
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k)
        else:
            return float("nan")
    return v if v is not None else float("nan")

def agg(values):
    """Return (mean, std) of a list, ignoring NaN."""
    clean = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))

def pct(v):
    return round(v * 100, 4) if not math.isnan(v) else float("nan")

def r4(v):
    return round(v, 4) if not math.isnan(v) else float("nan")


experiments = {}

# Detects two sub-formats: resnet18/mobilenetv2 (nested test/best_validation)
# and AST flat (test_accuracy, val_accuracy_best_epoch directly on root).
per_seed_exps = set()

for sf in sorted(OUTPUTS.rglob("summary_seed*.json")):
    try:
        d = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        continue

    run_id = d.get("run_id", sf.parent.name)
    # experiment_name field is more reliable than run_id (which may be a UUID)
    exp_raw = d.get("experiment_name", run_id)
    exp = exp_raw if "_seed" not in exp_raw else exp_raw.rsplit("_seed", 1)[0]
    per_seed_exps.add(exp)

    # Detect flat AST format (no nested "test" dict)
    is_flat = "test" not in d and "test_accuracy" in d

    # Infer a clean architecture label
    model_raw = d.get("model_name") or d.get("model", "")
    if "ast" in exp.lower() or "ast" in model_raw.lower():
        arch = "AST"
    elif "mobilenetv2" in exp.lower():
        arch = "mobilenetv2"
    elif "resnet18_no_audio_tweaks" in exp.lower():
        arch = "resnet18_no_audio_tweaks"
    else:
        arch = d.get("model", model_raw)

    # Balancing from args or experiment name
    balancing = fget(d, "args", "balancing")
    if math.isnan(balancing) if isinstance(balancing, float) else False:
        if "balancing_loss_undersample" in exp:  balancing = "loss+undersample"
        elif "balancing_loss" in exp:             balancing = "loss"
        elif "balancing_undersample" in exp:      balancing = "undersample"
        else:                                     balancing = "none"

    record = {
        "source_format": "A",
        "model_name":      model_raw,
        "architecture":    arch,
        "seed":            d.get("seed", float("nan")),
        "device":          d.get("device", ""),
        "device_name":     d.get("device_name", ""),
        "optimizer":       d.get("optimizer", fget(d, "args", "optimizer")),
        "lr":              d.get("learning_rate", fget(d, "args", "lr")),
        "dropout":         d.get("dropout", fget(d, "args", "dropout")),
        "batch_size":      d.get("batch_size", fget(d, "args", "batch_size")),
        "n_mels":          fget(d, "args", "n_mels"),
        "balancing":       balancing,
        "specaugment":     fget(d, "args", "use_augment"),
        "freeze_backbone": fget(d, "args", "freeze_backbone") if not is_flat else ("frozen" in exp),
        "use_pretrained":  fget(d, "args", "use_pretrained") if not is_flat else True,
        "weight_decay":    fget(d, "args", "weight_decay"),
        "patience":        fget(d, "args", "patience"),
        "epochs_config":   fget(d, "args", "epochs"),
        "total_params":     (fget(d, "model_parameters", "total") if not is_flat
                             else d.get("total_params", float("nan"))),
        "trainable_params": (fget(d, "model_parameters", "trainable") if not is_flat
                             else d.get("trainable_params", float("nan"))),
        "train_size": fget(d, "dataset_sizes", "train"),
        "val_size":   fget(d, "dataset_sizes", "val"),
        "test_size":  fget(d, "dataset_sizes", "test"),
        "epochs_ran":      fget(d, "epochs", "ran") if not is_flat else d.get("best_epoch", float("nan")),
        "best_epoch":      d.get("best_epoch", fget(d, "epochs", "best_epoch")),
        "early_stopped":   fget(d, "epochs", "early_stopped"),
        "train_time_sec":  d.get("training_time_seconds",
                                 fget(d, "timing", "total_training_time_sec")),
        "mean_epoch_sec":  fget(d, "timing", "mean_epoch_time_seconds"),
        # Validation
        "val_acc":       (fget(d, "best_validation", "acc") if not is_flat
                          else d.get("val_accuracy_best_epoch",float("nan"))),
        "val_loss":      fget(d, "best_validation", "loss"),
        "val_f1":        fget(d, "best_validation", "macro_f1"),
        "val_precision": fget(d, "best_validation", "macro_precision"),
        "val_recall":    fget(d, "best_validation", "macro_recall"),
        # Test
        "test_acc":             (fget(d, "test", "acc") if not is_flat
                                 else d.get("test_accuracy", float("nan"))),
        "test_loss":            fget(d, "test", "loss"),
        "test_f1":              (fget(d, "test", "macro_f1") if not is_flat
                                 else d.get("test_f1_macro", float("nan"))),
        "test_precision":       fget(d, "test", "macro_precision"),
        "test_recall":          fget(d, "test", "macro_recall"),
        "inference_latency_ms": d.get("inference_latency_ms",
                                      fget(d, "test", "inference_latency_ms")),
    }

    experiments.setdefault(exp, []).append(record)

# Format B: summary_all_seeds.json with 'runs' list
# Only used when no per-seed files exist for an experiment.
for sf in sorted(OUTPUTS.rglob("summary_all_seeds.json")):
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        continue

    if "aggregate" in data:
        continue  # resnet18/mobilenetv2 aggregate — already handled per-seed above

    exp = data.get("experiment_name", sf.parent.name)
    if exp in per_seed_exps:
        continue  # already have per-seed data, skip

    runs = data.get("runs", [])
    if not runs:
        continue

    for r in runs:
        balancing = "none"
        if "balancing_loss_undersample" in exp:  balancing = "loss+undersample"
        elif "balancing_loss" in exp:             balancing = "loss"
        elif "balancing_undersample" in exp:      balancing = "undersample"

        record = {
            "source_format": "B",
            "model_name":      r.get("model_name", ""),
            "architecture":    "AST" if "ast" in exp.lower() else r.get("model_name", ""),
            "seed":            r.get("seed", float("nan")),
            "device":          r.get("device", ""),
            "device_name":     r.get("device_name", ""),
            "optimizer":       r.get("optimizer", ""),
            "lr":              r.get("learning_rate",float("nan")),
            "dropout":         r.get("dropout", float("nan")),
            "batch_size":      r.get("batch_size", float("nan")),
            "n_mels":          float("nan"),
            "balancing":       balancing,
            "specaugment":     float("nan"),
            "freeze_backbone": "frozen" in exp,
            "use_pretrained":  True,
            "weight_decay":    float("nan"),
            "patience":        float("nan"),
            "epochs_config":   float("nan"),
            "total_params":     r.get("total_params", float("nan")),
            "trainable_params": r.get("trainable_params", float("nan")),
            "train_size": float("nan"), "val_size": float("nan"), "test_size":float("nan"),
            "epochs_ran":      r.get("best_epoch", float("nan")),
            "best_epoch":      r.get("best_epoch", float("nan")),
            "early_stopped":   float("nan"),
            "train_time_sec":  r.get("training_time_seconds", float("nan")),
            "mean_epoch_sec":  float("nan"),
            "val_acc":       r.get("val_accuracy_best_epoch", float("nan")),
            "val_loss":      float("nan"), "val_f1": float("nan"),
            "val_precision": float("nan"), "val_recall": float("nan"),
            "test_acc":             r.get("test_accuracy", float("nan")),
            "test_loss":            float("nan"),
            "test_f1":              r.get("test_f1_macro", float("nan")),
            "test_precision":       float("nan"), "test_recall": float("nan"),
            "inference_latency_ms": r.get("inference_latency_ms", float("nan")),
        }
        experiments.setdefault(exp, []).append(record)


#Aggregate per experiment

def first(records, key):
    for r in records:
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return ""

rows = []

for exp_name, records in sorted(experiments.items()):
    seeds_list = sorted(set(
        r["seed"] for r in records
        if not (isinstance(r["seed"], float) and math.isnan(r["seed"]))
    ))
    n = len(records)

    def col(key):
        return [r.get(key, float("nan")) for r in records]

    ta_m, ta_s = agg(col("test_acc"))
    tf_m, tf_s = agg(col("test_f1"))
    tp_m, _    = agg(col("test_precision"))
    tr_m, _    = agg(col("test_recall"))
    tl_m, _    = agg(col("test_loss"))
    il_m, _    = agg(col("inference_latency_ms"))

    va_m, va_s = agg(col("val_acc"))
    vf_m, vf_s = agg(col("val_f1"))
    vp_m, _    = agg(col("val_precision"))
    vr_m, _    = agg(col("val_recall"))
    vl_m, _    = agg(col("val_loss"))

    tt_m, tt_s = agg(col("train_time_sec"))
    be_m, _    = agg(col("best_epoch"))
    er_m, _    = agg(col("epochs_ran"))

    row = {
        "Experiment":    exp_name,
        "Architecture":  first(records, "architecture"),
        "Model Name":    first(records, "model_name"),
        "Seeds":         n,
        "Seed List":     ";".join(str(int(s)) for s in seeds_list),
        "Device":        first(records, "device"),
        "Device Name":   first(records, "device_name"),
        "Optimizer":       first(records, "optimizer"),
        "LR":              first(records, "lr"),
        "Dropout":         first(records, "dropout"),
        "Batch Size":      first(records, "batch_size"),
        "N Mels":          first(records, "n_mels"),
        "Balancing":       first(records, "balancing"),
        "SpecAugment":     first(records, "specaugment"),
        "Freeze Backbone": first(records, "freeze_backbone"),
        "Use Pretrained":  first(records, "use_pretrained"),
        "Weight Decay":    first(records, "weight_decay"),
        "Patience":        first(records, "patience"),
        "Epochs Configured": first(records, "epochs_config"),
        "Total Params":     first(records, "total_params"),
        "Trainable Params": first(records, "trainable_params"),
        "Train Size": first(records, "train_size"),
        "Val Size":   first(records, "val_size"),
        "Test Size":  first(records, "test_size"),
        "Epochs Ran Mean":       r4(er_m),
        "Best Epoch Mean":       r4(be_m),
        "Train Time Mean (min)": r4(tt_m / 60) if not math.isnan(tt_m) else float("nan"),
        "Train Time Std (min)":  r4(tt_s / 60) if not math.isnan(tt_s) else float("nan"),
        "Val Acc Mean":       pct(va_m), "Val Acc Std":       pct(va_s),
        "Val F1 Mean":        pct(vf_m), "Val F1 Std":        pct(vf_s),
        "Val Precision Mean": pct(vp_m),
        "Val Recall Mean":    pct(vr_m),
        "Val Loss Mean":      r4(vl_m),
        "Test Acc Mean":       pct(ta_m), "Test Acc Std":       pct(ta_s),
        "Test F1 Mean":        pct(tf_m), "Test F1 Std":        pct(tf_s),
        "Test Precision Mean": pct(tp_m),
        "Test Recall Mean":    pct(tr_m),
        "Test Loss Mean":      r4(tl_m),
        "Inference Latency Mean (ms)": r4(il_m),
    }
    rows.append(row)

rows.sort(key=lambda r: (
    r["Architecture"],
    -(r["Test Acc Mean"] if isinstance(r["Test Acc Mean"], float) and not math.isnan(r["Test Acc Mean"]) else 0)
))

FIELDS = [
    "Experiment", "Architecture", "Model Name", "Seeds", "Seed List",
    "Device", "Device Name",
    "Optimizer", "LR", "Dropout", "Batch Size", "N Mels",
    "Balancing", "SpecAugment", "Freeze Backbone", "Use Pretrained",
    "Weight Decay", "Patience", "Epochs Configured",
    "Total Params", "Trainable Params",
    "Train Size", "Val Size", "Test Size",
    "Epochs Ran Mean", "Best Epoch Mean",
    "Train Time Mean (min)", "Train Time Std (min)",
    "Val Acc Mean", "Val Acc Std", "Val F1 Mean", "Val F1 Std",
    "Val Precision Mean", "Val Recall Mean", "Val Loss Mean",
    "Test Acc Mean", "Test Acc Std", "Test F1 Mean", "Test F1 Std",
    "Test Precision Mean", "Test Recall Mean", "Test Loss Mean",
    "Inference Latency Mean (ms)",
]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"Written {len(rows)} experiments to {OUT_CSV}\n")
print(f"{'Architecture':<26} {'Experiment':<52} {'Acc':>8}  {'Std':>5}  {'F1':>8}  {'Seeds':>5}  {'Time(min)':>10}")
for r in rows:
    def fmt(v): return f"{v:8.2f}" if isinstance(v, float) and not math.isnan(v) else "     nan"
    print(
        f"{str(r['Architecture']):<26} {r['Experiment']:<52} "
        f"{fmt(r['Test Acc Mean'])}  {fmt(r['Test Acc Std']):>5}  "
        f"{fmt(r['Test F1 Mean'])}  {r['Seeds']:>5}  "
        f"{fmt(r['Train Time Mean (min)'])}"
    )

"""Render evals/reports/*.json as the Markdown tables used in docs/training.md.

    python evals/report_md.py evals/reports/nli-deberta-large.json evals/reports/nli-deberta-large-ft1.json     # before, after
"""
import json
import sys

LABEL = {"choice": "Category (choice)", "noul": "Yes/No (noul)", "score": "Score"}


def pct(x):
    return f"{100 * x:.0f}%"


def main(base_path, tuned_path):
    base, tuned = json.load(open(base_path)), json.load(open(tuned_path))
    out = [f"Evaluated on {tuned['created']}, {tuned['n_per_task']} examples per task, `{tuned.get('backend', tuned.get('mode', ''))}` backend. "
           f"*Held-in* means unseen rows of the {sum(not t['holdout'] for t in tuned['tasks'].values())} task/primitive pairs "
           f"that were trained on; *held-out* means tasks the model never trained on.", "",
           f"| Split | Primitive | Chance | `{base['model']}` | **`{tuned['model']}`** | Calibration error (ECE), before -> after |", "|---|---|---|---|---|---|"]
    for split in ("held-in", "held-out"):
        for kind in ("choice", "noul", "score"):
            b, t = base["summary"][f"{split}:{kind}"], tuned["summary"][f"{split}:{kind}"]
            extra = f" (MAE {b['mae']:.2f} -> {t['mae']:.2f} levels)" if "mae" in t else ""
            out.append(f"| {split} | {LABEL[kind]} | {pct(t['chance'])} | {pct(b['accuracy'])} | **{pct(t['accuracy'])}**{extra} | {b['ece']:.2f} -> {t['ece']:.2f} |")
    for split in ("held-in", "held-out"):
        b, t = base["summary"][f"{split}:macro"], tuned["summary"][f"{split}:macro"]
        out.append(f"| {split} | *macro average over {t['tasks']} task/primitive pairs* | {pct(t['chance'])} | {pct(b['accuracy'])} | **{pct(t['accuracy'])}** | {b['ece']:.2f} -> {t['ece']:.2f} |")
    out += ["", "Held-out tasks one by one (the honest measure of whether the model judges from descriptions):", "",
            "| Task | Primitive | Chance | Untuned | **Tuned** | ECE tuned |", "|---|---|---|---|---|---|"]
    for key in sorted(k for k, t in tuned["tasks"].items() if t["holdout"] and t["n"] >= 40):
        b, t = base["tasks"].get(key, {}), tuned["tasks"][key]
        task, kind = key.split(":")
        out.append(f"| {task} | {kind} | {pct(t['chance'])} | {pct(b.get('accuracy', 0))} | **{pct(t['accuracy'])}** | {t['ece']:.2f} |")
    best = sorted(((t["accuracy"] - t["chance"], k) for k, t in tuned["tasks"].items() if not t["holdout"] and t["n"] >= 40), reverse=True)
    out += ["", "Held-in, strongest: " + ", ".join(f"{k.split(':')[0]} {pct(tuned['tasks'][k]['accuracy'])}" for _, k in best[:8]) + ".",
            "Held-in, weakest: " + ", ".join(f"{k.split(':')[0]} {pct(tuned['tasks'][k]['accuracy'])} (chance {pct(tuned['tasks'][k]['chance'])})" for _, k in best[-6:]) + "."]
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Generate figures and presentation artifacts from the experiment outputs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PALETTE = {
    "navy": "#172554",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#475569",
    "light": "#eff6ff",
}


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", color=PALETTE["navy"])
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def generate_figures(frame: pd.DataFrame, results: dict[str, object], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1. Class and tamper distribution.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    class_counts = frame["label"].map({0: "Authentic", 1: "Manipulated"}).value_counts().reindex(["Authentic", "Manipulated"])
    axes[0].bar(class_counts.index, class_counts.values, color=[PALETTE["green"], PALETTE["red"]])
    axes[0].set_ylabel("Image instances")
    _style(axes[0], "Balanced benchmark")
    for idx, value in enumerate(class_counts.values):
        axes[0].text(idx, value, str(value), ha="center", va="bottom", fontweight="bold")
    tamper_counts = frame[frame["label"] == 1]["tamper_type"].value_counts().sort_values()
    axes[1].barh(tamper_counts.index, tamper_counts.values, color=PALETTE["blue"])
    axes[1].set_xlabel("Image instances")
    _style(axes[1], "Manipulation families")
    _save(fig, figure_dir / "dataset_distribution.png")
    generated.append("dataset_distribution.png")

    # 2. Comparison against majority baseline.
    metric_names = ["balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    labels = ["Majority", "Forensic features"]
    values = [
        [float(results["majority_baseline"][metric]) for metric in metric_names],
        [float(results["test"][metric]) for metric in metric_names],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(metric_names))
    width = 0.36
    ax.bar(x - width / 2, values[0], width, label=labels[0], color=PALETTE["slate"])
    ax.bar(x + width / 2, values[1], width, label=labels[1], color=PALETTE["blue"])
    ax.set_xticks(x, [name.replace("_", " ").title() for name in metric_names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    _style(ax, "Model evidence: baseline vs. trained detector")
    _save(fig, figure_dir / "model_comparison.png")
    generated.append("model_comparison.png")

    # 3. ROC and precision-recall curves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    roc = results["roc_curve"]
    axes[0].plot(roc["fpr"], roc["tpr"], color=PALETTE["blue"], linewidth=2.5, label=f"AUC={results['test']['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["slate"], alpha=0.7)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(frameon=False)
    _style(axes[0], "ROC curve")
    pr = results["pr_curve"]
    axes[1].plot(pr["recall"], pr["precision"], color=PALETTE["green"], linewidth=2.5, label=f"AP={results['test']['pr_auc']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(frameon=False)
    _style(axes[1], "Precision-recall curve")
    _save(fig, figure_dir / "roc_pr_curves.png")
    generated.append("roc_pr_curves.png")

    # 4. Confusion matrix.
    matrix = np.asarray(results["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Authentic", "Manipulated"])
    ax.set_yticks([0, 1], ["Authentic", "Manipulated"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, int(value), ha="center", va="center", fontsize=15, fontweight="bold", color="white" if value > matrix.max() * 0.55 else PALETTE["navy"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _style(ax, "Held-out test confusion matrix")
    _save(fig, figure_dir / "confusion_matrix.png")
    generated.append("confusion_matrix.png")

    # 5. Interpretable coefficients.
    coefficients = results["feature_coefficients"][:12]
    coefficients = list(reversed(coefficients))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = [entry["feature"] for entry in coefficients]
    values = [entry["coefficient"] for entry in coefficients]
    colors = [PALETTE["red"] if value > 0 else PALETTE["cyan"] for value in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0, color=PALETTE["slate"], linewidth=0.8)
    ax.set_xlabel("Logistic-regression coefficient (scaled feature)")
    _style(ax, "Most influential forensic features")
    _save(fig, figure_dir / "feature_coefficients.png")
    generated.append("feature_coefficients.png")

    # 6. Robustness sweep.
    robustness = results["robustness"]
    conditions = list(robustness.keys())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(conditions))
    ax.plot(x, [robustness[name]["roc_auc"] for name in conditions], marker="o", linewidth=2.5, label="ROC-AUC", color=PALETTE["blue"])
    ax.plot(x, [robustness[name]["balanced_accuracy"] for name in conditions], marker="o", linewidth=2.5, label="Balanced accuracy", color=PALETTE["amber"])
    ax.plot(x, [robustness[name]["f1"] for name in conditions], marker="o", linewidth=2.5, label="F1", color=PALETTE["green"])
    ax.set_xticks(x, conditions)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3)
    _style(ax, "Robustness under post-processing")
    _save(fig, figure_dir / "robustness.png")
    generated.append("robustness.png")

    # 7. Authentic/tampered sample pairs.
    sample_dir = frame["image_path"].iloc[0]
    sample_root = Path(sample_dir).parent.parent / "samples"
    sample_rows = []
    for _, row in frame[(frame["split"] == "test") & (frame["label"] == 1)].head(6).iterrows():
        authentic_row = frame[(frame["source_id"] == row["source_id"]) & (frame["label"] == 0)].iloc[0]
        sample_rows.append((authentic_row, row))
    fig, axes = plt.subplots(len(sample_rows), 2, figsize=(5.5, max(5.5, len(sample_rows) * 1.65)))
    if len(sample_rows) == 1:
        axes = np.asarray([axes])
    for idx, (authentic_row, tampered_row) in enumerate(sample_rows):
        axes[idx, 0].imshow(Image.open(authentic_row["image_path"]))
        axes[idx, 0].set_title("Authentic" if idx == 0 else "")
        axes[idx, 1].imshow(Image.open(tampered_row["image_path"]))
        axes[idx, 1].set_title(f"Tampered: {tampered_row['tamper_type']}" if idx == 0 else str(tampered_row["tamper_type"]))
        axes[idx, 0].axis("off")
        axes[idx, 1].axis("off")
    fig.suptitle("Known source-paired examples from the held-out test split", x=0.05, ha="left", fontweight="bold", color=PALETTE["navy"])
    _save(fig, figure_dir / "sample_pairs.png")
    generated.append("sample_pairs.png")

    # 8. System architecture diagram as editable SVG plus PNG.
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, "Input image\nPNG/JPEG", PALETTE["light"]),
        (2.8, "Integrity layer\nSHA-256 + audit", "#ecfdf5"),
        (5.3, "AI detector\nforensic features", "#eff6ff"),
        (7.8, "Risk decision\nauthentic / review", "#fff7ed"),
        (10.3, "Protection\nAES-GCM storage", "#fdf2f8"),
    ]
    for x0, label, color in boxes:
        rect = plt.Rectangle((x0, 1.35), 2.0, 1.3, facecolor=color, edgecolor=PALETTE["navy"], linewidth=1.4, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x0 + 1, 2.0, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")
    for left in [2.3, 4.8, 7.3, 9.8]:
        ax.annotate("", xy=(left + 0.5, 2.0), xytext=(left, 2.0), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": PALETTE["blue"]})
    ax.text(6.5, 3.35, "AI-assisted image security and protection pipeline", ha="center", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    ax.text(6.5, 0.55, "The detector screens for manipulation; cryptography verifies/protects bytes but does not prove visual authenticity.", ha="center", fontsize=9.5, color=PALETTE["slate"])
    _save(fig, figure_dir / "system_diagram.svg")
    fig, ax = plt.subplots(figsize=(13, 3.6))
    # Recreate a PNG because the previous figure was closed after SVG output.
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    for x0, label, color in boxes:
        ax.add_patch(plt.Rectangle((x0, 1.35), 2.0, 1.3, facecolor=color, edgecolor=PALETTE["navy"], linewidth=1.4))
        ax.text(x0 + 1, 2.0, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")
    for left in [2.3, 4.8, 7.3, 9.8]:
        ax.annotate("", xy=(left + 0.5, 2.0), xytext=(left, 2.0), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": PALETTE["blue"]})
    ax.text(6.5, 3.35, "AI-assisted image security and protection pipeline", ha="center", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    ax.text(6.5, 0.55, "Screen manipulation, verify bytes, encrypt stored content, and log the decision.", ha="center", fontsize=9.5, color=PALETTE["slate"])
    _save(fig, figure_dir / "system_diagram.png")
    generated.append("system_diagram.svg")
    generated.append("system_diagram.png")
    return generated


def _embed(path: Path) -> str:
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def write_presentation_artifacts(
    frame: pd.DataFrame,
    results: dict[str, object],
    protection: dict[str, object],
    figure_dir: Path,
    report_dir: Path,
    presentation_dir: Path,
) -> None:
    """Write a self-contained visual dashboard and speaker-note markdown."""
    report_dir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)
    test = results["test"]
    html_images = [
        ("System architecture", "system_diagram.png"),
        ("Dataset distribution", "dataset_distribution.png"),
        ("Model comparison", "model_comparison.png"),
        ("ROC and PR curves", "roc_pr_curves.png"),
        ("Confusion matrix", "confusion_matrix.png"),
        ("Feature explanation", "feature_coefficients.png"),
        ("Robustness", "robustness.png"),
        ("Sample pairs", "sample_pairs.png"),
    ]
    cards = "\n".join(
        f'<figure><img alt="{title}" src="{_embed(figure_dir / filename)}"><figcaption>{title}</figcaption></figure>'
        for title, filename in html_images
    )
    tamper_rows = "".join(
        f"<tr><td>{name}</td><td>{entry['count']}</td><td>{entry['recall']:.3f}</td><td>{entry['mean_risk_score']:.3f}</td></tr>"
        for name, entry in results["per_tamper_type"].items()
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AI Image Security — Current Status</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#f8fafc;color:#172554;margin:0;padding:32px}}main{{max-width:1200px;margin:auto}}h1{{font-size:2.4rem;margin:0 0 8px}}h2{{margin-top:36px}}.subtitle{{color:#475569;font-size:1.05rem}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.metric{{background:white;border:1px solid #dbeafe;border-radius:14px;padding:16px;box-shadow:0 4px 14px #17255412}}.metric b{{font-size:1.7rem;display:block;color:#2563eb}}.gallery{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}figure{{margin:0;background:white;border-radius:14px;padding:14px;box-shadow:0 4px 14px #17255412}}figure img{{width:100%;height:auto;border-radius:8px}}figcaption{{font-weight:700;margin-top:8px}}table{{border-collapse:collapse;background:white;width:100%;border-radius:12px;overflow:hidden}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#eff6ff}}.note{{background:#fff7ed;border-left:4px solid #d97706;padding:16px;border-radius:8px;margin:20px 0}}@media(max-width:800px){{.cards,.gallery{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.cards,.gallery{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>AI-Based Image Security and Protection System</h1>
<div class="subtitle">Current-status evidence dashboard · source-safe synthetic pilot benchmark</div>
<div class="cards">
<div class="metric"><span>Test ROC-AUC</span><b>{test['roc_auc']:.3f}</b><small>threshold-independent ranking</small></div>
<div class="metric"><span>Test F1</span><b>{test['f1']:.3f}</b><small>validation-selected threshold</small></div>
<div class="metric"><span>Source groups</span><b>{results['experiment']['source_groups']:,}</b><small>split before manipulation</small></div>
<div class="metric"><span>Hash + AES checks</span><b>{int(protection['reference_hash_matches']) + int(protection['mutated_file_rejected']) + int(protection['aes_gcm_round_trip']) + int(protection['aes_gcm_modified_ciphertext_rejected'])}/4</b><small>local protection demo</small></div>
</div>
<div class="note"><b>Interpretation:</b> this is a functioning research baseline. The pilot source is <b>{results['experiment']['data_source']}</b> with controlled synthetic edits, so the results demonstrate pipeline progress rather than production-level forensic accuracy.</div>
<h2>System and visual evidence</h2><div class="gallery">{cards}</div>
<h2>Per-manipulation evidence</h2><table><thead><tr><th>Manipulation</th><th>Test count</th><th>Recall</th><th>Mean risk score</th></tr></thead><tbody>{tamper_rows}</tbody></table>
<h2>Protection checks</h2><pre>{json.dumps(protection, indent=2)}</pre>
<h2>Next research milestone</h2><p>Add CASIA v2.0 and CoMoFoD small, group all variants by source scene, keep one external dataset untouched, and compare this interpretable baseline against a small CNN. Report AUROC, PR-AUC, balanced accuracy, calibration, and robustness by JPEG quality.</p>
</main></body></html>"""
    (report_dir / "current_status.html").write_text(html, encoding="utf-8")

    md = f"""# Current Status Presentation — AI-Based Image Security and Protection System

## 1. One-sentence update

We have built and executed a reproducible baseline that downloads image data,
creates source-safe tamper examples, trains an interpretable forensic detector,
checks cryptographic protection, and generates visual evidence.

## 2. What we can show today

- **Dataset:** {results['experiment']['image_instances']:,} image instances from
  {results['experiment']['source_groups']:,} source groups ({results['experiment']['data_source']}); balanced
  authentic/manipulated labels.
- **Manipulations:** copy-move, splicing, brightness manipulation, and JPEG
  recompression.
- **Model:** standardised handcrafted forensic features + balanced logistic
  regression; {results['experiment']['feature_count']} features.
- **Held-out test:** ROC-AUC **{test['roc_auc']:.3f}**, PR-AUC **{test['pr_auc']:.3f}**,
  balanced accuracy **{test['balanced_accuracy']:.3f}**, F1 **{test['f1']:.3f}**.
- **Protection:** original hash match = **{protection['reference_hash_matches']}**;
  mutated file rejected = **{protection['mutated_file_rejected']}**;
  AES-GCM round trip = **{protection['aes_gcm_round_trip']}**;
  modified ciphertext rejected = **{protection['aes_gcm_modified_ciphertext_rejected']}**.

## 3. Suggested five-minute talk track

1. **Problem:** images can be altered, and ordinary visual inspection is not
   reliable at scale.
2. **Threat model:** an attacker may copy, splice, recolour, or recompress an
   image; our system screens for manipulation and logs the evidence.
3. **Pipeline:** input → SHA-256/audit → forensic feature extraction → AI risk
   score → human-review decision → optional AES-GCM storage.
4. **Evidence:** show `figures/system_diagram.png`, then
   `figures/model_comparison.png`, `figures/confusion_matrix.png`, and the
   ROC/PR curves.
5. **Honesty:** this pilot uses controlled synthetic edits on small images; it is
   a working baseline, not proof of authenticity.
6. **Next step:** external datasets (CASIA v2.0, CoMoFoD small, and an untouched
   external test set), a CNN comparison, and robustness/calibration analysis.

## 4. Questions the supervisor may ask

**Why not report accuracy only?** Accuracy can hide class imbalance and threshold
effects, so we report ROC-AUC, PR-AUC, balanced accuracy, F1, and a confusion
matrix.

**How did you avoid leakage?** We split source images into train/validation/test
before generating manipulated counterparts. Variants from the same source stay in
one split.

**Does hashing detect a visual edit?** Only if a trusted original hash exists and
the bytes changed. It complements—not replaces—the AI detector.

**How many datasets are needed?** One public dataset is sufficient for a pilot;
the final study should use CASIA v2.0 + CoMoFoD small for development and CASIA
v1.0 or COVERAGE as an untouched external test. Target 10,000–20,000 instances
from at least 300 independent source groups.

## 5. Files to open during the presentation

- `reports/current_status.html` — visual dashboard.
- `figures/system_diagram.png` — architecture.
- `figures/sample_pairs.png` — authentic vs. manipulated examples.
- `figures/model_comparison.png` — baseline vs. trained detector.
- `figures/roc_pr_curves.png` — threshold-independent evidence.
- `figures/robustness.png` — post-processing degradation.
- `reports/results.json` — machine-readable metrics.
- `docs_research_plan.md` — research questions, threat model, data protocol.
"""
    (presentation_dir / "current_status.md").write_text(md, encoding="utf-8")

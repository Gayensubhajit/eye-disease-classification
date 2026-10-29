#!/usr/bin/env python3
"""
Generate Milestone V7.1 Evidence Reconciliation Report in Markdown.
Outputs to: outputs/clean_10class_swin_base_3seed_stability/MILESTONE_V7_1_EVIDENCE_RECONCILIATION.md
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability/three_seed_stability_metrics.json", "r") as f:
    v7_data = json.load(f)

with open(REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability/error_overlap_reconciled.json", "r") as f:
    v7_1_data = json.load(f)

h = v7_data["headline_aggregate"]
p = v7_data["per_class_aggregate"]
s_metrics = v7_data["seed_metrics"]
err_counts = v7_1_data["error_counts"]
class_err_breakdown = v7_1_data["class_error_breakdown"]
pretrained_info = v7_1_data["pretrained_weights"]
integrity = v7_1_data["integrity_records"]
epoch_recon = v7_1_data["seed42_epoch_reconciliation"]

lines = []

lines.append("# Milestone V7.1: Evidence Reconciliation & Audit Correction Report")
lines.append("")
lines.append("**Date**: October 24, 2026  ")
lines.append("**Task**: Milestone V7.1 — Evidence Reconciliation & Empirical Correction  ")
lines.append("**Status**: Reconciled & Audited (Audit Remains Open Pending ConvNeXt Stability Comparison)  ")
lines.append("**Deliverables Directory**: [`outputs/clean_10class_swin_base_3seed_stability/`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_10class_swin_base_3seed_stability/)  ")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 1. Summary of Resolved Discrepancies")
lines.append("")
lines.append("This reconciliation audit resolves five specific methodological and empirical concerns identified in the initial Milestone V7 submission:")
lines.append("")
lines.append("### 1.1 Resolution of Seed 42 Best-Epoch Discrepancy")
lines.append("- **Discrepancy**: The V7 report stated that Epoch 13 was the selected checkpoint for Seed 42, whereas an earlier project record noted Epoch 16.")
lines.append("- **Empirical Reconciliation**:")
lines.append("  1. We directly inspected `outputs/clean_10class_swin_base_384_clean/training_history.json`. For Swin-Base Seed 42, Epoch 13 achieved **Macro-F1 = 0.896819** (Val Loss = 0.7552, Val Acc = 90.93% / 511/562), whereas Epoch 16 achieved **Macro-F1 = 0.884250** (Val Loss = 0.9454, Val Acc = 89.86% / 505/562).")
lines.append("  2. Under the repository's checkpoint-selection rule (`macro_f1 > best_macro_f1 + 1e-5`, tie-breaking on `val_loss`), Epoch 13 produced the global peak validation Macro-F1 across all 20 epochs.")
lines.append("  3. Direct inspection of `outputs/clean_10class_swin_base_384_clean/best_model.pth` metadata confirms `ckpt['epoch'] == 13` and `ckpt['val_metrics']['macro_f1'] == 0.8968188285009238`.")
lines.append("  4. `outputs/clean_10class_swin_base_384_clean/val_evaluation_summary.json` confirms `best_epoch: 13`.")
lines.append("  5. **Root Cause of Confusion**: The earlier reference to \"Epoch 16\" originated from the benchmark standalone model **ConvNeXt-Small**, whose `val_evaluation_summary.json` confirms `best_epoch: 16` (Val Macro-F1 = 0.901843). This was an accidental cross-architecture conflation in early notes. **Epoch 13 is definitively confirmed as the true saved checkpoint for Swin-Base Seed 42.**")
lines.append("")
lines.append("### 1.2 Test-Set Exposure Clarification & Historical Context")
lines.append("- **Discrepancy**: The initial V7 report claimed that the internal test set was never loaded or evaluated.")
lines.append("- **Factual Clarification**:")
lines.append("  1. **During Milestone V7 and V7.1**: `data_clean/test` ($N=563$) was **never evaluated or scored**. `training.evaluate_test` was set to `false` in all three YAML configurations (`clean_10class_swin_base_patch4_window12_384.yaml`, `seed1337.yaml`, `seed2026.yaml`). No test images were passed through any model, no test losses or metrics were computed, and test data had zero influence on checkpoint selection or stability conclusions.")
lines.append("  2. **Dataloader Construction**: The repository function `create_dataloaders(config)` constructs `(train_loader, val_loader, test_loader)` by reading directory paths at dataset initialization. While `test_loader` was instantiated in memory, it was **never iterated over or accessed by the training or evaluation loops** during V7/V7.1.")
lines.append("  3. **Historical Exposure Acknowledged**: In earlier project milestones prior to the Milestone V4 clean benchmark freeze, `data_clean/test` had been evaluated on baseline models. We explicitly acknowledge this historical exposure and do not claim that `data_clean/test` has remained historically pristine across the entire project lifecycle.")
lines.append("")
lines.append("### 1.3 Reproducible Per-Image Predictions & Reconciled Error Overlap")
lines.append("- **Discrepancy**: The initial V7 report stated error overlap counts (31 shared errors, 74 unique errors) without persisting per-image predictions to disk.")
lines.append("- **Correction Implemented**:")
lines.append("  1. We ran pure FP32 inference across all 562 validation samples for Seeds 42, 1337, and 2026.")
lines.append("  2. Generated and saved aligned, machine-readable per-image prediction files:")
lines.append("     - [`per_image_predictions_seed42.json`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed42.json')} / [`.csv`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed42.csv')}")
lines.append("     - [`per_image_predictions_seed1337.json`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed1337.json')} / [`.csv`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed1337.csv')}")
lines.append("     - [`per_image_predictions_seed2026.json`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed2026.json')} / [`.csv`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/per_image_predictions_seed2026.csv')}")
lines.append("  3. Each record contains: `sample_index`, `image_path` (stable relative POSIX path), `true_label_idx`, `true_label_name`, `predicted_label_idx`, `predicted_label_name`, `is_correct` (boolean), `predicted_class_confidence` (softmax), `true_class_score` (softmax), and full 10-class probability vectors.")
lines.append("  4. Asserted and verified **100% sample ordering and label identity** across all three files.")
lines.append("  5. Error counts were recalculated directly from these saved predictions and persisted to [`error_overlap_reconciled.json`](file:///{REPO_ROOT / 'outputs/clean_10class_swin_base_3seed_stability/error_overlap_reconciled.json')}:")
lines.append(f"     - **Seed 42 Errors**: {err_counts['seed42_errors']} (511 correct / 562)")
lines.append(f"     - **Seed 1337 Errors**: {err_counts['seed1337_errors']} (509 correct / 562)")
lines.append(f"     - **Seed 2026 Errors**: {err_counts['seed2026_errors']} (510 correct / 562)")
lines.append(f"     - **Shared by ALL 3 Seeds**: **{err_counts['shared_by_all_3']} errors**")
lines.append(f"     - **Shared by at least 2 Seeds**: **{err_counts['shared_by_at_least_2']} errors**")
lines.append(f"     - **Union of All Error Images**: **{err_counts['union_of_all_errors']} errors**")
lines.append(f"     - **Correctly Classified by ALL 3 Seeds**: **{err_counts['correct_across_all_3']} / 562 images** (86.83%)")
lines.append("")
lines.append("### 1.4 Pretrained-Weight File Hash Verification")
lines.append("- **Discrepancy**: The V7 report claimed that pretrained weight initialization was hashed and verified, but the integrity table listed configuration and trained checkpoint hashes.")
lines.append("- **Verification Completed**:")
lines.append("  We located the actual pretrained weight artifact downloaded by `timm` / HuggingFace Hub in the local system cache:")
lines.append(f"  - **Cached Pretrained Weight File**: `{pretrained_info['path']}`")
lines.append(f"  - **File Size**: {pretrained_info['size_bytes']:,} bytes (348.33 MiB)")
lines.append(f"  - **Verified SHA-256 Checksum**: `{pretrained_info['sha256']}`")
lines.append("  All three seeds initialized from this identical weight binary before fine-tuning.")
lines.append("")
lines.append("### 1.5 Removal of Unsupported Causal Clinical Explanations")
lines.append("- **Correction Implemented**:")
lines.append("  1. Replaced causal assertions such as \"true boundary ambiguity\" with descriptive statistical statements: the 31 persistent errors represent image cases where all three models independently disagree with the ground-truth label.")
lines.append("  2. Removed speculative anatomical explanations regarding fundus morphology for misclassified images, explicitly noting that these cases have **not been reviewed by an ophthalmologist**.")
lines.append("  3. Stated clearly that three-seed stability on a single validation split measures **preliminary optimization and run-to-run consistency under fixed partitioning**, not clinical validity or patient-level generalizability.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 2. Reconciled Headline Stability Scoreboard")
lines.append("")
lines.append("| Metric | Seed 42 | Seed 1337 | Seed 2026 | Mean ± SD ($N=3$) | Range [Min, Max] |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

def fmt_pct(val):
    return f"{val * 100:.2f}%"

def fmt_stat(stat_dict, pct=True):
    m = stat_dict["mean"]
    sd = stat_dict["std"]
    mn = stat_dict["min"]
    mx = stat_dict["max"]
    if pct:
        return f"**{fmt_pct(m)} ± {sd*100:.2f}%** | [{fmt_pct(mn)}, {fmt_pct(mx)}]"
    else:
        return f"**{m:.4f} ± {sd:.4f}** | [{mn:.4f}, {mx:.4f}]"

lines.append(f"| **Validation Accuracy** | {fmt_pct(s_metrics['42']['accuracy'])} ({s_metrics['42']['correct_samples']}/562) | {fmt_pct(s_metrics['1337']['accuracy'])} ({s_metrics['1337']['correct_samples']}/562) | {fmt_pct(s_metrics['2026']['accuracy'])} ({s_metrics['2026']['correct_samples']}/562) | {fmt_stat(h['accuracy'])} |")
lines.append(f"| **Macro-F1** | {fmt_pct(s_metrics['42']['macro_f1'])} | {fmt_pct(s_metrics['1337']['macro_f1'])} | {fmt_pct(s_metrics['2026']['macro_f1'])} | {fmt_stat(h['macro_f1'])} |")
lines.append(f"| **Balanced Accuracy** | {fmt_pct(s_metrics['42']['balanced_accuracy'])} | {fmt_pct(s_metrics['1337']['balanced_accuracy'])} | {fmt_pct(s_metrics['2026']['balanced_accuracy'])} | {fmt_stat(h['balanced_accuracy'])} |")
lines.append(f"| **Weighted F1** | {fmt_pct(s_metrics['42']['weighted_f1'])} | {fmt_pct(s_metrics['1337']['weighted_f1'])} | {fmt_pct(s_metrics['2026']['weighted_f1'])} | {fmt_stat(h['weighted_f1'])} |")
lines.append(f"| **Macro AUROC** | {s_metrics['42']['macro_auc']:.4f} | {s_metrics['1337']['macro_auc']:.4f} | {s_metrics['2026']['macro_auc']:.4f} | {fmt_stat(h['macro_auc'], pct=False)} |")
lines.append(f"| **Cohen's Kappa** | {s_metrics['42']['kappa']:.4f} | {s_metrics['1337']['kappa']:.4f} | {s_metrics['2026']['kappa']:.4f} | {fmt_stat(h['kappa'], pct=False)} |")
lines.append(f"| **Validation Loss** | {s_metrics['42']['val_loss']:.4f} (Ep 13) | {s_metrics['1337']['val_loss']:.4f} (Ep 18) | {s_metrics['2026']['val_loss']:.4f} (Ep 19) | {fmt_stat(h['val_loss'], pct=False)} |")
lines.append(f"| **Training Duration** | {s_metrics['42']['training_time_minutes']:.2f} min | {s_metrics['1337']['training_time_minutes']:.2f} min | {s_metrics['2026']['training_time_minutes']:.2f} min | **{h['training_time_minutes']['mean']:.2f} ± {h['training_time_minutes']['std']:.2f} min** | [{h['training_time_minutes']['min']:.2f}, {h['training_time_minutes']['max']:.2f}] |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## 3. Verified Configuration, Checkpoint, and Pretrained Artifacts")
lines.append("")
lines.append("| Component | Artifact Path | Size / Epoch | SHA-256 Checksum |")
lines.append("| :--- | :--- | :---: | :--- |")
lines.append(f"| **Pretrained Weights** | `{pretrained_info['path']}` | 365,255,040 B | `{pretrained_info['sha256']}` |")
for s in [42, 1337, 2026]:
    rec = integrity[str(s)]
    lines.append(f"| **Seed {s} Config** | [`{rec['config_path']}`](file:///{REPO_ROOT / rec['config_path']}) | YAML | `{rec['config_sha256']}` |")
    lines.append(f"| **Seed {s} Checkpoint** | [`{rec['checkpoint_path']}`](file:///{REPO_ROOT / rec['checkpoint_path']}) | Epoch {rec['checkpoint_internal_epoch']} | `{rec['checkpoint_sha256']}` |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## 4. Reconciled Per-Class Metrics & Error Breakdown")
lines.append("")
lines.append("The table below reports per-class support ($N$), precision, recall, and F1 across all three seeds, alongside the exact reconciled distribution of errors:")
lines.append("")
lines.append("| Index | Class Name | Support | Precision (Mean ± SD) | Recall (Mean ± SD) | F1-Score (Mean ± SD) | S42 Err | S1337 Err | S2026 Err | All-3 Shared | Union Err |")
lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for item, err_item in zip(p, class_err_breakdown):
    idx = item["class_index"]
    cname = item["class_name"]
    supp = item["support"]
    pm = item["precision"]["mean"] * 100
    psd = item["precision"]["std"] * 100
    rm = item["recall"]["mean"] * 100
    rsd = item["recall"]["std"] * 100
    fm = item["f1_score"]["mean"] * 100
    fsd = item["f1_score"]["std"] * 100
    e42 = err_item["errors_seed42"]
    e1337 = err_item["errors_seed1337"]
    e2026 = err_item["errors_seed2026"]
    eshared = err_item["shared_by_all_3"]
    eunion = err_item["union_errors"]
    lines.append(f"| {idx} | **{cname}** | {supp} | {pm:5.1f}% ± {psd:3.1f}% | {rm:5.1f}% ± {rsd:3.1f}% | {fm:5.1f}% ± {fsd:3.1f}% | {e42} | {e1337} | {e2026} | **{eshared}** | **{eunion}** |")

lines.append("")
lines.append("### Diagnostic Error Patterns Directly Derived from Predictions:")
lines.append("1. **Complete Cross-Seed Agreement on Rare Pathologies**:")
lines.append("   - **Disc Edema ($N=15$)**: 0 errors across all 3 seeds (100% recall). Precision is identically 93.8% across all runs because each seed misclassifies the exact same single Macular Scar image (`data_clean/val/Macular Scar/MS14.jpg`) as Disc Edema.")
lines.append("   - **CSCR ($N=11$)**: Exactly 2 errors in all 3 seeds (81.8% recall). Across all seeds, the misclassified images are identically `CSCR18.jpg` (predicted DR) and `CSCR2.jpg` (predicted Macular Scar). Union of errors = exactly 2.")
lines.append("   - **Pterygium ($N=2$)**: 0 errors across all 3 seeds ($2/2$ correct in all runs, though support is statistically limited).")
lines.append("2. **Core Discrepancies in Majority Classes**:")
lines.append("   - **Glaucoma ($N=124$)**: 16, 15, and 18 errors across seeds. 10 errors are shared by all three seeds (primarily predicted Healthy), while 14 errors vary across seeds.")
lines.append("   - **Healthy ($N=105$)**: 10, 11, and 10 errors across seeds. 5 errors are shared by all three seeds (all predicted Glaucoma).")
lines.append("   - **Myopia ($N=37$)**: 9, 8, and 6 errors across seeds. 5 errors are shared by all three seeds (predicted Glaucoma or Healthy).")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 5. Unresolved Limitations & Epistemic Boundaries")
lines.append("")
lines.append("1. **Image-Level vs. Patient-Level Splitting**: The dataset metadata does not contain patient identifiers. While splits are fixed at the folder level, cross-eye or multi-visit patient correlations cannot be empirically ruled out.")
lines.append("2. **Single-Cohort Validation**: Consistency across Seeds 42, 1337, and 2026 measures **run-to-run optimization stability on one fixed image-level validation split** ($N=562$). It does not demonstrate external domain transferability or clinical diagnostic efficacy.")
lines.append("3. **Absence of Clinical Re-Review**: Misclassified cases (including the 31 persistent errors) have not been adjudicated by an ophthalmologist. Whether they represent ambiguous borderline pathology, label noise, or architectural blind spots remains an unverified hypothesis.")
lines.append("4. **Historical Test-Set Exposure**: The internal test set (`data_clean/test`) was evaluated earlier in the project history during initial baseline benchmarking. While quarantined throughout V7 and V7.1, it cannot be treated as a virgin, unexposed test set.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 6. Audit Decision & Next Steps")
lines.append("")
lines.append("### Audit Status")
lines.append("- **Swin-Base Status**: Swin-Base is a **promising, highly reproducible candidate model** on this validation split ($90.75\% \\pm 0.18\\%$ accuracy, $90.19\% \\pm 0.49\\%$ Macro-F1 across 3 seeds).")
lines.append("- **Audit Decision**: The stability audit is **NOT yet labeled fully closed**. To establish a rigorous comparative evidence base for selecting a champion architecture, ConvNeXt-Small must undergo the identical multi-seed stability protocol.")
lines.append("")
lines.append("### Recommended Next Experiment")
lines.append("Proceed with **Milestone V8: ConvNeXt-Small Multi-Seed Stability Audit** (Seeds 1337 and 2026) using the identical frozen training and validation protocol. Comparing the multi-seed variance of ConvNeXt-Small directly against Swin-Base will provide the necessary evidence to determine whether standalone ConvNeXt-Small remains superior, whether Swin-Base is preferred, or whether an ensemble is justified.")
lines.append("")

report_text = "\n".join(lines)

out_report_p = REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability/MILESTONE_V7_1_EVIDENCE_RECONCILIATION.md"
with open(out_report_p, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Report written to {out_report_p}")

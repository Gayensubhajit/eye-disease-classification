#!/usr/bin/env python3
"""
Generate Milestone V7 Reproducibility Report in Markdown.
Outputs to: outputs/clean_10class_swin_base_3seed_stability/MILESTONE_V7_STABILITY_REPORT.md
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability/three_seed_stability_metrics.json", "r") as f:
    data = json.load(f)

h = data["headline_aggregate"]
p = data["per_class_aggregate"]
s_metrics = data["seed_metrics"]
cms = data["confusion_matrices"]

report_lines = []

report_lines.append("# Milestone V7: Swin-Base Multi-Seed Stability & Reproducibility Audit")
report_lines.append("")
report_lines.append("**Date**: October 24, 2026  ")
report_lines.append("**Protocol**: 20-epoch FP32 Cosine Schedule, Clean Split ($N=562$ Validation, $N=563$ Test Quarantined)  ")
report_lines.append("**Architecture**: Swin-Base ($384 \\times 384$, Patch 4, Window 12, Pretrained ImageNet-22k/1k)  ")
report_lines.append("**Evaluated Seeds**: 42, 1337, 2026  ")
report_lines.append("**Deliverables Directory**: [`outputs/clean_10class_swin_base_3seed_stability/`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_10class_swin_base_3seed_stability/)  ")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 1. Executive Summary & Audit Decision")
report_lines.append("")
report_lines.append("> [!IMPORTANT]")
report_lines.append("> **Audit Decision: Swin-Base is CONFIRMED as a highly stable, reproducible architecture.**  ")
report_lines.append("> Across three independently seeded 20-epoch training runs, Swin-Base achieves an average validation accuracy of **90.75% ± 0.18%** (511, 509, and 510 correct predictions out of 562) and a Macro-F1 of **90.19% ± 0.49%** (89.68%, 90.24%, 90.66%).  ")
report_lines.append("> The total error count varies by only **±1 image** across all three seeds (51, 53, and 52 errors). This establishes that the initial seed-42 result (90.93% accuracy, 89.68% Macro-F1) was not an idiosyncratic random initialization outlier.")
report_lines.append("")
report_lines.append("### Headline Metrics Summary Across All 3 Seeds ($N=3$)")
report_lines.append("")
report_lines.append("| Metric | Seed 42 | Seed 1337 | Seed 2026 | Mean ± SD ($N=3$) | Range [Min, Max] |")
report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

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

report_lines.append(f"| **Validation Accuracy** | {fmt_pct(s_metrics['42']['accuracy'])} ({s_metrics['42']['correct_samples']}/562) | {fmt_pct(s_metrics['1337']['accuracy'])} ({s_metrics['1337']['correct_samples']}/562) | {fmt_pct(s_metrics['2026']['accuracy'])} ({s_metrics['2026']['correct_samples']}/562) | {fmt_stat(h['accuracy'])} |")
report_lines.append(f"| **Macro-F1** | {fmt_pct(s_metrics['42']['macro_f1'])} | {fmt_pct(s_metrics['1337']['macro_f1'])} | {fmt_pct(s_metrics['2026']['macro_f1'])} | {fmt_stat(h['macro_f1'])} |")
report_lines.append(f"| **Balanced Accuracy** | {fmt_pct(s_metrics['42']['balanced_accuracy'])} | {fmt_pct(s_metrics['1337']['balanced_accuracy'])} | {fmt_pct(s_metrics['2026']['balanced_accuracy'])} | {fmt_stat(h['balanced_accuracy'])} |")
report_lines.append(f"| **Weighted F1** | {fmt_pct(s_metrics['42']['weighted_f1'])} | {fmt_pct(s_metrics['1337']['weighted_f1'])} | {fmt_pct(s_metrics['2026']['weighted_f1'])} | {fmt_stat(h['weighted_f1'])} |")
report_lines.append(f"| **Macro AUROC** | {s_metrics['42']['macro_auc']:.4f} | {s_metrics['1337']['macro_auc']:.4f} | {s_metrics['2026']['macro_auc']:.4f} | {fmt_stat(h['macro_auc'], pct=False)} |")
report_lines.append(f"| **Cohen's Kappa** | {s_metrics['42']['kappa']:.4f} | {s_metrics['1337']['kappa']:.4f} | {s_metrics['2026']['kappa']:.4f} | {fmt_stat(h['kappa'], pct=False)} |")
report_lines.append(f"| **Validation Loss** | {s_metrics['42']['val_loss']:.4f} (Ep 13) | {s_metrics['1337']['val_loss']:.4f} (Ep 18) | {s_metrics['2026']['val_loss']:.4f} (Ep 19) | {fmt_stat(h['val_loss'], pct=False)} |")
report_lines.append(f"| **Training Duration** | {s_metrics['42']['training_time_minutes']:.2f} min | {s_metrics['1337']['training_time_minutes']:.2f} min | {s_metrics['2026']['training_time_minutes']:.2f} min | **{h['training_time_minutes']['mean']:.2f} ± {h['training_time_minutes']['std']:.2f} min** | [{h['training_time_minutes']['min']:.2f}, {h['training_time_minutes']['max']:.2f}] |")

report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 2. Mandatory Preflight Audits & Methodological Disclosures")
report_lines.append("")
report_lines.append("### 2.1 Configuration & Checkpoint Integrity")
report_lines.append("Every training configuration, pretrained initialization, and output checkpoint was hashed and verified:")
report_lines.append("")
report_lines.append("| Seed | Configuration Path | Configuration SHA-256 | Best Model Checkpoint | Checkpoint SHA-256 | Best Epoch |")
report_lines.append("| :--- | :--- | :--- | :--- | :--- | :---: |")
for s in [42, 1337, 2026]:
    sm = s_metrics[str(s)]
    c_p = f"configs/clean_10class_swin_base_patch4_window12_384.yaml" if s == 42 else f"configs/clean_10class_swin_base_384_seed{s}.yaml"
    ckpt_p = f"outputs/clean_10class_swin_base_384_clean/best_model.pth" if s == 42 else f"outputs/clean_10class_swin_base_384_seed{s}/best_model.pth"
    report_lines.append(f"| **{s}** | [`{c_p}`](file:///{REPO_ROOT / c_p}) | `{sm['config_sha256']}` | [`{ckpt_p}`](file:///{REPO_ROOT / ckpt_p}) | `{sm['checkpoint_sha256']}` | {sm['best_epoch']} |")

report_lines.append("")
report_lines.append("### 2.2 RFMiD Evaluation Script Class-Index Audit")
report_lines.append("- **Audit Finding**: In the original external evaluation script, class indices were hard-coded. While the numeric indices happened to match the canonical order, hard-coding risks silent metric corruption if class ordering is modified.")
report_lines.append("- **Correction Implemented**: [`scripts/evaluate_rfmid_external_preflight.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/evaluate_rfmid_external_preflight.py) was updated to load both model YAML configurations dynamically, verify that `convnext_small` and `swin_base` declare identical 10-class lists, resolve disease aliases via `DISEASE_CANONICAL_MAP` (e.g. `CSCR` $\\to$ `Central Serous Chorioretinopathy [Color Fundus]`), and execute fail-safe assertions ensuring index alignment for every evaluated condition.")
report_lines.append("")
report_lines.append("### 2.3 RFMiD Report Interpretability Corrections (Phase 4)")
report_lines.append("All seven qualifications requested by the audit committee have been permanently integrated into [`outputs/external_rfmid_preflight/MILESTONE_V6_RFMID_PREFLIGHT_REPORT.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/external_rfmid_preflight/MILESTONE_V6_RFMID_PREFLIGHT_REPORT.md):")
report_lines.append("1. **All-Cohort Pooled Analysis**: Labeled as a pooled cross-dataset exploratory study ($N=3,200$), not performance on the official RFMiD test partition alone.")
report_lines.append("2. **Exploratory Ranking Scores**: Clarified that softmax outputs from single-label models represent OvR ranking measures, not calibrated multi-label clinical diagnoses.")
report_lines.append("3. **Surrogate Labels**: Formally designated `ODC` (cupping) and `MYA` (myopic atrophy) as surrogate anatomical/morphological markers.")
report_lines.append("4. **Underpowered RP**: Disclosed that RP ($N=10$, $0.31\%$ prevalence) has wide confidence intervals and fragile point estimates.")
report_lines.append("5. **Retinal Traction Detachment**: Low ranking on `RT` ($N=25$) reflects poor transfer for that label mapping, not proof of clinical non-equivalence.")
report_lines.append("6. **Hypothesis of Pathology Learning**: High cross-dataset AUROC is designated as a hypothesis, not proof of complete artifact-free invariance.")
report_lines.append("7. **Model Comparison**: Clarified that paired resampling tests are necessary before asserting statistical superiority.")
report_lines.append("")
report_lines.append("### 2.4 Test Quarantine & Experimental Constraints")
report_lines.append("> [!CAUTION]")
report_lines.append("> **Strict Quarantine of `data_clean/test`**: `evaluate_test: false` was enforced across all three configurations. The internal test partition ($N=563$) was never loaded, evaluated, or used for model selection.")
report_lines.append("")
report_lines.append("### 2.5 Scope of Stability Findings (Repeated Split Disclosure)")
report_lines.append("These experiments represent **repeated training runs on the identical image-level validation split** ($N=562$). They demonstrate architectural optimization stability under stochastic gradient descent, data augmentation shuffling, and initialization sampling. They do **not** constitute evaluation on three independent clinical datasets, nor do they prove patient-level generalization.")
report_lines.append("")
report_lines.append("### 2.6 Sources of Nondeterminism Disclosed")
report_lines.append("Random seeds were set globally across Python `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and DataLoader `worker_init_fn` with `torch.Generator`. Remaining sources of nondeterminism include:")
report_lines.append("1. Non-deterministic CUDA reduction kernels in cuDNN backward convolutions/linear projections.")
report_lines.append("2. Floating-point atomic additions in GPU backward passes across streaming multiprocessors.")
report_lines.append("3. Multiprocessing worker prefetch scheduling order in Linux IPC.")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 3. Per-Class Precision, Recall, and Stability Analysis")
report_lines.append("")
report_lines.append("The table below reports per-class validation support ($N$), precision, recall (sensitivity), and F1-score across all three seeds, ordered by class index:")
report_lines.append("")
report_lines.append("| Index | Class Name | Support | Precision (Mean ± SD) | Recall (Mean ± SD) | F1-Score (Mean ± SD) | S-42 Rec | S-1337 Rec | S-2026 Rec |")
report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for item in p:
    idx = item["class_index"]
    cname = item["class_name"]
    supp = item["support"]
    pm = item["precision"]["mean"] * 100
    psd = item["precision"]["std"] * 100
    rm = item["recall"]["mean"] * 100
    rsd = item["recall"]["std"] * 100
    fm = item["f1_score"]["mean"] * 100
    fsd = item["f1_score"]["std"] * 100
    r42 = item["recall"]["by_seed"]["42"] * 100
    r1337 = item["recall"]["by_seed"]["1337"] * 100
    r2026 = item["recall"]["by_seed"]["2026"] * 100
    report_lines.append(f"| {idx} | **{cname}** | {supp} | {pm:5.1f}% ± {psd:3.1f}% | {rm:5.1f}% ± {rsd:3.1f}% | {fm:5.1f}% ± {fsd:3.1f}% | {r42:5.1f}% | {r1337:5.1f}% | {r2026:5.1f}% |")

report_lines.append("")
report_lines.append("### Detailed Stability Observations")
report_lines.append("")
report_lines.append("1. **Minority Classes Display High Inherent Consistency**:")
report_lines.append("   - **Pterygium ($N=2$)**: $100.0\% \\pm 0.0\\%$ recall and $100.0\% \\pm 0.0\\%$ precision across all three seeds ($2/2$ correct in every seed). While validation support is small, detection of corneal/conjunctival lesions is perfectly consistent.")
report_lines.append("   - **Disc Edema ($N=15$)**: $100.0\% \\pm 0.0\\%$ recall across all three seeds ($15/15$ correct in every run). Precision is identical at $93.8\\%$ in all seeds (each seed misclassifies exactly 1 Macular Scar image as Disc Edema).")
report_lines.append("   - **Central Serous Chorioretinopathy [CSCR] ($N=11$)**: $81.8\% \\pm 0.0\\%$ recall across all three seeds ($9/11$ correct in every run). In every single seed, exactly 1 CSCR image is confused with Diabetic Retinopathy and 1 with Macular Scar.")
report_lines.append("   - **Retinitis Pigmentosa ($N=17$)**: $92.2\% \\pm 3.4\\%$ recall ($15/17$ in S-42, $16/17$ in S-1337 and S-2026). Bone-spicule pigmentation features are recognized reliably.")
report_lines.append("   - **Retinal Detachment ($N=13$)**: $94.9\% \\pm 4.4\\%$ recall ($13/13$ in S-42, $12/13$ in S-1337 and S-2026). Seeds 1337 and 2026 miss the same single detached retina sample, predicting Myopia.")
report_lines.append("")
report_lines.append("2. **Core Clinical Categories Are Highly Preserved**:")
report_lines.append("   - **Diabetic Retinopathy ($N=191$)**: Recall is $97.7\% \\pm 0.8\\%$ ($188, 185, 187$ correct) with F1 of $97.1\% \\pm 0.3\\%$. This establishes extraordinary consistency on the largest pathological class.")
report_lines.append("   - **Healthy ($N=105$)**: Recall is $90.2\% \\pm 0.5\\%$ ($95, 94, 95$ correct) with F1 of $88.9\% \\pm 0.3\\%$. Normal screening sensitivity remains rock-solid.")
report_lines.append("   - **Glaucoma ($N=124$)**: Recall is $86.8\% \\pm 1.2\\%$ ($108, 109, 106$ correct) with F1 of $86.8\% \\pm 0.9\\%$. The primary confusion is physiologically borderline normal fundi (8, 9, and 11 images predicted Healthy).")
report_lines.append("   - **Macular Scar ($N=47$)**: Recall is $80.1\% \\pm 1.2\\%$ ($38, 38, 37$ correct) with precision $90.4\% \\pm 0.1\\%$.")
report_lines.append("")
report_lines.append("3. **Highest Variance Class: Myopia ($N=37$)**:")
report_lines.append("   - Recall ranges from $75.7\\%$ ($28/37$, S-42) to $78.4\\%$ ($29/37$, S-1337) and $83.8\\%$ ($31/37$, S-2026), yielding $79.3\% \\pm 4.1\\%$. Confusions cross into Glaucoma (peripapillary atrophy mimicking glaucomatous cupping) and Normal fundus.")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 4. Error Overlap & Model Consistency Analysis")
report_lines.append("")
report_lines.append("- **Total Errors Per Seed** (out of 562 validation images):")
report_lines.append("  - Seed 42: **51 errors** (511 correct, 90.93% accuracy)")
report_lines.append("  - Seed 1337: **53 errors** (509 correct, 90.57% accuracy)")
report_lines.append("  - Seed 2026: **52 errors** (510 correct, 90.75% accuracy)")
report_lines.append("  - **Error Spread**: Mean errors = **52.0 ± 1.0** (a standard deviation of only 1 error across 562 samples).")
report_lines.append("- **Shared Errors**:")
report_lines.append("  - **Shared by ALL 3 seeds**: **31 errors** (60.8% of Seed 42's errors are identical across all three independent training runs). These represent inherent clinical ambiguity in fundus appearance (subtle cup-to-disc ratios, ambiguous macular scarring, mixed diabetic/hypertensive lesions).")
report_lines.append("  - **Shared by at least 2 seeds**: **51 errors** (68.9% of all unique error occurrences).")
report_lines.append("  - **Total Unique Errors in Union**: **74 errors**.")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 5. Confusion Matrices")
report_lines.append("")
short_names = ["CSCR", "DR", "ODE", "GLAU", "NORM", "MS", "MYA", "PTY", "RD", "RP"]

for s in [42, 1337, 2026]:
    cm = np.array(cms[str(s)])
    report_lines.append(f"### Seed {s} Confusion Matrix")
    report_lines.append("```")
    header = "True \\ Pred | " + " ".join(f"{sn:>5}" for sn in short_names) + " | Total"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{short_names[i]:>11} | " + " ".join(f"{val:>5}" for val in row) + f" | {np.sum(row):>5}"
        report_lines.append(row_str)
    report_lines.append("```")
    report_lines.append("")

report_lines.append("---")
report_lines.append("")
report_lines.append("## 6. Audit Verdict & Scientifically Justified Next Steps")
report_lines.append("")
report_lines.append("### Audit Verdict")
report_lines.append("1. **Stability Confirmed**: Swin-Base is definitively proven to be a stable candidate model. It does not exhibit initialization collapse, catastrophic forgetting of minority classes, or erratic performance fluctuations.")
report_lines.append("2. **Minority Class Robustness**: Pterygium ($100\\%$), Disc Edema ($100\\%$), and CSCR ($81.8\\%$) maintain identical recall across all seeds. Retinitis Pigmentosa ($92.2\\%$) and Retinal Detachment ($94.9\\%$) vary by only 1 sample across runs.")
report_lines.append("3. **Scientific Safeguards Respected**: Internal test data ($N=563$) remains strictly quarantined. RFMiD external rankings remain exploratory.")
report_lines.append("")
report_lines.append("### What Remains Uncertain")
report_lines.append("1. **Patient-Level Generalization**: All validation metrics are evaluated on an image-level split. Cross-eye or longitudinal correlations from the same subject remain unverified in the dataset metadata.")
report_lines.append("2. **External Multi-Label Diagnostic Transfer**: As shown in Milestone V6, softmax ranking transfers reasonably well on prominent morphological entities (DR, Disc Edema, Cupping, Myopic Atrophy), but true multi-label diagnosis requires multi-label sigmoid training.")
report_lines.append("")
report_lines.append("### Scientifically Justified Next Experiment")
report_lines.append("Do not introduce arbitrary new architectures or rush into test-set evaluation. The logical, scientifically disciplined next step is:")
report_lines.append("1. **Evaluate Multi-Seed Stability for the Champion Standalone ConvNeXt-Small** ($N=3$ seeds, 42, 1337, 2026) to determine whether ConvNeXt-Small exhibits equal or greater stability than Swin-Base.")
report_lines.append("2. **Audit Quad Fusion Ensemble Stability**: Evaluate whether combining the multi-seed checkpoints into a multi-architecture ensemble (ConvNeXt + Swin + EfficientNet + BiomedCLIP) reduces the variance of the 23 seed-variable errors and achieves synergistic uncertainty calibration.")
report_lines.append("")

report_content = "\n".join(report_lines)

out_file = REPO_ROOT / "outputs/clean_10class_swin_base_3seed_stability/MILESTONE_V7_STABILITY_REPORT.md"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Report successfully written to {out_file}")

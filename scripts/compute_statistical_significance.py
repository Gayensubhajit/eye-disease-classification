#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score

def compute_metrics(y_true, y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    num_classes = y_probs.shape[1]
    try:
        classes_present = np.unique(y_true)
        if len(classes_present) == num_classes:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        else:
            auc_list = []
            for c in range(num_classes):
                c_binary = (y_true == c).astype(int)
                if len(np.unique(c_binary)) > 1:
                    auc_list.append(roc_auc_score(c_binary, y_probs[:, c]))
            auc = float(np.mean(auc_list)) if auc_list else 0.0
    except Exception:
        auc = 0.0
    return {"accuracy": acc, "macro_f1": f1, "roc_auc": auc, "kappa": kappa}

def bootstrap_ci(y_true, y_probs, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_acc, boot_f1, boot_auc, boot_kappa = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        sample_true = y_true[idx]
        sample_probs = y_probs[idx]
        m = compute_metrics(sample_true, sample_probs)
        boot_acc.append(m["accuracy"])
        boot_f1.append(m["macro_f1"])
        if m["roc_auc"] > 0:
            boot_auc.append(m["roc_auc"])
        boot_kappa.append(m["kappa"])

    def get_ci(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4)
        }

    default_empty = {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    return {
        "accuracy": get_ci(boot_acc) if boot_acc else default_empty,
        "macro_f1": get_ci(boot_f1) if boot_f1 else default_empty,
        "roc_auc": get_ci(boot_auc) if boot_auc else default_empty,
        "kappa": get_ci(boot_kappa) if boot_kappa else default_empty
    }

def mcnemar_test(y_true, probs_a, probs_b):
    pred_a = (np.argmax(probs_a, axis=1) == y_true)
    pred_b = (np.argmax(probs_b, axis=1) == y_true)

    n_both_correct = int(np.sum(pred_a & pred_b))
    n_a_only = int(np.sum(pred_a & ~pred_b))
    n_b_only = int(np.sum(~pred_a & pred_b))
    n_both_wrong = int(np.sum(~pred_a & ~pred_b))

    total_discordant = n_a_only + n_b_only
    if total_discordant == 0:
        return {
            "contingency_table": {
                "both_correct": n_both_correct,
                "model_a_only": 0,
                "model_b_only": 0,
                "both_wrong": n_both_wrong
            },
            "chi2": 0.0,
            "p_value_exact": 1.0,
            "significant_05": False,
            "significant_01": False
        }

    chi2_stat = ((abs(n_a_only - n_b_only) - 1.0) ** 2) / total_discordant
    binom_res = stats.binomtest(n_a_only, total_discordant, p=0.5, alternative="two-sided")
    p_val_exact = float(binom_res.pvalue)

    return {
        "contingency_table": {
            "both_correct": n_both_correct,
            "model_a_only": n_a_only,
            "model_b_only": n_b_only,
            "both_wrong": n_both_wrong
        },
        "chi2": round(float(chi2_stat), 4),
        "p_value_exact": p_val_exact,
        "significant_05": bool(p_val_exact < 0.05),
        "significant_01": bool(p_val_exact < 0.01)
    }

def main():
    print("=" * 80)
    print(" STATISTICAL SIGNIFICANCE & 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)

    # 1. 10-Class Benchmark
    print("\n>>> 1. 10-Class Unified Benchmark (600 held-out test images)")
    d10 = np.load("outputs/cached_10class_5model_probs.npz")
    y_test_10 = d10["test_targets"]

    models_10 = {
        "ConvNeXt-Small 384": d10["test_convnext"],
        "EfficientNet-B3 384": d10["test_effnet"],
        "ResNet-50d 384": d10["test_resnet"],
        "BiomedCLIP-CBAM 224": d10["test_biomed"],
        "ViT-Base-384": d10["test_vit"],
    }

    quad_10 = (
        0.429 * d10["test_effnet"] +
        0.286 * d10["test_resnet"] +
        0.143 * d10["test_convnext"] +
        0.143 * d10["test_biomed"]
    )
    quad_10 = quad_10 / np.sum(quad_10, axis=1, keepdims=True)
    models_10["Unified Quad Ensemble (SOTA)"] = quad_10

    results_10 = {}
    for name, probs in models_10.items():
        point = compute_metrics(y_test_10, probs)
        ci = bootstrap_ci(y_test_10, probs, n_bootstraps=1000)
        results_10[name] = {"point_estimates": point, "confidence_intervals_95": ci}
        acc_s = f"{point['accuracy']*100:.2f}% [{ci['accuracy']['ci_lower']*100:.2f}%, {ci['accuracy']['ci_upper']*100:.2f}%]"
        f1_s = f"{point['macro_f1']*100:.2f}% [{ci['macro_f1']['ci_lower']*100:.2f}%, {ci['macro_f1']['ci_upper']*100:.2f}%]"
        auc_s = f"{point['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}]"
        print(f"  {name:30} | Acc: {acc_s} | F1: {f1_s} | AUC: {auc_s}")

    mcnemar_10 = {}
    print("\nMcNemar Significance Tests vs Unified Quad Ensemble (10-Class):")
    for name, probs in models_10.items():
        if name == "Unified Quad Ensemble (SOTA)":
            continue
        res = mcnemar_test(y_test_10, quad_10, probs)
        mcnemar_10[name] = res
        ct = res["contingency_table"]
        sig = "p < 0.01 (Extremely Sig)" if res["significant_01"] else ("p < 0.05 (Sig)" if res["significant_05"] else "p >= 0.05")
        print(f"  vs {name:20}: +{ct['model_a_only']:2d} / -{ct['model_b_only']:2d} discordant | chi2={res['chi2']:.2f} | p={res['p_value_exact']:.4e} -> {sig}")

    # 2. 4-Class Kaggle Benchmark
    print("\n>>> 2. 4-Class Kaggle Benchmark (423 held-out test images)")
    d4 = np.load("outputs/cached_4class_model_probs.npz")
    y_test_4 = d4["ground_truth"]

    models_4 = {
        "BiomedCLIP + CBAM": d4["BiomedCLIP_CBAM"],
        "ConvNeXt-Small v2": d4["ConvNeXt_Small_v2"],
        "ResNet-50d": d4["ResNet_50d"],
        "EfficientNet-B3": d4["EfficientNet_B3"],
        "DenseNet-121": d4["DenseNet_121"],
        "ViT-Base-384": d4["ViT_Base_384"],
    }

    quad_4 = (
        0.312 * d4["BiomedCLIP_CBAM"] +
        0.260 * d4["ConvNeXt_Small_v2"] +
        0.234 * d4["ResNet_50d"] +
        0.195 * d4["EfficientNet_B3"]
    )
    quad_4 = quad_4 / np.sum(quad_4, axis=1, keepdims=True)
    models_4["ResNet-Integrated Quad Ensemble (SOTA)"] = quad_4

    results_4 = {}
    for name, probs in models_4.items():
        point = compute_metrics(y_test_4, probs)
        ci = bootstrap_ci(y_test_4, probs, n_bootstraps=1000)
        results_4[name] = {"point_estimates": point, "confidence_intervals_95": ci}
        acc_s = f"{point['accuracy']*100:.2f}% [{ci['accuracy']['ci_lower']*100:.2f}%, {ci['accuracy']['ci_upper']*100:.2f}%]"
        f1_s = f"{point['macro_f1']*100:.2f}% [{ci['macro_f1']['ci_lower']*100:.2f}%, {ci['macro_f1']['ci_upper']*100:.2f}%]"
        auc_s = f"{point['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}]"
        print(f"  {name:35} | Acc: {acc_s} | F1: {f1_s} | AUC: {auc_s}")

    mcnemar_4 = {}
    print("\nMcNemar Significance Tests vs ResNet-Integrated Quad Ensemble (4-Class):")
    for name, probs in models_4.items():
        if name == "ResNet-Integrated Quad Ensemble (SOTA)":
            continue
        res = mcnemar_test(y_test_4, quad_4, probs)
        mcnemar_4[name] = res
        ct = res["contingency_table"]
        sig = "p < 0.01 (Extremely Sig)" if res["significant_01"] else ("p < 0.05 (Sig)" if res["significant_05"] else "p >= 0.05")
        print(f"  vs {name:20}: +{ct['model_a_only']:2d} / -{ct['model_b_only']:2d} discordant | chi2={res['chi2']:.2f} | p={res['p_value_exact']:.4e} -> {sig}")

    final_payload = {
        "ten_class": {
            "n_test_samples": len(y_test_10),
            "results": results_10,
            "mcnemar_against_ensemble": mcnemar_10
        },
        "four_class": {
            "n_test_samples": len(y_test_4),
            "results": results_4,
            "mcnemar_against_ensemble": mcnemar_4
        }
    }

    out_json = Path("outputs/statistical_significance_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    # Markdown Report Generation
    md_lines = [
        "# Empirical Statistical Rigor & Hypothesis Testing Report",
        "",
        "> **Project:** Retinal Disease Classification System  ",
        "> **Institution:** Department of Information Technology, Jadavpur University  ",
        "> **Supervisor:** Dr. Pawan Kumar Singh  ",
        "> **Methodology:** 1,000-iteration Bootstrap 95% Confidence Intervals & McNemar's Paired Discordance Test (Two-Sided Exact Binomial)",
        "",
        "---",
        "",
        "## 1. 10-Class Unified Benchmark (600 Untouched Test Images)",
        "",
        "### 95% Bootstrap Confidence Intervals",
        "",
        "| Architecture | Test Accuracy (95% CI) | Macro F1-Score (95% CI) | Macro ROC-AUC (95% CI) | Cohen's $\\kappa$ (95% CI) |",
        "|---|:---:|:---:|:---:|:---:|"
    ]

    for name, data in results_10.items():
        pt = data["point_estimates"]
        ci = data["confidence_intervals_95"]
        acc_s = f"**{pt['accuracy']*100:.2f}%** [{ci['accuracy']['ci_lower']*100:.2f}%, {ci['accuracy']['ci_upper']*100:.2f}%]"
        f1_s = f"{pt['macro_f1']*100:.2f}% [{ci['macro_f1']['ci_lower']*100:.2f}%, {ci['macro_f1']['ci_upper']*100:.2f}%]"
        auc_s = f"{pt['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}]"
        k_s = f"{pt['kappa']:.4f} [{ci['kappa']['ci_lower']:.4f}, {ci['kappa']['ci_upper']:.4f}]"
        bold_name = f"**{name}**" if "Ensemble" in name else name
        md_lines.append(f"| {bold_name} | {acc_s} | {f1_s} | {auc_s} | {k_s} |")

    md_lines.extend([
        "",
        "### McNemar's Paired Significance Tests (Unified Quad Ensemble vs Standalone Backbones)",
        "",
        "| Compared Model | Ensemble Only Wins ($b$) | Model Only Wins ($c$) | Continuity $\\chi^2$ | Exact Two-Sided $p$-value | Statistical Significance ($\\alpha=0.05$) |",
        "|---|:---:|:---:|:---:|:---:|:---:|"
    ])

    for name, res in mcnemar_10.items():
        ct = res["contingency_table"]
        sig_label = "**$p < 0.001$ (Highly Significant)**" if res["p_value_exact"] < 0.001 else ("**$p < 0.01$ (Very Significant)**" if res["p_value_exact"] < 0.01 else ("**$p < 0.05$ (Significant)**" if res["p_value_exact"] < 0.05 else "Not Significant ($p \\ge 0.05$)"))
        md_lines.append(f"| {name} | +{ct['model_a_only']} | -{ct['model_b_only']} | {res['chi2']:.2f} | {res['p_value_exact']:.4e} | {sig_label} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. 4-Class Kaggle Benchmark (423 Untouched Test Images)",
        "",
        "### 95% Bootstrap Confidence Intervals",
        "",
        "| Architecture | Test Accuracy (95% CI) | Macro F1-Score (95% CI) | Macro ROC-AUC (95% CI) | Cohen's $\\kappa$ (95% CI) |",
        "|---|:---:|:---:|:---:|:---:|"
    ])

    for name, data in results_4.items():
        pt = data["point_estimates"]
        ci = data["confidence_intervals_95"]
        acc_s = f"**{pt['accuracy']*100:.2f}%** [{ci['accuracy']['ci_lower']*100:.2f}%, {ci['accuracy']['ci_upper']*100:.2f}%]"
        f1_s = f"{pt['macro_f1']*100:.2f}% [{ci['macro_f1']['ci_lower']*100:.2f}%, {ci['macro_f1']['ci_upper']*100:.2f}%]"
        auc_s = f"{pt['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}]"
        k_s = f"{pt['kappa']:.4f} [{ci['kappa']['ci_lower']:.4f}, {ci['kappa']['ci_upper']:.4f}]"
        bold_name = f"**{name}**" if "Ensemble" in name else name
        md_lines.append(f"| {bold_name} | {acc_s} | {f1_s} | {auc_s} | {k_s} |")

    md_lines.extend([
        "",
        "### McNemar's Paired Significance Tests (ResNet-Integrated Quad Ensemble vs Standalone Backbones)",
        "",
        "| Compared Model | Ensemble Only Wins ($b$) | Model Only Wins ($c$) | Continuity $\\chi^2$ | Exact Two-Sided $p$-value | Statistical Significance ($\\alpha=0.05$) |",
        "|---|:---:|:---:|:---:|:---:|:---:|"
    ])

    for name, res in mcnemar_4.items():
        ct = res["contingency_table"]
        sig_label = "**$p < 0.001$ (Highly Significant)**" if res["p_value_exact"] < 0.001 else ("**$p < 0.01$ (Very Significant)**" if res["p_value_exact"] < 0.01 else ("**$p < 0.05$ (Significant)**" if res["p_value_exact"] < 0.05 else "Not Significant ($p \\ge 0.05$)"))
        md_lines.append(f"| {name} | +{ct['model_a_only']} | -{ct['model_b_only']} | {res['chi2']:.2f} | {res['p_value_exact']:.4e} | {sig_label} |")

    out_md = Path("docs/statistical_significance_analysis.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSUCCESS: Report saved to: {out_md}")
    print(f"SUCCESS: Results JSON saved to: {out_json}")

if __name__ == "__main__":
    main()

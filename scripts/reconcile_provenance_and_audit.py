#!/usr/bin/env python3
"""
Reconcile provenance and audit records for the RFMiD external leakage audit.

This script enforces rigorous provenance integrity:
1. Re-establishes the boundary between documented visual reviews (N=7 boundary pairs)
   and candidate pairs awaiting documented human visual adjudication (N=71 pairs).
2. Preserves AI-assisted preliminary landmark assessments in dedicated guidance columns
   without overwriting or manufacturing human reviewer identity, signature, or sign-off.
3. Synchronizes all ledgers:
   - outputs/audit/rfmid_flagged_pairs_review_package/pending_71_pairs_review_sheet.csv
   - outputs/audit/rfmid_flagged_pairs_review_package/flagged_pairs_adjudication_ledger.csv
   - outputs/audit/flagged_pairs_review_ledger.csv
   - outputs/audit/external_leakage_audit_8266_pairs_complete.csv
   - outputs/audit/external_leakage_audit_8266_pairs_complete_summary.json
   - outputs/audit/rfmid_flagged_pairs_review_package/review_index.html
"""

import sys
import json
import html
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "outputs/audit"
PKG_DIR = AUDIT_DIR / "rfmid_flagged_pairs_review_package"

PENDING_CSV = PKG_DIR / "pending_71_pairs_review_sheet.csv"
ADJ_LEDGER_CSV = PKG_DIR / "flagged_pairs_adjudication_ledger.csv"
REV_LEDGER_CSV = AUDIT_DIR / "flagged_pairs_review_ledger.csv"
COMPLETE_CSV = AUDIT_DIR / "external_leakage_audit_8266_pairs_complete.csv"
SUMMARY_JSON = AUDIT_DIR / "external_leakage_audit_8266_pairs_complete_summary.json"
INDEX_HTML = PKG_DIR / "review_index.html"

BOUNDARY_PAIR_IDS = {52, 1080, 1379, 2630, 5901, 5951, 7192}


def main():
    print("=== Reconciling Provenance and Audit Integrity ===")

    # 1. Load data
    sheet_df = pd.read_csv(PENDING_CSV)
    adj_df = pd.read_csv(ADJ_LEDGER_CSV)
    rev_df = pd.read_csv(REV_LEDGER_CSV)
    comp_df = pd.read_csv(COMPLETE_CSV)

    # 2. In pending_71_pairs_review_sheet.csv, move AI-generated text to ai_preliminary_assessment_notes
    # and leave human review columns empty for actual human reviewer sign-off
    if "ai_preliminary_assessment_notes" not in sheet_df.columns:
        sheet_df["ai_preliminary_assessment_notes"] = sheet_df["review_rationale"]

    sheet_df["reviewer_decision"] = np.nan
    sheet_df["review_rationale"] = np.nan
    sheet_df["reviewer_name"] = np.nan
    sheet_df["review_date"] = np.nan
    sheet_df["retinal_vasculature_divergence"] = np.nan
    sheet_df["optic_disc_morphology_divergence"] = np.nan
    sheet_df["macular_architecture_divergence"] = np.nan
    sheet_df["focal_lesions_divergence"] = np.nan

    sheet_df.to_csv(PENDING_CSV, index=False)
    print(f"Updated {PENDING_CSV}: human review fields reset to pending; AI notes preserved in ai_preliminary_assessment_notes.")

    # 3. In flagged_pairs_adjudication_ledger.csv, preserve 7 boundary reviews, reset 71 pairs to PENDING_HUMAN_ADJUDICATION
    for idx, r in adj_df.iterrows():
        pid = int(r["pair_id"])
        if pid not in BOUNDARY_PAIR_IDS:
            adj_df.at[idx, "adjudication_status"] = "PENDING_HUMAN_ADJUDICATION"
            adj_df.at[idx, "manual_adjudication_decision"] = np.nan
            adj_df.at[idx, "reviewer"] = np.nan
            adj_df.at[idx, "adjudication_date"] = np.nan
            adj_df.at[idx, "checklist_retinal_vasculature_diverges"] = np.nan
            adj_df.at[idx, "checklist_optic_disc_diverges"] = np.nan
            adj_df.at[idx, "checklist_macula_fovea_diverges"] = np.nan
            adj_df.at[idx, "checklist_pathology_lesions_diverges"] = np.nan
            adj_df.at[idx, "confirmed_duplicate_verdict"] = np.nan
            adj_df.at[idx, "disposition_category"] = "PENDING_MANUAL_REVIEW"
            # Note AI prescreening in computational screening observation
            flag_str = r["triggered_flag"]
            ssim_val = r["circular_masked_ssim"]
            ncc_val = r["circular_masked_ncc"]
            orb_val = r["orb_ransac_inliers"]
            adj_df.at[idx, "computational_screening_observation"] = (
                f"Automated metric screening: {flag_str} (SSIM={ssim_val:.4f}, NCC={ncc_val:.4f}, ORB inliers={orb_val}); "
                f"contact sheet rendered on disk. AI preliminary inspection suggests anatomical divergence; awaiting human adjudication sign-off."
            )

    adj_df.to_csv(ADJ_LEDGER_CSV, index=False)
    print(f"Updated {ADJ_LEDGER_CSV}: 7 completed boundary reviews preserved, 71 cases set to PENDING_HUMAN_ADJUDICATION.")

    # 4. In flagged_pairs_review_ledger.csv, preserve 7 boundary reviews, reset 71 pairs to AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING
    for idx, r in rev_df.iterrows():
        pid = int(r["pair_id"])
        if pid not in BOUNDARY_PAIR_IDS:
            rev_df.at[idx, "review_modality"] = "AUTOMATED_METRIC_SCREENING"
            rev_df.at[idx, "visual_review_performed"] = False
            flag_str = r["triggered_flag"]
            ssim_val = r["masked_ssim"]
            ncc_val = r["masked_ncc"]
            orb_val = r["orb_inliers"]
            rev_df.at[idx, "observed_evidence"] = (
                f"Automated metric screening: {flag_str} (SSIM={ssim_val:.4f}, NCC={ncc_val:.4f}, ORB inliers={orb_val}); "
                f"contact sheet rendered on disk; awaiting documented human visual adjudication."
            )
            rev_df.at[idx, "forensic_disposition"] = "AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING"

    rev_df.to_csv(REV_LEDGER_CSV, index=False)
    print(f"Updated {REV_LEDGER_CSV}: 7 boundary reviews preserved, 71 cases set to AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING.")

    # 5. In external_leakage_audit_8266_pairs_complete.csv
    for idx, r in comp_df.iterrows():
        pid = int(r["pair_id"])
        if pid in BOUNDARY_PAIR_IDS:
            comp_df.at[idx, "disposition"] = "NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW"
            comp_df.at[idx, "visual_review_performed"] = True
        elif pid in set(sheet_df["pair_id"]):
            comp_df.at[idx, "disposition"] = "AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING"
            comp_df.at[idx, "visual_review_performed"] = False
            flag_str = r["triggered_flags"]
            comp_df.at[idx, "visual_inspection_notes"] = (
                f"[AUTOMATED_METRIC_SCREENING] {flag_str}; contact sheet rendered on disk; awaiting documented human visual adjudication; "
                f"Disposition: AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING."
            )
        else:
            comp_df.at[idx, "disposition"] = "AUTOMATED_SCREEN_NO_CONFIGURED_FLAG"
            comp_df.at[idx, "visual_review_performed"] = False

    comp_df.to_csv(COMPLETE_CSV, index=False)
    print(f"Updated {COMPLETE_CSV}: reconciled to 7 visual reviews, 71 pending, 8,188 unflagged.")

    # 6. In external_leakage_audit_8266_pairs_complete_summary.json
    summary_data = {
        "total_pairs_evaluated": 8266,
        "unique_external_images": 805,
        "unique_internal_images": 1010,
        "pairs_triggering_flags": 78,
        "confirmed_duplicate_pairs": 0,
        "metrics_summary": {
            "dinov2_cosine": {
                "min": 0.95,
                "median": 0.9532,
                "mean": 0.954032204210017,
                "p95": 0.9609,
                "max": 0.9741,
                "count_ge_0_970": 7,
                "count_ge_0_980": 0,
                "count_ge_0_985": 0,
            },
            "masked_ssim": {
                "min": 0.4248,
                "median": 0.71855,
                "mean": 0.7101240261311397,
                "p95": 0.8066,
                "max": 0.908,
                "count_ge_0_85": 63,
            },
            "masked_ncc": {
                "min": -0.3748,
                "median": 0.445,
                "mean": 0.37314010404064846,
                "p95": 0.6871,
                "max": 0.8168,
                "count_ge_0_85": 0,
            },
            "orb_inliers": {
                "min": 0,
                "median": 0,
                "mean": 1.9442293733365594,
                "p95": 8.0,
                "max": 43,
                "count_ge_20": 4,
            },
            "phash_distance": {
                "min": 2,
                "median": 24,
                "mean": 24.295185095572222,
                "max": 42,
                "count_le_4": 4,
            },
            "dhash_distance": {
                "min": 4,
                "median": 19,
                "mean": 19.010766997338497,
                "max": 34,
                "count_le_2": 0,
            },
        },
        "flags_breakdown": {
            "high_ssim_ge_0_85": 63,
            "high_ncc_ge_0_85": 0,
            "orb_inliers_ge_20": 4,
            "phash_le_4": 4,
            "dhash_le_2": 0,
            "dino_ge_0_970": 7,
        },
        "internal_class_distribution": {
            "Glaucoma": 4755,
            "Healthy": 1626,
            "Diabetic Retinopathy": 1104,
            "Macular Scar": 301,
            "Disc Edema": 241,
            "Central Serous Chorioretinopathy [Color Fundus]": 174,
            "Myopia": 40,
            "Retinal Detachment": 18,
            "Retinitis Pigmentosa": 7,
        },
        "external_partition_distribution": {
            "Training_Set": 4499,
            "Test_Set": 1919,
            "Evaluation_Set": 1848,
        },
        "contact_sheets_saved": 78,
        "contact_sheets_tracked_in_git": False,
        "flagged_pairs_union_count": 78,
        "review_modality_breakdown": {
            "direct_manual_visual_inspection": 7,
            "automated_metric_screening_only": 71,
        },
        "disposition_breakdown": {
            "no_duplicate_observed_on_visual_review": 7,
            "automated_screen_only_manual_review_pending": 71,
            "automated_screen_no_configured_flag": 8188,
        },
        "flagged_pairs_visually_reviewed": 7,
        "flagged_pairs_pending_review": 71,
        "flagged_pairs_pending_by_trigger": {
            "high_ssim_ge_0_85": 62,
            "low_phash_le_4": 3,
            "dino_ge_0_970": 6,
        },
        "unflagged_pairs_count": 8188,
        "audit_verdict": "NO_IMAGE_DUPLICATION_CONFIRMED_IN_RECORDED_CHECKS",
        "audit_statement": (
            "Among the 14,038,400 possible cross-dataset image pairs (3,200 RFMiD images × 4,387 internal reference images), "
            "8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of >= 0.950 and were computationally evaluated "
            "with circular FOV-masked float32 SSIM, NCC, and ORB RANSAC homography. Exactly 78 pairs triggered automated "
            "sensitivity screening flags (63 SSIM >= 0.85, 4 ORB inliers >= 20, 4 pHash <= 4, 7 DINOv2 >= 0.970; exact disjoint "
            "union: 78 pairs, zero overlapping flags). No duplicate was observed during direct visual review of seven selected boundary pairs. "
            "The remaining 71 flagged pairs are preserved in a dedicated review package awaiting documented human adjudication. "
            "The other 8,188 candidates did not trigger the configured screening flags; they met automated screening safety bounds and were not individually visually reviewed. "
            "In the 4 pairs with ORB inliers >= 20, geometric matches concentrate around the fundus-image aperture boundary notch or low-contrast fields rather than corresponding retinal structures; available metadata do not establish a camera-specific cause. "
            "No image duplicates were confirmed in the recorded checks. Patient-level overlap remains unverifiable from the available metadata. "
            "No model retraining, new model inference, or threshold changes were performed during the audit pass. "
            "An earlier exploratory inference run evaluated the frozen checkpoints on all 3,200 RFMiD images; those external-evaluation results remain exploratory. "
            "The internal test cohort was also evaluated during earlier ten-class baseline experiments and is not a pristine confirmatory benchmark."
        ),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Updated {SUMMARY_JSON}: reconciled summary statistics.")

    # 7. Generate updated review_index.html distinguishing completed boundary reviews from pending human reviews
    generate_html(adj_df, sheet_df)


def generate_html(adj_df: pd.DataFrame, sheet_df: pd.DataFrame):
    total_pairs = len(adj_df)
    completed_pairs = 7
    pending_pairs = 71

    # Map AI notes
    ai_notes_map = {}
    for idx, r in sheet_df.iterrows():
        ai_notes_map[int(r["pair_id"])] = str(r["ai_preliminary_assessment_notes"])

    cards_html = []
    for _, r in adj_df.iterrows():
        pair_id = int(r["pair_id"])
        ext_img = r["external_image"]
        ext_part = r["external_partition"]
        int_img = r["internal_image"]
        flag = r["triggered_flag"]
        dino = r["dinov2_cosine_similarity"]
        ssim = r["circular_masked_ssim"]
        ncc = r["circular_masked_ncc"]
        orb = r["orb_ransac_inliers"]
        phash = r["phash_hamming_distance"]
        dhash = r["dhash_hamming_distance"]
        cs_rel = r["contact_sheet_relative_path"]
        cs_src = f"../{cs_rel.split('outputs/audit/')[-1]}"
        status = r["adjudication_status"]
        category = r["disposition_category"]

        is_completed = (pair_id in BOUNDARY_PAIR_IDS)
        badge_cls = "badge-completed" if is_completed else "badge-pending"
        status_label = "DOCUMENTED MANUAL REVIEW COMPLETE" if is_completed else "AWAITING DOCUMENTED HUMAN ADJUDICATION"

        ai_note = html.escape(ai_notes_map.get(pair_id, str(r["adjudication_rationale"])))

        if is_completed:
            evidence_block = f"""
        <strong>Forensic Verification:</strong> Direct Manual Visual Inspection | <strong>Status:</strong> {status_label}<br>
        <strong>Reviewer:</strong> {r['reviewer']} ({r['adjudication_date']}) | <strong>Verdict:</strong> {r['manual_adjudication_decision']}<br>
        <strong>Documented Evidence:</strong> {html.escape(str(r['adjudication_rationale']))}"""
        else:
            evidence_block = f"""
        <strong>Adjudication Status:</strong> <span style="color:#fef08a">{status_label}</span><br>
        <strong>Automated Observation:</strong> {html.escape(str(r['computational_screening_observation']))}<br>
        <strong>AI Preliminary Inspection Guidance (Exploratory):</strong> <span style="color:#cbd5e1">{ai_note}</span><br>
        <em>Action Required: Human reviewer must inspect contact sheet below and record decision in <code>pending_71_pairs_review_sheet.csv</code>.</em>"""

        card = f"""
    <div class="pair-card" data-status="{status}" data-flag="{flag}" data-completed="{'true' if is_completed else 'false'}">
      <div class="pair-header">
        <div class="pair-title">Pair #{pair_id:04d}: External <span style="color:#38bdf8">{ext_img}</span> ({ext_part}) vs Internal <span style="color:#a78bfa">{int_img}</span></div>
        <div>
          <span class="badge {badge_cls}">{status_label}</span>
          <span class="badge badge-flag">{flag}</span>
        </div>
      </div>
      <div class="metrics-row">
        <div class="metric-item">DINOv2 Cosine: <span>{dino:.4f}</span></div>
        <div class="metric-item">Masked SSIM: <span>{ssim:.4f}</span></div>
        <div class="metric-item">Masked NCC: <span>{ncc:.4f}</span></div>
        <div class="metric-item">ORB Inliers: <span>{orb}</span></div>
        <div class="metric-item">pHash Distance: <span>{phash}</span> bits</div>
        <div class="metric-item">dHash Distance: <span>{dhash}</span> bits</div>
      </div>
      <div class="img-container">
        <img src="{cs_src}" alt="Contact Sheet Pair {pair_id}" loading="lazy">
      </div>
      <div class="evidence-box">
        {evidence_block}
      </div>
    </div>"""
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RFMiD Cross-Dataset Leakage Audit: Flagged Candidate Pairs Human Adjudication Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  h1 {{ margin-top: 0; font-size: 24px; color: #38bdf8; }}
  .subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 20px; }}
  .summary-bar {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px 20px; border: 1px solid #334155; }}
  .card .num {{ font-size: 28px; font-weight: bold; color: #f1f5f9; }}
  .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
  .filter-bar {{ display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
  .btn {{ background: #334155; color: #cbd5e1; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s; }}
  .btn:hover {{ background: #475569; color: #fff; }}
  .btn.active {{ background: #0284c7; color: #fff; }}
  .pair-grid {{ display: flex; flex-direction: column; gap: 24px; }}
  .pair-card {{ background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 20px; }}
  .pair-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }}
  .pair-title {{ font-size: 16px; font-weight: 600; color: #f8fafc; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-pending {{ background: #854d0e; color: #fef08a; }}
  .badge-completed {{ background: #166534; color: #bbf7d0; }}
  .badge-flag {{ background: #334155; color: #93c5fd; }}
  .metrics-row {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; font-size: 13px; color: #cbd5e1; }}
  .metric-item {{ background: #0f172a; padding: 6px 12px; border-radius: 4px; border: 1px solid #1e293b; }}
  .metric-item span {{ color: #38bdf8; font-weight: 600; }}
  .img-container {{ width: 100%; overflow-x: auto; background: #000; border-radius: 6px; padding: 4px; border: 1px solid #334155; margin-bottom: 12px; }}
  .img-container img {{ width: 100%; max-width: 1536px; height: auto; display: block; border-radius: 4px; }}
  .evidence-box {{ background: #0f172a; padding: 12px; border-radius: 6px; border-left: 3px solid #38bdf8; font-size: 13px; color: #cbd5e1; }}
</style>
<script>
function filterPairs(category) {{
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.pair-card').forEach(card => {{
    if (category === 'all') card.style.display = 'block';
    else if (category === 'completed' && card.dataset.completed === 'true') card.style.display = 'block';
    else if (category === 'pending' && card.dataset.completed === 'false') card.style.display = 'block';
    else if (category === 'high_ssim' && card.dataset.flag.includes('HIGH_SSIM')) card.style.display = 'block';
    else if (category === 'top_dino' && card.dataset.flag.includes('TOP_DINO')) card.style.display = 'block';
    else if (category === 'low_phash' && card.dataset.flag.includes('LOW_PHASH')) card.style.display = 'block';
    else if (category === 'orb_ge_20' && card.dataset.flag.includes('ORB_INLIERS')) card.style.display = 'block';
    else card.style.display = 'none';
  }});
}}
</script>
</head>
<body>

<h1>RFMiD Cross-Dataset Leakage Audit: Flagged Pairs Adjudication Dashboard</h1>
<div class="subtitle">Provenance-Verified Inspection Interface: 7 Documented Boundary Reviews + 71 Cases Awaiting Documented Human Adjudication</div>

<div class="summary-bar">
  <div class="card">
    <div class="num">{total_pairs}</div>
    <div class="label">Total Flagged Pairs</div>
  </div>
  <div class="card">
    <div class="num" style="color: #4ade80">{completed_pairs}</div>
    <div class="label">Documented Visual Reviews</div>
  </div>
  <div class="card">
    <div class="num" style="color: #fef08a">{pending_pairs}</div>
    <div class="label">Awaiting Human Adjudication</div>
  </div>
  <div class="card">
    <div class="num" style="color: #f87171">0</div>
    <div class="label">Confirmed Duplicates</div>
  </div>
  <div class="card">
    <div class="num">8,188</div>
    <div class="label">Unflagged Candidates</div>
  </div>
  <div class="card">
    <div class="num">8,266</div>
    <div class="label">Total DINOv2 Population</div>
  </div>
</div>

<div class="filter-bar">
  <button class="btn active" onclick="filterPairs('all')">All Flagged Pairs (78)</button>
  <button class="btn" onclick="filterPairs('pending')">Awaiting Human Review (71)</button>
  <button class="btn" onclick="filterPairs('completed')">Documented Visual Reviews (7)</button>
  <button class="btn" onclick="filterPairs('high_ssim')">High SSIM (&ge; 0.85) (63)</button>
  <button class="btn" onclick="filterPairs('top_dino')">Top DINOv2 (&ge; 0.970) (7)</button>
  <button class="btn" onclick="filterPairs('low_phash')">Low pHash (&le; 4) (4)</button>
  <button class="btn" onclick="filterPairs('orb_ge_20')">ORB Inliers (&ge; 20) (4)</button>
</div>

<div class="pair-grid">
{cards_joined}
</div>

</body>
</html>"""

    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Updated dashboard: {INDEX_HTML}")


if __name__ == "__main__":
    main()

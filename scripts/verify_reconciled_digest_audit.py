#!/usr/bin/env python3
"""
Reconciled Path-Keyed Digest Duplicate Screening Audit
=====================================================

Performs path-keyed cryptographic digest comparisons between the external RFMiD cohort
(N=3,200) and internal reference cohorts:
  1. Canonical frozen benchmark (data_clean/, N=3,747) via SHA-256
  2. Eligible reference pool (clean_split_manifest.csv, N=4,387) via SHA-256 and MD5

Key Fix:
  - Uses full relative path keys (e.g. "Training/1.png", "Evaluation/1.png", "Test/1.png")
    to guarantee zero key-collision overwriting across partitions.
  - Verifies that all 3,200 RFMiD images, 3,747 clean benchmark images, and 4,387
    eligible pool images are physically present on disk and successfully digested.
"""

import hashlib
import json
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RFMID_ROOT = REPO_ROOT / "data_external" / "rfmid_v1"
CLEAN_ROOT = REPO_ROOT / "data_clean"
POOL_ROOT = REPO_ROOT / "Eye Disease Image Dataset" / "Original Dataset" / "Original Dataset"
MANIFEST_PATH = REPO_ROOT / "outputs" / "audit" / "clean_split_manifest.csv"
OUT_SUMMARY_JSON = REPO_ROOT / "outputs" / "audit" / "reconciled_digest_audit_summary.json"


def main():
    t0 = time.time()
    print("=" * 70)
    print("RECONCILED PATH-KEYED CRYPTOGRAPHIC DIGEST DUPLICATE SCREENING AUDIT")
    print("=" * 70)

    # 1. RFMiD external cohort
    print("\n[1/4] Indexing and hashing RFMiD cohort (N=3,200)...")
    rfmid_paths = sorted(RFMID_ROOT.rglob("*.png"))
    assert len(rfmid_paths) == 3200, f"Expected 3200 RFMiD images, found {len(rfmid_paths)}"

    rfmid_sha256 = {}
    rfmid_md5 = {}
    for p in rfmid_paths:
        rel_key = str(p.relative_to(RFMID_ROOT))
        data = p.read_bytes()
        rfmid_sha256[rel_key] = hashlib.sha256(data).hexdigest()
        rfmid_md5[rel_key] = hashlib.md5(data).hexdigest()

    assert len(rfmid_sha256) == 3200, f"Expected 3200 unique path-keyed hashes, got {len(rfmid_sha256)}"
    assert len(rfmid_md5) == 3200
    print(f"  -> Successfully digested 3,200 RFMiD images using unique relative paths.")

    # 2. Canonical frozen benchmark (data_clean/, N=3,747)
    print("\n[2/4] Indexing and hashing canonical frozen benchmark (data_clean/, N=3,747)...")
    clean_paths = sorted([p for p in CLEAN_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    assert len(clean_paths) == 3747, f"Expected 3747 data_clean images, found {len(clean_paths)}"

    clean_sha256 = {}
    clean_md5 = {}
    canonical_splits = {"train": 0, "val": 0, "test": 0}
    pterygium_splits = {"train": 0, "val": 0, "test": 0}
    fundus_splits = {"train": 0, "val": 0, "test": 0}

    for p in clean_paths:
        rel_key = str(p.relative_to(CLEAN_ROOT))
        data = p.read_bytes()
        clean_sha256[rel_key] = hashlib.sha256(data).hexdigest()
        clean_md5[rel_key] = hashlib.md5(data).hexdigest()
        parts = p.relative_to(CLEAN_ROOT).parts
        split = parts[0]
        if split in canonical_splits:
            canonical_splits[split] += 1
            if len(parts) > 1 and parts[1] == "Pterygium":
                pterygium_splits[split] += 1
            else:
                fundus_splits[split] += 1

    assert canonical_splits == {"train": 2622, "val": 562, "test": 563}, f"Unexpected clean split counts: {canonical_splits}"
    assert pterygium_splits == {"train": 10, "val": 2, "test": 2}, f"Unexpected pterygium split counts: {pterygium_splits}"
    assert fundus_splits == {"train": 2612, "val": 560, "test": 561}, f"Unexpected 9-class fundus split counts: {fundus_splits}"

    assert len(clean_sha256) == 3747, f"Expected 3747 unique clean hashes, got {len(clean_sha256)}"
    assert len(clean_md5) == 3747
    print(f"  -> Successfully digested 3,747 canonical benchmark images (Train: {canonical_splits['train']}, Val: {canonical_splits['val']}, Test: {canonical_splits['test']}).")
    print(f"     * Anterior Pterygium: {sum(pterygium_splits.values())} (Train: {pterygium_splits['train']}, Val: {pterygium_splits['val']}, Test: {pterygium_splits['test']})")
    print(f"     * Posterior Fundus 9-class: {sum(fundus_splits.values())} (Train: {fundus_splits['train']}, Val: {fundus_splits['val']}, Test: {fundus_splits['test']})")

    # 3. Eligible reference pool (clean_split_manifest.csv, N=4,387)
    print("\n[3/4] Indexing and hashing eligible reference pool (N=4,387)...")
    df_manifest = pd.read_csv(MANIFEST_PATH)
    assert len(df_manifest) == 4387, f"Expected 4387 manifest rows, found {len(df_manifest)}"

    pool_sha256 = {}
    pool_md5 = {}
    manifest_md5_matches = 0

    for _, row in df_manifest.iterrows():
        rel_path = row["original_path"]
        full_p = POOL_ROOT / rel_path
        assert full_p.exists(), f"Missing pool image file: {full_p}"
        data = full_p.read_bytes()
        s = hashlib.sha256(data).hexdigest()
        m = hashlib.md5(data).hexdigest()
        pool_sha256[rel_path] = s
        pool_md5[rel_path] = m
        if m == row["md5"]:
            manifest_md5_matches += 1

    assert len(pool_sha256) == 4387, f"Expected 4387 unique pool hashes, got {len(pool_sha256)}"
    assert len(pool_md5) == 4387
    assert manifest_md5_matches == 4387, f"Manifest MD5 verification failed: {manifest_md5_matches}/4387 matched"

    pool_splits_series = df_manifest["split"].value_counts()
    pool_splits = {
        "train": int(pool_splits_series.get("train", 0)),
        "val": int(pool_splits_series.get("val", 0)),
        "test": int(pool_splits_series.get("test", 0)),
    }
    assert pool_splits == {"train": 3073, "val": 657, "test": 657}, f"Unexpected pool splits: {pool_splits}"
    print(f"  -> Successfully digested 4,387 pool images (Train: {pool_splits['train']}, Val: {pool_splits['val']}, Test: {pool_splits['test']}); 100% (4,387/4,387) match clean_split_manifest.csv MD5s.")

    # 4. Cross-cohort digest comparisons
    print("\n[4/4] Evaluating pairwise cryptographic digest collisions...")

    # Comparison 1: SHA-256 RFMiD vs data_clean (11,990,400 comparisons)
    rfmid_sha_set = set(rfmid_sha256.values())
    clean_sha_set = set(clean_sha256.values())
    sha_matches_clean = rfmid_sha_set.intersection(clean_sha_set)

    # Comparison 2: SHA-256 RFMiD vs Eligible Pool (14,038,400 comparisons)
    pool_sha_set = set(pool_sha256.values())
    sha_matches_pool = rfmid_sha_set.intersection(pool_sha_set)

    # Comparison 3: MD5 RFMiD vs Eligible Pool (14,038,400 comparisons)
    rfmid_md5_set = set(rfmid_md5.values())
    pool_md5_set = set(pool_md5.values())
    md5_matches_pool = rfmid_md5_set.intersection(pool_md5_set)

    comp_clean_count = len(rfmid_sha256) * len(clean_sha256)
    comp_pool_count = len(rfmid_sha256) * len(pool_sha256)

    print(f"  - SHA-256: RFMiD (3,200) vs Canonical Benchmark (3,747): {comp_clean_count:,} comparisons | Matches: {len(sha_matches_clean)}")
    print(f"  - SHA-256: RFMiD (3,200) vs Eligible Pool (4,387):       {comp_pool_count:,} comparisons | Matches: {len(sha_matches_pool)}")
    print(f"  - MD5:     RFMiD (3,200) vs Eligible Pool (4,387):       {comp_pool_count:,} comparisons | Matches: {len(md5_matches_pool)}")

    assert len(sha_matches_clean) == 0, f"Found {len(sha_matches_clean)} SHA-256 matching digests against data_clean!"
    assert len(sha_matches_pool) == 0, f"Found {len(sha_matches_pool)} SHA-256 matching digests against eligible pool!"
    assert len(md5_matches_pool) == 0, f"Found {len(md5_matches_pool)} MD5 matching digests against eligible pool!"

    summary = {
        "audit_name": "Reconciled Path-Keyed Cryptographic Digest Duplicate Screening Audit",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ZERO_MATCHING_DIGESTS_CONFIRMED",
        "rfmid_external_cohort": {
            "total_images_found": len(rfmid_paths),
            "unique_path_keyed_hashes_retained": len(rfmid_sha256),
            "missing_images": 0,
            "partition_breakdown": {
                "Training": len([p for p in rfmid_paths if "Training" in str(p)]),
                "Evaluation": len([p for p in rfmid_paths if "Evaluation" in str(p)]),
                "Test": len([p for p in rfmid_paths if "Test" in str(p)]),
            }
        },
        "canonical_frozen_benchmark": {
            "path": "data_clean",
            "total_images_found": len(clean_paths),
            "unique_path_keyed_hashes_retained": len(clean_sha256),
            "missing_images": 0,
            "canonical_split_breakdown": canonical_splits,
            "includes_anterior_pterygium_images": 14,
            "pterygium_split_breakdown": pterygium_splits,
            "posterior_fundus_9class_images": 3733,
            "fundus_9class_split_breakdown": fundus_splits,
            "sha256_comparisons": comp_clean_count,
            "sha256_matching_digests": len(sha_matches_clean)
        },
        "eligible_reference_pool": {
            "path": "Eye Disease Image Dataset/Original Dataset/Original Dataset",
            "manifest": "outputs/audit/clean_split_manifest.csv",
            "total_images_found": len(pool_sha256),
            "unique_path_keyed_hashes_retained": len(pool_sha256),
            "missing_images": 0,
            "eligible_pool_split_breakdown": pool_splits,
            "manifest_md5_bitwise_verification_rate": f"{manifest_md5_matches}/{len(df_manifest)} (100%)",
            "sha256_comparisons": comp_pool_count,
            "sha256_matching_digests": len(sha_matches_pool),
            "md5_comparisons": comp_pool_count,
            "md5_matching_digests": len(md5_matches_pool)
        },
        "methodological_conclusions": {
            "zero_exact_duplicates_across_canonical_benchmark": True,
            "zero_exact_duplicates_across_eligible_pool_sha256": True,
            "zero_exact_duplicates_across_eligible_pool_md5": True,
            "patient_level_overlap_limitation": "Patient-level overlap between RFMiD and the internal dataset cannot be assessed from the available public metadata because patient identifiers or validated cross-dataset linkage information are unavailable."
        },
        "elapsed_seconds": round(time.time() - t0, 2)
    }

    OUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PASS] Reconciled digest audit summary saved to: {OUT_SUMMARY_JSON}")
    print(f"Completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()

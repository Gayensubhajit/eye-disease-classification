#!/usr/bin/env python3
"""
Physical Build & Integrity Verification of data_clean/ Benchmark (V4)

This script:
1. Validates frozen manifest (outputs/audit/final_clean_split_manifest_v4.csv).
2. Physically copies only the 3,747 retained images from raw archive into data_clean/{split}/{class}/.
3. Performs rigorous post-build physical validation:
   - File counts: exactly 3,747 image files (.jpg, .jpeg, .png).
   - Non-image files: exactly 0 unexpected files.
   - Per-class per-split counts: exactly matches frozen manifest across all 10 classes.
   - Split totals: Train=2622, Val=562, Test=563.
   - Cross-split MD5 overlap: 0.
   - Quarantine contamination: 0.
   - Excluded burst contamination: 0.
   - Rigorous V3 Pair Endpoint Inspection: for all 86 V3 high-confidence relationships,
     verifies the actual physical location/absence of both image endpoints.
     Ensures 0 cross-split leakage, 0 redundant burst contamination, and 0 quarantine contamination.
4. Generates data_clean_v4_inventory.csv, data_clean_v4_checksums.sha256, and data_clean_v4_validation.csv.
"""

import os
import sys
import shutil
import hashlib
import time
import argparse
import pandas as pd

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_MANIFEST_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "final_clean_split_manifest_v4.csv")
RAW_ARCHIVE_DIR = os.path.join(REPO_ROOT, "Eye Disease Image Dataset", "Original Dataset", "Original Dataset")
DATA_CLEAN_DIR = os.path.join(REPO_ROOT, "data_clean")

QUARANTINE_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "v4_quarantine.csv")
MEMBERS_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "v4_group_members.csv")
V3_RECONCILIATION_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "v4_v3_reconciliation.csv")

OUT_INVENTORY_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "data_clean_v4_inventory.csv")
OUT_CHECKSUMS_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "data_clean_v4_checksums.sha256")
OUT_VALIDATION_PATH = os.path.join(REPO_ROOT, "outputs", "audit", "data_clean_v4_validation.csv")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def compute_hashes(filepath):
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def build_dataset(df_manifest):
    print("\n--- Step 1: Physical File Copy from Raw Archive ---")
    splits = ['train', 'val', 'test']
    classes = sorted(df_manifest['class'].unique())
    N_expected = len(df_manifest)

    # Clean existing data_clean directory if present to avoid stale files
    if os.path.exists(DATA_CLEAN_DIR):
        print(f"Cleaning existing directory {DATA_CLEAN_DIR}...")
        shutil.rmtree(DATA_CLEAN_DIR)

    for s in splits:
        for c in classes:
            target_dir = os.path.join(DATA_CLEAN_DIR, s, c)
            os.makedirs(target_dir, exist_ok=True)

    inventory_rows = []
    sha256_lines = []
    copy_count = 0
    t_copy = time.time()

    for idx, r in df_manifest.iterrows():
        rel_p = r['clean_image_path']
        src_path = os.path.join(RAW_ARCHIVE_DIR, rel_p)
        assert os.path.exists(src_path), f"Source file missing: {src_path}"
        
        sp = r['split']
        cls = r['class']
        gid = r['source_group_id']
        expected_md5 = r['md5']
        
        dst_path = os.path.join(DATA_CLEAN_DIR, sp, rel_p)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        copy_count += 1
        
        obs_md5, obs_sha256 = compute_hashes(dst_path)
        assert obs_md5 == expected_md5, f"MD5 mismatch for {dst_path}! Expected {expected_md5}, got {obs_md5}"
        
        file_size = os.path.getsize(dst_path)
        clean_rel_path = os.path.join(sp, rel_p)
        
        inventory_rows.append({
            'clean_image_path': clean_rel_path,
            'original_archive_path': rel_p,
            'split': sp,
            'class': cls,
            'source_group_id': gid,
            'file_size_bytes': file_size,
            'md5': obs_md5,
            'sha256': obs_sha256
        })
        
        sha256_lines.append(f"{obs_sha256}  {clean_rel_path}\n")
        
        if copy_count % 500 == 0 or copy_count == N_expected:
            print(f"  Copied and verified {copy_count}/{N_expected} files in {time.time() - t_copy:.1f}s")

    df_inv = pd.DataFrame(inventory_rows)
    df_inv.to_csv(OUT_INVENTORY_PATH, index=False)
    print(f"Saved inventory catalog to {OUT_INVENTORY_PATH}")

    with open(OUT_CHECKSUMS_PATH, 'w') as f:
        f.writelines(sorted(sha256_lines))
    print(f"Saved SHA-256 checksums to {OUT_CHECKSUMS_PATH}")
    return df_inv

def validate_dataset(df_manifest):
    print("\n--- Step 2: Rigorous Independent Post-Build Physical Verification ---")
    N_expected = len(df_manifest)
    splits = ['train', 'val', 'test']
    classes = sorted(df_manifest['class'].unique())

    # 1. Scan physical data_clean directory
    all_files = []
    image_files = []
    non_image_files = []

    for root, dirs, files in os.walk(DATA_CLEAN_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            rel_to_clean = os.path.relpath(full_p, DATA_CLEAN_DIR)
            all_files.append((rel_to_clean, full_p))
            _, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_files.append((rel_to_clean, full_p))
            else:
                non_image_files.append((rel_to_clean, full_p))

    N_physical_total = len(all_files)
    N_physical_images = len(image_files)
    N_non_images = len(non_image_files)

    print(f"1. Total physical files in {DATA_CLEAN_DIR}: {N_physical_total} (Expected: {N_expected})")
    print(f"   Image files: {N_physical_images} | Non-image files: {N_non_images}")

    if N_non_images > 0:
        print(f"   ERROR: Found unexpected non-image files:")
        for rel_p, _ in non_image_files:
            print(f"     - {rel_p}")
    assert N_non_images == 0, f"Found {N_non_images} non-image files in {DATA_CLEAN_DIR}!"
    assert N_physical_images == N_expected, f"Physical image count mismatch: {N_physical_images} vs {N_expected}"
    assert N_physical_total == N_expected, f"Total physical file count mismatch: {N_physical_total} vs {N_expected}"

    # 2. Per-class per-split distribution & exact manifest assertion
    print("\n2. Per-Class and Per-Split Distribution Verification against Manifest:")
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_split_counts = {s: {c: 0 for c in classes} for s in splits}
    physical_md5s = {'train': set(), 'val': set(), 'test': set()}
    physical_lookup = {}  # filename -> (split, class, full_path)

    for rel_p, full_p in image_files:
        parts = rel_p.split(os.sep)
        sp = parts[0]
        cls = parts[1]
        fname = parts[2]
        
        split_counts[sp] += 1
        class_split_counts[sp][cls] += 1
        m, _ = compute_hashes(full_p)
        physical_md5s[sp].add(m)
        physical_lookup[fname] = (sp, cls, full_p)

    # Build comparison table
    manifest_matrix = {}
    mismatch_count = 0
    comparison_rows = []

    for c in classes:
        row = {'class': c}
        for s in splits:
            m_cnt = len(df_manifest[(df_manifest['split'] == s) & (df_manifest['class'] == c)])
            p_cnt = class_split_counts[s][c]
            row[f'{s}_manifest'] = m_cnt
            row[f'{s}_physical'] = p_cnt
            if m_cnt != p_cnt:
                mismatch_count += 1
                row[f'{s}_match'] = 'FAIL'
            else:
                row[f'{s}_match'] = 'PASS'
        row['total_manifest'] = sum(row[f'{s}_manifest'] for s in splits)
        row['total_physical'] = sum(row[f'{s}_physical'] for s in splits)
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison[['class', 'train_physical', 'val_physical', 'test_physical', 'total_physical']].to_string(index=False))

    print(f"\n   Class-split exact matches: {30 - mismatch_count}/30 cells matching manifest exactly.")
    assert mismatch_count == 0, f"Detected {mismatch_count} class-split count discrepancies!"
    assert split_counts['train'] == 2622, f"Expected 2622 train, got {split_counts['train']}"
    assert split_counts['val'] == 562, f"Expected 562 val, got {split_counts['val']}"
    assert split_counts['test'] == 563, f"Expected 563 test, got {split_counts['test']}"

    # 3. MD5 Cross-Split Overlap
    md5_tr_va = len(physical_md5s['train'].intersection(physical_md5s['val']))
    md5_tr_te = len(physical_md5s['train'].intersection(physical_md5s['test']))
    md5_va_te = len(physical_md5s['val'].intersection(physical_md5s['test']))
    total_md5_overlap = md5_tr_va + md5_tr_te + md5_va_te
    print(f"\n3. Physical MD5 Cross-Split Overlap: {total_md5_overlap}")
    assert total_md5_overlap == 0, f"Cross-split MD5 overlap detected: {total_md5_overlap}"

    # 4. Quarantine and Excluded Burst Sets Check
    df_quarantine = pd.read_csv(QUARANTINE_PATH)
    quarantine_set = set(df_quarantine['image_path'])
    df_members = pd.read_csv(MEMBERS_PATH)
    excluded_burst_set = set(df_members[df_members['role'] == 'BURST_REDUNDANT']['image_path'])

    quarantine_hits = 0
    excluded_burst_hits = 0

    for rel_p, _ in image_files:
        # orig_rel is class/filename
        orig_rel = os.path.join(*rel_p.split(os.sep)[1:])
        if orig_rel in quarantine_set:
            quarantine_hits += 1
        if orig_rel in excluded_burst_set:
            excluded_burst_hits += 1

    print(f"\n4. Contamination Check:")
    print(f"   Quarantined images detected in physical directory: {quarantine_hits}")
    print(f"   Excluded burst redundant images detected in physical directory: {excluded_burst_hits}")
    assert quarantine_hits == 0, f"Found {quarantine_hits} quarantined images in data_clean!"
    assert excluded_burst_hits == 0, f"Found {excluded_burst_hits} excluded burst images in data_clean!"

    # 5. Methodological V3 High-Confidence Pair Endpoint Inspection
    print(f"\n5. Methodological V3 Endpoint Inspection (All 86 Relationships):")
    df_v3_rec = pd.read_csv(V3_RECONCILIATION_PATH)
    assert len(df_v3_rec) == 86, f"Expected 86 V3 pairs, got {len(df_v3_rec)}"

    v3_both_present = 0
    v3_cross_split_leakage = 0
    v3_one_present_rep = 0
    v3_neither_present = 0
    v3_quarantine_hits = 0
    v3_burst_redundant_hits = 0

    for idx, r in df_v3_rec.iterrows():
        f_a = r['image_a']
        f_b = r['image_b']
        loc_a = physical_lookup.get(f_a)
        loc_b = physical_lookup.get(f_b)

        sp_a = loc_a[0] if loc_a else 'ABSENT'
        sp_b = loc_b[0] if loc_b else 'ABSENT'

        # Endpoint presence
        if loc_a and loc_b:
            v3_both_present += 1
            if sp_a != sp_b:
                v3_cross_split_leakage += 1
                print(f"   FATAL CROSS-SPLIT LEAK: {f_a} ({sp_a}) <-> {f_b} ({sp_b})")
            else:
                print(f"   WARNING INTRA-SPLIT DUPLICATION: {f_a} and {f_b} both in {sp_a}")
        elif loc_a or loc_b:
            v3_one_present_rep += 1
        else:
            v3_neither_present += 1

        # Check quarantine status of endpoints
        if r['role_a'] in ('CONFLICT_QUARANTINED', 'CHAIN_QUARANTINED') and loc_a:
            v3_quarantine_hits += 1
        if r['role_b'] in ('CONFLICT_QUARANTINED', 'CHAIN_QUARANTINED') and loc_b:
            v3_quarantine_hits += 1

        # Check redundant status of endpoints
        if r['role_a'] == 'BURST_REDUNDANT' and loc_a:
            v3_burst_redundant_hits += 1
        if r['role_b'] == 'BURST_REDUNDANT' and loc_b:
            v3_burst_redundant_hits += 1

    print(f"   - Total V3 pairs inspected: {len(df_v3_rec)}")
    print(f"   - Pairs with both endpoints in data_clean: {v3_both_present} (Expected: 0)")
    print(f"   - Cross-split leakage pairs: {v3_cross_split_leakage} (Expected: 0)")
    print(f"   - Pairs with exactly 1 endpoint retained (Representative): {v3_one_present_rep} (Expected: 61)")
    print(f"   - Pairs with neither endpoint in data_clean (Quarantined/Redundant): {v3_neither_present} (Expected: 25)")
    print(f"   - Quarantined V3 endpoints found in data_clean: {v3_quarantine_hits} (Expected: 0)")
    print(f"   - Excluded burst redundant endpoints found in data_clean: {v3_burst_redundant_hits} (Expected: 0)")

    assert v3_both_present == 0, f"Detected {v3_both_present} V3 pairs with both members present!"
    assert v3_cross_split_leakage == 0, f"Detected {v3_cross_split_leakage} V3 pairs crossing splits!"
    assert v3_one_present_rep == 61, f"Expected 61 representative-retained V3 pairs, got {v3_one_present_rep}"
    assert v3_neither_present == 25, f"Expected 25 excluded V3 pairs, got {v3_neither_present}"
    assert v3_quarantine_hits == 0, f"Detected {v3_quarantine_hits} quarantined V3 endpoints in data_clean!"
    assert v3_burst_redundant_hits == 0, f"Detected {v3_burst_redundant_hits} redundant V3 endpoints in data_clean!"

    # 6. Save comprehensive validation report
    val_records = [
        {'Check': 'Total Physical Image Files Copied', 'Observed': N_physical_images, 'Expected': 3747, 'Status': 'PASS'},
        {'Check': 'Non-Image Stray Files', 'Observed': N_non_images, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'Train Split Total Count', 'Observed': split_counts['train'], 'Expected': 2622, 'Status': 'PASS'},
        {'Check': 'Validation Split Total Count', 'Observed': split_counts['val'], 'Expected': 562, 'Status': 'PASS'},
        {'Check': 'Test Split Total Count', 'Observed': split_counts['test'], 'Expected': 563, 'Status': 'PASS'},
        {'Check': 'Per-Class Distribution Cells Matching Manifest', 'Observed': 30 - mismatch_count, 'Expected': 30, 'Status': 'PASS'},
        {'Check': 'MD5 Cross-Split Overlap', 'Observed': total_md5_overlap, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'Quarantine File Contamination', 'Observed': quarantine_hits, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'Excluded Burst Redundant Contamination', 'Observed': excluded_burst_hits, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'V3 Pairs Cross-Split Leakage (Endpoint Inspected)', 'Observed': v3_cross_split_leakage, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'V3 Pairs Both Endpoints Present', 'Observed': v3_both_present, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'V3 Pairs Representative Retained Exactly 1', 'Observed': v3_one_present_rep, 'Expected': 61, 'Status': 'PASS'},
        {'Check': 'V3 Pairs Both Endpoints Excluded', 'Observed': v3_neither_present, 'Expected': 25, 'Status': 'PASS'}
    ]
    df_val = pd.DataFrame(val_records)
    df_val.to_csv(OUT_VALIDATION_PATH, index=False)
    print(f"\nSaved updated physical build validation to {OUT_VALIDATION_PATH}")
    print(df_val.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Build and validate data_clean V4 benchmark.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation without copying images.")
    args = parser.parse_args()

    t0 = time.time()
    assert os.path.exists(FROZEN_MANIFEST_PATH), f"Manifest missing: {FROZEN_MANIFEST_PATH}"
    df_manifest = pd.read_csv(FROZEN_MANIFEST_PATH)
    assert len(df_manifest) == 3747, f"Expected 3747 images, got {len(df_manifest)}"

    if not args.validate_only:
        build_dataset(df_manifest)

    validate_dataset(df_manifest)
    print(f"\nTotal execution time: {time.time() - t0:.1f}s.")

if __name__ == '__main__':
    main()

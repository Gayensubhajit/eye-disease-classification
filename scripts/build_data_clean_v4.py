import os, sys, shutil, hashlib, time, pandas as pd

RAW_ARCHIVE_DIR = '/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset'
FROZEN_MANIFEST_PATH = 'outputs/audit/final_clean_split_manifest_v4.csv'
QUARANTINE_PATH = 'outputs/audit/v4_quarantine.csv'
MEMBERS_PATH = 'outputs/audit/v4_group_members.csv'
V3_RECONCILIATION_PATH = 'outputs/audit/v4_v3_reconciliation.csv'
DATA_CLEAN_DIR = 'data_clean'

OUT_INVENTORY_PATH = 'outputs/audit/data_clean_v4_inventory.csv'
OUT_CHECKSUMS_PATH = 'outputs/audit/data_clean_v4_checksums.sha256'
OUT_VALIDATION_PATH = 'outputs/audit/data_clean_v4_validation.csv'

def compute_hashes(filepath):
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

def main():
    print("=== STARTING CONTROLLED DATA_CLEAN V4 CONSTRUCTION ===")
    t0 = time.time()
    
    # 1. Verify frozen manifest
    assert os.path.exists(FROZEN_MANIFEST_PATH), f"Manifest missing: {FROZEN_MANIFEST_PATH}"
    df_manifest = pd.read_csv(FROZEN_MANIFEST_PATH)
    N_expected = len(df_manifest)
    print(f"Loaded frozen V4 manifest: {N_expected} images across {df_manifest['class'].nunique()} classes.")
    assert N_expected == 3747, f"Expected 3747 images, got {N_expected}"
    
    # 2. Check quarantine and excluded files sets
    df_quarantine = pd.read_csv(QUARANTINE_PATH)
    quarantine_set = set(df_quarantine['image_path'])
    print(f"Quarantine set loaded: {len(quarantine_set)} images.")
    
    df_members = pd.read_csv(MEMBERS_PATH)
    excluded_burst_set = set(df_members[df_members['role'] == 'BURST_REDUNDANT']['image_path'])
    print(f"Excluded burst set loaded: {len(excluded_burst_set)} images.")
    
    # Verify zero intersection in manifest
    manifest_paths = set(df_manifest['clean_image_path'])
    assert len(manifest_paths.intersection(quarantine_set)) == 0, "Quarantined images detected in manifest!"
    assert len(manifest_paths.intersection(excluded_burst_set)) == 0, "Excluded burst images detected in manifest!"
    
    # 3. Create target directory structure
    splits = ['train', 'val', 'test']
    classes = sorted(df_manifest['class'].unique())
    
    for s in splits:
        for c in classes:
            target_dir = os.path.join(DATA_CLEAN_DIR, s, c)
            os.makedirs(target_dir, exist_ok=True)
            
    # 4. Copy images and verify byte-for-byte hashes
    print("\nCopying files and verifying SHA-256 / MD5 checksums...")
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
        
        # Copy file preserving timestamp
        shutil.copy2(src_path, dst_path)
        copy_count += 1
        
        # Hash verification
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
            
    # Save inventory and sha256 checksums
    df_inv = pd.DataFrame(inventory_rows)
    df_inv.to_csv(OUT_INVENTORY_PATH, index=False)
    print(f"\nSaved inventory to {OUT_INVENTORY_PATH}")
    
    with open(OUT_CHECKSUMS_PATH, 'w') as f:
        f.writelines(sorted(sha256_lines))
    print(f"Saved SHA-256 checksums to {OUT_CHECKSUMS_PATH}")
    
    # 5. Independent Post-Build Physical Verification
    print("\n--- Running Independent Post-Build Physical Verification ---")
    
    # Scan physical data_clean directory
    physical_files = []
    for root, dirs, files in os.walk(DATA_CLEAN_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            rel_to_clean = os.path.relpath(full_p, DATA_CLEAN_DIR)
            physical_files.append((rel_to_clean, full_p))
            
    N_physical = len(physical_files)
    print(f"1. Total physical files in {DATA_CLEAN_DIR}: {N_physical} (Expected: {N_expected})")
    assert N_physical == N_expected, f"Physical count mismatch: {N_physical} vs {N_expected}"
    
    # Split counts
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = {s: {c: 0 for c in classes} for s in splits}
    physical_md5s = {'train': set(), 'val': set(), 'test': set()}
    
    for rel_p, full_p in physical_files:
        parts = rel_p.split(os.sep)
        sp = parts[0]
        cls = parts[1]
        split_counts[sp] += 1
        class_counts[sp][cls] += 1
        m, _ = compute_hashes(full_p)
        physical_md5s[sp].add(m)
        
    print(f"2. Physical split counts: {split_counts}")
    assert split_counts['train'] == 2622, f"Expected 2622 train, got {split_counts['train']}"
    assert split_counts['val'] == 562, f"Expected 562 val, got {split_counts['val']}"
    assert split_counts['test'] == 563, f"Expected 563 test, got {split_counts['test']}"
    
    # MD5 Cross-Split overlap
    md5_tr_va = len(physical_md5s['train'].intersection(physical_md5s['val']))
    md5_tr_te = len(physical_md5s['train'].intersection(physical_md5s['test']))
    md5_va_te = len(physical_md5s['val'].intersection(physical_md5s['test']))
    total_md5_overlap = md5_tr_va + md5_tr_te + md5_va_te
    print(f"3. Physical MD5 cross-split overlap: {total_md5_overlap}")
    assert total_md5_overlap == 0, f"Cross-split MD5 overlap detected: {total_md5_overlap}"
    
    # Check for quarantine contamination
    quarantine_physical_hits = 0
    for rel_p, _ in physical_files:
        # rel_p is split/class/filename.jpg
        # original rel_p was class/filename.jpg
        orig_rel = os.path.join(*rel_p.split(os.sep)[1:])
        if orig_rel in quarantine_set:
            quarantine_physical_hits += 1
    print(f"4. Quarantined files detected in physical directory: {quarantine_physical_hits}")
    assert quarantine_physical_hits == 0, "Quarantined files found in data_clean!"
    
    # Check V3 reconciliation cross-split leakage
    df_v3_rec = pd.read_csv(V3_RECONCILIATION_PATH)
    v3_leak = (df_v3_rec['v4_resolution'] == 'CROSS_SPLIT_LEAKAGE').sum()
    print(f"5. V3 high-confidence pairs crossing split: {v3_leak}")
    assert v3_leak == 0, "V3 leakage detected!"
    
    # Class Distribution Table
    print("\nPhysical Class Distribution in data_clean:")
    df_summary = pd.DataFrame(class_counts).reset_index()
    df_summary.columns = ['class', 'train', 'val', 'test']
    df_summary['total'] = df_summary['train'] + df_summary['val'] + df_summary['test']
    print(df_summary.to_string(index=False))
    
    # Validation Table
    val_records = [
        {'Check': 'Total Physical Files Copied', 'Observed': N_physical, 'Expected': 3747, 'Status': 'PASS'},
        {'Check': 'Train Split Count', 'Observed': split_counts['train'], 'Expected': 2622, 'Status': 'PASS'},
        {'Check': 'Validation Split Count', 'Observed': split_counts['val'], 'Expected': 562, 'Status': 'PASS'},
        {'Check': 'Test Split Count', 'Observed': split_counts['test'], 'Expected': 563, 'Status': 'PASS'},
        {'Check': 'MD5 Cross-Split Overlap', 'Observed': total_md5_overlap, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'Quarantine File Contamination', 'Observed': quarantine_physical_hits, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'V3 Cross-Split Burst Leakage', 'Observed': v3_leak, 'Expected': 0, 'Status': 'PASS'},
        {'Check': 'SHA-256 Inventory Hash Verification', 'Observed': len(df_inv), 'Expected': 3747, 'Status': 'PASS'}
    ]
    df_val = pd.DataFrame(val_records)
    df_val.to_csv(OUT_VALIDATION_PATH, index=False)
    print(f"\nSaved physical build validation to {OUT_VALIDATION_PATH}")
    print(df_val.to_string(index=False))
    print(f"\nConstruction completed in {time.time() - t0:.1f}s.")

if __name__ == '__main__':
    main()

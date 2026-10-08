import os
import re
import cv2
import numpy as np
import pandas as pd
import networkx as nx

RAW_ARCHIVE_DIR = '/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset'
CLEAN_MANIFEST_PATH = 'outputs/audit/clean_split_manifest.csv'
CROSS_SPLIT_CANDIDATES_PATH = 'outputs/audit/cross_split_near_duplicate_candidates.csv'

OUTPUT_CANDIDATE_CLUSTERS_PATH = 'outputs/audit/source_candidate_clusters.csv'
OUTPUT_SOURCE_MANIFEST_PATH = 'outputs/audit/final_source_cluster_manifest.csv'
OUTPUT_FINAL_SPLIT_PATH = 'outputs/audit/final_clean_split_manifest.csv'
OUTPUT_VALIDATION_PATH = 'outputs/audit/final_clean_split_validation.csv'

def parse_filename_num(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r'(\d+)$', base)
    if m:
        num = int(m.group(1))
        prefix = base[:m.start(1)].strip()
        return prefix, num
    return base, None

def get_filename_adjacency(path_a, path_b):
    p_a, num_a = parse_filename_num(path_a)
    p_b, num_b = parse_filename_num(path_b)
    if num_a is not None and num_b is not None:
        if p_a == p_b and abs(num_a - num_b) <= 5:
            return 'ADJACENT_FILENAME_SUPPORT'
        else:
            return 'NON_ADJACENT'
    return 'UNKNOWN'

def compute_ssim(img1, img2, mask=None):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mask is not None:
        eroded_mask = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        if eroded_mask.sum() > 0:
            return float(np.mean(ssim_map[eroded_mask]))
    return float(np.mean(ssim_map))

def main():
    print("=== Starting Source-Image / Burst-Cluster Audit ===")
    manifest = pd.read_csv(CLEAN_MANIFEST_PATH)
    print(f"Loaded clean split manifest: {len(manifest)} images.")
    
    df_cross = pd.read_csv(CROSS_SPLIT_CANDIDATES_PATH)
    t2 = df_cross[df_cross['tier'] == 'TIER_2_HIGHLY_SUSPICIOUS'].copy()
    print(f"Loaded cross-split candidate pairs: {len(t2)} Tier-2 pairs.")
    
    # 1. Image cache & feature extractor
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    img_cache = {}
    
    def get_img_data(rel_path):
        if rel_path in img_cache:
            return img_cache[rel_path]
        full_p = os.path.join(RAW_ARCHIVE_DIR, rel_path)
        bgr = cv2.imread(full_p)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {full_p}")
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_512 = cv2.resize(gray, (512, 512)).astype(np.float32)
        mask_512 = gray_512 > 15
        kp, des = orb.detectAndCompute(gray, None)
        data = {
            'h': h, 'w': w,
            'gray_512': gray_512,
            'mask_512': mask_512,
            'kp': kp, 'des': des
        }
        img_cache[rel_path] = data
        return data

    # 2. Detailed edge evaluation
    evaluated_edges = []
    print("Evaluating geometric consistency & masked SSIM across candidate pairs...")
    for idx, row in t2.iterrows():
        p_a = row['image_a']
        p_b = row['image_b']
        d_a = get_img_data(p_a)
        d_b = get_img_data(p_b)
        
        cmask = d_a['mask_512'] & d_b['mask_512']
        if cmask.sum() > 100:
            v1 = d_a['gray_512'][cmask]
            v2 = d_b['gray_512'][cmask]
            v1_c = v1 - v1.mean()
            v2_c = v2 - v2.mean()
            denom = np.sqrt(np.sum(v1_c**2) * np.sum(v2_c**2))
            m_ncc = float(np.sum(v1_c * v2_c) / denom) if denom > 0 else 0.0
            m_pix_diff = float(np.mean(np.abs(v1 - v2)))
            m_ssim = compute_ssim(d_a['gray_512'], d_b['gray_512'], cmask)
        else:
            m_ncc, m_pix_diff, m_ssim = 0.0, 255.0, 0.0
            
        des1, des2 = d_a['des'], d_b['des']
        kp1, kp2 = d_a['kp'], d_b['kp']
        tot_matches, good_matches, ransac_inliers, inlier_ratio = 0, 0, 0, 0.0
        if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
            raw_matches = bf.knnMatch(des1, des2, k=2)
            tot_matches = len(raw_matches)
            gm = [m for m, n in raw_matches if m.distance < 0.75 * n.distance] if raw_matches else []
            good_matches = len(gm)
            if good_matches >= 4:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)
                H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                if inlier_mask is not None:
                    ransac_inliers = int(inlier_mask.sum())
                    inlier_ratio = float(ransac_inliers / good_matches)
                    
        fn_adj = get_filename_adjacency(p_a, p_b)
        
        # Edge classification
        # SAME_SOURCE_CAPTURE: near-identical photograph
        if m_ssim >= 0.95 and m_ncc >= 0.99 and m_pix_diff <= 6.0:
            edge_type = 'SAME_SOURCE_CAPTURE'
            evidence = 'CONFIRMED'
        # SAME_CAPTURE_SEQUENCE: burst shot with geometric verification
        elif (ransac_inliers >= 15 and inlier_ratio >= 0.20) or (m_ssim >= 0.90 and fn_adj == 'ADJACENT_FILENAME_SUPPORT') or m_ssim >= 0.92:
            edge_type = 'SAME_CAPTURE_SEQUENCE'
            evidence = 'HIGH_CONFIDENCE'
        # POSSIBLE_RELATED: moderate similarity
        elif m_ssim >= 0.86 or (ransac_inliers >= 8 and inlier_ratio >= 0.15):
            edge_type = 'POSSIBLE_RELATED'
            evidence = 'REVIEW_REQUIRED'
        else:
            edge_type = 'DISTINCT'
            evidence = 'DISTINCT'
            
        evaluated_edges.append({
            'image_a': p_a,
            'image_b': p_b,
            'class_a': row['class_a'],
            'class_b': row['class_b'],
            'split_a': row['split_a'],
            'split_b': row['split_b'],
            'dhash_distance': row['dhash_distance'],
            'phash_distance': row['phash_distance'],
            'masked_ncc': round(m_ncc, 4),
            'masked_pixel_diff': round(m_pix_diff, 2),
            'masked_ssim': round(m_ssim, 4),
            'orb_total_matches': tot_matches,
            'orb_good_matches': good_matches,
            'orb_ransac_inliers': ransac_inliers,
            'orb_inlier_ratio': round(inlier_ratio, 4),
            'filename_adjacency': fn_adj,
            'edge_type': edge_type,
            'evidence_level': evidence
        })
        
    df_eval_edges = pd.DataFrame(evaluated_edges)
    print(f"Edge evaluation complete. Edge types breakdown:\n{df_eval_edges['edge_type'].value_counts()}")
    
    # 3. Build candidate connected components graph from all Tier-2 edges (Step 2 requirement)
    G_raw = nx.Graph()
    for _, r in df_eval_edges.iterrows():
        G_raw.add_edge(r['image_a'], r['image_b'], **r.to_dict())
        
    path_to_class = dict(zip(manifest['original_path'], manifest['class']))
    path_to_split = dict(zip(manifest['original_path'], manifest['split']))
    path_to_md5 = dict(zip(manifest['original_path'], manifest['md5']))
    
    candidate_clusters_records = []
    for i, comp in enumerate(nx.connected_components(G_raw)):
        comp_list = sorted(list(comp))
        classes = sorted(list(set(path_to_class[p] for p in comp_list)))
        splits = [path_to_split[p] for p in comp_list]
        split_dist = {s: splits.count(s) for s in ['Train', 'Val', 'Test'] if splits.count(s) > 0}
        split_dist_str = ', '.join(f'{k}:{v}' for k, v in split_dist.items())
        
        subG = G_raw.subgraph(comp)
        edges = list(subG.edges(data=True))
        
        ncc_vals = [d.get('masked_ncc', 0.0) for _, _, d in edges]
        pix_vals = [d.get('masked_pixel_diff', 255.0) for _, _, d in edges]
        ssim_vals = [d.get('masked_ssim', 0.0) for _, _, d in edges]
        inlier_vals = [d.get('orb_ransac_inliers', 0) for _, _, d in edges]
        
        stat_summary = (
            f"ssim:[{min(ssim_vals):.3f}-{max(ssim_vals):.3f}], "
            f"ncc:[{min(ncc_vals):.3f}-{max(ncc_vals):.3f}], "
            f"pix_diff:[{min(pix_vals):.1f}-{max(pix_vals):.1f}], "
            f"inliers:[{min(inlier_vals)}-{max(inlier_vals)}]"
        )
        
        candidate_clusters_records.append({
            'cluster_id': f'CAND_{i+1:03d}',
            'number_of_images': len(comp_list),
            'classes_present': '; '.join(classes),
            'current_split_distribution': split_dist_str,
            'filenames': '; '.join(comp_list),
            'pairwise_similarity_statistics': stat_summary
        })
        
    df_candidate_clusters = pd.DataFrame(candidate_clusters_records)
    df_candidate_clusters.to_csv(OUTPUT_CANDIDATE_CLUSTERS_PATH, index=False)
    print(f"Wrote candidate clusters to {OUTPUT_CANDIDATE_CLUSTERS_PATH} ({len(df_candidate_clusters)} clusters).")

    # 4. Resolve Convincing Clusters vs Spurious Links
    # A true cluster link requires CONFIRMED (SAME_SOURCE_CAPTURE) or HIGH_CONFIDENCE (SAME_CAPTURE_SEQUENCE)
    G_convincing = nx.Graph()
    # Add all images in the dataset as potential nodes
    for p in manifest['original_path']:
        G_convincing.add_node(p)
        
    for _, r in df_eval_edges.iterrows():
        if r['edge_type'] in ['SAME_SOURCE_CAPTURE', 'SAME_CAPTURE_SEQUENCE', 'POSSIBLE_RELATED']:
            G_convincing.add_edge(r['image_a'], r['image_b'], **r.to_dict())
            
    # Classify each connected component of G_convincing
    final_manifest_records = []
    cluster_idx = 1
    
    retained_images = []
    excluded_conflict_images = []
    excluded_redundant_images = []
    
    # We will process multi-node components and singletons
    for comp in nx.connected_components(G_convincing):
        comp_list = sorted(list(comp))
        if len(comp_list) == 1:
            p = comp_list[0]
            final_manifest_records.append({
                'cluster_id': f'CLUST_SINGLE_{cluster_idx:04d}',
                'original_path': p,
                'class': path_to_class[p],
                'md5': path_to_md5[p],
                'cluster_decision': 'RETAIN_SINGLE',
                'representative': True,
                'evidence_level': 'DISTINCT'
            })
            retained_images.append(p)
            cluster_idx += 1
            continue
            
        # Multi-image cluster
        c_id = f'CLUST_MULTI_{cluster_idx:04d}'
        cluster_idx += 1
        
        classes = set(path_to_class[p] for p in comp_list)
        subG = G_convincing.subgraph(comp_list)
        edges = list(subG.edges(data=True))
        
        # Check if any edge has cross-class conflict
        has_cross_class = len(classes) > 1
        
        # Check highest evidence level in cluster
        edge_types = set(d.get('edge_type') for _, _, d in edges)
        
        if 'SAME_SOURCE_CAPTURE' in edge_types:
            clust_evidence = 'CONFIRMED'
        elif 'SAME_CAPTURE_SEQUENCE' in edge_types:
            clust_evidence = 'HIGH_CONFIDENCE'
        else:
            clust_evidence = 'REVIEW_REQUIRED'
            
        if has_cross_class:
            # Rule 7: Exclude complete cluster
            for p in comp_list:
                final_manifest_records.append({
                    'cluster_id': c_id,
                    'original_path': p,
                    'class': path_to_class[p],
                    'md5': path_to_md5[p],
                    'cluster_decision': 'EXCLUDE_CROSS_CLASS_CONFLICT',
                    'representative': False,
                    'evidence_level': clust_evidence
                })
                excluded_conflict_images.append(p)
        else:
            # Same class cluster
            # Rule 8: Keep exactly ONE representative
            # Deterministic selection: pick image with lowest numerical suffix or sorted path
            rep_img = comp_list[0]
            for p in comp_list:
                is_rep = (p == rep_img)
                decision = 'RETAIN_REPRESENTATIVE' if is_rep else 'EXCLUDE_REDUNDANT_BURST'
                final_manifest_records.append({
                    'cluster_id': c_id,
                    'original_path': p,
                    'class': path_to_class[p],
                    'md5': path_to_md5[p],
                    'cluster_decision': decision,
                    'representative': is_rep,
                    'evidence_level': clust_evidence
                })
                if is_rep:
                    retained_images.append(p)
                else:
                    excluded_redundant_images.append(p)

    df_final_cluster_manifest = pd.DataFrame(final_manifest_records)
    df_final_cluster_manifest.to_csv(OUTPUT_SOURCE_MANIFEST_PATH, index=False)
    print(f"Wrote final source cluster manifest to {OUTPUT_SOURCE_MANIFEST_PATH} ({len(df_final_cluster_manifest)} images).")
    print(f"Decisions breakdown:\n{df_final_cluster_manifest['cluster_decision'].value_counts()}")
    print(f"Evidence levels breakdown:\n{df_final_cluster_manifest['evidence_level'].value_counts()}")
    
    # 5. Create new cluster-level stratified split (70 / 15 / 15) on retained independent images
    retained_df = df_final_cluster_manifest[df_final_cluster_manifest['representative'] == True].copy()
    print(f"\nRetained unique source images/representatives: {len(retained_df)}")
    print(f"Class breakdown of retained pool:\n{retained_df['class'].value_counts()}")
    
    # Stratified split with fixed seed = 42
    np.random.seed(42)
    split_assignments = {}
    
    for cls, grp in retained_df.groupby('class'):
        cls_items = grp.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(cls_items)
        
        # For integer counts targeting 70/15/15:
        # Train ~ 0.70, Val ~ 0.15, Test ~ 0.15
        n_val = int(round(n * 0.15))
        n_test = int(round(n * 0.15))
        # Ensure at least 1 image in Val and Test if n >= 3
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        n_train = n - n_val - n_test
        
        for idx, row in cls_items.iterrows():
            if idx < n_train:
                s = 'Train'
            elif idx < n_train + n_val:
                s = 'Val'
            else:
                s = 'Test'
            split_assignments[row['original_path']] = s

    retained_df['split'] = retained_df['original_path'].map(split_assignments)
    
    # Save clean split manifest
    final_clean_split = retained_df[['original_path', 'class', 'md5', 'split', 'cluster_id', 'evidence_level']].copy()
    final_clean_split.to_csv(OUTPUT_FINAL_SPLIT_PATH, index=False)
    print(f"\nWrote final clean split manifest to {OUTPUT_FINAL_SPLIT_PATH} ({len(final_clean_split)} images).")
    print(f"Split distribution:\n{final_clean_split['split'].value_counts()}")
    print(f"Class by split crosstab:\n{pd.crosstab(final_clean_split['class'], final_clean_split['split'])}")

    # 6. Final Leakage Validation
    val_records = []
    
    # A. Exact MD5 overlaps
    train_md5 = set(final_clean_split[final_clean_split['split'] == 'Train']['md5'])
    val_md5 = set(final_clean_split[final_clean_split['split'] == 'Val']['md5'])
    test_md5 = set(final_clean_split[final_clean_split['split'] == 'Test']['md5'])
    
    tt_md5 = len(train_md5.intersection(test_md5))
    tv_md5 = len(train_md5.intersection(val_md5))
    vt_md5 = len(val_md5.intersection(test_md5))
    
    val_records.append({'check_category': 'EXACT', 'check_name': 'MD5 Train-Test Overlap', 'observed_value': tt_md5, 'target_value': 0, 'status': 'PASS' if tt_md5 == 0 else 'FAIL'})
    val_records.append({'check_category': 'EXACT', 'check_name': 'MD5 Train-Val Overlap', 'observed_value': tv_md5, 'target_value': 0, 'status': 'PASS' if tv_md5 == 0 else 'FAIL'})
    val_records.append({'check_category': 'EXACT', 'check_name': 'MD5 Val-Test Overlap', 'observed_value': vt_md5, 'target_value': 0, 'status': 'PASS' if vt_md5 == 0 else 'FAIL'})
    
    # B. Cluster cross-split
    clust_split_counts = final_clean_split.groupby('cluster_id')['split'].nunique()
    multi_split_clusters = (clust_split_counts > 1).sum()
    val_records.append({'check_category': 'CLUSTER', 'check_name': 'Clusters Crossing Split Boundary', 'observed_value': multi_split_clusters, 'target_value': 0, 'status': 'PASS' if multi_split_clusters == 0 else 'FAIL'})

    # C. Near-duplicate cross-split check
    # Check if any CONFIRMED or HIGH_CONFIDENCE edge exists between images in different splits
    path_to_new_split = dict(zip(final_clean_split['original_path'], final_clean_split['split']))
    
    cross_split_confirmed = 0
    cross_split_high_conf = 0
    cross_split_review_req = 0
    
    for _, r in df_eval_edges.iterrows():
        p_a = r['image_a']
        p_b = r['image_b']
        if p_a in path_to_new_split and p_b in path_to_new_split:
            s_a = path_to_new_split[p_a]
            s_b = path_to_new_split[p_b]
            if s_a != s_b:
                if r['edge_type'] == 'SAME_SOURCE_CAPTURE':
                    cross_split_confirmed += 1
                elif r['edge_type'] == 'SAME_CAPTURE_SEQUENCE':
                    cross_split_high_conf += 1
                elif r['edge_type'] == 'POSSIBLE_RELATED':
                    cross_split_review_req += 1
                    
    val_records.append({'check_category': 'NEAR_DUPLICATE', 'check_name': 'Cross-Split CONFIRMED Pairs', 'observed_value': cross_split_confirmed, 'target_value': 0, 'status': 'PASS' if cross_split_confirmed == 0 else 'FAIL'})
    val_records.append({'check_category': 'NEAR_DUPLICATE', 'check_name': 'Cross-Split HIGH_CONFIDENCE Burst Pairs', 'observed_value': cross_split_high_conf, 'target_value': 0, 'status': 'PASS' if cross_split_high_conf == 0 else 'FAIL'})
    val_records.append({'check_category': 'NEAR_DUPLICATE', 'check_name': 'Cross-Split REVIEW_REQUIRED Pairs', 'observed_value': cross_split_review_req, 'target_value': 0, 'status': 'PASS' if cross_split_review_req == 0 else 'PASS_WITH_FLAG'})

    # D. Excluded Cross-Class Conflict Check
    excluded_in_split = sum(1 for p in excluded_conflict_images if p in path_to_new_split)
    val_records.append({'check_category': 'CONFLICT', 'check_name': 'Excluded Cross-Class Conflicts in Split', 'observed_value': excluded_in_split, 'target_value': 0, 'status': 'PASS' if excluded_in_split == 0 else 'FAIL'})

    # E. Provenance Check
    all_orig = all(os.path.exists(os.path.join(RAW_ARCHIVE_DIR, p)) for p in final_clean_split['original_path'])
    val_records.append({'check_category': 'PROVENANCE', 'check_name': 'Original Dataset Provenance', 'observed_value': '100% Original' if all_orig else 'Mixed', 'target_value': '100% Original', 'status': 'PASS' if all_orig else 'FAIL'})
    aug_in_split = any('Augmented' in p for p in final_clean_split['original_path'])
    val_records.append({'check_category': 'PROVENANCE', 'check_name': 'Augmented Dataset Images Present', 'observed_value': 1 if aug_in_split else 0, 'target_value': 0, 'status': 'PASS' if not aug_in_split else 'FAIL'})

    val_df = pd.DataFrame(val_records)
    val_df.to_csv(OUTPUT_VALIDATION_PATH, index=False)
    print(f"\nWrote validation results to {OUTPUT_VALIDATION_PATH}:\n{val_df.to_string()}")

if __name__ == '__main__':
    main()

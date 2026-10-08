import os, re, cv2, time, numpy as np, pandas as pd

RAW_ARCHIVE_DIR = '/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset'
CLEAN_MANIFEST_PATH = 'outputs/audit/clean_split_manifest.csv'
PHASH_CACHE_PATH = 'outputs/audit/phash_cache.npz'
RAW_INDEXED_CACHE_PATH = 'outputs/audit/raw_indexed_cache.npz'
DINO_CACHE_PATH = 'outputs/audit/dinov2_embeddings.npz'
V4_DINO_EDGES_PATH = 'outputs/audit/v4_all_candidate_edges.csv'
OUT_COMBINED_EDGES = 'outputs/audit/v4_combined_candidate_edges.csv'

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
    print("=== Combining Hash Pre-Filter (V2) and DINOv2 (V3/V4) Edges ===")
    t0 = time.time()
    
    manifest = pd.read_csv(CLEAN_MANIFEST_PATH)
    eligible_paths = manifest['original_path'].tolist()
    path_to_idx = {p: i for i, p in enumerate(eligible_paths)}
    path_to_class = dict(zip(manifest['original_path'], manifest['class']))
    N = len(eligible_paths)
    
    # 1. Load DINOv2 embeddings to look up cosine similarities
    dino_data = np.load(DINO_CACHE_PATH)
    embs = dino_data['embeddings']
    
    # 2. Find hash candidate pairs
    p_data = np.load(PHASH_CACHE_PATH)
    r_data = np.load(RAW_INDEXED_CACHE_PATH)
    idx_in_raw = [r_data['ids'].tolist().index(p) for p in eligible_paths]
    
    dhash_mat = r_data['bits_mat'][idx_in_raw].astype(np.float32)
    phash_mat = p_data['phash_mat'].astype(np.float32)
    
    d_sums = dhash_mat.sum(axis=1)
    d_dists = (d_sums[:, None] + d_sums[None, :] - 2 * np.dot(dhash_mat, dhash_mat.T)).astype(np.int16)
    p_sums = phash_mat.sum(axis=1)
    p_dists = (p_sums[:, None] + p_sums[None, :] - 2 * np.dot(phash_mat, phash_mat.T)).astype(np.int16)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    cand_mask = (d_dists[i_idx, j_idx] <= 2) | ((d_dists[i_idx, j_idx] <= 4) & (p_dists[i_idx, j_idx] <= 4))
    hash_cand_i = i_idx[cand_mask]
    hash_cand_j = j_idx[cand_mask]
    print(f"Total hash candidate pairs: {len(hash_cand_i)}")
    
    # 3. Load DINOv2 edges already evaluated
    df_dino = pd.read_csv(V4_DINO_EDGES_PATH)
    print(f"Loaded {len(df_dino)} DINOv2 evaluated edges.")
    
    # Existing evaluated pairs map: (p_a, p_b) -> row dict
    existing_pairs = {}
    for _, r in df_dino.iterrows():
        pair_key = tuple(sorted([r['image_a'], r['image_b']]))
        existing_pairs[pair_key] = r.to_dict()
        
    # Check which hash candidate pairs are not in existing_pairs
    missing_i = []
    missing_j = []
    for k in range(len(hash_cand_i)):
        pa = eligible_paths[hash_cand_i[k]]
        pb = eligible_paths[hash_cand_j[k]]
        pair_key = tuple(sorted([pa, pb]))
        if pair_key not in existing_pairs:
            missing_i.append(hash_cand_i[k])
            missing_j.append(hash_cand_j[k])
            
    print(f"Hash candidates not evaluated in DINOv2 top-sim list: {len(missing_i)}")
    
    # Evaluate missing hash candidates if any
    if len(missing_i) > 0:
        unique_indices = sorted(list(set(missing_i).union(set(missing_j))))
        print(f"Loading {len(unique_indices)} images for missing hash candidates...")
        orb = cv2.ORB_create(1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        img_cache = {}
        for idx in unique_indices:
            p = eligible_paths[idx]
            full_p = os.path.join(RAW_ARCHIVE_DIR, p)
            bgr = cv2.imread(full_p)
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray_512 = cv2.resize(gray, (512, 512)).astype(np.float32)
            mask_512 = gray_512 > 15
            kp, des = orb.detectAndCompute(gray, None)
            img_cache[idx] = {
                'gray_512': gray_512, 'mask_512': mask_512, 'kp': kp, 'des': des
            }
            
        print("Evaluating missing hash pairs...")
        for k in range(len(missing_i)):
            i = missing_i[k]
            j = missing_j[k]
            p_a = eligible_paths[i]
            p_b = eligible_paths[j]
            dino_sim = float(np.dot(embs[i], embs[j]))
            
            d_a = img_cache[i]
            d_b = img_cache[j]
            cmask = d_a['mask_512'] & d_b['mask_512']
            if cmask.sum() > 100:
                v1 = d_a['gray_512'][cmask]
                v2 = d_b['gray_512'][cmask]
                v1_c = v1 - v1.mean()
                v2_c = v2 - v2.mean()
                denom = np.sqrt(np.sum(v1_c**2) * np.sum(v2_c**2))
                m_ncc = float(np.sum(v1_c * v2_c) / denom) if denom > 0 else 0.0
                m_pix_diff = float(np.mean(np.abs(v1 - v2)))
            else:
                m_ncc, m_pix_diff = 0.0, 255.0
                
            fn_adj = get_filename_adjacency(p_a, p_b)
            m_ssim = 0.0
            inliers, inlier_ratio = 0, 0.0
            
            if m_ncc >= 0.80 or m_pix_diff <= 25.0:
                if cmask.sum() > 100:
                    m_ssim = compute_ssim(d_a['gray_512'], d_b['gray_512'], cmask)
                des1, des2 = d_a['des'], d_b['des']
                kp1, kp2 = d_a['kp'], d_b['kp']
                if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                    raw_m = bf.knnMatch(des1, des2, k=2)
                    gm = [m for m, n in raw_m if m.distance < 0.75 * n.distance] if raw_m else []
                    if len(gm) >= 4:
                        pts1 = np.float32([kp1[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
                        pts2 = np.float32([kp2[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)
                        H, inlier_msk = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                        if inlier_msk is not None:
                            inliers = int(inlier_msk.sum())
                            inlier_ratio = float(inliers / len(gm))
                            
            if m_ssim >= 0.95 and m_ncc >= 0.99 and m_pix_diff <= 6.0:
                classification = 'CONFIRMED_SAME_SOURCE'
            elif (inliers >= 15 and inlier_ratio >= 0.20) or (m_ssim >= 0.90 and fn_adj == 'ADJACENT_FILENAME_SUPPORT') or (m_ssim >= 0.92):
                classification = 'HIGH_CONFIDENCE_CAPTURE_SEQUENCE'
            elif m_ssim >= 0.86 or (inliers >= 8 and inlier_ratio >= 0.15):
                classification = 'POSSIBLE_RELATED'
            else:
                classification = 'DISTINCT'
                
            pair_key = tuple(sorted([p_a, p_b]))
            existing_pairs[pair_key] = {
                'image_a': p_a,
                'image_b': p_b,
                'class_a': path_to_class[p_a],
                'class_b': path_to_class[p_b],
                'same_class': (path_to_class[p_a] == path_to_class[p_b]),
                'dino_cosine_similarity': round(dino_sim, 5),
                'masked_ncc': round(m_ncc, 4),
                'masked_pixel_diff': round(m_pix_diff, 2),
                'masked_ssim': round(m_ssim, 4),
                'orb_ransac_inliers': inliers,
                'orb_inlier_ratio': round(inlier_ratio, 4),
                'filename_adjacency': fn_adj,
                'classification': classification
            }
            
    # Add dhash and phash distance to all existing pairs
    combined_rows = []
    for pair_key, r in existing_pairs.items():
        pa, pb = r['image_a'], r['image_b']
        ia = path_to_idx[pa]
        ib = path_to_idx[pb]
        dh = int(d_dists[ia, ib])
        ph = int(p_dists[ia, ib])
        r['dhash_distance'] = dh
        r['phash_distance'] = ph
        combined_rows.append(r)
        
    df_combined = pd.DataFrame(combined_rows)
    df_combined.to_csv(OUT_COMBINED_EDGES, index=False)
    print(f"Saved combined edges ({len(df_combined)} rows) to {OUT_COMBINED_EDGES} in {time.time() - t0:.1f}s")
    print(f"Classifications in combined set:\n{df_combined['classification'].value_counts()}")
    high_conf = df_combined[df_combined['classification'].isin(['CONFIRMED_SAME_SOURCE', 'HIGH_CONFIDENCE_CAPTURE_SEQUENCE'])]
    print(f"Total High-Confidence Edges: {len(high_conf)}")
    print(f"High-confidence breakdown by same_class:\n{high_conf['same_class'].value_counts()}")

if __name__ == '__main__':
    main()

import os, re, cv2, time, numpy as np, pandas as pd

RAW_ARCHIVE_DIR = '/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset'
CLEAN_MANIFEST_PATH = 'outputs/audit/clean_split_manifest.csv'
DINO_CACHE_PATH = 'outputs/audit/dinov2_embeddings.npz'
OUT_EDGES_PATH = 'outputs/audit/v4_all_candidate_edges.csv'

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
    print("=== Scanning Complete Pool for Unified V4 Candidate Edges ===")
    t0 = time.time()
    
    manifest = pd.read_csv(CLEAN_MANIFEST_PATH)
    paths = manifest['original_path'].tolist()
    path_to_class = dict(zip(manifest['original_path'], manifest['class']))
    path_to_md5 = dict(zip(manifest['original_path'], manifest['md5']))
    N = len(paths)
    
    # 1. Load DINOv2 embeddings and find all pairs with Sim >= 0.9584
    print("Loading cached DINOv2 embeddings...")
    dino_data = np.load(DINO_CACHE_PATH)
    embs = dino_data['embeddings'] # (4387, 384)
    S = np.dot(embs, embs.T)
    np.fill_diagonal(S, -1.0)
    
    triu_i, triu_j = np.triu_indices(N, k=1)
    s_vals = S[triu_i, triu_j]
    high_sim_mask = s_vals >= 0.9584
    
    cand_i = triu_i[high_sim_mask]
    cand_j = triu_j[high_sim_mask]
    cand_s = s_vals[high_sim_mask]
    print(f"Total DINOv2 candidate pairs (Sim >= 0.9584): {len(cand_i)}")
    
    # 2. Also include V2 hash candidates from all_pool_source_clusters
    # So we have complete coverage from both perceptual hash AND vision embeddings
    # Let's collect unique images to load
    unique_indices = sorted(list(set(cand_i).union(set(cand_j))))
    print(f"Unique images to load: {len(unique_indices)} / {N}")
    
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    img_cache = {}
    
    t_load = time.time()
    for idx_num, img_idx in enumerate(unique_indices):
        rel_p = paths[img_idx]
        full_p = os.path.join(RAW_ARCHIVE_DIR, rel_p)
        bgr = cv2.imread(full_p)
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_512 = cv2.resize(gray, (512, 512)).astype(np.float32)
        mask_512 = gray_512 > 15
        kp, des = orb.detectAndCompute(gray, None)
        img_cache[img_idx] = {
            'w': w, 'h': h,
            'gray_512': gray_512,
            'mask_512': mask_512,
            'kp': kp, 'des': des
        }
        if (idx_num + 1) % 500 == 0 or (idx_num + 1) == len(unique_indices):
            print(f"  Loaded {idx_num + 1}/{len(unique_indices)} images in {time.time() - t_load:.1f}s")
            
    # 3. Evaluate pairs
    print("Evaluating structural similarity and geometric consistency on all candidates...")
    records = []
    
    for k in range(len(cand_i)):
        i = cand_i[k]
        j = cand_j[k]
        p_a = paths[i]
        p_b = paths[j]
        dino_sim = float(cand_s[k])
        
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
        
        # Fast filter: only compute SSIM & RANSAC if noticeable similarity
        m_ssim = 0.0
        tot_m, good_m, inliers, inlier_ratio = 0, 0, 0, 0.0
        
        if m_ncc >= 0.80 or m_pix_diff <= 25.0:
            if cmask.sum() > 100:
                m_ssim = compute_ssim(d_a['gray_512'], d_b['gray_512'], cmask)
                
            des1, des2 = d_a['des'], d_b['des']
            kp1, kp2 = d_a['kp'], d_b['kp']
            if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                raw_m = bf.knnMatch(des1, des2, k=2)
                tot_m = len(raw_m)
                gm = [m for m, n in raw_m if m.distance < 0.75 * n.distance] if raw_m else []
                good_m = len(gm)
                if good_m >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)
                    H, inlier_msk = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                    if inlier_msk is not None:
                        inliers = int(inlier_msk.sum())
                        inlier_ratio = float(inliers / good_m)
                        
        # Classification
        if m_ssim >= 0.95 and m_ncc >= 0.99 and m_pix_diff <= 6.0:
            classification = 'CONFIRMED_SAME_SOURCE'
        elif (inliers >= 15 and inlier_ratio >= 0.20) or (m_ssim >= 0.90 and fn_adj == 'ADJACENT_FILENAME_SUPPORT') or (m_ssim >= 0.92):
            classification = 'HIGH_CONFIDENCE_CAPTURE_SEQUENCE'
        elif m_ssim >= 0.86 or (inliers >= 8 and inlier_ratio >= 0.15):
            classification = 'POSSIBLE_RELATED'
        else:
            classification = 'DISTINCT'
            
        if classification in ['CONFIRMED_SAME_SOURCE', 'HIGH_CONFIDENCE_CAPTURE_SEQUENCE', 'POSSIBLE_RELATED']:
            records.append({
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
            })
            
    df_edges = pd.DataFrame(records)
    df_edges.to_csv(OUT_EDGES_PATH, index=False)
    print(f"Saved {len(df_edges)} non-distinct candidate edges to {OUT_EDGES_PATH} in {time.time() - t0:.1f}s.")
    print(f"Classifications breakdown:\n{df_edges['classification'].value_counts()}")
    print(f"Same-class vs Cross-class:\n{df_edges['same_class'].value_counts()}")

if __name__ == '__main__':
    main()

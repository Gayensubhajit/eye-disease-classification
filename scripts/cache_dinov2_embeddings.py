import os, time, torch, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

RAW_ARCHIVE_DIR = '/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset'
CLEAN_MANIFEST_PATH = 'outputs/audit/clean_split_manifest.csv'
OUT_PATH = 'outputs/audit/dinov2_embeddings.npz'

class FundusImageDataset(Dataset):
    def __init__(self, paths, root_dir, transform):
        self.paths = paths
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        rel_p = self.paths[idx]
        full_p = os.path.join(self.root_dir, rel_p)
        img = Image.open(full_p).convert('RGB')
        return self.transform(img), rel_p

def main():
    print("=== Caching DINOv2 Embeddings for All 4,387 Images ===")
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    manifest = pd.read_csv(CLEAN_MANIFEST_PATH)
    paths = manifest['original_path'].tolist()
    
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    dataset = FundusImageDataset(paths, RAW_ARCHIVE_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=(device.type=='cuda'))
    
    feats_list = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            f = model(imgs)
            f = torch.nn.functional.normalize(f, p=2, dim=-1)
            feats_list.append(f.cpu().numpy())
            
    embeddings = np.concatenate(feats_list, axis=0) # (4387, 384)
    np.savez_compressed(OUT_PATH, embeddings=embeddings, paths=np.array(paths))
    print(f"Saved {embeddings.shape} to {OUT_PATH} in {time.time() - t0:.1f}s.")

if __name__ == '__main__':
    main()

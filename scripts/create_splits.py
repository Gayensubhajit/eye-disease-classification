"""Script to generate reproducible stratified train, validation, and test split manifests.

Reads original dataset directory structure, performs stratified splitting (80/10/10),
and writes CSV split files to data/splits/.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def create_splits(
    dataset_dir: str,
    output_dir: str,
    test_ratio: float = 0.10,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path.resolve()}")

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No subdirectories found in dataset path: {dataset_path}")

    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

    records = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
            records.append({
                "image_path": str(img_path),
                "class_name": class_name,
                "label": class_idx,
            })

    df = pd.DataFrame(records)
    print(f"Total images found: {len(df)} across {len(class_to_idx)} classes.")

    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: train vs val
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    test_csv = output_path / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Splits saved to {output_path.resolve()}:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    print("\nClass distribution per split:")
    summary = pd.DataFrame({
        "Train": train_df["class_name"].value_counts(),
        "Val": val_df["class_name"].value_counts(),
        "Test": test_df["class_name"].value_counts(),
    }).fillna(0).astype(int)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        dataset_dir = config["data"]["dataset_dir"]
        output_dir = str(Path(config["data"]["train_csv"]).parent)
        seed = config.get("seed", 42)
    else:
        dataset_dir = "Eye Disease Image Dataset/Original Dataset/Original Dataset"
        output_dir = "data/splits"
        seed = 42

    create_splits(dataset_dir=dataset_dir, output_dir=output_dir, seed=seed)


if __name__ == "__main__":
    main()

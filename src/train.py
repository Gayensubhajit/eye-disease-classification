"""Training entry point placeholder.

Implement after the dataset format, class labels, and baseline protocol are agreed.
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not config["data"]["num_classes"]:
        raise ValueError("Set data.num_classes and data.class_names before training.")
    print("Configuration loaded. Implement dataset and training loop for the selected dataset.")


if __name__ == "__main__":
    main()

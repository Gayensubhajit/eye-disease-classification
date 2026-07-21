# Classification of Eye Diseases from Color Fundus Images

Research repository for an 8th-semester B.Tech project on multi-class retinal disease classification from colour fundus images.

> **Research-use only.** This project is not a clinical diagnostic device and must not be used for patient care.

## Objectives

- Build reproducible data preparation and evaluation pipelines.
- Establish fair CNN baselines before proposing an improved architecture.
- Evaluate models with macro F1, ROC-AUC, Cohen's kappa, sensitivity, and specificity—not accuracy alone.
- Use explainability (Grad-CAM) to inspect model behaviour.

## Research workflow

1. Review the supplied literature and define one dataset/task precisely.
2. Train and document a baseline such as EfficientNet-B0.
3. Identify a measured limitation, then add one justified architectural change.
4. Run controlled ablations and report confidence intervals where possible.

The proposed architecture name is intentionally not fixed yet. It should follow the literature review and baseline results.

## Repository layout

```text
configs/       Experiment configuration
data/          Local datasets and split manifests (images are not versioned)
docs/          Proposal, literature review, and experiment records
experiments/   Per-experiment configs, metrics, and notes
notebooks/     Exploratory work only; move reusable code into src/
scripts/       Repeatable command-line helpers
src/           Installable Python package
tests/         Lightweight checks
outputs/       Local checkpoints, logs, and figures
```

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .[dev]
python -m src.train --config configs/config.yaml
```

Update `configs/config.yaml` with the selected dataset paths and class names before training.

## Team and supervision

- Team members: _to be added_
- Supervisor: Dr. Pawan Kumar Singh

## License

MIT. Check every dataset's licence and citation requirements separately before use.

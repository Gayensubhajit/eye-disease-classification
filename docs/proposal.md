# Initial proposal

## Working title

Classification of Eye Diseases from Colour Fundus Images using Deep Learning

## Research question

Can a carefully motivated enhancement to a strong baseline improve generalisation and class-balanced performance for the selected fundus-image classification task?

## Scope for first review

- Select one public dataset and state its licence, labels, and split policy.
- Reproduce at least one baseline on an untouched test set.
- Record macro F1, AUC, kappa, sensitivity, specificity, confusion matrix, and training cost.
- Propose an improvement only after evidence from the baseline and literature review.

## Non-negotiable methodology

- Prevent patient-level data leakage where patient IDs are available.
- Never tune on the test set.
- Report per-class metrics because medical datasets are often imbalanced.
- Treat Grad-CAM as a sanity-check tool, not clinical evidence.

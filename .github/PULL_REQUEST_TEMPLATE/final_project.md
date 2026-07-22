<!--
Final Project submission.
See assignments/FINAL_PROJECT.md for the full assignment.
-->

## Group members

<!-- List everyone in the group. -->

## Project title

## One-paragraph summary

<!-- Problem, approach, and key result. -->

## Submission checklist

- [ ] `final_project.ipynb` is in the `assignments` folder and runs end to end
- [ ] Presentation slides (PDF) are committed to `final_project_files`
- [ ] All code is linted and formatted with `ruff`
- [ ] No AI-generated cruft (placeholder comments, redundant docstrings)
- [ ] Commits are organized and logical
- [ ] Large files (model weights, raw data) are gitignored, not committed
- [ ] All group members' names appear at the top of the notebook

## Method checklist

- [ ] Baseline implemented and compared against the primary model
- [ ] Train/test split is spatial (geographic holdout, spatial block CV, or leave-one-region-out), not a random pixel split
- [ ] If using a pretrained or foundation model, embeddings inspected (t-SNE, PCA, or nearest-neighbor)
- [ ] If using several numeric predictors, multicollinearity checked (correlation matrix or VIF)

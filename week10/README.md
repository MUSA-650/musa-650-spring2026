# Week 10 — Practical Deep Learning: Augmentation, Generators & Regularization

**Topics covered:**
- The overfitting problem and how to detect it from training curves
- Data augmentation with `ImageDataGenerator` (hands-on with CIFAR-10 cat images)
- Data generators: `flow()` and `flow_from_directory()`
- Dropout regularization
- Batch Normalization

## Files

| File | Description |
|------|-------------|
| `week10_presentation.ipynb` | Lecture slides (convert with `jupyter nbconvert --to slides`) |
| `week10_lab_student.ipynb` | Student lab — fill in `___` blanks and `# TODO` items |
| `week10_lab_solutions.ipynb` | Complete solutions |

## Lab Summary

Students build four versions of a cats-vs-dogs binary classifier on CIFAR-10:
1. **Baseline CNN** — observe overfitting
2. **+ Data Augmentation** — reduce overfitting with `ImageDataGenerator`
3. **+ Dropout** — add regularization layers
4. **+ Batch Normalization** — stabilize and speed up training

## Converting Slides

```bash
jupyter nbconvert --to slides week10_presentation.ipynb --post serve
```

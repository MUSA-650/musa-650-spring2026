## Final Project

For this final project, you will design and implement a machine learning-based remote sensing solution to address a specific problem in urban planning. Building on your proposal, you will develop a working solution, evaluate its performance, and present your work in a professional lightning talk.

_You may want to review [our list of tooling, methods, and open data catalogs](./FINAL_PROJECT_RESOURCES.md)._

### Submission Guidelines

**All assignments must be completed in groups of 2-3 students.** One member of the group should submit on behalf of everyone, making sure to include all group members' names at the top of the notebook.

Your assignment should be submitted via a pull request to the main branch of this repository. Please make sure it is in the `assignments` subdirectory, titled `final_project.ipynb`, with any your class presentation and any additional files included in a subfolder named `final_project_files`. Additionally:
- Your notebook must contain the complete assignment instructions, followed by the relevant code chunks
- Your notebook must include group members' names and submission date
- Your notebook must be committed to the repository and run end to end. Documentation or output without the code that produced it counts as an incomplete submission
- All code must be well-formatted with appropriate code chunks (no overly long code blocks); no single cell should exceed roughly 200 lines
- All code must be linted and formatted using [`ruff`](https://docs.astral.sh/ruff/) before submission
- Clean up AI-generated cruft before submitting: remove placeholder comments, redundant docstrings, and inconsistent style
- Keep a logical commit history: not one giant commit, and not hundreds of micro-commits
- Large files (model weights, raw imagery) belong in `.gitignore`, not in the repository

_A note on visualizations_: If your your analysis includes interactive visualizations (e.g., `geemap`), please include a .gif of you clicking through each layer in your map. Embed this .gif in your notebook and include it in your submission folder. Do not include the interactive widget itself, as this often doesn't render well on GitHub.

#### Submission Checklist

Before you open your pull request, verify each item:

- [ ] `final_project.ipynb` is in the `assignments` folder and runs end to end
- [ ] Presentation slides (PDF) are committed to `final_project_files`
- [ ] All code is linted and formatted with `ruff`
- [ ] No AI-generated cruft (placeholder comments, redundant docstrings)
- [ ] Commits are organized and logical
- [ ] Large files (model weights, raw data) are gitignored, not committed
- [ ] All group members' names appear at the top of the notebook

### Project Requirements

Your final project notebook should include the following sections. Much of this builds directly on your proposal—refine and expand that work based on feedback you received.

#### Problem Definition & Technical Justification

Begin by restating the problem you're solving, your target user, and the specific output your solution produces. Then, refined from your proposal, explain the logical chain from problem to method: what your model is actually learning, why your chosen approach is appropriate for the task, and what failure modes you anticipated. Now that you've implemented your solution, discuss whether you observed any of these failure modes in practice and how you addressed them.

#### Methodological Precedent

Summarize the three or more sources that informed your approach and explain how they shaped your design decisions. Update this section if your reading evolved during implementation.

#### Data & Preprocessing

Describe your dataset(s), including source, spatial and temporal extent, resolution, and bands or features used. Discuss any data quality issues you encountered and how you addressed them. Explain your preprocessing steps and feature engineering choices, with justification for each decision.

#### Modeling Approach

Present your baseline model first—the simple, non-deep-learning approach you used to establish a performance floor. Report its results before describing your primary model. For your primary model, explain the architecture, training process, and any hyperparameter tuning you performed. Compare your primary model to the baseline: how much did the added complexity improve performance, and was that improvement worth the cost?

#### Evaluation & Analysis

Report your evaluation metrics with appropriate context. Beyond the numbers, discuss what your model gets right and wrong. Include example predictions that illustrate both successes and failures—visual examples are particularly valuable here. Be honest about limitations: constraints from labeled data availability, computational resources, or generalizability to other regions or time periods.

#### Evaluation Requirements

Your data is spatial. Neighboring pixels are correlated, so a random pixel or point split leaks information across the train/test boundary and inflates your metrics. Random splits are not acceptable. Split by geography instead: a geographic holdout (latitude or longitude strips), spatial block cross-validation, or leave-one-region-out. State which you used and why it isolates the test set from the training footprint.

If you use a pretrained or foundation model, inspect the embeddings before you trust them. Include a t-SNE, PCA, or nearest-neighbor analysis showing whether the model separates your target classes. When the model was trained on one dataset and applied to another region or sensor, discuss the domain shift and how you validated the transfer.

#### Future Work

Describe what you would do with more time or resources. How could this solution be scaled, improved, or operationalized for real-world use?

### Lightning Talk Presentation

Prepare a five-minute lightning talk that professionally summarizes your problem, solution, and key findings. This presentation should be polished and suitable for inclusion in a professional portfolio.

You may present your work as a video demonstration, a blog post, a LinkedIn article, or a slide deck with speaker notes. Whatever format you choose, your talk should clearly define the problem and why it matters, explain your approach in accessible terms, present key results both quantitatively and visually, discuss limitations honestly, and convey one or two key insights or takeaways.

### Rubric (30 points)

#### Code and Analysis (20 points)

| Category | Weight | Excellent | Satisfactory | Unsatisfactory |
|----------|--------|-----------|--------------|----------------|
| Problem Definition & Technical Justification | 5 pts | Clear problem with well-justified approach; accurate description of what model learns; failure modes revisited | Adequate definition; gaps in justification | Problem unclear; approach not justified |
| Data & Preprocessing | 3 pts | Data appropriate for task; preprocessing documented and justified | Data adequate; preprocessing weakly justified | Data inappropriate; preprocessing unexplained |
| Baseline Implementation | 3 pts | Baseline implemented and evaluated; clear comparison to primary model | Baseline present but comparison weak | No baseline |
| Primary Model Implementation | 5 pts | Well-designed architecture; clear documentation; training process explained; pretrained or foundation models inspected with embedding diagnostics | Functional with minor issues | Poor implementation |
| Evaluation & Analysis | 4 pts | Comprehensive metrics; spatial train/test split justified; honest discussion of successes and failures | Basic evaluation with some analysis; split strategy unjustified | Evaluation missing or superficial; random split with no spatial control |

#### Presentation (10 points)

| Category | Weight | Excellent | Satisfactory | Unsatisfactory |
|----------|--------|-----------|--------------|----------------|
| Content & Narrative | 5 pts | Clear story; problem well-motivated; approach accessible; honest about limitations | Adequate structure; some unclear elements | Poorly organized; key elements missing |
| Results & Professionalism | 5 pts | Results clearly presented with strong visuals; portfolio-ready quality | Results present but visualization weak | Unprofessional or poorly prepared |
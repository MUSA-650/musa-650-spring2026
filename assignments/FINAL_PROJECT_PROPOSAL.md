## Final Project Proposal

In this assignment, you will outline the proposal for your final project. You will describe the problem you want to solve, justify why your chosen approach fits that problem, and demonstrate that you understand how your model will actually work. The proposal is your chance to think through the conceptual foundations of the project before you begin implementation.

_Before you start, review [our list of tooling, methods, and open data catalogs](./FINAL_PROJECT_RESOURCES.md)._

### Proposal Requirements

The proposal should run about 1,500 words and must address the areas below. Several sections ask a direct question you must answer. Grading rewards clear, honest answers to these questions, so treat them as a checklist rather than a formality.

#### Problem Definition & Use Case

Describe the problem you are solving and why it matters. Identify the target user, whether a city agency, a research institution, or another stakeholder, and explain which decisions your output will inform. Be specific about the output the user needs: a binary mask of affected areas, a multi-class land cover map, a change detection layer comparing two time periods, or something else. The clearer you are about the required output, the easier it becomes to design an appropriate solution.

**State your framing.** Decide whether this is a methods contribution or an applied tool, and say so plainly:

- **Methods:** you are testing whether some technique improves a task ("Does adding SAR bands improve flood segmentation?"). Frame the research question clearly.
- **Applied:** you are building something a practitioner could deploy. Explain how the output would be used and who would use it.

Either lane is valid. Ambiguity between them is not, because it hides what "success" means for your project.

#### Technical Justification

This is the most important section of the proposal. You must explain the logical chain from problem to method, and show that you understand why your chosen approach fits.

First, describe what your model actually learns. What does a "positive" prediction mean in your context? Which features will the model rely on? A model trained to separate "flooded" from "not flooded" images learns to detect the presence of water, which is not the same as detecting flooding. Flooding means water where water should not be, so it requires either change detection or comparison against a baseline.

Second, explain why your task type fits. Why classification, segmentation, change detection, or regression? What would happen under a different task type? Would it still answer the user's question?

Third, identify failure modes. Describe at least two realistic scenarios where your approach breaks or misleads. A water segmentation model might flag permanent lakes as floods. A land cover classifier trained in one climate zone might fail in another because the spectral signatures differ. Knowing where an approach breaks is essential to using it responsibly.

**Why deep learning?** Explain why simpler approaches (thresholding, spectral indices, change detection, a random forest on spectral bands) cannot do the job. Deep learning earns its place when the pattern is too complex or too contextual for these methods, not by default. If you cannot answer this question convincingly, reconsider whether you need deep learning at all.

#### Existing Work Check

Before you propose a new model, verify what already exists. Reinventing a published product wastes your effort and weakens the proposal.

- [ ] **Foundation model embeddings** (Clay, Prithvi, SatCLIP): could you use these instead of training from scratch?
- [ ] **Existing products** that already solve this (JRC Global Surface Water, GHSL, Dynamic World, and similar): do they cover your area and target?
- [ ] **Commercial or operational systems** in this space.
- [ ] **Academic work with code or weights** you could run or fine-tune.

If a solution already exists, explain why you are not using it or how your work extends it.

#### Methodological Precedent

Reference at least three rigorous sources — academic papers, professional white papers, or well-documented challenge datasets — that inform your methodology. For each source, summarize the methods, the data, the evaluation approach, and any limitations the authors identify. Then explain how each source shapes your own approach and what you borrow or adapt.

#### Data Plan

Identify your datasets and document them in the tables below. Include spatial and temporal extent, resolution, and the bands or features you need.

**Input data**

| Source | Resolution | Temporal coverage | Access method |
|--------|------------|-------------------|---------------|
|        |            |                   |               |

**Labels**

| Source | How derived? | Known limitations |
|--------|--------------|-------------------|
|        |              |                   |

**Check for label circularity.** If your labels come from the same source as your inputs — for example, predicting temperature-derived heat labels from the thermal bands that produced them — explain why this is not circular. A model that recovers its own inputs proves nothing.

Confirm that the data supports the task. If you plan change detection, confirm you have suitable pre-event and post-event imagery. Discuss data quality risks (cloud cover, access restrictions, gaps), and describe a fallback plan if your primary dataset proves unavailable or unsuitable.

#### Modeling Approach

Describe your proposed architecture, including preprocessing and feature engineering. Before the primary model, describe a simple baseline: a non-deep-learning method that establishes a performance floor, such as NDWI thresholding for water or a random forest on spectral bands. Explain the alternatives you considered and why you chose your preferred method.

Check that your input resolution matches your prediction target. Predicting individual buildings from 30 m Landsat is not feasible; if your target is finer than your input, say how you will handle the mismatch or revise the target.

#### Evaluation Strategy

Identify the metrics you will use and explain why they fit your task and use case. Then address two requirements specific to geospatial models:

- **Spatial cross-validation.** Describe how you split data so that spatial autocorrelation does not inflate your metrics. Random pixel splits leak information between train and test; spatial blocking or region holdouts do not.
- **Baseline comparison.** State which simpler baseline you will compare against, and what margin over it would count as a real improvement.

Finally, define "good enough." What level of accuracy and reliability would make the solution useful to your target user?

#### Pre-Submission Checklist

Confirm each item before you submit:

- [ ] I can explain why this problem requires deep learning.
- [ ] My labels are not derived from my input features, or I have justified why the overlap is not circular.
- [ ] I have checked for existing solutions and positioned my work relative to them.
- [ ] I have a baseline comparison planned.
- [ ] I have a spatial cross-validation strategy.
- [ ] My input resolution is appropriate for my prediction target.
- [ ] I have a fallback plan if data is unavailable or insufficient.

### Rubric (10 points)

| Category | Weight | Excellent | Satisfactory | Unsatisfactory |
|----------|--------|-----------|--------------|----------------|
| Problem Definition & Use Case | 2 pts | Clearly articulates the problem, required output, and target user; states methods-vs-applied framing; well-scoped | Problem defined but required output, target user, or framing unclear | Problem vague; no clear output specification |
| Technical Justification | 3 pts | Clear logical chain from problem to method; accurately describes what the model learns; convincingly justifies the need for deep learning; identifies realistic failure modes | Some justification but gaps in reasoning; DL need or failure modes thin | No justification; task type inappropriate; no case for deep learning |
| Existing Work & Precedent | 2 pts | Checks foundation models and existing products and positions the work relative to them; 3+ rigorous sources with thorough summaries linked to the approach | Landscape check or sources present but shallow, or fewer than 3 sources | Missing landscape check; missing or irrelevant sources |
| Data Plan | 1.5 pts | Input and label tables complete; label provenance addressed with no circularity; feasible fallback plan | Datasets identified but tables incomplete or fit to task unclear | Datasets inappropriate or not identified |
| Modeling & Evaluation | 1.5 pts | Clear architecture and baseline; resolution matched to target; spatial CV and appropriate metrics justified | Model, metrics, or spatial CV described but weakly justified | Approach unclear; no baseline; no spatial CV; metrics inappropriate |

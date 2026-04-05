## Final Project Proposal

In this assignment, you will outline your proposal for the final project. You will describe the problem you're seeking to solve, justify why your chosen approach is appropriate for that problem, and demonstrate that you understand how your model will actually work. This proposal is your opportunity to think through the conceptual foundations of your project before you begin implementation.

_You may find it to check [our list of tooling, possible methods, and open commercial data catalogs.](../assignments/FINAL_PROJECT_PROPOSAL.md)_

### Proposal Requirements

Your proposal should be approximately 1,500 words and must address the following areas.

#### Problem Definition & Use Case

Describe the problem you're solving and why it matters. Identify your target user—whether a city agency, a research institution, or another stakeholder—and explain what decisions your output will inform. Be specific about what output your user actually needs: a binary mask of affected areas, a multi-class land cover map, a change detection layer comparing two time periods, or something else. The clearer you are about the required output, the easier it will be to design an appropriate solution.

#### Technical Justification

This is the most important section of your proposal. You must explain the logical chain from problem to method, demonstrating that you understand why your chosen approach is appropriate.

First, describe what your model is actually learning. What does a "positive" prediction mean in your context? What features will the model rely on to make predictions? For example, a model trained to detect "flooded" vs. "not flooded" images is really learning to detect the presence of water—which is not the same as detecting flooding. Flooding implies water where it shouldn't be, which requires either change detection or comparison to a baseline.

Second, explain why your chosen task type is appropriate. Why did you select classification, segmentation, change detection, or regression? What would happen if you chose a different task type—would it still answer your user's question?

Third, identify potential failure modes. Describe at least two realistic scenarios where your approach might fail or produce misleading results. For instance, a water segmentation model might flag permanent water bodies as floods, or a land cover classifier trained in one climate zone might fail in another due to different spectral signatures. Understanding where your approach might break is essential to using it responsibly.

#### Methodological Precedent

Reference at least three rigorous sources—academic papers, professional white papers, or well-documented challenge datasets—that inform your methodology. For each source, summarize the methods used, the data sources, the evaluation approach, and any limitations the authors identify. Explain how these sources shape your own approach and what you're borrowing or adapting from each.

#### Data Plan

Identify the dataset(s) you plan to use, including spatial and temporal extent, resolution, and relevant bands or features. Explain how your data supports the task you've defined—if you're doing change detection, for example, confirm that you have suitable pre- and post-event imagery. Discuss any potential data quality issues and describe your backup plan if your primary dataset proves unavailable or unsuitable.

#### Modeling Approach

Describe your proposed model architecture, including preprocessing steps and feature engineering. Before discussing your primary model, describe a simple baseline—a non-deep-learning approach you could use to establish a performance floor, such as NDWI thresholding for water detection or a random forest classifier on spectral bands. Explain what alternatives you considered and why you chose your preferred method.

#### Evaluation Strategy

Identify the metrics you will use to evaluate your model and explain why they are appropriate for your task and use case. Describe how you will split your data for training and validation. Finally, consider what "good enough" performance would look like for your user—what level of accuracy or reliability would make this solution practically useful?

### Rubric (10 points)

| Category | Weight | Excellent | Satisfactory | Unsatisfactory |
|----------|--------|-----------|--------------|----------------|
| Problem Definition & Use Case | 2 pts | Clearly articulates problem, required output, and target users; well-scoped and relevant | Problem defined but required output or target user unclear | Problem vague; no clear output specification |
| Technical Justification | 3 pts | Clear logical chain from problem to method; accurately describes what model learns; identifies realistic failure modes | Some justification but gaps in reasoning; failure modes generic | No justification; task type inappropriate for problem |
| Methodological Precedent | 2 pts | 3+ rigorous sources with thorough summaries; clear connection to proposed approach | Fewer than 3 sources or superficial summaries | Missing or irrelevant sources |
| Data Plan | 1.5 pts | Dataset clearly appropriate for task; feasible backup plan | Dataset identified but fit to task unclear | Dataset inappropriate or not identified |
| Modeling Approach & Evaluation | 1.5 pts | Clear architecture with baseline; appropriate metrics justified | Model and metrics described but weakly justified | Approach unclear; no baseline; metrics inappropriate |
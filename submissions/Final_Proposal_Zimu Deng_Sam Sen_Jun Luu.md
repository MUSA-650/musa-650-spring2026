**Building Damage Classifier for Post-Hurricane Assessment for Fort Myers, Florida**

**1. Problem Definition & Use Case**

This project proposes a deep learning framework for rapid post-hurricane building damage assessment for **Fort Myers, Lee County, Florida**. Fort Myers is a hurricane-prone coastal city in southwest Florida, where storm surge and heavy rainfall create recurring flood risk and make rapid post-disaster building damage assessment especially valuable for local response and recovery. The primary target user is the Lee County Emergency Management Division, including the **Emergency Operations Center (EOC)**, which is responsible for coordinating disaster preparedness, response, and recovery activities across Fort Myers and the rest of Lee County. In many smaller municipalities, detailed field-based damage assessment is slow and labor-intensive, yet emergency managers still need fast estimates of which buildings are likely affected and which appear more severely damaged in order to prioritize inspections, allocate recovery resources, and communicate impacts to decision-makers.

This motivation is closely aligned with the original xBD dataset and the xView2 model challenge, which were developed to advance rapid building damage assessment from pre- and post-disaster satellite imagery through benchmark deep learning models and model competition (Gupta et al., 2019). However, those models were primarily designed and compared under a shared dataset setting, while practical deployment in a specific local city raises a different challenge. Prior work has shown that when these models are applied to specific disaster events or new areas of interest, performance can decline because cross-event generalization remains limited and domain shift is substantial. As a result, an off-the-shelf xBD/xView2 model may not be sufficiently robust for direct local deployment. This project therefore tests whether a limited locally labeled dataset can be used to fine-tune a pretrained model for one disaster-prone small city, while also evaluating the practical value of transfer learning for urban disaster monitoring.

The required output is **building-level classification**. For each building footprint in the study area, the model will output:  
(1) a binary prediction of **damage / no damage**, and  
(2) for damaged buildings only, a second binary severity label of **light damage / severe damage**.

This output is intended to support triage rather than final legal or insurance assessment. It would allow a local agency to generate a preliminary map of likely affected properties within hours of image availability, helping staff target on-the-ground verification more efficiently.

**  **

**2. Technical Justification**

**2.1 What the model is actually learning**

In this project, a positive prediction means that a building shows evidence of **post-disaster impact relative to its pre-disaster condition and local context**. That distinction matters.

The model will learn visual cues such as abnormal debris around buildings, changes in roof condition or parcel appearance between pre- and post-event imagery, contextual signs of disruption in nearby streets or lots, and other patterns associated with hurricane-related building impact. The xBD paper also notes that damage varies by disaster type, building type, and place, and that contextual environmental information such as water and smoke can be important to interpretation (Gupta et al., 2019).

**2.2 Why classification is the right task**

The task type is **building-level classification with bi-temporal input**. This is appropriate because the user’s decision problem is parcel-oriented: they need to know which buildings likely need attention.

The proposed two-step classification design is also practical. First, a binary classifier identifies buildings likely affected. Second, a severity classifier separates light from severe damage among damaged buildings. This simplifies the original xBD four-level scale (no damage; minor damage; major damage, destroyed) into a more realistic operational scheme for limited local labels. xBD also highlights a practical limitation of the original four-class damage scheme: some damage categories are much less common than others, creating a strongly imbalanced classification problem (Gupta et al., 2019). In such cases, a model can achieve high overall accuracy by over-predicting the dominant no-damage class, while still performing poorly on the minority damage classes that matter most for response. To make the task more robust and more useful for local practice, this project simplifies the label structure into a two-stage binary framework during the fine-tune process: **first**, damage versus no damage; **second**, light versus severe damage among affected buildings.

**2.3 Potential failure modes**

One likely failure mode is false positives caused by temporary post-hurricane conditions such as standing water, debris, fallen vegetation, or blue roof tarps near buildings. These features may make lightly affected or even undamaged buildings appear more severely impacted than they are in overhead imagery, which is one reason why pre/post comparison is preferable to post-only inference.

A second failure mode is **label ambiguity for light damage**. Severe destruction is often more visible than light damage. If local labels are noisy, the model may confuse lightly damaged and undamaged buildings. This is another reason to collapse the label space into a more operational binary-plus-severity workflow instead of forcing the full xBD four-class system.

**  **

**3. Methodological Precedent**

The first key precedent is **xBD** itself. Gupta et al. (2019) introduced xBD as a large-scale dataset for building damage assessment using pre- and post-disaster imagery, building polygons, and ordinal damage labels across multiple disaster types. The paper’s baseline used building localization plus classification, and its broader contribution was to establish a benchmark for remote post-disaster assessment. Its limitations are also relevant: subtle class differences, possible label noise, and variation across disaster types and built environments. This project borrows the core idea of building-level damage classification and the use of xBD-trained models as a starting point.

The second precedent is **Zheng et al. (2024)**, which frames transferable building damage assessment as a domain adaptation problem. Their study emphasizes that building damage assessment suffers from distribution mismatch between source and target domains and proposes adaptation methods that use target pre-disaster imagery to reduce dependence on immediately available post-disaster labels. Even though our project will be simpler and supervised, this paper strengthens the conceptual claim that domain shift is not just a minor inconvenience but a central methodological challenge. My project borrows that framing, though with a more practical course-scale implementation based on local fine-tuning.

Thirdly, more application-oriented precedent is **Cao and Choe (2020)**, who used known building coordinates and square image crops from Hurricane Harvey imagery to classify buildings as damaged/flooded or undamaged. That study is useful because it shows that a building-centered patch classification workflow is feasible when building locations are already known from public data. My project similarly assumes that building footprints will be available, allowing me to skip localization and focus on classification and adaptation.

**  **

**4. Data Plan**

The primary imagery source for this project is the Maxar Open Data Program, which released both pre- and post-event high-resolution satellite imagery for Hurricane Ian under a Creative Commons 4.0 license, freely accessible via an AWS S3-hosted STAC catalog. Imagery was collected by **Maxar's WorldView** constellation at approximately 30–50 cm panchromatic resolution, sufficient for building-level damage classification. 

**The study area will focus on Fort Myers Beach, Lee County, Florida**, where storm surge and Category 4 winds destroyed or damaged an estimated 90% of structures — providing a spatially concentrated area of interest with high damage variability across the damage spectrum. Building footprints will be sourced from OpenStreetMap and supplemented where necessary with Microsoft's US Building Footprints dataset, a nationally comprehensive polygon layer derived from aerial imagery. Pre-storm OSM footprints will be used as the unit of analysis, with each building polygon used to extract paired pre- and post-event image patches for model input.

The primary spectral input will likely be RGB imagery, since the xBD baseline model is built around RGB data. If available, extra inputs such as NIR, NDWI/NDBI-style indices, or flood extent layers may be tested as auxiliary features, but the core proposal does not depend on them.

**  **

**5. Modeling Approach**

The preprocessing pipeline will begin by aligning pre- and post-event imagery and clipping both to the study area. The original xView2 baseline uses a two-stage workflow: it first detects building polygons and then extracts post-disaster polygon images for damage classification. In this project, that detection step will be simplified by using existing OpenStreetMap building footprints as the analysis units. Based on these footprints, I will generate **target-building-centered image samples**. Each sample will contain a pre-event patch, a post-event patch, the target building footprint, and the local label. This design keeps the workflow focused on local adaptation rather than building extraction, while also allowing the model to use explicit pre/post change information and limited surrounding context.

Labels will be created manually for a subset of buildings and then reorganized into two tasks:

1.  **damage vs. no damage**, and

2.  **light vs. severe** among damaged cases.

This departs from the original xBD four-class setup and reflects the operational goal of producing a simpler, more robust local workflow under limited labels.

As a simple non-deep-learning **baseline 0**, we will use **Random Forest classification** on hand-crafted features such as pre/post spectral differences, local water-related indicators, and contextual statistics around each building. This matches course guidance that a baseline should establish a performance floor and that Random Forest is a common non-DL benchmark in remote sensing.

**Baseline 1** will be a deep model trained only on the local labeled dataset from scratch.  
**Baseline 2** will be an off-the-shelf xBD/xView2 pre-trained model applied directly to the local city without adaptation.  
The **final model** will initialize from the xBD-based classification model and then be fine-tuned on limited local labels.

Because the original xBD baseline classification model is more closely associated with post-disaster building classification, my adapted model will modify the input setup to use **pre/post paired patches** while retaining transferable learned features where possible. The fine-tuning strategy will be staged: first train the new task head, then selectively unfreeze deeper layers with a smaller learning rate. This follows standard transfer-learning logic and is appropriate because the new task is similar but not identical to the source task (Keras, 2023).

**  **

**6. Evaluation Strategy**

Evaluation will emphasize practical usefulness for emergency triage, not only overall accuracy. For the first-stage binary classifier, I will report **precision, recall, F1-score, and ROC-AUC**. Recall is especially important because missing truly damaged buildings could delay response. For the second-stage severity classifier, I will report **per-class precision/recall/F1**, macro-F1, and a confusion matrix. This aligns with course material stressing that imbalanced classes and unequal error costs make simple overall accuracy insufficient.

To avoid inflated results from spatial autocorrelation, I will use a **spatially block train/validation split**, separating labeled buildings by neighborhood, block group, or another subarea rather than random nearby samples. Course material specifically warns that dense nearby samples can overstate performance and recommends sampling across the whole scene.

For the target user, “good enough” performance would mean a model that reliably reduces manual review workload, even if it does not replace expert inspection. Practically, that would mean substantially better recall and F1 than the off-the-shelf pre-trained model and a useful ranking of likely damaged buildings for field verification.

**Reference List**

Cao, Q.D. and Choe, Y. (2020) ‘Building damage annotation on post-hurricane satellite imagery based on convolutional neural networks’, *arXiv*. Available at: [https://arxiv.org/abs/1807.01688](https://arxiv.org/abs/1807.01688?utm_source=chatgpt.com)

Gupta, R., Hosfelt, R., Sajeev, S., Patel, N., Goodman, B., Doshi, J., Heim, E., Choset, H. and Gaston, M. (2019) ‘xBD: A dataset for assessing building damage from satellite imagery’, *arXiv*. Available at: [https://arxiv.org/abs/1911.09296](https://arxiv.org/abs/1911.09296?utm_source=chatgpt.com)

Keras (2023) ‘Transfer learning & fine-tuning’. Available at: [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/?utm_source=chatgpt.com)

Zheng, Z., Tian, J., Geiß, C. and Taubenböck, H. (2024) ‘Towards transferable building damage assessment via single-temporal domain adaptive semantic change detection’, *Remote Sensing of Environment*, 317, 114514. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S0034425724004425](https://www.sciencedirect.com/science/article/abs/pii/S0034425724004425?utm_source=chatgpt.com)

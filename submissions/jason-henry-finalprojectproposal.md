# Final Project Proposal
**Course:** MUSA 6500: Geospatial Machine Learning in Remote Sensing

**Instructors:** Nissim Lebovits, Guray Erus

**Authors:** Jason Fan, Henry Sywulak-Herr

## Problem Definition & Use Case


Philadelphia’s aging row home stock presents a unique structural challenge: a single neglected roof can compromise the stability of an entire block. With over 40,000 vacant lots, buildings classified as Imminently Dangerous (ID) within the [Philadelphia Property Maintenance Code](https://codelibrary.amlegal.com/codes/philadelphia/latest/philadelphia_pa/0-0-0-271402) represent the most severe category of code violation, where a structure is at "imminent danger of failure or collapse."

While the Department of Licenses and Inspections (L&I) maintains a database of vacant property violations, administrative data is often reactive and lagged. L&I has [discontinued](https://www.inquirer.com/newsletters/morning/vacant-abandoned-rowhouses-dangerous-licenses-inspections-tool-mayor-parker-housing-plan-church-tour-20251208.html) a predictive tool for vacant lot identification and Mayor Cherelle Parker’s administration has [indefinitely paused](https://www.inquirer.com/opinion/commentary/mayor-parker-housing-plan-missing-data-20250625.html) data collection on vacant lots. This left a data vacuum regarding buildings which pose the greatest immediate threat to Philadelphia residents.

The target user for this project is L&I’s Emergency Services Unit and the Philadelphia Land Bank, though it would be broadly useful to most infrastructure- and service-related organizations in Philadelphia (PECO, USPS, Philly Water Dept., Planning Office, etc.) that require accurate accounting of vacant parcels. These users require a tool that can move beyond static tax records to identify physical signatures of structural failure (roof collapses, bowing walls, or missing structural members). The output is a binary segmentation mask (Imminently Dangerous vs. Stable) at the parcel level. This output will help the target users identify prime candidates for acquisition and stabilization before they cause irreparable injury to residents or adjacent structures.

## Technical Justification

Structural damage in high-resolution satellite imagery is defined by changes in geometry and texture rather than simple spectral signatures.

### Semantic Segmentation:

To identify a building as "dangerous," a model must recognize localized failure points (e.g., a hole in a roof) within* the context of the larger structure.

- **Task Selection:** We will utilize *Semantic Segmentation* powered by a pre-trained Building Damage Model (similar models to post-disaster recovery models). By segmenting the specific areas of a roof or facade that shows “damage,” we can calculate a “structural distress score” per parcel.

- **Structural Distress Score:** By calculating the ratio of "damaged" pixels to the total building footprint, we can generate a normalized distress score.

### Failure Modes & Mitigation: 

- **New construction:** New roof installations can look like holes or debris. We will mitigate this by filtering parcels with active eCLIPSE permits.

- **Parallax & Occlusion:** Philadelphia’s tall "row home canyons" create shadows that hide facade leaning. We will integrate LiDAR-derived Canopy Height model to detect vertical deviations that RGB imagery might miss.

## Methodological Precedent

1) **The Clay Foundation Model (v1.5):** Instead of training a CNN from scratch, we use Clay, a Vision Transformer (ViT) pre-trained on global Earth Observation data. Clay’s ability to "understand" urban textures allows for better generalization across Philly’s diverse neighborhoods, from Kensington’s industrial shells to South Philly’s brick row houses.
    
    - [Schroer, et al. (2025)](https://aws.amazon.com/blogs/machine-learning/revolutionizing-earth-observation-with-geospatial-foundation-models-on-aws/) provides the foundations on the usage of Vision Transformer (ViT) models, such as the Clay model,  instead of the traditional CNN processes. Through this approach, the computational overhead and amount of labeled data needed to extract spatial insights is significantly reduced.

    - Schroer, K., Adhikari, B., & Moise, I. (2025, May 29). Revolutionizing earth observation with geospatial foundation models on AWS. AWS Machine Learning Blog.

2) **Building Footprint Decoupling:** Recent research suggests that decoupling localization (where is the building?) from classification (is it damaged?) improves accuracy in dense urban areas. We use the Open Buildings dataset to provide static masks, allowing the model to focus purely on structural integrity.

    - [Hatić et. al. (2025)](https://ieeexplore.ieee.org/document/9554795) and [Liu et. al. (2021)](https://www.mdpi.com/2072-4292/17/24/3957) both explore this approach.

    - Hatić, D., Polushko, V., Rauhut, M., & Hagen, H. (2025). Post-Disaster Building Damage Assessment: Multi-Class Object Detection vs. Object Localization and Classification. Remote Sensing, 17(24), 3957.

    - C. Liu, L. Ge and S. M. E. Sepasgozar, "Post-Disaster Classification of Building Damage Using Transfer Learning," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, Brussels, Belgium, 2021, pp. 2194-2197

3) **Potential Model Improvements:** Although many existing building damage detection models are designed for post-disaster (e.g., hurricane) contexts, the classification challenges faced for are still applicable to non-disaster contexts. Therefore, some model enhancement methods explore for use with these types of models could aid us in our approach:

    - [Ahmadi et. al. (2023)](https://doi.org/10.3390/rs16010182) discusses potential elements of a building damage detection model that we could explore for our modeling approach, including the use of adaptive kernel sizes and/or data augmentation (to overcome the imbalance of buildings and background pixels).

    - Ahmadi, S. A., Mohammadzadeh, A., Yokoya, N., & Ghorbanian, A. (2024). BD-SKUNet: Selective-Kernel UNets for Building Damage Assessment in High-Resolution Satellite Images. Remote Sensing, 16(1), 182.

## Data Plan

- **Primary Imagery:** City of Philadelphia Aerial Imagery (most recent = 2025, past years available)
    
    - Resolution: 0.25ft (7.62cm)
    - Bands: 3-band (RGB)
    - Justification: Ultra-high resolution is essential for identifying precise details of the built environment.
    - [Source](https://www.pasda.psu.edu/download/philacity/data/Imagery2025/tiles/)

- **Vector Data:** Philadelphia Water Department (PWD) parcel layers and L&I "Vacant Lot" violation data will serve as the source for training labels. We’ll consider using clean and seal data to improve the accuracy of the ground truthed data.

    - [PWD Parcels](https://opendataphilly.org/datasets/pwd-stormwater-billing-parcels/)
    - [L&I Code Violations](https://opendataphilly.org/datasets/licenses-and-inspections-code-violations/)
    - [L&I Clean and Seal](https://opendataphilly.org/datasets/licenses-and-inspections-clean-and-seal/)

## Modeling Approach

Our model will derive from a pre-built, global foundation model from [Clay](https://clay-foundation.github.io/model/index.html) designed to “efficiently distill and synthesize vast amounts of environmental data.”

- **Structural Embedding (Clay):** For each building parcel, capture textural decay (rusting, debris, weathering).

- **Segmentation Head (UNet):** A UNet decoder will process these outputs to produce a binary damage mask.

- **Temporal Change Detection:** By comparing 2024 and 2025 imagery, we identify the rate of decay. A building that shows a sudden "darkening" or "texture change" in the roof area over 12 months will be flagged for immediate inspection.

## Evaluation Strategy

Given the high cost of sending an inspection team, we prioritize Precision over Recall.

- **Primary Metric:** *Mean Intersection over Union (mIoU)* for the "Damaged" class.

- **The "ID" Goal:** A Precision of *>0.85* for the "Imminently Dangerous" class. We want to ensure that 85% of the buildings we flag actually require an orange "Unsafe" or "ID" poster upon manual inspection.

- **Geographic Stratification:** We will evaluate the model separately in North, South, and West Philadelphia to ensure that differences in architectural style (e.g., wood frame vs. masonry) do not bias the results.

## Comparison of Considered Approaches

| Approach |   Pro    |   Con    |
|----------|----------|----------|
| **NDVI Baseline** | Finds "green" roofs (leaks leading to moss). | Misses structural failures without vegetation. |
| **Random Forest** | Fast; uses height features well. | Fails to capture the "shape" of a collapse. |
| **Clay + UNet** | Captures decay textures; high precision. | High GPU demand; requires high-res imagery. |
## Week 6: Validation, Interpretation, and Generalization

### Lecture
Slides: [TODO]

### Lab: Multi-class land cover classification and model generalization

In this lab, we'll extend last week's binary forest classification to a multi-class problem. We'll work with the same Vienna study area and Sentinel-2 data, but this time classify four land cover types: forest, agriculture, urban, and water. You'll think critically about sampling strategy, feature engineering, evaluation metrics, and what happens when you apply a model trained in one place to a completely different one.

#### Reference Materials

You'll continue building on [the Project Pythia classification lab](https://projectpythia.org/eo-datascience-cookbook/notebooks/templates/classification/) from last week.

#### Part 1: Training Data Design

1. You'll need to define training polygons for four classes: forest, agriculture, urban, and water. You can reuse the forest polygons from last week, but you'll need to create new ones for the other three classes. Use your RGB and false color composites to guide placement. Think carefully about where you place your polygons — are your urban samples capturing only dense city center, or also suburban areas? Are your agriculture samples capturing active cropland, or also fallow fields?

Training polygons were sourced from three cloud-hosted datasets: [Overture Maps land_cover](s3://overturemaps-us-west-2/release/2026-01-21.0/theme=base/type=land_cover/) (forest and urban classes), [Overture Maps water](s3://overturemaps-us-west-2/release/2026-01-21.0/theme=base/type=water/) (water class), and [fiboa Austrian crop boundaries](https://data.source.coop/fiboa/at-crop/at_crop_2024.parquet) (agriculture class). Overture polygons are natively in EPSG:4326; Austrian crop boundaries were reprojected from EPSG:31287. All polygons were filtered to the Vienna bounding box and a minimum size threshold (~400m in each dimension) to ensure they contain enough unmixed pixels for training at Sentinel-2 resolution. Data was queried using [DuckDB](https://duckdb.org/) with the spatial extension, reading directly from cloud-hosted Parquet files.

2. Examine the class balance of your training data. Are the classes roughly equal in size? If not, what problems might this cause? Implement a strategy to address any imbalance (e.g., undersampling the majority class or oversampling the minority class).

#### Part 2: Feature Engineering

3. Last week we used red, green, blue, and NIR. For a four-class problem, we'll likely need more discriminating features. Engineer at least two spectral indices (e.g., NDVI, MNDWI, NDBI) and add them as features alongside the raw bands. Justify your choices---how will your engineered features help you distinguish the four classes we're trying to map?

4. Train a Random Forest on your full feature set. Inspect feature importance. Which features contribute most? Are the indices you engineered more or less important than the raw bands?

#### Part 3: Multi-Class Classification and Evaluation

5. Split your data 70/30 and standardize as before. Train a Random Forest classifier and generate predictions for the full scene.

6. Plot the predicted land cover map. Visually inspect the results — do the class boundaries look reasonable? Where do you see obvious errors?

7. Report the confusion matrix and classification report. Which classes are easiest to distinguish? Which are most commonly confused with each other? Why do you think this is?

8. Last week we used simple accuracy to compare models. With four classes of potentially different sizes and importance, accuracy alone can be misleading. Compare overall accuracy, balanced accuracy, and macro-averaged F1 score. Why might these differ? Which metric would you report to a stakeholder, and why?

#### Part 4: Spatial Cross-Validation

9. So far, our training and testing pixels come from the same polygons, split randomly. Why might this inflate our accuracy estimates? (Hint: think about spatial autocorrelation — neighboring pixels tend to look alike.)

10. Implement a spatial cross-validation strategy using `GroupKFold`, where each fold holds out entire polygons rather than random pixels. Compare your spatially cross-validated accuracy to your random-split accuracy. How much does performance drop? What does this tell you about how the model would perform on truly new areas?

#### Part 5: Model Transfer

11. Load Sentinel-2 imagery for Damascus, Syria for the same time period. Apply your Vienna-trained Random Forest to this new scene and map the predictions. 

12. Visually assess the results (map your predictions next to the RGB image). Does the model produce a reasonable land cover map for Damascus? Where does it succeed, and where does it clearly fail? Why? Consider differences in climate, vegetation types, urban form, and agricultural practices. What would you need to do to make this model work in a new geography?

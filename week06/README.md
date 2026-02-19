## Week 6: Validation, Interpretation, and Generalization

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vQvSDXR0Y9ZJuUwOCcEaIcEia_Mqoxky7gQXfZXXpxoOhXvIn3ZRU1IqBkHodytHNs3ZIMjCXtbWRvY/pub?start=false&loop=false&delayms=3000

### Additional Resources

[Supervised Machine Learning for Science](https://ml-science-book.com/)

[MLU Explain](https://mlu-explain.github.io/)

["Accuracy, Precision, and Recall in Multi-Class Classification,"](https://www.evidentlyai.com/classification-metrics/multi-class-metrics) EvidentlyAI

[Mapbiomas accuracy assessment](https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2024/08/ACCURACY-ASSESSMENT-Appendix-Collection-9.pdf)

### Lab: Multi-class land cover classification and model generalization

In this lab, we'll extend last week's binary forest classification to a multi-class problem. We'll work with the same Vienna study area and Sentinel-2 data, but this time classify six land cover types: water, wetland, urban/built-up, cropland, grassland, and forest/shrubland. In this lab, you'll think critically about sampling, feature engineering and selection, evaluation metrics, feature importance, and generalizability.

#### Reference Materials

You'll continue building the lab from Week 5; reference the README and solutions from last week as needed.

#### Part 1: Data Aquisiton, Preprocessing, and Feature Engineering

1. Load our imagery from the same area of Vienna but in March of 2020. I had some issues getting cloud-free imagery in March; you can use my query like this:

```{python}
# Define spatial and temporal extent for Vienna
dx = 0.0006  # ~60m resolution
epsg = 4326

# Vienna area bounds
vienna_bounds = (16.32, 47.86, 16.9, 48.407)

# Temporal extent: March 2020
start_date = datetime(year=2020, month=3, day=1)
end_date = start_date + timedelta(days=30)

date_query = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

# Search for Sentinel-2 data
stac_client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

vienna_items = stac_client.search(
    bbox=vienna_bounds,
    collections=["sentinel-2-l2a"],
    datetime=date_query,
    query={"eo:cloud_cover": {"lt": 50}},  # skip heavily cloudy scenes
    limit=100,
).item_collection()

print(f"Vienna: {len(vienna_items)} scenes found")
``` 

2. Load the training data from `vienna_samples.parquet` in the repo. If you haven't worked with GeoParquet data before, this is very simple. Just use `gpd.read_parquet()`. These data were derived from OSMlanduse, a contiguous land use map of the EU, current as of March 2020.[^Schultz, M., Li, H., Wu, Z. et al. OSMlanduse a dataset of European Union land use at 10 m resolution derived from OpenStreetMap and Sentinel-2. Sci Data 12, 750 (2025). https://doi.org/10.1038/s41597-025-04703-8] In the real world, this is not the most reliable way to generate training data. But for our lab, it'll be okay. :) 

Examine the class balance of the dataset. How many total samples are there? How many samples per class? How should this inform our sampling and evaluation strategies for our model?

3. Load in all the bands for our AOI (not just a select few like last time). Give some thought to feature selection + engineering. Compute a couple of indices (e.g., NDVI, MNDWI) that you think will be useful and add them to the feature set. Plot a correlation heatmap of all the bands and use it to help select your final feature set for your model.

#### Part 2: Multi-Class Classification and Evaluation

4. Using a 70/30 split and your chosen sampling approach, compare four different random forest models using samples of 100, 1,000, 5,000, and all 10,000 points, respectively. What validation metric(s) will you use to compare these models? How does performance compare across each? What does this tell you about the importance of sample size in this context?

5. Compare the results of your best 70/30 split model to two more validation approaches: regular 5-fold cross validation and 5-fold spatial cross validation. How does the macro-averaged accuracy compare for each? 

6. For the spatial cross fold validation, report full validation metrics, including: macro-averaged accuracy, per-class precision amd recall, and a full confusion matrix. What do you make of the model performance? When you plot the confusion matrix, which classes appear hardest to separate? Does this make sense to you? What does the actual predicted map look like, and how does it compare to the RGB or false color image?

7. Plot feature importance for the model, including: MID, permutation importance, and SHAP values. Which features are most important? Does this match your expectations? Do you notice any comparisons across feature importance methods that suggest colinearity in your features? Would you keep this feature set or go back and retrain with a different feature set?

#### Part 3: Generalization

8. Load Sentinel-2 imagery for Damascus, Syria for the same time period (March 2020). Apply your Vienna-trained Random Forest to this new scene and map the predictions. Visually assess the results (map your predictions next to the RGB image). Does the model produce a reasonable land cover map for Damascus? Where does it succeed, and where does it clearly fail? Why? Consider differences in climate, vegetation types, urban form, and agricultural practices. What would you need to do to make this model work in a new geography?
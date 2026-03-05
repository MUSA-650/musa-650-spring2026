## Week 7: Midterm Exam

See the [midterm exam description](../assignments/MIDTERM.md).

### Lab: Supervised Land Cover Classification with Landsat 8 and NLCD Data

In this lab, you'll learn to use Google Earth Engine (GEE) for supervised land cover classification. GEE is a powerful platform that enables efficient planetary-scale analysis of raster data and is therefore widely used in remote sensing applications. While we won't go into depth on the content this year, you can reference [this presentation from the 2024 class](https://docs.google.com/presentation/d/e/2PACX-1vTch18CngNjfOjh2ChnYbsW4In6Cbdiz1S8iucGW7fe3cYQvUmO4CJ5bq6DqHgXVIfAA8E_Tb3XuKDn/pub?start=false&loop=false&delayms=3000) and the [Google Earth Engine tutorials book](https://google-earth-engine.com/).

The lab itself is adapted from [Dr. Qiusheng Wu's NLCD tutorial](https://geemap.org/notebooks/32_supervised_classification/). The hardest part will likely be getting set up and authenticated, which can be a little annoying, and getting used to the different ergonomics of handling data in Earth Engine.

#### 1. Set up and authenticate your Google Earth Engine account using `geemap`. 
Accounts are free for academic/non-profit use. Make sure you copy your project ID correctly; it can be a bit tricky to figure out what it actually is, but you can find it in the top right corner of the Earth Engine Code Editor in the browser: https://code.earthengine.google.com/.

#### 2. Load data and labels.
Referencing last week's lab, select imagery with the same cloud cover, AOI, and date range over Vienna. Load our training data into an Earth Engine asset. (This time, though, use native 10m sentinel resolution instead of 60m.) 

Note that you'll have to convert the parquet data to an Earth Engine asset using `gdf_to_ee`. This will take maybe 15 seconds.

Plot an interactive map with the median true color and false color composites. Play with the map UI--try using the transparency slider and clicking the layers on and off.

#### 3. Train a Random Forest model with an 80/20 split.
GEE has no tooling for cross-validation, spatial or otherwise. The best option would be exporting the data _post hoc_ to sklearn for true cross validation, but since we covered that last week, we'll skip it here.

#### 4. Output metrics and classification.
Output the metrics available from Earth Engine, including the confusion matrix, overall accuracy, kappa, and producer and consumer's accuracy (i.e., recall and precision) per class.

Apply the trained model to the entire Vienna scene. How quickly does this happen? How do the results look? Plot them interactively as before.

#### 5. Feature importance.
Using the built-in MDI, print the feature importance for our model. How does this compare to last week's results? What do we recall about MDI and how it differs from permutation importance or SHAP?

#### 6. Compare the classifier to ESA WorldCover across all of Austria.
The best part of GEE is how well it scales. Apply our classifier to Sentinel data across all of Austria and then plot it over ESA WorldCover data. How quickly does it run? How do our results compare to the WorldCover classifications?
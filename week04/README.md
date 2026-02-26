## Week 4: Introduction to Machine Learning

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vS7BXOvSJxg_rjSBYnAzn2xL_yD8Ko8b7qTVTsa_Gv_8qweeuK_qPwy6uA7aZIeiJq1B0ic1f5spLXe/pub?start=false&loop=false&delayms=3000

[Download the PDF](https://docs.google.com/presentation/d/1gqe8qXBKMq0Hr5_LGvVr9WEzw2GV5O05XG1goqzGDoc/export/pdf)

### Lab: Unsupervised and Supervised ML for Water Extent Monitoring

This week, we'll apply k-means clustering and a simple linear classifier to Landsat 8 imagery to monitor changes in a lake in Nevada over time. As usual, you'll generate your own code, working mostly from this [spectral clustering tutorial from the Landsat ML Cookbook](https://projectpythia.org/landsat-ml-cookbook/notebooks/spectral-clustering-pc/).

A couple of small notes here:
- For ease of visualization, we recommend using static `matplotlib` plots rather than interactive `hvplot` ones.
- Instead of spectral clustering, we'll use k-means clustering from `dask-ml`, since there seems to be some kind of bug in the spectral clustering in the latest version of Dask.

### Part 1: Data Preparation

1. Per the Landsat ML Cookbook tutorial, query Landsat 8 data from Planetary Computer using the following parameters: 

```
bbox = [-118.89, 38.54, -118.57, 38.84]  # Region over a lake in Nevada, USA
datetime = "2017-06-01/2017-09-30"  # Summer months of 2017
collection = "landsat-c2-l2"
platform = "landsat-8"
cloudy_less_than = 1  # percent
```
Examine the metadata (e.g., bands) and visualize the pre-rendered overview of the scene.

2. Load the data with `odc-stac`. What does `da_2017` look like when you view it? (Just run `da_2017` in a cell to see the output.) What dimensions does it have? What size is it, and how does its chunk size relate to the total size?

3. Flatten your array so that it is two-dimensional: `n_samples` by `n_features`. View this the same way you did before--what does it look like now?

4. Standardize your data following the Landsat ML Cookbook tutorial. Why is standardization important for k-means clustering? (Hint: Euclidian distance.)

### Part 2: K-Means Clustering

5. Initialize your Dask cluster and use `dask_ml` to fit a KMeans model to the data. The tutorial uses 4 clusters, which is fine for our purposes. What bands are we loading into our clusters? 

6. Map the resulting clusters next to the original image (you can just use a grayscale band for this). How do the clusters correspond to land versus water? Do you notice any inconsistencies or noise?

### Part 3: Supervised Classification

7. Generate 200 training points, evenly split between water and land classes, based on points close to the lake center and points far away from it. (Hint: the approach used to map the lake center in the tutorial is a good starting point.) Map these points over a basemap to verify that they look correct. (Be mindful that this approach would _not_ hold up in academia or industry--we're just doing this for demonstration purposes.)

8. Use the training points to train a Dask linear classifier on the 2017 image. Validate it with stratified 5-fold cross-validation. Plot a classification report--what's the precision and recall? What does the confusion matrix look like? (Bear in mind that these are likely inflated, as they don't take spatial autocorrelation into account--more on this over the next two weeks.) How does the map of water produced by the linear classifier compare to the k-means results?

9. Use the trained classifier to predict a binary mask for the original image. How do the results look? Do you notice any inconsistencies or noise?

### Part 4: Index + Otsu's Threshold

10. Calculate the MNDWI (Modified Normalized Difference Water Index). Use this to generate a binary water mask (Otsu's method works well here) and compare the result to the k-means and classifier outputs. How do they compare? What are the pros and cons of each method?


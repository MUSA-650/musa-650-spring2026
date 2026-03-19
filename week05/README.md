## Week 5: Data Generation and Model Selection

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vR4RgLroTDM9ZBZ1U9afBOOFcHvPOKjorr9WKDQKCsU6gOwriaMwy9VbY62-uHRh-gHayxmc-vM3bn0/pub?start=false&loop=false&delayms=3000

[Download the PDF](https://docs.google.com/presentation/d/1RelyyOyaA8yL-IgplErF9M8wKRnTeh1cBlxRfX-0r8c/export/pdf)

### Recommend Resources
- [End-to-end land cover classification workflow](https://brasil.mapbiomas.org/en/atbd-entenda-cada-etapa/) from Mapbiomas
- [Fundamentals of Machine Learning for Earth Science](https://appliedsciences.nasa.gov/get-involved/training/english/arset-fundamentals-machine-learning-earth-science), NASA ARSET

### Lab: Supervised land cover classification with Sentinel 2

In this lab, we'll compare several different traditional ML models applied to classifying Sentinel 2 data over Vienna, Austria in order to produce a forest cover map. You'll generate your own code based largely on [this lab from Project Pythia](https://projectpythia.org/eo-datascience-cookbook/notebooks/templates/classification/).

_Note: as in last week's lab, you'll want to use `dask` to speed up data loading. None of your cells should take longer than a minute to run, barring bad network connection. If you find that this is happening, in particular when calculating median composites, pause and double-check where you're calling `.compute()`. Feel free to check the solutions notebook if necessary._

#### Part 1: Data Aquisition + Preprocessing
1. Following the code in the lab, pull Sentinel 2 data over Vienna from May of 2024. Load the data into an `xarray` dataset and visualize the RGB image and false color composite.

#### Part 2: Baseline -- NDVI + Otsu's Method
2. Using the appropriate Sentinel 2 bands, calculate and plot NDVI for the image. Apply Otsu's method to produce a binary vegetation mask. Recall that our task for this lab is to classify _forest_ cover. Given the land cover in and around Vienna (take a look at your RGB image if you don't know what it looks like), why might NDVI + Otsu's rule not be appropriate here?

#### Part 3: Classification
3. Plot the training samples provided by the Project Pythia cookbook, as they do. Examine the polygons. What do you notice about their size, placement, and composition? How might this impact the results of our model? (Hint: what land cover classes does each polygon actually include?)

4. You'll be using the red, green, blue, and NIR bands for training and testing. Based on what you've learned in previous labs, why do you think we've selected these?

5. Split your data into training and testing sets (use a 70/30 split here, not 50/50 as the Project Pythia notebook does). Standardize your input features as we did in the previous week's lab. Since we're going to be training a linear classifier in addition to the naive Bayes and Random Forest, we'll need to standardize (although we wouldn't have to do this if we were just making the naive Bayes and Random Forest).

6. Train a linear classifier, naive Bayes, and Random Forest model. For each, examine the classification report, confusion matrix, and a map of the predictions.

7. Compare the results of each classifier. Which model is most accurate? 

8. Map a 2x2 grid of the outputs of the threshold approach and all three classifiers. Following the Project Pythia cookbook, map a classification comparison of the threshold and each of the models (this should give you three such comparisons). Where does each model differ from the threshold? What types of pixels seem to generate disagreement?

9. Pick your most accurate model and add the NDVI as an additional feature. Retrain the model and test the model. Does performance improve? When you map the output, do you see any changes in the accuracy of the map, or in the areas that it classifies? Would you say that including NDVI as a feature improves performance?
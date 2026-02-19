## Week 2: Remote Sensing Data Sources and IO

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vRuiwyYrl8nfDl6x5N5yzepKcBLyHUiS-Ofrpg2Y_Ikn7UfaAV_lDlxwa0mEUsez5NRfnXS96MfVqpK/pub?start=false&loop=false&delayms=3000

### Additional Resources
Landsat ML Cookbook, Demetris Roumis. https://projectpythia.org/landsat-ml-cookbook/

STAC Spec: https://stacspec.org/en/ 

Cloud-Optimized Geospatial Formats Guide, https://guide.cloudnativegeo.org/ 

### Lab: Landsat Data Discovery and Access with STAC
In this lab, you will retrieve Landsat imagery for Singapore and produce a cloud-free composite image. You will work through the full workflow of discovering, inspecting, loading, and exporting satellite data using modern cloud-native tools.

You will not be given any code. Instead, you will work from documentation and tutorials to figure out how to accomplish each step. This mirrors real-world practice: the tools change constantly, but the workflow patterns remain consistent. Your job is to learn to navigate documentation and adapt examples to your specific problem.

**Study area**: Singapore  
**Time period**: 2025 (January through December)  
**Dataset**: Landsat Collection 2 Level-2 
**Goal**: Produce and export a cloud-free true-color composite

#### Reference Materials

This tutorial is based on the following resources. You are not limited to using these, but they will be helpful in pulling together the code you need to complete the lab.

[STAC Spec Tutorial: Reading STAC with Planetary Computer](https://stacspec.org/en/tutorials/reading-stac-planetary-computer/)

[Project Pythia: Data Ingestion (Geospatial)](https://projectpythia.org/landsat-ml-cookbook/notebooks/data-ingestion-geospatial/)

[Planetary Computer: Cloudless Sentinel-2 Mosaic](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/cloudless-mosaic-sentinel2.ipynb)

#### Part 1: Understand what data is available and how STAC organizes it.

1. Connect to the Planetary Computer STAC catalog using `pystac-client`.

2. Find the Landsat Collection 2 Level-2 dataset. Per [Project Pythia](https://projectpythia.org/landsat-ml-cookbook/notebooks/data-ingestion-geospatial/):
> We’ll use the `landsat-c2-l2` dataset, which stands for Collection 2 Level-2. It contains data from several landsat missions and has better data quality than Level 1 (`landsat-c2-l1`). Microsoft Planetary Computer has descriptions of [Level 1](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l1) and [Level 2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2), but a direct and succinct comparison can be found in [this community post](https://gis.stackexchange.com/questions/439767/landsat-collections), and the information can be verified with [USGS](https://www.usgs.gov/landsat-missions/landsat-collection-2).

3. Explore the metadata by simply examining the collection object. What's the description? STAC version? Check out the links, the ID, the license, keywords, etc. List the assets available in this collection. What bands are available? What other assets (e.g., QA bands, thumbnails) are included?

#### Part 2: Query for imagery over your study area and verify the results make sense.

4. Define a bounding box for Singapore. Hint: we deliberately chose Singapore because it's small enough to fit within a single Landsat tile; this simplifies loading.

5. Set up a Dask cluster for parallel processing. Use `odc.stac.configure_rio(cloud_defaults=True, client=client)` to configure `rasterio` for efficient cloud COG access. This step is critical—without it, loading data from the cloud will be extremely slow.

6. Search for Landsat scenes within your bounding box for 2025. Filter for scenes with less than 50% cloud cover. (Singapore is cloudy; you may need to adjust this threshold to get enough scenes.)

7. Preview your results. How many scenes did you get? Look at some thumbnails to verify you're getting what you expect.

8. Visualize the STAC item footprints on a map. Do the tiles cover your entire bounding box? Are there gaps?

9. Create a time series plot showing cloud cover percentage per scene across the year.

#### Part 3: Combine multiple scenes into a single cloud-free image.

10. Load the data using `odc.stac.load`. Include the spectral bands (red, green, blue, nir08) and qa_pixel for cloud masking. Experiment with different resolutions and chunk sizes to get the data to load efficiently. A couple of hints:
 - We deliberately selected Singapore to avoid the need to load multiple tiles to cover the AOI.
 - Landsat data is 30m resolution. You _can_ coarsen this to load more quickly, but if you optimize your chunking, you should be able to load all of your data at 30m resolution in a couple of minutes.
 - If you get _really_ stuck, feel free to look at the solutions notebook.

11. Examine the resulting dataset. What are its dimensions? What's the CRS? How much data would you be loading if you called `.compute()` right now?

12. Create a cloud mask using the `qa_pixel` band. Bits 3 and 4 indicate cloud and cloud shadow respectively. Mask out cloudy pixels so they become NaN.

13. Create a median composite across all time steps. Because cloudy pixels are NaN, the median will ignore them and select from clear observations only. Call `.compute()` to trigger the actual download and computation.

#### Part 4: Visualize and export your results.

14. Plot individual bands (red, green, blue, NIR) as separate grayscale images.

15. Plot a true-color composite and a false-color composite. You'll need to stack the bands and normalize the values appropriately.

16. Export your true-color composite as a Cloud-Optimized GeoTIFF (COG). Open in QGIS or ArcGIS and compare it to a satellite basemap of your choice.



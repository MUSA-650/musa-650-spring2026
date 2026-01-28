## Week 3: Non-ML Approaches to Remote Sensing

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vTOTYqwvctmoNefZbcLZhek4vaHxzxiqc4KJCjrBGV1-TrTWnoKF3SMTmdE3-mcEzAKqkZxNuO3DT7f/pub?start=false&loop=false&delayms=3000

### Additional Resources
Applied Remote Sensing Training Program, NASA, https://www.earthdata.nasa.gov/data/projects/arset/learn 

Google Earth Engine Tutorials, https://google-earth-engine.com/ 


### Lab: Wildfire Detection and Burn Severity Assessment

In this lab, you'll analyze a real wildfire event using techniques that don't require machine learning: spectral composites, multi-sensor integration, and spectral indices. These methods form the foundation of remote sensing analysis and remain widely used in practice. Before reaching for a neural network, you should always ask: can I solve this problem with an index, a threshold, or a simple comparison?

You will not be given any code. Instead, you will work through tutorials developed by ESA's Earth Observation Processing Framework (EOPF) team and adapt their code to complete the tasks below. The tutorials are thorough—your job is to read carefully, understand what each step does, and apply it.

**Study area**: Province of Nuoro, Sardinia, Italy  
**Event**: Wildfire, June 10, 2025 (~1000 hectares burned)  
**Datasets**: Sentinel-2 L2A, Sentinel-3 SLSTR LST  

#### Reference Materials

These three tutorials contain everything you need. You don't need to work through them linearly; pull what you need for each task.

- [Fire in Sardinia - Part 1](https://eopf-docs.eodc.eu/eopf-data-book/EOPF%20Zarr%20in%20Action/sardinia-fire-p1.html): True and false color composites, pre/post comparison
- [Fire in Sardinia - Part 2](https://eopf-docs.eodc.eu/eopf-data-book/EOPF%20Zarr%20in%20Action/sardinia-fire-p2.html): Sentinel-3 LST integration, thermal anomaly detection
- [Fire in Sardinia - Part 3](https://eopf-docs.eodc.eu/eopf-data-book/EOPF%20Zarr%20in%20Action/sardinia-fire-p3.html): Normalized Burn Ratio (NBR) and burn severity classification

You may also find the broader [EOPF 101 documentation](https://eopf-docs.eodc.eu/eopf-data-book/) helpful for understanding STAC, Zarr, and the data catalog.

For the time series section, this Project Pythia tutorial demonstrates useful visualization patterns:

- [Project Pythia: NDMI Time Series](https://projectpythia.org/landsat-ml-cookbook/notebooks/ndmi/)

#### Part 1: Pre-Fire Baseline

1. Connect to the EOPF STAC catalog and search for Sentinel-2 L2A imagery over the study area for June 3, 2025 (one week before the fire).

2. Open the retrieved item and explore its structure. What groups are available? Where are the spectral bands stored? Where is the Scene Classification Layer (SCL)?

3. Create a cloud mask using the SCL band. What pixel types does this mask remove, and why is this necessary?

4. Produce a **true color composite** (RGB) for the pre-fire date. Apply histogram equalization to improve visual contrast.

5. Produce a **false color composite** using SWIR (B12), NIR (B8A), and Blue (B02). How does vegetation appear in this band combination? Why is this combination useful for distinguishing land cover types?

#### Part 2: Post-Fire Comparison

6. Search for and load Sentinel-2 imagery from June 21, 2025 (after the fire).

7. Produce true color and false color composites for the post-fire date using the same methods as Part 1.

8. Compare your pre-fire and post-fire false color composites. Where is the burn scar visible? How does the spectral signature of burned area differ from healthy vegetation in this band combination?

#### Part 3: Thermal Anomaly Detection

9. Search for Sentinel-3 SLSTR LST data from June 10, 2025 (the day of the fire). Note: Sentinel-2 has a 5-day revisit, so optical imagery isn't available for the exact fire date—this is why thermal data is valuable.

10. Extract Land Surface Temperature (LST) for your area of interest. Filter to show only pixels above 312 Kelvin. What are we isolating with this threshold?

11. Overlay the filtered LST data on your Sentinel-2 true color composite from the closest available date. What does this combined view tell you that neither dataset alone could show?

#### Part 4: Burn Severity Assessment

12. Calculate the Normalized Burn Ratio (NBR) for your pre-fire image using the formula:

$$NBR = \frac{NIR - SWIR}{NIR + SWIR}$$

What bands does this use (specifically, which Sentinel-2 band numbers)? Why are NIR and SWIR sensitive to fire damage?

13. Calculate NBR for your post-fire image.

14. Calculate the differenced NBR (dNBR) by subtracting post-fire NBR from pre-fire NBR. Multiply by 1000 to match standard severity thresholds.

15. Using the EFFIS burn severity classification below, what severity class does the Sardinia fire fall into?

| Class | dNBR range (×1000) |
|-------|-------------------|
| Unburned / Regrowth | < 100 |
| Low severity | 100–270 |
| Moderate-low severity | 270–440 |
| Moderate-high severity | 440–660 |
| High severity | ≥ 660 |

#### Part 5: Time Series Analysis

Two snapshots tell you about change, but a time series tells you about recovery. In this section, you'll extend your analysis to track NBR over a full year.

16. Query the EOPF catalog for all available Sentinel-2 L2A scenes over your AOI from January 2025 through December 2025. How many scenes are available?

17. Calculate NBR for each scene. You'll want to write a helper function or loop to do this efficiently. Make sure to apply cloud masking to each scene.

18. Create a time series plot showing mean NBR within the burn scar area over the year. You can select a small buffer around the fire center coordinates to extract values. What pattern do you see? When does the fire occur in your time series? Is there any evidence of vegetation recovery by the end of the year?

19. Create a faceted plot (small multiples) showing NBR maps for each month, similar to the NDMI example in Project Pythia. This gives you a visual sense of how the landscape changes through time.
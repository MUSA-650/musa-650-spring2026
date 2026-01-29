## Week 3: Non-ML Approaches to Remote Sensing

### Lecture
Slides: https://docs.google.com/presentation/d/e/2PACX-1vTOTYqwvctmoNefZbcLZhek4vaHxzxiqc4KJCjrBGV1-TrTWnoKF3SMTmdE3-mcEzAKqkZxNuO3DT7f/pub?start=false&loop=false&delayms=3000

### Additional Resources
Applied Remote Sensing Training Program, NASA, https://www.earthdata.nasa.gov/data/projects/arset/learn 

Google Earth Engine Tutorials, https://google-earth-engine.com/ 


### Lab: Wildfire Detection and Burn Severity Assessment

In this lab, you'll analyze a real wildfire event using techniques non-ML techniques, including indices and change detection. These methods form the foundation of remote sensing analysis and remain widely used in practice.

You will not be given any code. Instead, you will work through tutorials developed by ESA's Earth Observation Processing Framework (EOPF) team and adapt their code to complete the tasks below.

**Study area**: Province of Nuoro, Sardinia, Italy  
**Event**: Wildfire, June 10, 2025 (~1000 hectares burned)  
**Dataset**: Sentinel-2 L2A


#### Reference Materials

_Note: the EoPF toolikit is under active development and is, as of Jan. 29, 2026, not rendering properly. The text and code still remain accessible, though. For our tutorial, you'll want to make the following modifications:
- Use only the first `search_box` as your AOI, not the larger `bbox_vis` that they define later.
- Access the Sentinel 2 data via Microsoft Planetary Computer, not EoPF's Zarr store, since the latter is not yet functional._


These tutorials contain everything you need. You don't need to work through them linearly; pull what you need for each task.

- [Fire in Sardinia - Part 1](https://eopf-toolkit.github.io/eopf-101/06_eopf_zarr_in_action/61_sardinia_s2_tfci.html): True and false color composites, pre/post comparison
- [Fire in Sardinia - Part 3](https://eopf-toolkit.github.io/eopf-101/06_eopf_zarr_in_action/63_sardinia_dNBR.html): Normalized Burn Ratio (NBR) and burn severity classification

For the time series section, this Project Pythia tutorial demonstrates useful visualization patterns:

- [Project Pythia: NDMI Time Series](https://projectpythia.org/landsat-ml-cookbook/notebooks/ndmi/)

#### Part 1: Pre-Fire Baseline

1. Connect to the Microsoft Planetary Computer STAC catalog and search for Sentinel-2 L2A imagery over the study area for June 3, 2025 (one week before the fire).

2. Open the retrieved item and explore its structure. What groups are available? Where are the spectral bands stored? Plot a rendered preview, as we did in the previous week's lab.

3. Create a cloud mask using the Scene Classification Layer (SCL) band. What pixel types does this mask remove, and why is this necessary?

4. Produce a **true color composite** (RGB) for the pre-fire date. Make sure to normalize the input, per the EoPF toolkit. Apply histogram equalization to improve visual contrast. Compare the un-equalized and equalized images. What differences do you observe?

5. Produce a **false color composite** using SWIR (B12), NIR (B8A), and Blue (B02). How does vegetation appear in this band combination?

#### Part 2: Post-Fire Comparison

6. Search for and load Sentinel-2 imagery from June 21, 2025 (after the fire).

7. Produce true color and false color composites for the post-fire date using the same methods as Part 1.

8. Compare your pre-fire and post-fire false color composites. Where is the burn scar visible? How does the spectral signature of burned area differ from healthy vegetation in this band combination?

#### Part 3: Burn Severity Assessment

9. Calculate the Normalized Burn Ratio (NBR) for your pre-fire image using the formula:

$$NBR = \frac{NIR - SWIR}{NIR + SWIR}$$

What bands does this use (specifically, which Sentinel-2 band numbers)? Why are NIR and SWIR sensitive to fire damage?

10. Calculate NBR for your post-fire image.

11. Calculate the differenced NBR (dNBR) by subtracting post-fire NBR from pre-fire NBR. Multiply by 1000 to match standard severity thresholds.

12. Using the EFFIS burn severity classification below, plot the severity map for the burn scar area. Which areas experienced the highest severity?

| Class | dNBR range (×1000) |
|-------|-------------------|
| Unburned / Regrowth | < 100 |
| Low severity | 100–270 |
| Moderate-low severity | 270–440 |
| Moderate-high severity | 440–660 |
| High severity | ≥ 660 |

#### Part 4: Time Series Analysis

Two snapshots tell you about change, but a time series tells you about recovery. In this section, you'll extend your analysis to track NBR over the period before and after the fire.

16. Query the EOPF catalog for all available Sentinel-2 L2A scenes over your AOI from May 2025 through July 2025. Calculate NBR for each scene. (You'll want to write a helper function or loop to do this efficiently. Make sure to apply cloud masking to each scene.)

17. Create a faceted plot (small multiples) showing NBR maps for each month, similar to the NDMI example in Project Pythia. This gives you a visual sense of how the landscape changes through time.
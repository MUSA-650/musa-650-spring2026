import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Week 6: Multi-Class Land Cover Classification and Model Generalization

        This lab extends last week's binary forest classification to a **four-class problem**: forest, agriculture, urban, and water. We'll explore training data design, feature engineering with spectral indices, spatial cross-validation, and model transfer to a new geography (Damascus, Syria).

        **Objectives:**
        - Design training data for multi-class classification
        - Engineer spectral indices (NDVI, MNDWI, NDBI) as features
        - Evaluate with appropriate multi-class metrics
        - Implement spatial cross-validation with GroupKFold
        - Test model generalization to a new region
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Imports
    from datetime import datetime, timedelta

    import geopandas as gpd
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import odc.stac
    import pandas as pd
    import pystac_client
    import rioxarray  # noqa: F401
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from shapely.geometry import Polygon
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    return (
        ConfusionMatrixDisplay,
        GeoBox,
        GroupKFold,
        Polygon,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        cross_val_score,
        datetime,
        f1_score,
        gpd,
        mcolors,
        np,
        odc,
        pd,
        plt,
        pystac_client,
        rioxarray,
        timedelta,
        train_test_split,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Data Acquisition

        We reuse Vienna Sentinel-2 data from Week 5, but now load additional bands needed for spectral indices.
        """
    )
    return


@app.cell
def _(GeoBox, datetime, odc, pystac_client, timedelta):
    # Define spatial and temporal extent for Vienna
    dx = 0.0006  # ~60m resolution
    epsg = 4326

    # Vienna area bounds
    vienna_bounds = (16.32, 47.86, 16.9, 48.407)

    # Temporal extent: May 2024
    start_date = datetime(year=2024, month=5, day=1)
    end_date = start_date + timedelta(days=10)

    date_query = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    # Search for Sentinel-2 data
    stac_client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

    vienna_items = stac_client.search(
        bbox=vienna_bounds,
        collections=["sentinel-2-l2a"],
        datetime=date_query,
        limit=100,
    ).item_collection()

    print(f"Vienna: {len(vienna_items)} scenes found")

    # Create geobox and load data (including SWIR for water index)
    vienna_geobox = GeoBox.from_bbox(vienna_bounds, crs=f"epsg:{epsg}", resolution=dx)

    dc_vienna = odc.stac.load(
        vienna_items,
        bands=["scl", "red", "green", "blue", "nir", "swir16"],
        chunks={"time": 5, "x": 600, "y": 600},
        geobox=vienna_geobox,
        resampling="bilinear",
    )
    dc_vienna
    return (
        date_query,
        dc_vienna,
        dx,
        end_date,
        epsg,
        stac_client,
        start_date,
        vienna_bounds,
        vienna_geobox,
        vienna_items,
    )


@app.cell
def _(dc_vienna):
    # Cloud masking
    def is_valid_pixel(data):
        """Include vegetated, not_vegetated, water, and snow pixels (SCL classes 4-6 and 11)"""
        return ((data > 3) & (data < 7)) | (data == 11)

    valid_mask = is_valid_pixel(dc_vienna.scl)
    dc_valid = dc_vienna.where(valid_mask)
    return dc_valid, is_valid_pixel, valid_mask


@app.cell
def _(dc_valid, end_date, plt, start_date):
    # Compute median composites
    rgb_median = (
        dc_valid[["red", "green", "blue"]]
        .to_dataarray(dim="band")
        .median(dim="time")
        .astype(int)
    )

    fc_median = (
        dc_valid[["nir", "green", "blue"]]
        .to_dataarray(dim="band")
        .transpose(..., "band")
        .median(dim="time")
        .astype(int)
    )

    # Plot both
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

    rgb_median.plot.imshow(ax=axes1[0], robust=True)
    axes1[0].set_title(f"RGB Composite\n{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
    axes1[0].set_aspect("equal")

    fc_median.plot.imshow(ax=axes1[1], robust=True)
    axes1[1].set_title("False Color (NIR-G-B)")
    axes1[1].set_aspect("equal")

    plt.tight_layout()
    fig1
    return axes1, fc_median, fig1, rgb_median


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 1: Training Data Design

        We define training polygons for four classes: **forest**, **agriculture**, **urban**, and **water**.

        Key considerations:
        - **Forest**: Dense tree cover in Vienna Woods (southeast)
        - **Agriculture**: Active cropland and vineyards (north and west)
        - **Urban**: Dense city center AND suburban areas
        - **Water**: Danube River and lakes
        """
    )
    return


@app.cell
def _(Polygon, gpd):
    # Define training polygons for 4 classes
    # Class labels: 0=forest, 1=agriculture, 2=urban, 3=water

    forest_polys = {
        0: Polygon([(16.482772, 47.901753), (16.465133, 47.870124), (16.510142, 47.874382)]),
        1: Polygon([(16.594079, 47.938855), (16.581914, 47.894454), (16.620233, 47.910268)]),
        2: Polygon([(16.67984, 47.978998), (16.637263, 47.971091), (16.660376, 47.929123)]),
        3: Polygon([(16.756477, 48.000286), (16.723024, 47.983256), (16.739446, 47.972916)]),
        4: Polygon([(16.80696, 48.135923), (16.780806, 48.125583), (16.798445, 48.115243)]),
        5: Polygon([(16.684097, 48.144438), (16.664634, 48.124366), (16.690788, 48.118892)]),
    }

    agriculture_polys = {
        0: Polygon([(16.35, 48.30), (16.38, 48.30), (16.38, 48.27), (16.35, 48.27)]),
        1: Polygon([(16.40, 48.35), (16.43, 48.35), (16.43, 48.32), (16.40, 48.32)]),
        2: Polygon([(16.45, 48.38), (16.48, 48.38), (16.48, 48.35), (16.45, 48.35)]),
        3: Polygon([(16.70, 48.35), (16.73, 48.35), (16.73, 48.32), (16.70, 48.32)]),
    }

    urban_polys = {
        # Dense city center
        0: Polygon([(16.36, 48.21), (16.39, 48.21), (16.39, 48.19), (16.36, 48.19)]),
        1: Polygon([(16.40, 48.22), (16.43, 48.22), (16.43, 48.20), (16.40, 48.20)]),
        # Suburban areas
        2: Polygon([(16.28, 48.18), (16.31, 48.18), (16.31, 48.16), (16.28, 48.16)]),
        3: Polygon([(16.50, 48.25), (16.53, 48.25), (16.53, 48.23), (16.50, 48.23)]),
    }

    water_polys = {
        # Danube River
        0: Polygon([(16.40, 48.24), (16.42, 48.24), (16.42, 48.235), (16.40, 48.235)]),
        1: Polygon([(16.45, 48.245), (16.47, 48.245), (16.47, 48.24), (16.45, 48.24)]),
        2: Polygon([(16.50, 48.24), (16.52, 48.24), (16.52, 48.235), (16.50, 48.235)]),
    }

    # Create GeoDataFrames
    class_names = ["forest", "agriculture", "urban", "water"]
    class_colors = ["forestgreen", "gold", "gray", "dodgerblue"]

    all_polys = [forest_polys, agriculture_polys, urban_polys, water_polys]
    training_gdfs = []

    for _class_id, (_polys, _name) in enumerate(zip(all_polys, class_names)):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": list(_polys.values()),
                "class_id": _class_id,
                "class_name": _name,
                "polygon_id": list(_polys.keys()),
            },
            crs="EPSG:4326",
        )
        training_gdfs.append(gdf)

    training_gdf = gpd.GeoDataFrame(pd.concat(training_gdfs, ignore_index=True))
    training_gdf
    return (
        agriculture_polys,
        all_polys,
        class_colors,
        class_names,
        forest_polys,
        training_gdf,
        training_gdfs,
        urban_polys,
        water_polys,
    )


@app.cell
def _(class_colors, class_names, plt, rgb_median, training_gdf):
    # Plot training polygons
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    rgb_median.plot.imshow(ax=ax2, robust=True)

    for _class_id, (_name, _color) in enumerate(zip(class_names, class_colors)):
        _subset = training_gdf[training_gdf["class_id"] == _class_id]
        _subset.plot(ax=ax2, facecolor="none", edgecolor=_color, linewidth=2, label=_name)

    ax2.legend(loc="upper right")
    ax2.set_title("Training Polygons: 4-Class Land Cover")
    ax2.set_aspect("equal")
    plt.tight_layout()
    fig2
    return ax2, fig2


@app.cell
def _(plt, training_gdf):
    # Examine class balance
    class_counts = training_gdf.groupby("class_name").size()
    print("Number of polygons per class:")
    print(class_counts)
    print()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    class_counts.plot(kind="bar", ax=ax3, color=["forestgreen", "gold", "gray", "dodgerblue"])
    ax3.set_ylabel("Number of Polygons")
    ax3.set_title("Training Polygon Count by Class")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig3
    return ax3, class_counts, fig3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Class Balance Considerations

        **Observations:**
        - Forest has more polygons than other classes
        - Water has fewest polygons (Danube is a thin linear feature)

        **Potential problems with imbalance:**
        - Model may be biased toward majority class
        - Minority class (water) may have poor recall

        **Strategies to address:**
        - Undersample majority class
        - Oversample minority class (e.g., SMOTE)
        - Use class weights in the classifier
        - Ensure evaluation metrics account for imbalance (balanced accuracy, macro F1)

        We'll use `class_weight='balanced'` in Random Forest to handle this automatically.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 2: Feature Engineering

        We'll compute three spectral indices to help distinguish our four classes:

        1. **NDVI** (Normalized Difference Vegetation Index): Distinguishes vegetation from non-vegetation
           - Formula: (NIR - Red) / (NIR + Red)
           - High for vegetation, low for water/urban

        2. **MNDWI** (Modified Normalized Difference Water Index): Identifies water bodies
           - Formula: (Green - SWIR) / (Green + SWIR)
           - High for water, low for land

        3. **NDBI** (Normalized Difference Built-up Index): Highlights urban areas
           - Formula: (SWIR - NIR) / (SWIR + NIR)
           - High for built-up areas, low for vegetation
        """
    )
    return


@app.cell
def _(dc_valid, np):
    # Compute spectral indices
    def normalized_difference(a, b):
        """Compute (a - b) / (a + b)"""
        a_float = a.astype(float)
        b_float = b.astype(float)
        return (a_float - b_float) / (a_float + b_float)

    # Compute median bands first
    red_med = dc_valid.red.median(dim="time")
    green_med = dc_valid.green.median(dim="time")
    blue_med = dc_valid.blue.median(dim="time")
    nir_med = dc_valid.nir.median(dim="time")
    swir_med = dc_valid.swir16.median(dim="time")

    # Compute indices
    ndvi = normalized_difference(nir_med, red_med)
    mndwi = normalized_difference(green_med, swir_med)
    ndbi = normalized_difference(swir_med, nir_med)

    print(f"NDVI range: {float(np.nanmin(ndvi)):.3f} to {float(np.nanmax(ndvi)):.3f}")
    print(f"MNDWI range: {float(np.nanmin(mndwi)):.3f} to {float(np.nanmax(mndwi)):.3f}")
    print(f"NDBI range: {float(np.nanmin(ndbi)):.3f} to {float(np.nanmax(ndbi)):.3f}")
    return (
        blue_med,
        green_med,
        mndwi,
        ndbi,
        ndvi,
        nir_med,
        normalized_difference,
        red_med,
        swir_med,
    )


@app.cell
def _(mndwi, ndbi, ndvi, plt):
    # Visualize spectral indices
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

    ndvi.plot.imshow(ax=axes4[0], cmap="RdYlGn", vmin=-1, vmax=1)
    axes4[0].set_title("NDVI (Vegetation)")
    axes4[0].set_aspect("equal")

    mndwi.plot.imshow(ax=axes4[1], cmap="RdBu", vmin=-1, vmax=1)
    axes4[1].set_title("MNDWI (Water)")
    axes4[1].set_aspect("equal")

    ndbi.plot.imshow(ax=axes4[2], cmap="RdGy_r", vmin=-1, vmax=1)
    axes4[2].set_title("NDBI (Built-up)")
    axes4[2].set_aspect("equal")

    plt.tight_layout()
    fig4
    return axes4, fig4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How do these indices help distinguish classes?

        | Index | Forest | Agriculture | Urban | Water |
        |-------|--------|-------------|-------|-------|
        | **NDVI** | High (0.4-0.8) | Medium-High (0.3-0.6) | Low (0-0.2) | Negative |
        | **MNDWI** | Negative | Negative | Negative | High (>0.2) |
        | **NDBI** | Negative | Low | High (>0) | Variable |

        - **Forest vs Agriculture**: Both have high NDVI, but forest is denser (higher values)
        - **Urban vs Others**: High NDBI, low NDVI
        - **Water vs Others**: High MNDWI, negative NDVI
        """
    )
    return


@app.cell
def _(
    blue_med,
    dc_valid,
    green_med,
    mndwi,
    ndbi,
    ndvi,
    nir_med,
    red_med,
    swir_med,
    xr,
):
    # Create feature dataset with all bands and indices
    ds_features = xr.Dataset(
        {
            "red": red_med,
            "green": green_med,
            "blue": blue_med,
            "nir": nir_med,
            "swir16": swir_med,
            "ndvi": ndvi,
            "mndwi": mndwi,
            "ndbi": ndbi,
        }
    ).fillna(0)

    feature_names = ["red", "green", "blue", "nir", "swir16", "ndvi", "mndwi", "ndbi"]
    ds_features
    return ds_features, feature_names


@app.cell
def _(ds_features, feature_names, np, training_gdf, xr):
    # Extract training data from polygons
    def extract_pixels_from_polygon(ds, geometry, feature_list):
        """Extract pixel values from a polygon"""
        try:
            clipped = ds.rio.clip([geometry], all_touched=False, drop=True)
            arr = clipped.to_array(dim="band").values
            # Reshape to (n_pixels, n_features)
            n_features = len(feature_list)
            pixels = arr.reshape(n_features, -1).T
            return pixels
        except Exception:
            return np.array([]).reshape(0, len(feature_list))

    # Extract all training pixels
    X_list = []
    y_list = []
    group_list = []  # For spatial CV - polygon IDs

    for idx, row in training_gdf.iterrows():
        pixels = extract_pixels_from_polygon(ds_features, row.geometry, feature_names)
        if len(pixels) > 0:
            # Remove NaN rows
            _valid_mask = ~np.isnan(pixels).any(axis=1)
            pixels = pixels[_valid_mask]
            if len(pixels) > 0:
                X_list.append(pixels)
                y_list.append(np.full(len(pixels), row.class_id))
                # Unique group ID for each polygon (for spatial CV)
                group_list.append(np.full(len(pixels), idx))

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.concatenate(group_list)

    print(f"Total training pixels: {len(X_all)}")
    print(f"Pixels per class:")
    for _i, _name in enumerate(["forest", "agriculture", "urban", "water"]):
        print(f"  {_name}: {(y_all == _i).sum()}")
    return (
        X_all,
        X_list,
        extract_pixels_from_polygon,
        group_list,
        groups_all,
        y_all,
        y_list,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 3: Multi-Class Classification and Evaluation
        """
    )
    return


@app.cell
def _(
    RandomForestClassifier,
    StandardScaler,
    X_all,
    groups_all,
    train_test_split,
    y_all,
):
    # Split data 70/30, stratified by class
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X_all, y_all, groups_all, test_size=0.3, random_state=42, stratify=y_all
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with balanced class weights
    rf_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return (
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        groups_test,
        groups_train,
        rf_clf,
        scaler,
        y_test,
        y_train,
    )


@app.cell
def _(X_test_scaled, rf_clf, y_test):
    # Predictions
    y_pred = rf_clf.predict(X_test_scaled)
    return (y_pred,)


@app.cell
def _(ds_features, feature_names, np, rf_clf, scaler):
    # Predict on full image
    image_data = (
        ds_features[feature_names]
        .to_array(dim="band")
        .transpose("latitude", "longitude", "band")
    )

    img_shape = (ds_features.sizes["latitude"], ds_features.sizes["longitude"])
    X_image = image_data.values.reshape(-1, len(feature_names))
    X_image_scaled = scaler.transform(X_image)

    y_pred_img = rf_clf.predict(X_image_scaled).reshape(img_shape)
    return X_image, X_image_scaled, image_data, img_shape, y_pred_img


@app.cell
def _(class_colors, class_names, mcolors, plt, y_pred_img):
    # Plot predicted land cover map
    cmap_lc = mcolors.ListedColormap(class_colors)

    fig5, ax5 = plt.subplots(figsize=(12, 10))
    im = ax5.imshow(y_pred_img, cmap=cmap_lc, vmin=0, vmax=3)
    ax5.set_title("Predicted Land Cover Map (Vienna)")
    ax5.axis("off")

    # Add colorbar with class labels
    cbar = plt.colorbar(im, ax=ax5, ticks=[0.375, 1.125, 1.875, 2.625], shrink=0.7)
    cbar.set_ticklabels(class_names)

    plt.tight_layout()
    fig5
    return ax5, cbar, cmap_lc, fig5, im


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Visual Inspection

        **Look for:**
        - Do forest areas (Vienna Woods in SE) appear green?
        - Is the Danube River classified as water (blue)?
        - Are dense urban areas correctly identified (gray)?
        - Are agricultural areas in the north classified correctly (gold)?

        **Common errors:**
        - Agricultural and forest confusion (both vegetation)
        - Urban parks misclassified as forest
        - River edges mixed with urban
        """
    )
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    classification_report,
    confusion_matrix,
    plt,
    y_pred,
    y_test,
):
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig6, ax6 = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax6, cmap="Blues", colorbar=False
    )
    ax6.set_title("Confusion Matrix")
    plt.tight_layout()
    fig6
    return ax6, cm, fig6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Which classes are most commonly confused?

        **Typical patterns:**
        - **Forest ↔ Agriculture**: Both have high vegetation signal; spectral overlap
        - **Urban ↔ Agriculture**: Bare fields can look like urban spectrally
        - **Water edges**: Often confused with dark urban surfaces

        The confusion matrix reveals where class boundaries are weakest.
        """
    )
    return


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    class_names,
    f1_score,
    y_pred,
    y_test,
):
    # Compare evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Evaluation Metrics Comparison")
    print("=" * 40)
    print(f"Overall Accuracy:    {acc:.4f}")
    print(f"Balanced Accuracy:   {balanced_acc:.4f}")
    print(f"Macro-averaged F1:   {macro_f1:.4f}")
    return acc, balanced_acc, macro_f1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Why might these metrics differ?

        - **Overall Accuracy**: Proportion correct, but dominated by majority class
        - **Balanced Accuracy**: Average recall across classes — treats all classes equally
        - **Macro F1**: Average F1 across classes — balances precision and recall

        **When to use each:**
        - **Accuracy**: When classes are balanced and all equally important
        - **Balanced Accuracy**: When class sizes differ but you care equally about all
        - **Macro F1**: When you want to penalize poor performance on any class

        **For a stakeholder**: Report balanced accuracy or macro F1, since they don't hide poor performance on minority classes (water).
        """
    )
    return


@app.cell
def _(feature_names, np, plt, rf_clf):
    # Feature importance
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    ax7.bar(range(len(feature_names)), importances[indices], color="steelblue")
    ax7.set_xticks(range(len(feature_names)))
    ax7.set_xticklabels([feature_names[i] for i in indices])
    ax7.set_ylabel("Feature Importance")
    ax7.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    fig7
    return ax7, fig7, importances, indices


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Feature Importance Analysis

        **Key questions:**
        - Are the spectral indices (NDVI, MNDWI, NDBI) more important than raw bands?
        - Which features contribute most to classification?

        **Typically:**
        - NIR and SWIR are highly important (key for vegetation vs urban vs water)
        - NDVI often ranks high for distinguishing vegetation
        - MNDWI is crucial for water detection
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 4: Spatial Cross-Validation

        **Problem with random splits:**
        - Pixels from the same polygon are in both train and test sets
        - Due to **spatial autocorrelation**, neighboring pixels are very similar
        - This inflates accuracy estimates — the model may just be memorizing local patterns

        **Solution: GroupKFold**
        - Hold out entire polygons, not random pixels
        - Tests model's ability to generalize to truly unseen areas
        """
    )
    return


@app.cell
def _(
    GroupKFold,
    X_all,
    groups_all,
    np,
    rf_clf,
    scaler,
    y_all,
):
    # Spatial cross-validation with GroupKFold
    n_splits = 5
    group_kfold = GroupKFold(n_splits=n_splits)

    spatial_cv_scores = []

    for train_idx, test_idx in group_kfold.split(X_all, y_all, groups_all):
        X_train_cv = X_all[train_idx]
        X_test_cv = X_all[test_idx]
        y_train_cv = y_all[train_idx]
        y_test_cv = y_all[test_idx]

        # Scale
        scaler_cv = scaler.__class__()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler_cv.transform(X_test_cv)

        # Train and evaluate
        rf_cv = rf_clf.__class__(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf_cv.fit(X_train_cv_scaled, y_train_cv)
        score = rf_cv.score(X_test_cv_scaled, y_test_cv)
        spatial_cv_scores.append(score)

    print("Spatial Cross-Validation (GroupKFold)")
    print("=" * 40)
    print(f"Fold accuracies: {[f'{s:.4f}' for s in spatial_cv_scores]}")
    print(f"Mean accuracy: {np.mean(spatial_cv_scores):.4f} ± {np.std(spatial_cv_scores):.4f}")
    return (
        X_test_cv,
        X_test_cv_scaled,
        X_train_cv,
        X_train_cv_scaled,
        group_kfold,
        n_splits,
        rf_cv,
        scaler_cv,
        score,
        spatial_cv_scores,
        test_idx,
        train_idx,
        y_test_cv,
        y_train_cv,
    )


@app.cell
def _(acc, np, plt, spatial_cv_scores):
    # Compare random vs spatial CV
    fig8, ax8 = plt.subplots(figsize=(8, 5))

    metrics = ["Random Split\n(70/30)", "Spatial CV\n(GroupKFold)"]
    values = [acc, np.mean(spatial_cv_scores)]
    errors = [0, np.std(spatial_cv_scores)]

    bars = ax8.bar(metrics, values, yerr=errors, capsize=5, color=["steelblue", "coral"])
    ax8.set_ylabel("Accuracy")
    ax8.set_title("Random Split vs Spatial Cross-Validation")
    ax8.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center")

    plt.tight_layout()
    fig8
    return ax8, bars, errors, fig8, metrics, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What does the performance drop tell us?

        **Interpretation:**
        - If spatial CV accuracy is much lower than random split accuracy, the model is **overfitting to local patterns**
        - The drop indicates how much accuracy would decrease when applying the model to truly new areas
        - This is a more realistic estimate of real-world performance

        **The larger the gap, the more the model relies on memorizing training polygon characteristics rather than learning generalizable patterns.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Part 5: Model Transfer to Damascus, Syria

        Now we test whether our Vienna-trained model can generalize to a completely different geography.
        """
    )
    return


@app.cell
def _(GeoBox, datetime, dx, epsg, odc, stac_client, timedelta):
    # Load Sentinel-2 data for Damascus
    damascus_bounds = (36.2, 33.4, 36.5, 33.6)

    # Same time period as Vienna
    damascus_start = datetime(year=2024, month=5, day=1)
    damascus_end = damascus_start + timedelta(days=10)
    damascus_date_query = f"{damascus_start.strftime('%Y-%m-%d')}/{damascus_end.strftime('%Y-%m-%d')}"

    damascus_items = stac_client.search(
        bbox=damascus_bounds,
        collections=["sentinel-2-l2a"],
        datetime=damascus_date_query,
        limit=100,
    ).item_collection()

    print(f"Damascus: {len(damascus_items)} scenes found")

    damascus_geobox = GeoBox.from_bbox(damascus_bounds, crs=f"epsg:{epsg}", resolution=dx)

    dc_damascus = odc.stac.load(
        damascus_items,
        bands=["scl", "red", "green", "blue", "nir", "swir16"],
        chunks={"time": 5, "x": 600, "y": 600},
        geobox=damascus_geobox,
        resampling="bilinear",
    )
    dc_damascus
    return (
        damascus_bounds,
        damascus_date_query,
        damascus_end,
        damascus_geobox,
        damascus_items,
        damascus_start,
        dc_damascus,
    )


@app.cell
def _(dc_damascus, is_valid_pixel):
    # Cloud masking for Damascus
    valid_mask_dam = is_valid_pixel(dc_damascus.scl)
    dc_damascus_valid = dc_damascus.where(valid_mask_dam)
    return dc_damascus_valid, valid_mask_dam


@app.cell
def _(dc_damascus_valid, normalized_difference, xr):
    # Compute features for Damascus
    red_dam = dc_damascus_valid.red.median(dim="time")
    green_dam = dc_damascus_valid.green.median(dim="time")
    blue_dam = dc_damascus_valid.blue.median(dim="time")
    nir_dam = dc_damascus_valid.nir.median(dim="time")
    swir_dam = dc_damascus_valid.swir16.median(dim="time")

    ndvi_dam = normalized_difference(nir_dam, red_dam)
    mndwi_dam = normalized_difference(green_dam, swir_dam)
    ndbi_dam = normalized_difference(swir_dam, nir_dam)

    ds_features_dam = xr.Dataset(
        {
            "red": red_dam,
            "green": green_dam,
            "blue": blue_dam,
            "nir": nir_dam,
            "swir16": swir_dam,
            "ndvi": ndvi_dam,
            "mndwi": mndwi_dam,
            "ndbi": ndbi_dam,
        }
    ).fillna(0)
    return (
        blue_dam,
        ds_features_dam,
        green_dam,
        mndwi_dam,
        ndbi_dam,
        ndvi_dam,
        nir_dam,
        red_dam,
        swir_dam,
    )


@app.cell
def _(dc_damascus_valid, plt):
    # Damascus RGB composite
    rgb_damascus = (
        dc_damascus_valid[["red", "green", "blue"]]
        .to_dataarray(dim="band")
        .median(dim="time")
        .astype(int)
    )

    fig9, ax9 = plt.subplots(figsize=(10, 8))
    rgb_damascus.plot.imshow(ax=ax9, robust=True)
    ax9.set_title("Damascus, Syria - RGB Composite")
    ax9.set_aspect("equal")
    plt.tight_layout()
    fig9
    return ax9, fig9, rgb_damascus


@app.cell
def _(ds_features_dam, feature_names, np, rf_clf, scaler):
    # Apply Vienna model to Damascus
    image_data_dam = (
        ds_features_dam[feature_names]
        .to_array(dim="band")
        .transpose("latitude", "longitude", "band")
    )

    img_shape_dam = (ds_features_dam.sizes["latitude"], ds_features_dam.sizes["longitude"])
    X_image_dam = image_data_dam.values.reshape(-1, len(feature_names))
    X_image_dam_scaled = scaler.transform(X_image_dam)

    y_pred_dam = rf_clf.predict(X_image_dam_scaled).reshape(img_shape_dam)
    return X_image_dam, X_image_dam_scaled, image_data_dam, img_shape_dam, y_pred_dam


@app.cell
def _(
    class_colors,
    class_names,
    mcolors,
    plt,
    rgb_damascus,
    y_pred_dam,
):
    # Compare RGB and predictions for Damascus
    fig10, axes10 = plt.subplots(1, 2, figsize=(16, 7))

    rgb_damascus.plot.imshow(ax=axes10[0], robust=True)
    axes10[0].set_title("Damascus - RGB")
    axes10[0].set_aspect("equal")

    cmap_lc_dam = mcolors.ListedColormap(class_colors)
    im_dam = axes10[1].imshow(y_pred_dam, cmap=cmap_lc_dam, vmin=0, vmax=3)
    axes10[1].set_title("Damascus - Predicted Land Cover (Vienna Model)")
    axes10[1].axis("off")

    cbar_dam = plt.colorbar(im_dam, ax=axes10[1], ticks=[0.375, 1.125, 1.875, 2.625], shrink=0.7)
    cbar_dam.set_ticklabels(class_names)

    plt.tight_layout()
    fig10
    return axes10, cbar_dam, cmap_lc_dam, fig10, im_dam


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Model Transfer Analysis

        **Compare the RGB image to the predictions:**

        **Where does the model succeed?**
        - Water bodies (if present) may be correctly identified
        - Dense urban areas might be classified correctly

        **Where does the model fail?**
        - **Different vegetation types**: Syrian agriculture (orchards, irrigated fields) differs from Central European crops
        - **Different urban form**: Building materials, density, and layout differ
        - **Climate differences**: More arid → different spectral signatures for "bare" vs "vegetated"
        - **No forests**: Damascus region has very different forest types (if any)

        **Why does transfer fail?**
        - The model learned Vienna-specific spectral patterns
        - Same NDVI value might mean "healthy crop" in Vienna but "irrigated orchard" in Damascus
        - Urban materials reflect differently

        **To make this work in Damascus:**
        1. Collect new training data from Damascus
        2. Fine-tune the model on local examples
        3. Use domain adaptation techniques
        4. Consider climate/biome-specific models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary

        In this lab we:

        1. **Designed multi-class training data** for forest, agriculture, urban, and water
        2. **Engineered spectral indices** (NDVI, MNDWI, NDBI) as discriminating features
        3. **Trained a Random Forest classifier** with balanced class weights
        4. **Evaluated with appropriate metrics**: accuracy, balanced accuracy, and macro F1
        5. **Implemented spatial cross-validation** to get realistic performance estimates
        6. **Tested model transfer** to Damascus — revealing the challenges of geographic generalization

        **Key takeaways:**
        - Class imbalance requires careful metric selection and handling
        - Spatial autocorrelation inflates accuracy from random splits
        - Models trained in one geography often fail in another
        - Local training data is essential for reliable predictions
        """
    )
    return


if __name__ == "__main__":
    app.run()

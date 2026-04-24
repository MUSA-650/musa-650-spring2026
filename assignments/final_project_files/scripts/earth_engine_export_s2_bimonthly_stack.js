// Export a 2023 Sentinel-2 bi-monthly 4-band stack for model training.
// Official dataset IDs:
//   COPERNICUS/S2_SR_HARMONIZED
//   COPERNICUS/S2_CLOUD_PROBABILITY
//
// Before running:
// 1. Upload your AOI shapefile to Earth Engine Assets, or draw geometry manually.
// 2. Replace the asset path below.

var aoi = ee.FeatureCollection('users/your_username/your_aoi_asset');
var region = aoi.geometry();

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

var periods = [
  {name: 'jan_feb', start: '2023-01-01', end: '2023-03-01'},
  {name: 'mar_apr', start: '2023-03-01', end: '2023-05-01'},
  {name: 'may_jun', start: '2023-05-01', end: '2023-07-01'},
  {name: 'jul_aug', start: '2023-07-01', end: '2023-09-01'},
  {name: 'sep_oct', start: '2023-09-01', end: '2023-11-01'},
  {name: 'nov_dec', start: '2023-11-01', end: '2024-01-01'}
];

function maskClouds(img) {
  var cloudProb = ee.Image(
    clouds.filter(ee.Filter.eq('system:index', img.get('system:index'))).first()
  ).select('probability');
  var mask = cloudProb.lt(40);
  return img.updateMask(mask).clip(region);
}

function makePeriodComposite(period) {
  period = ee.Dictionary(period);
  var filtered = s2
    .filterBounds(region)
    .filterDate(period.getString('start'), period.getString('end'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskClouds)
    .select(['B2', 'B3', 'B4', 'B8']);

  var composite = filtered.median();
  return composite.rename([
    period.getString('name').cat('_B2'),
    period.getString('name').cat('_B3'),
    period.getString('name').cat('_B4'),
    period.getString('name').cat('_B8')
  ]);
}

var stack = ee.ImageCollection(periods.map(makePeriodComposite)).toBands();
var bandNames = stack.bandNames().map(function(name) {
  return ee.String(name).split('_').slice(1).join('_');
});
stack = stack.rename(bandNames).toUint16();

Map.centerObject(region, 8);
Map.addLayer(stack.select(['jan_feb_B4', 'jan_feb_B3', 'jan_feb_B2']), {min: 0, max: 3000}, 'S2 RGB');

Export.image.toDrive({
  image: stack,
  description: 'thailand_aoi_s2_2023_bimonthly_24ch',
  folder: 'earthengine',
  fileNamePrefix: 'thailand_aoi_s2_2023_bimonthly_24ch',
  region: region,
  scale: 10,
  // This AOI lies around 101E-102.6E, so UTM Zone 47N is the most practical grid.
  crs: 'EPSG:32647',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

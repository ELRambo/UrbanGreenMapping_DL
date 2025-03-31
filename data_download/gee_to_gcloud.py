# -*- coding: utf-8 -*-
'''
Created on Thu Dec 12 13:52:43 2024

@author: 10449
'''

import ee
import pandas as pd
from pansharpening import panSharpen

def maskS2clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def getNDVI(image):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    waterMask = ndwi.lt(0)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi.updateMask(waterMask)

def splitRec(geometry):
    
    ''' Split a rectangle into 4 tiles '''
    
    coordinates = geometry.bounds().getInfo()['coordinates'][0]
    min_x = min([coord[0] for coord in coordinates])
    max_x = max([coord[0] for coord in coordinates])
    min_y = min([coord[1] for coord in coordinates])
    max_y = max([coord[1] for coord in coordinates])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    rect1 = ee.Geometry.Rectangle([min_x, min_y, center_x, center_y])  # Bottom-left
    rect2 = ee.Geometry.Rectangle([center_x, min_y, max_x, center_y])  # Bottom-right
    rect3 = ee.Geometry.Rectangle([min_x, center_y, center_x, max_y])  # Top-left
    rect4 = ee.Geometry.Rectangle([center_x, center_y, max_x, max_y])  # Top-right
    
    grid = ee.FeatureCollection([rect1, rect2, rect3, rect4])
    
    return grid

def createMask(chunk):
    
    ''' Generate mask for each building chunk '''
    
    chunk = ee.FeatureCollection(chunk)
    mask = chunk.reduceToImage(
        properties=['boundary_id'],
        reducer=ee.Reducer.first()
    ).unmask(0).eq(0).reproject(
        crs='EPSG:4326',
        scale=10
    ).clip(geometry)
                
    return mask

def maskBuildings(country, geometry):
    
    ''' Mask building areas '''
    
    # Remove incorrect features in the building collection
    def isGlobalExtent(feature):
        bounds = feature.geometry().bounds()
        firstCoordinate = ee.List(bounds.coordinates().get(0))
        minLon = ee.List(firstCoordinate.get(0)).get(0)
        return ee.String(minLon).equals('-Infinity')
    
    buildings = ee.FeatureCollection( \
        'projects/sat-io/open-datasets/VIDA_COMBINED/' + country) \
        .filterBounds(geometry) \
        .map(lambda feature: feature.set('isGlobal', isGlobalExtent(feature))) \
        .filter(ee.Filter.eq('isGlobal', False))
        
    totalBuildings = buildings.size()
    
    # Generate an indexed building collection
    def assignIndex(f, idx):
        return f.set('index', idx)
    
    idxBuildings = buildings.map(lambda f:
          assignIndex(f,
            ee.Number(buildings.aggregate_array('system:index')
            .indexOf(f.get('system:index')))))
    
    # Split buildings and process each chunk
    chunkSize = 1e13
    indices = ee.List.sequence(0, totalBuildings.divide(chunkSize).ceil())
    
    def process_chunk(i):
      # Calculate the start and end index for this chunk
      start = ee.Number(i).multiply(chunkSize)
      end = start.add(chunkSize)
      chunk = idxBuildings.filter(ee.Filter.gte('index', start)).filter(ee.Filter.lt('index', end))
      return createMask(chunk)
  
    return ee.ImageCollection(indices.map(process_chunk)).min().unmask(1)


if __name__ == '__main__':
    
    ''' Execute the export process '''
    
    # Authenticate and initialize Earth Engine
    ee.Authenticate()
    ee.Initialize()
    print('ee initialised')
    
    df = pd.read_csv('D:/Msc/Thesis/Data/GEEDownload/newThresh.csv')
    zone = 'b'
    df = df[(df['zone'] == zone) & (df['exception'] == 2)]
    scale = 10
    
    # Loop through each city
    for index, row in df.iterrows():
        description = row['city']
        country = row['country']
        geometry = eval(row['geometry'])
        thresh = row['threshold']
                
        dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2019-01-01', '2020-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 8)) \
            .map(maskS2clouds) \
            .map(lambda image: image.clip(geometry)) \
            .median()
            
        sharpened = panSharpen({
            'image': dataset,
            'geometry': geometry,
            'best_effort': True
        })
            
        spectralBands = sharpened.select(['B4', 'B3', 'B2', 'B8', 'B12'])
        
        ndvi = getNDVI(dataset).rename('NDVI')
        ndviLabel = ndvi.gt(thresh).rename('NDVI_Label')
        
        # Mask buildings in each tile
        grid = splitRec(geometry)
        tiles = 4
        for i in range(tiles):
            feature = ee.Feature(grid.toList(1, i).get(0))
            tile_geometry = feature.geometry().bounds()
            buildingMask = maskBuildings(country, tile_geometry).clip(geometry)
            ndviLabel = ndviLabel.updateMask(buildingMask)
        
        ndviLabel =   ndviLabel.unmask(0).clip(geometry)
        image = spectralBands.addBands(ndviLabel.toFloat())
            
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=zone,
            fileNamePrefix=description,
            region=geometry,
            scale=scale,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )
        print(f'{description} export starts')
        task.start()
        
    print('Check https://code.earthengine.google.com/tasks for progress.')
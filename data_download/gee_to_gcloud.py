# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:52:43 2024

@author: JingyiZhang
"""

import ee
import pandas as pd
import time
from tqdm import tqdm

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

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

def check_export_status(task):
    
    with tqdm(total=100, desc="Export Progress", ncols=100, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        while task.active():            
            current_status = task.status()['state']
            
            if current_status == 'RUNNING': 
                pbar.update(1)
            elif current_status == 'READY':
                pbar.set_postfix_str("Ready to export")
                
            time.sleep(3)  # Check every x seconds

        final_status = task.status()["state"]
        pbar.set_postfix_str(f"Final Status: {final_status}")
        print(f'Final Export Status: {final_status}')
        if task.status()['state'] == 'COMPLETED':
            print('Export completed')
        else:
            print(f"Export failed due to: {task.status().get('error_message', 'Unknown error')}")

# In[]

df = pd.read_excel("D:/Msc/Thesis/Data/thresholds.xlsx")
df = df[df['zone'] == 'a']

for index, row in df.iterrows():
    geometry = eval(row['geometry'])
    thresh = row['threshold']

    dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2019-01-01', '2020-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 8)) \
        .map(maskS2clouds) \
        .map(lambda image: image.clip(geometry)) \
        .median()
        
    spectral_bands = dataset.select(['B4', 'B3', 'B2', 'B8'])
    ndvi = getNDVI(dataset).rename('NDVI')
    ndvi_label = ndvi.gt(thresh).rename('NDVI_Label')
    
    image = spectral_bands \
        .addBands(ndvi)
        
    spectral_task = ee.batch.Export.image.toDrive(
        image=image,
        folder='Tropical',
        fileNamePrefix=row['city'] + '_spectral', 
        region=geometry.bounds().getInfo()['coordinates'],
        scale=10,  # Resolution in meters
        maxPixels=1e9
    )
    # label_task = ee.batch.Export.image.toDrive(
    #     image=ndvi_label,
    #     folder='Tropical',
    #     fileNamePrefix=row['city'] + '_label', 
    #     region=geometry.bounds().getInfo()['coordinates'], 
    #     scale=10,  # Resolution in meters
    #     maxPixels=1e9
    # )
    
    print(row['city'])
    print("Spectral task starts")
    spectral_task.start()
    check_export_status(spectral_task)
    # print("Label task starts")
    # label_task.start()
    # check_export_status(label_task)
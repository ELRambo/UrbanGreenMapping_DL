# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:17:02 2025

@author: 10449
"""

import ee

def panSharpen(params):
    if 'image' not in params or not isinstance(params['image'], ee.Image):
        raise ValueError("pan_sharpen(params): You must provide a params dictionary with an 'image' key")

    image = params['image']
    geometry = params.get('geometry', image.geometry())
    crs = params.get('crs', image.select(0).projection())
    max_pixels = params.get('max_pixels', 1e13)
    best_effort = params.get('best_effort', False)
    tile_scale = params.get('tile_scale', 4)

    # Define bands
    bands_10m = ['B2', 'B3', 'B4', 'B8']
    bands_20m = ['B12']

    # Create panchromatic band
    panchromatic = image.select(bands_10m).reduce(ee.Reducer.mean())

    # Prepare 20m bands
    image_20m = image.select(bands_20m)

    # Calculate statistics for 20m bands
    stats_20m = image_20m.reduceRegion(
        reducer=ee.Reducer.stdDev().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=geometry,
        scale=20,
        crs=crs,
        bestEffort=best_effort,
        maxPixels=max_pixels,
        tileScale=tile_scale
    ).toImage()

    mean_20m = stats_20m.select('.*_mean').regexpRename('(.*)_mean', '$1')
    std_dev_20m = stats_20m.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

    # Create high-pass filter kernel
    kernel = ee.Kernel.fixed(
        width=5,
        height=5,
        weights=[
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ],
        x=-3,
        y=-3,
        normalize=False
    )

    # Calculate high-pass filter
    high_pass_filter = panchromatic.convolve(kernel).rename('highPassFilter')

    # Get standard deviation of high-pass filter
    std_dev_hpf = high_pass_filter.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=geometry,
        scale=10,
        crs=crs,
        bestEffort=best_effort,
        maxPixels=max_pixels,
        tileScale=tile_scale
    ).get('highPassFilter')
    
    std_dev_hpf = ee.Image.constant(ee.Number(std_dev_hpf))

    def calculate_output(band_name):
        band = ee.String(band_name)
        W = ee.Image().expression(
            'stdDev20m / stdDevHPF * factor', {
                'stdDev20m': std_dev_20m.select(band),
                'stdDevHPF': std_dev_hpf,
                'factor': 0.25
            }
        )
        return ee.Image().expression(
            'resampled + (hpf * W)', {
                'resampled': image_20m.select(band),
                'hpf': high_pass_filter,
                'W': W
            }
        ).float()

    # Process all 20m bands
    output = ee.ImageCollection(
        [calculate_output(band) for band in bands_20m]
    ).toBands().regexpRename('.*_(.*)', '$1')
    
    # Calculate output statistics
    stats_output = output.reduceRegion(
        reducer=ee.Reducer.stdDev().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=geometry,
        scale=10,
        crs=crs,
        bestEffort=best_effort,
        maxPixels=max_pixels,
        tileScale=tile_scale
    ).toImage()

    mean_output = stats_output.select('.*_mean').regexpRename('(.*)_mean', '$1')
    std_dev_output = stats_output.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

    # Normalize output
    sharpened = ee.Image().expression(
        '(output - mean_out) / std_dev_out * std_dev_20m + mean_20m', {
            'output': output,
            'mean_out': mean_output,
            'std_dev_out': std_dev_output,
            'std_dev_20m': std_dev_20m,
            'mean_20m': mean_20m
        }
    ).float()

    return image.addBands(sharpened, None, True).select(image.bandNames())
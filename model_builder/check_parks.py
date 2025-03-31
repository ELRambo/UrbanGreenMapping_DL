# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 09:58:56 2025

@author: 10449
"""

import pandas as pd
park_df = pd.read_csv("D:/Msc/Thesis/Data/parks.csv")
city_df = pd.read_csv("D:/Msc/Thesis/Data/GEEDownload/newThresh.csv")

parks_count = park_df.groupby('city').size().reset_index(name='park_count')

# Merge park count with city dataframe
city_park_counts = pd.merge(city_df, parks_count, on='city', how='left')

# If there are cities with no parks, fill the missing counts with 0
city_park_counts['park_count'] = city_park_counts['park_count'].fillna(0).astype(int)

# Optionally, you can save the result to a new CSV
city_park_counts.to_csv('D:/Msc/Thesis/Data/city_with_park_counts.csv', index=False)
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:59:34 2024

@author: JingyiZhang
"""

import requests
import pandas as pd

f = open("D:/Msc/Thesis/Data/GEEDownload/api.txt", 'r')
api_key = f.readline()
f.close()

def get_bounding_box(city_name):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    # Construct the request URL
    request_url = f"{base_url}?address={city_name}&key={api_key}"
    
    # Make the request to the API
    response = requests.get(request_url)
    data = response.json()
    
    if data['status'] == 'OK':
        # Extract the bounding box (viewport) if available
        viewport = data['results'][0]['geometry']['viewport']
        if viewport:
            # Extract northeast and southwest corners
            ne = viewport['northeast']
            sw = viewport['southwest']
            
            # Format as ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
            bounding_box = [sw['lng'], sw['lat'], ne['lng'], ne['lat']]
            log = []
            log.append(city_name)
            log.append(f"var geometry = ee.Geometry.Rectangle({bounding_box});")
            log.append(f"ee.Geometry.Rectangle({bounding_box})")
            return log
        else:
            return "Bounding box not available"
    else:
        return "No results found"

# In[]

tropical_cities = [
    "Singapore, Singapore",
    "Medellin, Colombia",
    "Santo Domingo, Dominican Republic",
    "Jayapura, Indonesia",
    "Kinshasa, DRC",
    "Vientiane, Laos",
    "Belmopan, Belize",
    "Port Louis, Mauritius",
    "Miami, United States",
    "Maceio, Brazil",
    "Mangalore, India",
    "Monrovia, Liberia",
    "Cape Coast, Ghana",
    "Fortaleza, Brazil",
    "Mombasa, Kenya",
]

arid_cities = [
    "Tucson, United States",
    "Cairo, Egypt",
    "Alice Springs, Australia",
    "Dubai, United Arab Emirates",  # custom
    "San Pedro de Atacama, Chile",
    "Yinchuan, China",
    "Isfahan, Iran",
    "Madrid, Spain",
    "Denver, United States",
    "Cochabamba, Bolivia",
    "Marrakesh, Morocco",
    "Alexandria, Egypt",
    "Bulawayo, Zimbabwe",
    "Luanda, Angola",
    "Niamey, Niger",
]

temperate_cities = [
    "Istanbul, Turkey",
    "Algiers, Algeria",
    "San Francisco, United States",
    "Victoria, Canada",
    "Sydney, Australia",
    "Buenos Aires, Argentina",
    "Sendai, Japan",
    "Munich, Germany",
    "Paris, France",
    "Ljubljana, Slovenia",
    "Guangzhou, China",
    "Lusaka, Zambia",
    "Salta, Argentina",
    "Shimla, India",
    "Potosi, Bolivia",
]

continental_cities = [
    "Chicago, United States",
    "Kiev, Ukraine",
    "Minneapolis, United States",
    "Beijing, China",
    "Harbin, China",
    "Arak, Iran",
    "Bishkek, Kyrgyzstan",
    "Moscow, Russia",
    "Ottawa, Canada",
    "Warsaw, Poland",
    "Astana, Kazakhstan",
    "Stockholm, Sweden",
    "Irkutsk, Russia",
    "Calgary, Canada",
    "Sivas, Turkey",
]

polar_cities = [
    "Reykjavik, Iceland",
    "Tromso, Norway",
    "Anchorage, USA",
    "Murmansk, Russia",
    "Fairbanks, USA",
    "Yellowknife, Canada",
    "Iqaluit, Canada",
    "Qaanaaq, Greenland",
    "Tasiilaq, Greenland",
    "Norilsk, Russia",  # custom
    "Upernavik, Greenland",
    "Yakutsk, Russia",
    "Hammerfest",
    "Tiksi, Russia",
    "Churchill, Canada"
]

res = pd.DataFrame(columns=["city", "bbox_js", "bbox_py"])

zones = [tropical_cities,arid_cities,temperate_cities,continental_cities,polar_cities]
climate = ["tropical", "arid", "temperate", "continental", "polar"]
count = 0
for zone in zones:
    res.loc[len(res)] = [climate[count], "", ""]
    count += 1
    for city in zone: 
        bounding_box = get_bounding_box(city)
        res.loc[len(res)] = bounding_box
    
res.to_csv("D:/Msc/Thesis/Data/GEEDownload/cityBbox.csv")

# In[pass geometry to threshold file]

df = pd.read_csv("D:/Msc/Thesis/Data/GEEDownload/thresholds.csv")
res['city'] = res['city'].str.split(',').str[0]
city_to_bbox = dict(zip(res['city'], res['bbox_py']))
def map_bbox_to_geometry(city):
    return city_to_bbox.get(city, None)  # Default to None if no match is found

# Apply the function to the 'geometry' column in df (thresholds.csv)
df['geometry'] = df['city'].apply(map_bbox_to_geometry)

# Save the updated df dataframe to a new CSV (optional)
df.to_csv('D:/Msc/Thesis/Data/GEEDownload/updated_thresholds.csv', index=False)

# In[get bounding box of a single city]

print(get_bounding_box("Bangkok"))
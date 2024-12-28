# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:59:34 2024

@author: JingyiZhang
"""

import requests
import pandas as pd

api_key = "AIzaSyCxLoWKBcCKanELXBDQRX6fsE9THmzPGOY"

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
            return log
        else:
            return "Bounding box not available"
    else:
        return "No results found"
    
# In[]

res = pd.DataFrame(columns=["city", "bbox"])

tropical_cities = [
    "Bangkok, Thailand",
    "Singapore, Singapore",
    "Medellín, Colombia",  # Replaced Kuala Lumpur
    "Jakarta, Indonesia",
    "Hanoi, Vietnam",
    "Manila Metropolitan, Philippines",
    "Ho Chi Minh City, Vietnam",
    "Dar es Salaam, Tanzania",
    "Rio de Janeiro, Brazil",
    "Mumbai, India"
]

arid_cities = [
    "Dubai, United Arab Emirates",
    "Cairo, Egypt",
    "Riyadh, Saudi Arabia",
    "Las Vegas, USA",
    "Lima, Peru",
    "Cochabamba, Bolivia",
    "Tucson, USA",
    "Amman, Jordan",
    "Baghdad, Iraq",
    "Algiers, Algeria"
]

temperate_cities = [
    "San Francisco, USA",
    "London, United Kingdom",
    "Tokyo, Japan",
    "Paris, France",
    "Seoul, South Korea",
    "Berlin, Germany",
    "New York City, USA",
    "Sydney, Australia",
    "Toronto, Canada",
    "Buenos Aires, Argentina"
]

continental_cities = [
    "Moscow, Russia",
    "Chicago, USA",
    "Ottawa, Canada",
    "Warsaw, Poland",
    "Astana (Nur-Sultan), Kazakhstan",
    "Beijing, China",
    "Almaty, Kazakhstan",
    "Ulaanbaatar, Mongolia",
    "Kiev, Ukraine",
    "Minneapolis, USA"
]

polar_cities = [
    "Reykjavik, Iceland",
    "Tromsø, Norway",
    "Barrow (Utqiaġvik), USA",
    "Anchorage, USA",
    "Murmansk, Russia",
    "Fairbanks, USA",
    "Svalbard, Norway",
    "Yellowknife, Canada",
    "Longyearbyen, Svalbard (Norway)",
    "Iqaluit, Canada"
]

zones = [tropical_cities,arid_cities,temperate_cities,continental_cities,polar_cities]
climate = ["tropical", "arid", "temperate", "continental", "polar"]
count = 0
for zone in zones:
    res.loc[len(res)] = [climate[count], ""]
    count += 1
    for city in zone: 
        bounding_box = get_bounding_box(city)
        res.loc[len(res)] = bounding_box
    
res.to_csv("D:/Msc/Thesis/Data/cityBbox.csv")

# In[]

print(get_bounding_box("Lagos"))
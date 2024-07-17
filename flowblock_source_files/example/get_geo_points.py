"""
name: "Get Geo Points"
requirements:
inputs:
    hh_metadata_path:
        type: Str
outputs:
    points:
        type: Sequence
description: "Gets the geographic points from metadata"
"""

import xml.etree.ElementTree as ET

def main(hh_metadata_path):
    print(f'Reading coords from {hh_metadata_path}')
    tree = ET.parse(hh_metadata_path)
    root = tree.getroot()
    points_path = './geolocationGrid/geolocationGridPointList/geolocationGridPoint'
    points_list = []
    for point in root.findall(points_path):
        points_list.append(
            {
                'lat': float(point.find('latitude').text),
                'lon': float(point.find('longitude').text),
                'y':   int(point.find('line').text),
                'x':   int(point.find('pixel').text),
            }
        )
    return points_list

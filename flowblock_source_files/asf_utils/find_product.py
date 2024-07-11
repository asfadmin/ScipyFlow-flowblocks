"""
name: "Find ASF Product"
requirements:
    - asf_search==7.0.8
inputs:
    scene_name:
        type: Str
        user_input: True
outputs:
    grd_product:
        type: !CustomClass asf_search.ASFProduct.ASFProduct
description: "Finds an ASF product from a scene name"
"""

import asf_search
import json

def main(scene_name):
    print(f'Searching for scene {scene_name}')
    results = asf_search.granule_search([scene_name])

    # Cherry pick @ download the GRD product
    grd_product = [ r for r in results if r.properties['processingLevel'].startswith('GRD') ][0]
    print(f'Found {scene_name} @ {grd_product.properties["url"]}')
    print(json.dumps(grd_product.properties, indent=2))

    # hack the url to overcome CORS
    #aosfhaspojgsopjf
    grd_product.properties["url"] = f'http://localhost:8080/download/{scene_name}.zip' #f'https://local.asf.alaska.edu/download/{scene_name}.zip'

    return grd_product

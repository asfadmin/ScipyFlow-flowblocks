"""
name: "Find ASF Product"
requirements:
    - asf_search
inputs:
    scene_name:
        type: Str
outputs:
    grd_product:
        type: !CustomClass asf_search.ASFProduct.ASFProduct
description: "Finds an ASF product from a scene name"
"""

import logging
import asf_search
import json

def main(scene_name):
    logging.info(f'Searching for scene {scene_name}')
    results = asf_search.granule_search([scene_name])

    # Cherry pick @ download the GRD product
    grd_product = [ r for r in results if r.properties['processingLevel'].startswith('GRD') ][0]
    logging.info(f'Found {scene_name} @ {grd_product.properties["url"]}')
    logging.info(json.dumps(grd_product.properties, indent=2))

    grd_product.properties["url"] = f'http://localhost:443/download/{scene_name}.zip' #f'https://local.asf.alaska.edu/download/{scene_name}.zip'

    return grd_product

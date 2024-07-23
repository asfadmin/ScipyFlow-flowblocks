"""
name: "Create Geotiff MD"
requirements:
inputs:
    points_list:
        type: Sequence
outputs:
    tiff_info:
        type: Map
description: "Creates geotiff metadata"
"""

def main (points_list):
    ### Create a geo-referenced image
    print('Creating Geotiff Metadata')

    # Geotiff Tag Keys
    gtk = {
        'GeoKeyDirectoryTag': 34735,
        'ModelTiepointTag': 33922,
        'GeoDoubleParamsTag': 34736,
        'GeoAsciiParamsTag': 34737,
    }

    # Add Tags
    tiff_info = {}

    ### Use 1/10th of points as GCP (Method 2)
    gcp_points = []

    print('Applying GCPs')

    #   GCP's are a tuple of
    #   ( X1, Y1, Z1, Lon1, Lat1, Z_cor1, X2, Y2, Z2, Lon2, Lat2, Z_cor2 ... )

    # Grab the coords of every 10th point from metadata
    for x in range(0,int(len(points_list)/10)):
       p = points_list[x*10]
       gcp_points += [float(int(p['x']/10)), float(int(p['y']/10)), 0.0, p['lon'], p['lat'], 0.0 ]

    # If we didn't get the last point, add it:
    if (len(points_list)/10)*10 < len(points_list):
       p = points_list[-1]
       gcp_points += [float(int(p['x']/10)), float(int(p['y']/10)), 0.0, p['lon'], p['lat'], 0.0 ]

    # Apply list of GCP's
    tiff_info[gtk['ModelTiepointTag']] = tuple (gcp_points)

    # Other GeoTIFF Headers
    tiff_info[gtk['GeoKeyDirectoryTag']] = (
	1, 1, 0, 7, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2049, 34737, 7, 0,
        2054, 0, 1, 9102, 2057, 34736, 1, 1, 2059, 34736, 1, 0,
    )
    tiff_info[gtk['GeoDoubleParamsTag']] = (298.257223563, 6378137.0,)
    tiff_info[gtk['GeoAsciiParamsTag']] = ('WGS 84|',)

    return tiff_info

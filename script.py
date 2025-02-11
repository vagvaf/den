import urllib.request
import geopandas
from geopandas.tools import sjoin
import momepy


#C:\Users\vagva\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyogrio\gdal_data

def get_buildings_and_streets(country:str, city_boundary:str, crs:int, download=True):
    """this function downloads the .pbf data from geofabrik and extracts the buildings

    TO DO:
        official city boundaries for sweden will be uploaded to osm soon
    """
    savetofile = f"{country}-latest.osm.pbf"
    dllink = f"https://download.geofabrik.de/europe/{country}-latest.osm.pbf"

    if download:
        urllib.request.urlretrieve(dllink, savetofile)
    
    #get the buildings
    df = geopandas.read_file(savetofile, engine="pyogrio", layer = 'multipolygons')
    df = df.to_crs(crs)
    df_filtered = df[df['building'].notnull()]
    df_filtered = df_filtered.to_crs(crs)
    
    ###calculate the Ground Space (area) of each building
    df_filtered['GS']=df_filtered.area
    
    city = geopandas.read_file(f"{city_boundary}.shp")
    city = city.to_crs(crs)

    ####keep only the buildings in our study area
    city_buildings = sjoin(df_filtered,city,how='inner')
  
    city_buildings['hght'] = city_buildings['height']
    city_buildings['hght'] = city_buildings['hght'].apply(height_level_conv)
    city_buildings['hght'] = city_buildings['hght'].astype('float')

    city_buildings['lvl'] = city_buildings['building_levels']
    city_buildings['lvl'] = city_buildings['lvl'].apply(height_level_conv)
    city_buildings['lvl'] = city_buildings['lvl'].astype('float')


    #get streets
    df = geopandas.read_file("sweden-latest.osm.pbf",engine="pyogrio",layer = 'lines')
    df = df.to_crs(3006)
    filtered_df=df[df['highway'].notnull()]
    filtered_df = filtered_df.to_crs(3006)

    city_streets = sjoin(filtered_df, city, how='inner')

    return [city_buildings, city_streets]

def height_level_conv(height):
    try:
        if ',' in height:
            float(height.split(",")[1].strip())
            return height.split(",")[1].strip()
        elif 'm' in height:
            float(height.split(" ")[0])
            return height.split(" ")[0]
        else:
            float(height)
            return(height)
    except (ValueError, TypeError):
        return None
    

def calculate_building_floorspace(buildings):
    """calculates FS (Floor Space)"""
    buildings['FS'] = buildings['GS']*buildings['lvl']

    return buildings
    
    pass

def compute_morphometric_indicators(buildings):
    """taken from https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%201%20-%20DATA%20ACQUISITION%20-%20FILTERS%20-%20MORPHOMETRY.ipynb"""


    # Calculating perimeter
    buildings.loc[:, 'P'] = buildings.geometry.length

    # Calculating elongation
    buildings.loc[:, 'E'] = momepy.Elongation(buildings).series

    # Convexity
    buildings.loc[:, 'C'] = momepy.Convexity(buildings).series

    # Product [1-E].C.S
    buildings.loc[:, 'ECA'] = (1 - buildings['E']) * buildings['GS'] * buildings['C']

    # [1-E].S
    buildings.loc[:, 'EA'] = (1 - buildings['E']) * buildings['GS']

    return buildings

    
a=get_buildings_and_streets("sweden", "gbg", 3006, download=False)
b=compute_morphometric_indicators(a[0])

b.to_file("bgbg_morph.shp")


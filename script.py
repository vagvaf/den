import urllib.request
import geopandas
from geopandas.tools import sjoin

def get_buildings_and_streets(country:str, city_boundary:str, crs:int):
    """this function downloads the .pbf data from geofabrik and extracts the buildings

    TO DO:
        official city boundaries for sweden will be uploaded to osm soon
    """

    
    dllink = f"https://download.geofabrik.de/europe/{country}-latest.osm.pbf"
    savetofile = f"{country}-latest.osm.pbf"
    #urllib.request.urlretrieve(dllink, savetofile)

    #get the buildings
    df = geopandas.read_file(savetofile, engine="pyogrio", layer = 'multipolygons')
    df = df.to_crs(crs)
    df_filtered = df[df['building'].notnull()]
    df_filtered = df_filtered.to_crs(crs)
    
    ###calculate the area of each building
    df_filtered['area']=df_filtered.area
    
    city = geopandas.read_file(f"{city_boundary}.shp")
    city = city.to_crs(crs)

    ####keep only the buildings in our study area
    city_buildings = sjoin(df_filtered,city,how='inner')




    #get streets
    df = geopandas.read_file("sweden-latest.osm.pbf",engine="pyogrio",layer = 'lines')
    df = df.to_crs(3006)
    filtered_df=df[df['highway'].notnull()]
    filtered_df = filtered_df.to_crs(3006)

    city_streets = sjoin(filtered_df, city, how='inner')

    return [city_buildings, city_streets]


def get_building_height():
    """"""
    pass

def get_streets(country:str, city_boundary:str, crs:int):
    pass
    
get_buildings_and_streets("sweden","gbg",3006)



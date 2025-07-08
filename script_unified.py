import urllib.request
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
from geopandas.tools import sjoin
import momepy
import libpysal
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import warnings
from sklearn.preprocessing import OneHotEncoder
import sys, array
import rasterio
from rasterstats import zonal_stats
from rasterio.features import shapes
from shapely.geometry import shape

import time

sys.path.append("C:/Users/vagva/OneDrive/Documents/PST_UC/pstqgis_3.3.1_2024-11-01/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii, DistanceType, OriginType

import warnings; 
warnings.simplefilter('ignore')


##################USER INPUT###########################################


input_data= {
            'country': 'sweden', #country name based on https://download.geofabrik.de/europe.html
            'study_area': 'esk.shp', #path to a polygon shapefile containing the study area
            'crs': 3006, #epsg number of the area's crs
            'min_building_area': 20, #exclude buildings with area less than this value
            'drop_btypes': False, #include or not building types
            'floor_levels':{
                    'copernicus_raster_file': r"cpnc/SE002_GÖTEBORG_UA2012_DHM_V010.tif", # path to copernicus raster file. get data from: https://land.copernicus.eu/en/products/urban-atlas/building-height-2012#download
                    'floors_levels_model_train_local':False,
                    'assumed_ceiling_heights': True, #do we provide assumptions regarding celing heights?
                    'assumed_floor_number': True, #do we provide assumptions regarding floor number?
                    'use_btypes_for_training': True, #do we want to use builging types for floor level estimation?
                    'building_floors': {
                        'allotment_house':{'keep':'yes','ceiling_height':None,'floor_numbers':None},
                        'annexe':{'keep':'yes','ceiling_height':3,'floor_numbers':None},
                        'apartments':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'bakehouse':{'keep':'yes','ceiling_height':3,'floor_numbers':None},
                        'barn':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'barracks':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'beach_hut':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'boathouse':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'bridge':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'bungalow':{'keep':'yes','ceiling_height':3.5,'floor_numbers':1},
                        'bunker':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'cabin':{'keep':'yes','ceiling_height':3.5,'floor_numbers':1},
                        'carport':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'castle':{'keep':'yes','ceiling_height':5,'floor_numbers':None},
                        'cathedral':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'chapel':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'church':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'civic':{'keep':'yes','ceiling_height':5,'floor_numbers':None},
                        'college':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'commercial':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'conservatory':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'construction':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'container':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'cowshed':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'detached':{'keep':'yes','ceiling_height':3.5,'floor_numbers':2},
                        'digester':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'dormitory':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'farm':{'keep':'yes','ceiling_height':3,'floor_numbers':1},
                        'farm_auxiliary':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'fire_station':{'keep':'yes','ceiling_height':None,'floor_numbers':1.5},
                        'garage':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'garages':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'gatehouse':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'ger':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'government':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'grandstand':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'greenhouse':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'guardhouse':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'hangar':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'hospital':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'hotel':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'house':{'keep':'yes','ceiling_height':3.5,'floor_numbers':1.5},
                        'houseboat':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'hut':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'industrial':{'keep':'yes','ceiling_height':6,'floor_numbers':None},
                        'kindergarten':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'kingdom_hall':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'kiosk':{'keep':'yes','ceiling_height':3,'floor_numbers':1},
                        'livestock':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'military':{'keep':'?','ceiling_height':None,'floor_numbers':None},
                        'monastery':{'keep':'yes','ceiling_height':None,'floor_numbers':1.5},
                        'mosque':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'museum':{'keep':'yes','ceiling_height':5,'floor_numbers':None},
                        'office':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'outbuilding':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'pagoda':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'parking':{'keep':'yes','ceiling_height':3,'floor_numbers':None},
                        'pavilion':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'presbytery':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'public':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'quonset_hut':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'religious':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'residential':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'retail':{'keep':'yes','ceiling_height':5,'floor_numbers':None},
                        'riding_hall':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'roof':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'ruins':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'school':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'semidetached_house':{'keep':'yes','ceiling_height':3.5,'floor_numbers':1.5},
                        'service':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'shed':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'ship':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'shrine':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'silo':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'slurry_tank':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'sports_centre':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'sports_hall':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'stable':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'stadium':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'static_caravan':{'keep':'no','ceiling_height':3,'floor_numbers':1},
                        'stilt_house':{'keep':'yes','ceiling_height':3.5,'floor_numbers':None},
                        'storage_tank':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'sty':{'keep':'no','ceiling_height':None,'floor_numbers':1},
                        'supermarket':{'keep':'yes','ceiling_height':5,'floor_numbers':None},
                        'synagogue':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'tech_cab':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'temple':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'tent':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'terrace':{'keep':'yes','ceiling_height':3.5,'floor_numbers':1.5},
                        'toilets':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'tower':{'keep':'?','ceiling_height':None,'floor_numbers':None},
                        'train_station':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'transformer_tower':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'transportation':{'keep':'yes','ceiling_height':None,'floor_numbers':1},
                        'tree_house':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'triumphal_arch':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'trullo':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'university':{'keep':'yes','ceiling_height':4.5,'floor_numbers':None},
                        'warehouse':{'keep':'yes','ceiling_height':6,'floor_numbers':1},
                        'water_tower':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'windmill':{'keep':'no','ceiling_height':None,'floor_numbers':None},
                        'yes':{'keep':'yes','ceiling_height':None,'floor_numbers':None},
                    }
                },
            'pst_params':{
                'radius_type':'walking', #choose between one: straight, walking, steps, angular or axmeter
                'radius_threshold':500, #units based on the radius_type selection
                'unlinks': '' #shapefile with unlinks
                },
            'cluster_centers': np.array([
                                        [0.11, 0.17],
                                        [0.18, 0.44],
                                        [0.34, 1.66],
                                        [0.35, 0.78],
                                        [0.19, 0.92],
                                        [0.12, 0.50],
                                        #[0.39, 3.32]
                                    ]),
            'outputfile': "script_unified_results_20250630.gpkg" #path to the output file
    }






####################FUNCTION DEFINITION#################################


#C:\Users\vagva\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyogrio\gdal_data\osmconf.ini

def get_buildings(country:str, crs:int, study_area:str = None, min_building_area=0, drop_btypes:bool=True, floors_levels_model_train_local:bool=True, download:bool=True):
    """Given the country and a shapefile as a city boundary
       this function downloads the .pbf data from geofabrik
       and extracts the buildings

    TO DO:
        official city boundaries for sweden will be uploaded to osm soon
    """

    starttime = time.time()
    print("Step A: Getting buildings.....", end='')
    
    savetofile = f"{country}-latest.osm.pbf"
    dllink = f"https://download.geofabrik.de/europe/{country}-latest.osm.pbf"

    if download:
        urllib.request.urlretrieve(dllink, savetofile)
    
    #get the buildings
    df = gpd.read_file(savetofile, engine="pyogrio", layer = 'multipolygons')
    df = df.to_crs(crs)
    df_filtered = df[df['building'].notnull()]
    df_filtered = df_filtered.to_crs(crs)

    #keep only the building types we are interested in
    if not drop_btypes:
        df_filtered = drop_building_types(df_filtered, input_data['floor_levels']['building_floors'])
    
    
    ###calculate the Ground Space (area) of each building
    df_filtered['GS']=df_filtered.area
    
    ###keep only buildings with area equal or larger than 15sqm
    df_filtered = df_filtered[df_filtered['GS'] >= min_building_area]
    


    df_filtered['hght'] = df_filtered['height']
    df_filtered['lvl'] = df_filtered['building_levels']

    # Ensure height & level columns are numeric
    df_filtered['hght'] = pd.to_numeric(df_filtered['hght'], errors='coerce')
    df_filtered['lvl'] = pd.to_numeric(df_filtered['lvl'], errors='coerce')

    #keep the buildings in our study area

    city = gpd.read_file(f"{study_area}")
    city = city.to_crs(df_filtered.crs)
    buildings = sjoin(df_filtered,city,how='inner')
    #compute the morphometric indicators
    compute_morphometric_indicators(buildings)
    
    ####If we choose to base the training of our building heights on another city, grab the buildings of that city too.
    if floors_levels_model_train_local:
        non_local_buildings = None
    else:
        non_local_buildings_area = vectorize_copernicus(input_data['floor_levels']['copernicus_raster_file'])
        non_local_buildings_area= non_local_buildings_area.to_crs(df_filtered.crs)
        non_local_buildings = sjoin(df_filtered,non_local_buildings_area,how='inner')
        compute_morphometric_indicators(non_local_buildings)

    

    endtime = time.time()
    combined_buildings = pd.concat([buildings, non_local_buildings])
    print(f"Done ({int(endtime-starttime)}sec) ")

    return combined_buildings

def drop_building_types(buildings, drop_btypes:dict):
        drop_types_list=[]
        for btype in drop_btypes.keys():
            if drop_btypes[btype]['keep']=='no':
                drop_types_list.append(btype)
        buildings=buildings.drop(buildings[buildings.building.isin(drop_types_list)].index)
        return buildings


def vectorize_copernicus(raster):
    #read the raster and get crs
    src = rasterio.open(raster)
    raster_crs = src.crs.to_authority()[1]

    data = src.read(1, masked=True)

    shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))

    gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)

    return gdf
    
    

def get_building_height_from_copernicus(buildings,raster):

    #read the raster and get crs
    src = rasterio.open(raster)
    raster_crs = src.crs.to_authority()[1]

    #read the vector and set its crs to the one of the raster
    buildings=buildings.to_crs(int(raster_crs))

    #calcualte the mean height value for each building 
    buildings["cop_mean_height"] = [x["mean"] for x in zonal_stats(buildings, raster, stats="mean")]

    return buildings
    

def calculate_building_floorspace(buildings):
    """calculates FS (Floor Space)"""

    buildings['FS'] = buildings['GS']*buildings['lvl']

    return buildings

def compute_morphometric_indicators(buildings):
    """taken from https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%201%20-%20DATA%20ACQUISITION%20-%20FILTERS%20-%20MORPHOMETRY.ipynb"""
    

    # Calculating perimeter
    buildings.loc[:, 'P'] = buildings.geometry.length

    # Calculating elongation
    buildings.loc[:, 'E'] = momepy.elongation(buildings)

    # Convexity
    buildings.loc[:, 'C'] = momepy.convexity(buildings)

    # Product [1-E].C.S
    buildings.loc[:, 'ECA'] = (1 - buildings['E']) * buildings['GS'] * buildings['C']

    # [1-E].S
    buildings.loc[:, 'EA'] = (1 - buildings['E']) * buildings['GS']

    # Shared walls
    warnings.filterwarnings("ignore", category=FutureWarning, module="momepy")
    buildings.loc[:, "SW"] = momepy.SharedWallsRatio(buildings).series

    return buildings




def calculate_floors_from_building_height(buildings, building_floors):

    #make sure we include additional building types from the dataset to our assumptions
    building_types = buildings['building'].unique()
    for btype in building_types:
        if btype  not in input_data['floor_levels']['building_floors'].keys():
            input_data['floor_levels']['building_floors'][btype] = {"keep":"yes","ceiling_height":None,"floor_numbers":None}

    
    buildings['lvl'] = buildings.apply(lambda row: round(row['height_final'] / input_data['floor_levels']['building_floors'][row['building']]['ceiling_height']) if pd.isna(row['lvl']) and not pd.isna(row['height_final']) and input_data['floor_levels']['building_floors'][row['building']]['ceiling_height']  else row['lvl'], axis=1)

    return buildings

def assign_assumed_floors(buildings, building_floors):
    
    #If we have an assumed floor number, buildinds can deviate at most +-1 floor from this number
    buildings['lvl'] = buildings.apply(lambda row: input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] + 1 if (not pd.isna(input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'])) and row['lvl'] > input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] + 1  and (not pd.isna(row['height_final']))  
                                       else  max(input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] - 1, 1) if (not pd.isna(input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'])) and row['lvl'] <= input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] - 1  and (not pd.isna(row['height_final']))
                                       else row['lvl'], axis=1)
    
    #If we don't know the building height but we know the building type assign number of floors for some building types based on assumptions
    buildings['lvl'] = buildings.apply(lambda row: input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] if pd.isna(row['lvl']) and pd.isna(row['height_final']) and input_data['floor_levels']['building_floors'][row['building']]['floor_numbers'] else row['lvl'], axis=1)

    return buildings

def predict_floor_number(buildings, floors_levels_model_train_local, study_area, use_btypes_for_training=True, Training_ratio = 0.7):
    """https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%203%20-%20FLOOR%20ESTIMATION.ipynb"""
    print(f"\t Calculating building floor space area")
        
    #calculate the building floorspace
    buildings = calculate_building_floorspace(buildings)

    ## 2. DECISION TREE CLASSIFIER TO EVALUATE THE MISSING NUMBER OF FLOORS
    print("\t Step 2: Decision tree classifier for missing floors")
    # 2.1 SUBSET DATA INTO TRAIN AND TEST DATA

    # List of columns to keep
    columns_to_keep = ['GS', 'P', 'E', 'C', 'FS', 'ECA', 'EA', 'SW', 'lvl']

    #if we want to use building types for the model training we have to create duummy variables for each type
    if use_btypes_for_training:
        columns_to_keep.append('building')
        
        # Subset the DataFrame
        building_filtered = buildings[columns_to_keep]
        
        #create dummy variables from building type
        building_filtered = pd.get_dummies(building_filtered, prefix='type', dtype=float)
    else:
        building_filtered = buildings[columns_to_keep]


    # Create two subsets: one with non-null 'FL' and one with null 'FL'
    building_non_null = building_filtered[building_filtered['lvl'].notnull()]
    building_null = building_filtered[building_filtered['lvl'].isnull()]

    # Set a random seed for reproducibility
    np.random.seed(45)

    # Create a boolean mask for selecting the data for training
    mask = np.random.rand(len(building_non_null)) < Training_ratio

    # Split the data into training and testing sets
    data_train = building_non_null[mask]
    data_test = building_non_null[~mask]

    # 2.2 CALCULATE DECISION TREE CLASSIFIER & PRINT ACCURACY
    print("\t Training decision tree classifier")
    # Initialize the Decision Tree Classifier
    np.random.seed(45)
    clf = DecisionTreeClassifier()

    # Explicitly cast FL to float64 before rounding and converting to categorical
    data_train = data_train.copy()
    data_test = data_test.copy()
    
    data_train['lvl'] = data_train['lvl'].astype(np.float64).round().astype('int32').astype('category')
    data_test['lvl'] = data_test['lvl'].astype(np.float64).round().astype('int32').astype('category')

    # Separate the target variable and features for the training set
    X_train = data_train.drop(columns=['lvl'])
    y_train = data_train['lvl']

    # Train the classifier
    clf.fit(X_train, y_train)

    # Separate the features and target variable for the test set
    X_test = data_test.drop(columns=['lvl'])
    y_test = data_test['lvl']

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\t Accuracy on test data: {accuracy:.2f}")

    # Apply the model to building_null
    print("\t Applying model to missing floor data")
    # Ensure that we are using the same features as those used during training
    X_null = building_null.drop(columns=['lvl'])

    # Make sure there are no additional columns
    X_null = X_null[X_train.columns]

    # Predict the types for building_null
    building_null = building_null.copy()
    building_null['lvl'] = clf.predict(X_null)

    # 2.3 APPLY THE TREE TO THE NULL VALUES
    print("\t Applying decision tree to the entire dataset")
    X_null = building_filtered.drop(columns=['lvl'])

    # Predict the types for building_null
    building_filtered = building_filtered.copy()  # Ensure we are working on a copy
    building_filtered.loc[:, 'lvl_pred'] = clf.predict(X_null)

    # Keep only one column from building_filtered
    type_pred = building_filtered[['lvl_pred']]

    # Concatenate along columns
    building_final = pd.concat([buildings.reset_index(drop=True), type_pred.reset_index(drop=True)], axis=1)

    # Create the 'FL_filled' column which take the non null values of FL, otherwise fill the null values with the model predictions
    building_final['lvl_filled'] = np.where(building_final['lvl'].notna(), 
                                           building_final['lvl'], 
                                           building_final['lvl_pred'])

    # Correction of FA (floor-area) using 'FL_filled'
    building_final['FS'] = building_final['lvl_filled'] * building_final['GS']

    building_final['B_Area']= building_final['GS']
    building_final['B_GFArea'] = building_final['lvl_filled'] * building_final['B_Area']

    if not floors_levels_model_train_local:
        city = gpd.read_file(f"{study_area}")
        city = city.to_crs(building_final.crs)
        building_final = building_final.drop(['index_right'], axis=1)
        building_final = sjoin(building_final,city,how='inner')


    return building_final


def floor_estimation(buildings, floors_levels_model_train_local, study_area, copernicus_data= None, building_assumptions = None,  assumed_ceiling_heights=False, assumed_floor_number=False, use_btypes_for_training=True):

        
    print(f"Step B: Running floor estimations")

    #Get the building height from Copernicus if there is available data
    if copernicus_data:
        print(f"Step B.1: Calculating building heights from copernicus data")
        buildings = get_building_height_from_copernicus(buildings, copernicus_data)
    else:
        buildings['cop_mean_height']=np.nan

    #If we have building height from OSM keep that, if we don't, use the Copernicus height, if we don't have Copernicus height, leave it empty
    buildings['height_final'] = buildings.apply(lambda row: row['hght'] if not pd.isna(row['hght'])  else row['cop_mean_height'] if not pd.isna(row['cop_mean_height']) else np.nan, axis=1)

    #If we know the building height and type, calculate the number of floors based on building height and celing height assumption for each building type
    if assumed_ceiling_heights:
        print(f"Step B.2: Calculating the number of floors based on building height and celing height assumption for each building type")
        buildings = calculate_floors_from_building_height(buildings, building_assumptions)

    #If we don't know the building height but we know the building type assign number of floors for some building types based on assumptions
    if assumed_floor_number:
        print(f"Step B.3: Assigning number of floors based on building assumptions")
        buildings = assign_assumed_floors(buildings, building_assumptions)

    buildings.to_file("eskilstuna_before_train.gpkg")

    #for the rest of the buildings where we don't know the building floors, train a model to predict it.
    print(f"Step B.4: Predicting floor number based on model training")
    buildings = predict_floor_number(buildings, floors_levels_model_train_local, study_area, use_btypes_for_training=True)

    return buildings
        
        


    
    

def get_streets(study_area:str, crs:int, download=True):
    #Flavias code

    """Given the country and a shapefile as a city boundary
       this function downloads the .pbf data from geofabrik
       and extracts the streets

    TO DO:
        official city boundaries for sweden will be uploaded to osm soon
    """
    starttime = time.time()
    print("Step C: Getting streets.....")

    # Import data
    area = gpd.read_file(study_area)
    buffer_non_motorized = area.buffer(5000)

    # Transform the non-motorized buffer to EPSG:4326 for OSMnx compatibility
    buffer_non_motorized = buffer_non_motorized.to_crs(epsg=4326)

    # Ensure the polygon is in the correct format (e.g., a single geometry)
    buffer_non_motorized = buffer_non_motorized.unary_union

    # Retrieve the non-motorized street network within the buffer
    multi_graph_non_motorized = ox.graph_from_polygon(buffer_non_motorized, network_type='all',
                                                      simplify=False, retain_all=True)
    print("\t Preprocess non motorized graph....", end="")
    simple_graph_non_motorized = preprocess_non_motorized_graph(multi_graph_non_motorized)
    print(f"Done")

    
    # Collect edges to keep
    edges_to_keep = [
        (u, v) for u, v, data in simple_graph_non_motorized.edges(data=True)
        if should_keep_non_motorized_edge(data)
    ]

    print("\t Keep non motorized edge....", end="")    
    # Create a new graph with only the edges to keep
    filtered_graph_non_motorized = simple_graph_non_motorized.edge_subgraph(edges_to_keep).copy()
    print(f"Done")

    print("\t Remove isolated islands....", end="")
    cleaned_graph_non_motorized = remove_isolated_islands(filtered_graph_non_motorized)
    print(f"Done")
    
    print("\t Graph to Geodataframe....", end="")
    gdf_non_motorized = graph_to_geodataframe(cleaned_graph_non_motorized)
    print(f"Done")
    
    endtime =  time.time()
    print(f"Done ({int(endtime-starttime)} sec)")

    return gdf_non_motorized

def preprocess_non_motorized_graph(multi_graph_non_motorized):

    # Convert to an undirected graph
    undirected_graph_non_motorized = ox.convert.to_undirected(multi_graph_non_motorized)

    # Create a simple graph and add nodes
    simple_graph_non_motorized = nx.Graph()
    simple_graph_non_motorized.add_nodes_from(undirected_graph_non_motorized.nodes(data=True))

    # Add edges, removing parallel edges with the same length
    for u, v, data in undirected_graph_non_motorized.edges(data=True):
        node_pair = tuple(sorted((u, v)))  # Sort nodes

        # Check for existing edges and remove if lengths are the same
        if simple_graph_non_motorized.has_edge(*node_pair):
            existing_length = simple_graph_non_motorized[node_pair[0]][node_pair[1]].get("length")
            if existing_length == data.get("length"):
                simple_graph_non_motorized.remove_edge(*node_pair)

        # Add the new edge
        edge_data = {key: data.get(key) for key in ['osmid', 'name', 'length', 'highway', 'maxspeed', 'service', 'access', 'junction', 'geometry'] if key in data}
        simple_graph_non_motorized.add_edge(*node_pair, **edge_data)

    return simple_graph_non_motorized

def should_keep_non_motorized_edge(edge_data):
    
    # Safely get attributes
    highway = edge_data.get('highway')
    service = edge_data.get('service')
    access = edge_data.get('access')
    junction = edge_data.get('junction')

    # Keep edges if they match the criteria
    # Keep if highway is in the list
    keep_highway = highway in ['primary', 'primary_link',
                               'secondary', 'secondary_link',
                               'tertiary', 'tertiary_link',
                               'residential', 'unclassified',
                               'pedestrian', 'living_street', 'service',
                               'path','footway','steps','track','cycleway' #my additions
                               ]

    # Keep if service is None or NOT in the exclusion list
    keep_service = service is None or service in ['alley', 'driveway']

    # Keep if access is None or NOT in the exclusion list
    keep_access = access is None or access in ['yes', 'permissive', 'destination', 'designated']

    # Keep if junction is None or NOT in the exclusion list
    keep_junction = junction is None or junction not in ['roundabout']

    # Return True if we should keep the edge
    return keep_highway and keep_service and keep_access and keep_junction

def remove_isolated_islands(graph):
   
    # Identify all connected components in the graph
    connected_components = list(nx.connected_components(graph))

    # Find the largest connected component
    largest_component = max(connected_components, key=len)

    # Create a subgraph that contains only the largest connected component
    cleaned_graph = graph.subgraph(largest_component).copy()

    return cleaned_graph


def graph_to_geodataframe(graph, crs='EPSG:4326'):
    
    # Extract the edges and their attributes
    edges_list = [
        (u, v, d.get('osmid'), d.get('name'), d.get('length'), d.get('highway'), d.get('maxspeed'), d.get('service'), d.get('access'), d.get('junction'), d.get('geometry'))
        for u, v, d in graph.edges(data=True)
    ]

    # Create a GeoDataFrame from the edges
    gdf = gpd.GeoDataFrame(
        edges_list,
        columns=['u', 'v', 'osmid', 'name', 'length', 'highway', 'maxspeed', 'service', 'access', 'junction', 'geometry'],
        geometry='geometry',
        crs=crs
    )

    gdf = gdf.to_crs(input_data['crs'])

    return gdf


def perform_Reach_Analysis(road_network, crs, radius_type, radius_threshold, unlinks=None, origin_points=None):
    starttime=time.time()
    print("Perform reach analysis....", end="")
    """ Perform a PST Reach Analysis
        Input:
            - road_network: shapefile with road network
            - crs: crs of the study area
            - radius_type: choose between one: straight, walking, steps, angular or axmeter
            - radius_threshold: the threshold based on the radius_type. straight (meters), walking (meters), steps (steps), angular (degrees) or axmeter (steps*m).
            - unlinks: shapefile with the unlinks
            - origin_points: shapefile with origin points from where the analysis should be based instead of the road network (points or polygons)
        Ouput:
            - geodataframe with 3 additional columns, each representing:
                - reached count: the total number of road segments reached
                - reached length: the total length of road segments reached
                - reached area: the total area reached 
    """
    radius_args = {radius_type:radius_threshold}
    radius = Radii(**radius_args)
    
    roads = gpd.read_file(road_network)
    roads = roads.to_crs(crs)


    road_coords = roads.get_coordinates()
    line_coords = []
    for index, row in road_coords.iterrows():
        line_coords.append(row['x'])
        line_coords.append(row['y'])

    line_coords = array.array('d', line_coords)
    line_count = int(len(line_coords) / 4)


    ### prepare the unlinks
    
    if unlinks:
        unlinks = gpd.read_file(unlinks)
        unlinks = unlinks.to_crs(crs)
        unlinks_coords = unlinks.get_coordinates()
        point_coords = []
        for index, row in unlinks_coords.iterrows():
            point_coords.append(row['x'])
            point_coords.append(row['y'])

        unlinks = array.array('d', point_coords)
        graph = pstalgo.CreateGraph(line_coords, None, unlinks, None, None)
    else:
       ### prepare the graph
        graph = pstalgo.CreateGraph(line_coords, None, None, None, None) 


    

    #If origin points were provided
    if origin_points is not None:
        
        ### Import origin poins and set crs
        buildings = gpd.read_file(origin_points)
        buildings = buildings.to_crs(crs)
    

        #validate the geometry
        buildings['geometry']=buildings['geometry'].make_valid()

        #calculate the building centroids
        buildings['cnt']=buildings['geometry'].centroid

        #get the x and y of the centroid
        buildings['x'] = buildings['cnt'].x
        buildings['y'] = buildings['cnt'].y

        b_coords = []


        for index, row in buildings.iterrows():
            b_coords.append(row['x'])
            b_coords.append(row['y'])

        
        b_coords_count = int(len(b_coords) / 2)
        b_coords = array.array('d', b_coords)

    # Create results arrays depending on whether we have origin points or not
    reached_count = array.array('I', [0]) * b_coords_count if origin_points else array.array('I', [0]) * line_count 
    reached_length = array.array('f', [0]) * b_coords_count if origin_points else array.array('I', [0]) * line_count 
    reached_area = array.array('f', [0]) * b_coords_count if origin_points else array.array('I', [0]) * line_count


    # Ask PST to do run a Reach analysis
    pstalgo.Reach(
        graph_handle = graph,
        radius = radius,
        origin_points = b_coords,
        out_reached_count = reached_count,
        out_reached_length = reached_length,
        out_reached_area = reached_area)

    # Free up the mnemory associated with the graph
    pstalgo.FreeGraph(graph)  # Or "FreeSegmentGraph"
    endtime=time.time()
    print(f"Done ({int(endtime-starttime)} sec)")
    if origin_points:
        buildings[f'RAc{radius_type[0]}{radius_threshold}']=pd.Series(reached_count)
        buildings[f'RAl{radius_type[0]}{radius_threshold}']=pd.Series(reached_length)
        buildings[f'RAa{radius_type[0]}{radius_threshold}']=pd.Series(reached_area)
        buildings = buildings.drop('cnt', axis=1)
        return buildings
    else:
        roads[f'RAc{radius_type[0]}{radius_threshold}']=pd.Series(reached_count)
        roads[f'RAl{radius_type[0]}{radius_threshold}']=pd.Series(reached_length)
        roads[f'RAa{radius_type[0]}{radius_threshold}']=pd.Series(reached_area)
        return roads

    


##################Attraction Reach analysis############################
def perform_Attraction_Reach_Analysis(road_network, crs, radius_type, radius_threshold, unlinks=None, origin_points=None, destinations=None, weight_attr=None,outputname=None):
    """ Perform a PST Reach Analysis
        Input:
            - road_network: shapefile with road network
            - crs: crs of the study area
            - radius_type: choose between one: straight, walking, steps, angular or axmeter
            - radius_threshold: the threshold based on the radius_type. Could be meters for straight or walking, number of steps for steps, degrees for angular
            - unlinks: shapefile with the unlinks
            - origin_points: shapefile with origin points from where the analysis should be based instead of the road network (points or polygons)
            - destinations: shapefile with destinations points to where the attraction will be calculated
            - weight_attr: an attribute on the above shapefile which will act as weight for the attraction calculation
        Ouput:
            - geodataframe with 1 additional columns, each representing:
                - score: the total weighted attraction reached from each origin point or road segment
    """

    starttime=time.time()
    print("Perform Attraction Reach Analysis....", end='')

    radius_args = {radius_type:radius_threshold}
    radius = Radii(**radius_args)
    
    roads = gpd.read_file(road_network)
    roads = roads.to_crs(crs)


    road_coords = roads.get_coordinates()
    line_coords = []
    for index, row in road_coords.iterrows():
        line_coords.append(row['x'])
        line_coords.append(row['y'])

    line_coords = array.array('d', line_coords)
    line_count = int(len(line_coords) / 4)


    ### prepare the unlinks
    
    if unlinks:
        unlinks = gpd.read_file(unlinks)
        unlinks = unlinks.to_crs(crs)
        unlinks_coords = unlinks.get_coordinates()
        point_coords = []
        for index, row in unlinks_coords.iterrows():
            point_coords.append(row['x'])
            point_coords.append(row['y'])

        unlinks = array.array('d', point_coords)
        

    #If origin points were provided
    if origin_points:
        
        ### Import origin poins and set crs
        origins = gpd.read_file(origin_points)
        origins = origins.to_crs(crs)
    

        #validate the geometry
        origins['geometry']=origins['geometry'].make_valid()

        #calculate the building centroids
        origins['cnt']=origins['geometry'].centroid

        #get the x and y of the centroid
        origins['x'] = origins['cnt'].x
        origins['y'] = origins['cnt'].y

        b_coords = []
        weight = []

        for index, row in origins.iterrows():
            b_coords.append(row['x'])
            b_coords.append(row['y'])
        
        b_coords_count = int(len(b_coords) / 2)
        origin_points = array.array('d', b_coords)


    ### prepare the graph
    graph = pstalgo.CreateGraph(line_coords, None, unlinks, origin_points, None)

    if destinations:
        
        ### Import origin poins and set crs
        destinations = gpd.read_file(destinations)
        destinations = destinations.to_crs(crs)
    

        #validate the geometry
        destinations['geometry']=destinations['geometry'].make_valid()

        #calculate the building centroids
        destinations['cnt']=destinations['geometry'].centroid

        #get the x and y of the centroid
        destinations['x'] = destinations['cnt'].x
        destinations['y'] = destinations['cnt'].y

        d_coords = []
        weight_scores = []

        for index, row in destinations.iterrows():
            d_coords.append(row['x'])
            d_coords.append(row['y'])
            weight_scores.append(row[weight_attr])
        
        d_coords_count = int(len(b_coords) / 2)
        destinations_points = array.array('d', d_coords)

        #values for destinations
        weight = array.array('f', weight_scores)


    graph = pstalgo.CreateGraph(line_coords, None, unlinks, origin_points, None)


    #output column in an array
    scores = array.array('f', [0]) * b_coords_count if origin_points else array.array('I', [0]) * line_count 


    pstalgo.AttractionReach(
                    graph_handle = graph,
                    origin_type = OriginType.POINTS,
                    distance_type=DistanceType.WALKING,
                    radius = radius,
                    attraction_points=destinations_points,
                    attraction_values=weight,
                    out_scores=scores)

    # Free up the memory associated with the graph
    pstalgo.FreeGraph(graph)  # Or "FreeSegmentGraph"

    endtime=time.time()
    print(f"Done ({int(endtime-starttime)} sec)")

    if origin_points:
        if outputname is None:
            origins[f'ARA{radius_type[0]}{radius_threshold}']=pd.Series(scores)
        else:
            origins[f'{outputname}']=pd.Series(scores)
        origins = origins.drop('cnt', axis=1)
        return origins
    
    else:
        roads[f'ARA{radius_type[0]}{radius_threshold}']=pd.Series(scores)
        return roads

def calculate_FSI(dataframe, catchment_area, total_floor_area):
    dataframe['FSI'] = dataframe[total_floor_area]/dataframe[catchment_area]
    return dataframe

def calculate_GSI(dataframe, catchment_area, total_ground_area):
    dataframe['GSI'] = dataframe[total_ground_area]/dataframe[catchment_area]
    return dataframe


def calculate_clusters (dataframe, cluster_centers):
    dataframe=dataframe[dataframe['RAaw500']>0]

    kmeans = KMeans(n_clusters=len(cluster_centers), init= cluster_centers, n_init=10, max_iter=300).fit(cluster_centers)
    
    kmeans_predict=kmeans.predict(dataframe[['GSI','FSI']].to_numpy())

    dataframe['clusters']=kmeans_predict+1
    return dataframe

def calculate_clusters_2 (dataframe, cluster_centers):

    kmeans = KMeans(n_clusters=len(cluster_centers), init= cluster_centers, n_init=50, max_iter=5000).fit(dataframe[['GSI','FSI']].to_numpy())


    dataframe['clusters']=kmeans.labels_+1
    print(kmeans.cluster_centers_)
    return dataframe
    


###################SCRIPT EXECUTION####################################

stepA=get_buildings(country=input_data['country'],
                    crs=input_data['crs'],
                    study_area=input_data['study_area'],
                    min_building_area=input_data['min_building_area'],
                    drop_btypes=input_data['drop_btypes'],
                    floors_levels_model_train_local=input_data['floor_levels']['floors_levels_model_train_local'],
                    download=False)

stepB=floor_estimation(stepA,
                       floors_levels_model_train_local = input_data['floor_levels']['floors_levels_model_train_local'],
                       study_area = input_data['study_area'],
                       copernicus_data=input_data['floor_levels']['copernicus_raster_file'],
                       building_assumptions = input_data['floor_levels']['building_floors'],
                      assumed_ceiling_heights = input_data['floor_levels']['assumed_ceiling_heights'],
                      assumed_floor_number = input_data['floor_levels']['assumed_floor_number'],
                       use_btypes_for_training=input_data['floor_levels']['use_btypes_for_training'] )
stepB.to_file("eskilstuna_without_gothenburg.gpkg")

#stepC=get_streets(input_data['study_area'], input_data['crs'])
#stepC.to_file("streetnonmotorized.shp")

#stepD=perform_Reach_Analysis(road_network="streetnonmotorized.shp",
#                             crs=input_data['crs'],
#                             radius_type=input_data['pst_params']['radius_type'],
#                             radius_threshold=input_data['pst_params']['radius_threshold'],
#                             origin_points="new_buildings_20250626_stepb.gpkg")
#stepD.to_file("new_buildings_20250626_stepd.gpkg")

#stepE=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
#                         crs=input_data['crs'],
#                         radius_type=input_data['pst_params']['radius_type'],
#                         radius_threshold=input_data['pst_params']['radius_threshold'],
#                         origin_points="new_buildings_20250626_stepd.gpkg",
#                         destinations="new_buildings_20250626_stepb.gpkg",
#                         weight_attr='B_Area',
#                         outputname='AGS')

#stepF=calculate_GSI(stepE,'RAaw500','AGS')
#stepF.to_file("new_buildings_20250626_stepf.gpkg")

#stepG=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
#                         crs=input_data['crs'],
#                         radius_type=input_data['pst_params']['radius_type'],
#                         radius_threshold=input_data['pst_params']['radius_threshold'],
#                         origin_points="new_buildings_20250626_stepf.gpkg",
#                         destinations="new_buildings_20250626_stepb.gpkg",
#                         weight_attr='B_GFArea',
#                         outputname='AFS')


#stepH=calculate_FSI(stepG,'RAaw500','AFS')
#stepH.to_file("new_buildings_20250626_steph.gpkg")

#stepI=calculate_clusters(stepH, input_data['cluster_centers'])

#stepI.to_file(input_data['outputfile'])






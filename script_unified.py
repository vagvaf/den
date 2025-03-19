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

import time

sys.path.append("C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii, DistanceType, OriginType

import warnings; 
warnings.simplefilter('ignore')


##################USER INPUT###########################################


input_data= {
            'country': 'sweden',
            'study_area': 'gbg.shp', #path to a polygon shapefile containing the study area
            'crs': 3006, #epsg number of the area's crs
            'copernicus_raster_file': r"C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/SE002_GÖTEBORG_UA2012_DHM_V010.tif",
            'pst_params':{
                'radius_type':'walking',
                'radius_threshold':500,
                'unlinks': '' #shapefile with unlinks
                },
            'building_floors': {
                "allotment_house":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "annexe":{"keep":"yes","ceiling_height":3,"floor_numbers":None},
                "apartments":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "barracks":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "barn":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "bungalow":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1},
                "beach_hut":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "boathouse":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "bridge":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "cabin":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1},
                "bunker":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "detached":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1.5},
                "carport":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "castle":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "dormitory":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "farm":{"keep":"yes","ceiling_height":3,"floor_numbers":1},
                "hotel":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "house":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1.5},
                "residential":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "semidetached_house":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1.5},
                "conservatory":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "construction":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "container":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "cowshed":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "static_caravan":{"keep":"yes","ceiling_height":3,"floor_numbers":1},
                "digester":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "stilt_house":{"keep":"yes","ceiling_height":3.5,"floor_numbers":None},
                "terrace":{"keep":"yes","ceiling_height":3.5,"floor_numbers":1.5},
                "farm_auxiliary":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "parking":{"keep":"yes","ceiling_height":3,"floor_numbers":None},
                "garage":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "garages":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "gatehouse":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "ger":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "bakehouse":{"keep":"yes","ceiling_height":3,"floor_numbers":None},
                "civic":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "greenhouse":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "guardhouse":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "hangar":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "college":{"keep":"yes","ceiling_height":None,"floor_numbers":1.5},
                "fire_station":{"keep":"yes","ceiling_height":None,"floor_numbers":1.5},
                "government":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "houseboat":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "hut":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "hospital":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "kindergarten":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "museum":{"keep":"yes","ceiling_height":None,"floor_numbers":2},
                "public":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "livestock":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "military":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "school":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "train_station":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "transportation":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "university":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "outbuilding":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "pagoda":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "commercial":{"keep":"yes","ceiling_height":4.5,"floor_numbers":None},
                "industrial":{"keep":"yes","ceiling_height":6,"floor_numbers":None},
                "kiosk":{"keep":"yes","ceiling_height":3,"floor_numbers":1},
                "office":{"keep":"yes","ceiling_height":4.5,"floor_numbers":None},
                "quonset_hut":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "retail":{"keep":"yes","ceiling_height":4.5,"floor_numbers":None},
                "supermarket":{"keep":"yes","ceiling_height":4.5,"floor_numbers":None},
                "warehouse":{"keep":"yes","ceiling_height":6,"floor_numbers":1},
                "cathedral":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "roof":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "ruins":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "chapel":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "church":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "service":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "shed":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "ship":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "kingdom_hall":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "silo":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "slurry_tank":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "monastery":{"keep":"yes","ceiling_height":None,"floor_numbers":None},
                "mosque":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "stable":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "presbytery":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "religious":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "shrine":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "storage_tank":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "sty":{"keep":"no","ceiling_height":None,"floor_numbers":1},
                "synagogue":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "temple":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "tech_cab":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "grandstand":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "tent":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "pavilion":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "toilets":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "tower":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "riding_hall":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "transformer_tower":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "sports_centre":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "tree_house":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "triumphal_arch":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "trullo":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "sports_hall":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "stadium":{"keep":"yes","ceiling_height":None,"floor_numbers":1},
                "water_tower":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "windmill":{"keep":"no","ceiling_height":None,"floor_numbers":None},
                "yes":{"keep":"yes","ceiling_height":None,"floor_numbers":None}
                },
            'cluster_centers': np.array([
                                        [0.11, 0.17],
                                        [0.18, 0.44],
                                        [0.34, 1.66],
                                        [0.35, 0.78],
                                        [0.19, 0.92],
                                        [0.12, 0.50],
                                        [0.39, 3.32]
                                    ]),
            'cluster_centers2': np.array([
                                        [0.11, 0.17],
                                        [0.18, 0.44],
                                        [0.34, 1.66],
                                        [0.35, 0.78],
                                        [0.19, 0.92],
                                        [0.12, 0.50]
                                    ]),
            'outputfile': "results.shp" #path to the output file
    }






####################FUNCTION DEFINITION#################################


#C:\Users\vagva\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyogrio\gdal_data

def get_buildings(country:str, crs:int, city_boundary = None, drop_btypes=None, download=True):
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
    if drop_btypes:
        drop_building_types(buildings, drop_btypes)
    
    
    ###calculate the Ground Space (area) of each building
    df_filtered['GS']=df_filtered.area
    ###keep only buildings with area equal or larger than 15sqm
    df_filtered = df_filtered[df_filtered['GS'] >= 15]
    


    df_filtered['hght'] = df_filtered['height']
    df_filtered['lvl'] = df_filtered['building_levels']

    # Ensure height & level columns are numeric
    df_filtered['hght'] = pd.to_numeric(df_filtered['hght'], errors='coerce')
    df_filtered['lvl'] = pd.to_numeric(df_filtered['lvl'], errors='coerce')
    
    ####keep only the buildings in our study area
    if city_boundary:
        city = gpd.read_file(f"{city_boundary}")
        city = city.to_crs(df_filtered.crs)
        buildings = sjoin(df_filtered,city,how='inner')
    else:
        buildings = df_filtered
        

    #compute the morphometric indicators
    compute_morphometric_indicators(buildings)

    endtime = time.time()
    print(f"Done ({int(endtime-starttime)}sec) ")

    return buildings

def drop_building_types(buildings, drop_btypes):
        drop_types_list=[]
        for btype in input_data['building_floors'].keys():
            if input_data['building_floors'][btype]['keep']=='no':
                drop_types_list.append(btype)
        df_filtered=df_filtered.drop(df[df.building.isin(drop_types_list)].index)
    

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
        if btype  not in input_data['building_floors'].keys():
            print(btype)
            input_data['building_floors'][btype] = {"keep":"yes","ceiling_height":None,"floor_numbers":None}

    
    buildings['lvl'] = buildings.apply(lambda row: round(row['hght'] / building_floors[row['building']]['ceiling_height']) if pd.isna(row['lvl']) and not pd.isna(row['hght']) and building_floors[row['building']]['ceiling_height']  else row['lvl'], axis=1)

    return buildings

def assign_assumed_floors(buildings, building_floors):

    
    buildings['lvl'] = buildings.apply(lambda row: building_floors[row['building']]['floor_numbers'] if pd.isna(row['lvl']) and pd.isna(row['hght']) and building_floors[row['building']]['floor_numbers'] else row['lvl'], axis=1)

    return buildings

def predict_floor_number(buildings, model_train = 'local', Training_ratio = 0.7):
    """https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%203%20-%20FLOOR%20ESTIMATION.ipynb"""
    print(f"\t Calculating building floor space area")
    #calculate the building floorspace
    buildings = calculate_building_floorspace(buildings)

    ## 2. DECISION TREE CLASSIFIER TO EVALUATE THE MISSING NUMBER OF FLOORS
    print("\t Step 2: Decision tree classifier for missing floors")
    # 2.1 SUBSET DATA INTO TRAIN AND TEST DATA

    # List of columns to keep
    columns_to_keep = ['building', 'GS', 'P', 'E', 'C', 'FS', 'ECA', 'EA', 'SW', 'lvl']


    # Subset the DataFrame
    building_filtered = buildings[columns_to_keep]

    #create dummy variables from building type
    building_filtered = pd.get_dummies(building_filtered, prefix='type', dtype=float) 


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


    return building_final


def floor_estimation(buildings, copernicus_data= None, building_assumptions = None,  assumed_ceiling_heights=False, assumed_floor_number=False, model_train='local'):

    print(f"Step B: Running floor estimations")

    #Get the building height from Copernicus if there is available data
    if copernicus_data:
        buildings = get_building_height_from_copernicus(buildings, copernicus_data)

    #If we have building height from copernicus keep that, if we don't, use the OSM height, if we don't have OSM height, leave it empty
    buildings['height_final'] = buildings.apply(lambda row: row['cop_mean_height'] if not pd.isna(row['cop_mean_height'])  else row['hght'] if not pd.isna(row['hght']) else np.nan, axis=1)

    #If we know the building height and type, calculate the number of floors based on building height and celing height assumption for each building type
    if assumed_ceiling_heights:
        buildings = calculate_floors_from_building_height(buildings, building_assumptions)

    #If we don't know the building height but we know the building type assign number of floors for some building types based on assumptions
    if assumed_floor_numbers:
        buildings = assign_assumed_floors(buildings, building_assumptions)

    #for the rest of the buildings where we don't know the building floors, train a model to predict it.
    buildings = predict_floor_number(buildings, model_train=model_train)

    return buildings


    
    

def get_streets(city_boundary:str, crs:int, download=True):

    """Given the country and a shapefile as a city boundary
       this function downloads the .pbf data from geofabrik
       and extracts the streets

    TO DO:
        official city boundaries for sweden will be uploaded to osm soon
    """
    starttime = time.time()
    print("Step C: Getting streets.....")

    # Import data
    area = gpd.read_file(city_boundary)
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

def calculate_FSI(dataframe, catchment_area, total_ground_area):
    dataframe['FSI'] = dataframe[total_ground_area]/dataframe[catchment_area]
    return dataframe

def calculate_GSI(dataframe, catchment_area, total_floor_area):
    dataframe['GSI'] = dataframe[total_floor_area]/dataframe[catchment_area]
    return dataframe




def calculate_clusters (dataframe, cluster_centers):
    #dataframe=dataframe[dataframe['RAaw500']>0]

    kmeans = KMeans(n_clusters=len(cluster_centers), init= cluster_centers, n_init=10, max_iter=300).fit(cluster_centers)
    print(kmeans.cluster_centers_)
    

    kmeans_predict=kmeans.predict(dataframe[['GSI','FSI']].to_numpy())


    dataframe['clusters']=kmeans_predict+1
    return dataframe
    


###################SCRIPT EXECUTION####################################

stepA=get_buildings(input_data['country'], input_data['crs'], input_data['study_area'],  download=False)
stepA.to_file("buildings_gbg_drop_btypes4.shp")
buildingsdf=gpd.read_file("buildings_gbg_drop_btypes4.shp")

stepB=floor_estimation(buildingsdf, copernicus_data=input_data['copernicus_raster_file'],building_assumptions = input_data['building_floors'], assumed_ceiling_heights = True, assumed_floor_number = True)
stepB.to_file("GBG_buildings_floors.shp")

#stepC=get_streets(input_data['study_area'], input_data['crs'])
#stepC.to_file("streetnonmotorized.shp")

#stepD=perform_Reach_Analysis(road_network="streetnonmotorized.shp",
#                             crs=input_data['crs'],
#                             radius_type=input_data['pst_params']['radius_type'],
#                             radius_threshold=input_data['pst_params']['radius_threshold'],
#                             origin_points="buildings_floors.shp")
#stepD.to_file("got_buildings_reach_test_Test2.shp")

#stepE=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
#                         crs=input_data['crs'],
#                         radius_type=input_data['pst_params']['radius_type'],
#                         radius_threshold=input_data['pst_params']['radius_threshold'],
#                         origin_points="got_buildings_reach_test_Test2.shp",
#                         destinations="buildings_floors.shp",
#                         weight_attr='B_Area',
#                         outputname='AFS')

#stepF=calculate_FSI(stepE,'RAaw500','AFS')
#stepF.to_file("got_buildings_reach_test_Test3.shp")

#stepG=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
#                         crs=input_data['crs'],
#                         radius_type=input_data['pst_params']['radius_type'],
#                         radius_threshold=input_data['pst_params']['radius_threshold'],
#                         origin_points="got_buildings_reach_test_Test3.shp",
#                         destinations="buildings_floors.shp",
#                         weight_attr='B_GFArea',
#                         outputname='AGS')
#stepH=calculate_GSI(stepG,'RAaw500','AGS')
#stepH.to_file("got_buildings_reach_test_Test4.shp")

#stepI=calculate_clusters(stepH, input_data['cluster_centers'])

#stepI.to_file(input_data['outputfile'])

#stepJ=calculate_clusters(stepH, input_data['cluster_centers2'])
#stepJ.to_file('results2.shp')

##


def calculate_clusters_2 (dataframe, cluster_centers):

    kmeans = KMeans(n_clusters=len(cluster_centers), init= cluster_centers, n_init=50, max_iter=5000).fit(dataframe[['GSI','FSI']].to_numpy())


    dataframe['clusters']=kmeans.labels_+1
    print(kmeans.cluster_centers_)
    return dataframe

#gbg_buildings=gpd.read_file("GOT_Building_Local_CL7_fixed_nonzero.shp")
#a=calculate_clusters_2(gbg_buildings, input_data['cluster_centers'])
#a.to_file("gbg_clusters_7.shp")
#b=calculate_clusters_2(gbg_buildings, input_data['cluster_centers2'])
#b.to_file("gbg_clusters_6.shp")
#c=calculate_clusters(gbg_buildings, input_data['cluster_centers'])
#c.to_file("gbg_clusters_noiter__7.shp")
#d=calculate_clusters(gbg_buildings, input_data['cluster_centers2'])
#d.to_file("gbg_clusters_noiter_6.shp")

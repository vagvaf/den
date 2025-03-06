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

import time

sys.path.append("C:/Users/vagva/OneDrive/Documents/PST_UC/pstqgis_3.3.1_2024-11-01/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii, DistanceType, OriginType

import warnings; 
warnings.simplefilter('ignore')


##################USER INPUT###########################################


input_data= {
            'country': 'sweden',
            'area': 'gbg.shp', #path to a polygon shapefile containing the study area
            'crs': 3006, #epsg number of the area's crs
            'pst_params':{
                'radius_type':'walking',
                'radius_threshold':500,
                'unlinks': '' #shapefile with unlinks
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
            'outputfile': "results.shp" #path to the output file
    }






####################FUNCTION DEFINITION#################################


#C:\Users\vagva\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyogrio\gdal_data

def get_buildings(country:str, crs:int, download=True):
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
    
    ###calculate the Ground Space (area) of each building
    df_filtered['GS']=df_filtered.area
    ###keep only buildings with area equal or larger than 15sqm
    df_filtered = df_filtered[df_filtered['GS'] >= 15]
    


    df_filtered['hght'] = df_filtered['height']
    df_filtered['lvl'] = df_filtered['building_levels']

    # Ensure height & level columns are numeric
    df_filtered['hght'] = pd.to_numeric(df_filtered['hght'], errors='coerce')
    df_filtered['lvl'] = pd.to_numeric(df_filtered['lvl'], errors='coerce')

    #compute the morphometric indicators
    compute_morphometric_indicators(df_filtered)

    endtime = time.time()
    print(f"Done ({int(endtime-starttime)}sec) ")

    return df_filtered


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

def height_level_conv(height):
    try:
        float(height)
        return(height)
    except (ValueError, TypeError):
        return None
    

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

def assign_type(building_type):
    if building_type == 'yes':
        return 0
    elif building_type in ['apartments', 'barracks', 'house', 'residential', 'bungalow', 'cabin', 'detached', 'dormitory', 'farm', 'static_caravan',
                              'semidetached_house', 'stilt_house']:
        return 1
    else:
        return 2
        
def assign_function(building_type):
    
    building_functions = {
                        'accommodation': {'apartments', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe', 'dormitory', 'farm',	
                                         'ger', 'hotel', 'house', 'houseboat', 'residential', 'semidetached_house', 'static_caravan',	
                                         'stilt_house', 'terrace', 'tree_house', 'trullo'},
                        'commercial':   {'commercial', 'industrial', 'kiosk', 'office', 'retail', 'supermarket', 'warehouse'},
                        'religious':    {'religious', 'cathedral', 'chapel', 'church', 'kingdom_hall', 'monastery', 'mosque', 'presbytery', 'shrine', 'synagogue', 'temple'},
                        'civic':        {'bakehouse', 'bridge', 'civic', 'college', 'fire_station', 'government', 'gatehouse', 'hospital',
                                        'kindergarten', 'museum', 'public', 'school', 'toilets', 'train_station', 'transportation', 'university'},
                        'agricultural': {'barn', 'conservatory', 'cowshed', 'farm_auxiliary', 'greenhouse', 'slurry_tank', 'stable', 'sty', 'livestock'},
                        'sports':       {'grandstand', 'pavilion', 'riding_hall', 'sports_hall', 'sports_centre', 'stadium'},
                        'storage':      {'allotment_house', 'boathouse', 'hangar', 'hut', 'shed'},
                        'cars':         {'carport', 'garage', 'garages', 'parking'},
                        'technical':    {'digester', 'service', 'tech_cab', 'transformer_tower', 'water_tower', 'storage_tank', 'silo'},
                        'other':        {'beach_hut', 'bunker', 'castle', 'construction', 'container', 'guardhouse', 'military', 'outbuilding',
                                          'pagoda', 'quonset_hut', 'roof', 'ruins', 'ship', 'tent', 'tower', 'triumphal_arch', 'windmill', 'yes'}
                        }

    for key in building_functions.keys():
        if building_type in building_functions[key]:
            return key

    
    

def floor_estimation(buildings, city_boundary:str, Training_ratio = 0.7):
    """https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%203%20-%20FLOOR%20ESTIMATION.ipynb"""

    starttime = time.time()
    print(f"Step B: Running floor estimations")


    
    # checks if height is Null and if floor is non Null
    # If both conditions are met : multiplies the value of floor by 3 and assigns it to height
    buildings['hght'] = buildings.apply(lambda row: row['lvl'] * 3 if pd.isna(row['hght']) and not pd.isna(row['lvl']) else row['hght'], axis=1)

    # checks if the height is not Null and if the floor is Null
    # If both conditions are met : divides the value of height by 3 and assigns it to level
    buildings['lvl'] = buildings.apply(lambda row: round(row['hght'] / 3) if pd.isna(row['lvl']) and not pd.isna(row['hght']) else row['lvl'], axis=1)

    #for some buildings we can guess the floor levels
    #buildings['lvl'] = buildings.apply(lambda row: 2 if pd.isna(row['lvl']) and row['building'] in ['house', 'detached', 'semidetached_house'] else row['lvl'], axis=1)
    #buildings['lvl'] = buildings.apply(lambda row: 1 if pd.isna(row['lvl']) and row['building'] in ['bungalow', 'cabin', 'static_caravan'] else row['lvl'], axis=1)

    
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
    print("\t Step 2.2: Training decision tree classifier")
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
    print("\t Step 2.2: Applying model to missing floor data")
    # Ensure that we are using the same features as those used during training
    X_null = building_null.drop(columns=['lvl'])

    # Make sure there are no additional columns
    X_null = X_null[X_train.columns]

    # Predict the types for building_null
    building_null = building_null.copy()
    building_null['lvl'] = clf.predict(X_null)

    # 2.3 APPLY THE TREE TO THE NULL VALUES
    print("\t Step 2.3: Applying decision tree to the entire dataset")
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

    ####keep only the buildings in our study area

    city = gpd.read_file(f"{city_boundary}")
    city = city.to_crs(building_final.crs)
    city_buildings = sjoin(building_final,city,how='inner')

    endtime = time.time()
    print(f"Done ({int(endtime-starttime)}sec) ")

    return city_buildings


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
    dataframe=dataframe[dataframe['RAaw500']>0]

    kmeans = KMeans(n_clusters=len(cluster_centers), init= cluster_centers, n_init=1, max_iter=1).fit(cluster_centers)
    

    kmeans_predict=kmeans.predict(dataframe[['GSI','FSI']].to_numpy())


    dataframe['clusters']=kmeans_predict+1
    return dataframe
    


###################SCRIPT EXECUTION####################################

#stepA=get_buildings(input_data['country'], input_data['crs'], download=False)
#stepA.to_file("buildings_sweden.shp")
buildingsdf=gpd.read_file("buildings_sweden.shp")

stepB=floor_estimation(buildingsdf, input_data['area'])
stepB.to_file("buildings_floors.shp")

stepC=get_streets(input_data['area'], input_data['crs'])
stepC.to_file("streetnonmotorized.shp")

stepD=perform_Reach_Analysis(road_network="streetnonmotorized.shp",
                             crs=input_data['crs'],
                             radius_type=input_data['pst_params']['radius_type'],
                             radius_threshold=input_data['pst_params']['radius_threshold'],
                             origin_points="buildings_floors.shp")
stepD.to_file("got_buildings_reach_test_Test2.shp")

stepE=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
                         crs=input_data['crs'],
                         radius_type=input_data['pst_params']['radius_type'],
                         radius_threshold=input_data['pst_params']['radius_threshold'],
                         origin_points="got_buildings_reach_test_Test2.shp",
                         destinations="buildings_floors.shp",
                         weight_attr='B_Area',
                         outputname='AFS')

stepF=calculate_FSI(stepE,'RAaw500','AFS')
stepF.to_file("got_buildings_reach_test_Test3.shp")

stepG=perform_Attraction_Reach_Analysis(road_network="streetnonmotorized.shp",
                         crs=input_data['crs'],
                         radius_type=input_data['pst_params']['radius_type'],
                         radius_threshold=input_data['pst_params']['radius_threshold'],
                         origin_points="got_buildings_reach_test_Test3.shp",
                         destinations="buildings_floors.shp",
                         weight_attr='B_GFArea',
                         outputname='AGS')
stepH=calculate_GSI(stepG,'RAaw500','AGS')
stepH.to_file("got_buildings_reach_test_Test4.shp")

stepI=calculate_clusters(stepH, input_data['cluster_centers'])

stepI.to_file(input_data['outputfile'])




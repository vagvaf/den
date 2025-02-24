import sys, array
import pandas as pd
import geopandas

#append pstalgo to path
#sys.path.append("/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii, DistanceType, OriginType


def perform_Reach_Analysis(road_network, crs, radius_type, radius_threshold, unlinks=None, origin_points=None):
    """ Perform a PST Reach Analysis
        Input:
            - road_network: shapefile with road network
            - crs: crs of the study area
            - radius_type: choose between one: straight, walking, steps, angular or axmeter
            - radius_threshold: the threshold based on the radius_type. Could be meters for straight or walking, number of steps for steps, degrees for angular
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
    
    roads = geopandas.read_file(road_network)
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
        unlinks = geopandas.read_file(unlinks)
        unlinks = unlinks.to_crs(crs)
        unlinks_coords = unlinks.get_coordinates()
        point_coords = []
        for index, row in unlinks_coords.iterrows():
            point_coords.append(row['x'])
            point_coords.append(row['y'])

        unlinks = array.array('d', point_coords)
        


    ### prepare the graph
    graph = pstalgo.CreateGraph(line_coords, None, unlinks, None, None)

    #If origin points were provided
    if origin_points:
        
        ### Import origin poins and set crs
        buildings = geopandas.read_file(origin_points)
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
def perform_Attraction_Reach_Analysis(road_network, crs, radius_type, radius_threshold, unlinks=None, origin_points=None, destinations=None, weight_attr=None):
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

    radius_args = {radius_type:radius_threshold}
    radius = Radii(**radius_args)
    
    roads = geopandas.read_file(road_network)
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
        unlinks = geopandas.read_file(unlinks)
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
        origins = geopandas.read_file(origin_points)
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
        destinations = geopandas.read_file(destinations)
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


    if origin_points:
        origins[f'ARA{radius_type[0]}{radius_threshold}']=pd.Series(scores)
        origins = origins.drop('cnt', axis=1)
        return origins
    
    else:
        roads[f'ARA{radius_type[0]}{radius_threshold}']=pd.Series(scores)
        return roads

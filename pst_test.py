import sys, array
import pandas as pd
import geopandas

sys.path.append("C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii

# Create a simple graph with 2 lines: (0,0)->(10,0) and (10,0)->(20,0)
line_coords = array.array('d', [0,0,10,0,10,0,20,0])  
graph = pstalgo.CreateGraph(line_coords, None, None, None, None)  # Or "CreateSegmentGraph" depending on analysis

# Create a radius definition
dist_type='walk'
dist = 500
radius = Radii(walking = 500)  

# Count number of lines in the line coordinate array. 
# There are 2 sets of coordinates for each line, with 
# 2 values for each coordinate => 4 values per line.
line_count = int(len(line_coords) / 4)

# Prepare an array for PST to write results into #in our case 2 because we have two lines
reached_count = array.array('I', [0]) * line_count #this is integer

# for attraction reach
#reached_count = array.array('f', [0]) * line_count #this is float



# Ask PST to do run a Reach analysis
pstalgo.Reach(
    graph_handle = graph,
    radius = radius,
    out_reached_count = reached_count)

# Print out the contents of the result array
print(reached_count)


# Free up the mnemory associated with the graph
pstalgo.FreeGraph(graph)  # Or "FreeSegmentGraph"


##################################REACH FOR AN ACTUAL NETWORK####################

roads = geopandas.read_file("C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/Density types_2017/roads/GOT/got_roads.shp")
pd.set_option('display.max_colwidth', None)

road_coords = roads.get_coordinates()
line_coords = []
for index, row in road_coords.iterrows():
    line_coords.append(row['x'])
    line_coords.append(row['y'])

line_coords = array.array('d', line_coords)

line_count = int(len(line_coords) / 4)

unlinks = geopandas.read_file("C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/Density types_2017/roads/GOT/got_unlinks.shp")
unlinks_coords = unlinks.get_coordinates()
point_coords = []
for index, row in unlinks_coords.iterrows():
    point_coords.append(row['x'])
    point_coords.append(row['y'])

point_coords = array.array('d', point_coords)

graph = pstalgo.CreateGraph(line_coords, None, point_coords, None, None)


# Prepare 2 arrays for PST to write results into. Size should the number of lines in the network
reached_count = array.array('I', [0]) * line_count #this is integer
reached_area = array.array('f', [0]) * line_count #this is integer

# for attraction reach
#reached_count = array.array('f', [0]) * line_count #this is float


# Ask PST to do run a Reach analysis
pstalgo.Reach(
    graph_handle = graph,
    radius = radius,
    out_reached_count = reached_count,
    out_reached_area = reached_area)


roads[f'RC{dist_type}{dist}']=pd.Series(reached_count)
roads[f'RA{dist_type}{dist}']=pd.Series(reached_area)

# Free up the mnemory associated with the graph
pstalgo.FreeGraph(graph)  # Or "FreeSegmentGraph"

print(roads.head(10))
print(roads.dtypes)




#####################################Attraction reach###########################

buildings = geopandas.read_file("C:/Users/evavaf/OneDrive - Chalmers/Buildingdensities/Density types_2017/roads/GOT/got_buildings.shp")

#calculate the building centroids
buildings['cnt']=buildings['geometry'].centroid

#get the x and y of the centroid
buildings['x'] = buildings['cnt'].x
buildings['y'] = buildings['cnt'].y

b_coords = []
weight = []

for index, row in buildings.iterrows():
    b_coords.append(row['x'])
    b_coords.append(row['y'])
    weight.append(row['B_Area'])

b_coords = array.array('d', b_coords)
weight = array.array('f', weight)

print("1")

scores = array.array('f', [0]) * line_count #this is integer
print("2")

print(len(b_coords),len(weight))
pstalgo.AttractionReach(
		graph_handle = graph,
                radius = radius,
		attraction_points=b_coords,
		attraction_values=weight,
		out_scores=scores)
print("3")
buildings['score'] = pd.Series(scores)



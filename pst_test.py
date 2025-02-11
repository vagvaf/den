import sys, array

sys.path.append("C:/Users/vagva/OneDrive/Documents/PST_UC/pstqgis_3.3.1_2024-11-01/pst/pstalgo/python")
import pstalgo

from pstalgo import Radii

# Create a simple graph with 2 lines: (0,0)->(10,0) and (10,0)->(20,0)
line_coords = array.array('d', [0,0,10,0,10,0,20,0])  
graph = pstalgo.CreateGraph(line_coords, None, None, None, None)  # Or "CreateSegmentGraph" depending on analysis

# Create a radius definition
radius = Radii(walking = 1000)  

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


pstalgo.AttractionReach(
		graph_handle, #this 
		origin_type=OriginType.LINES, 
		distance_type=DistanceType.WALKING, #walking
		radius=Radii(), #walking 
		weight_func=AttractionWeightFunction.CONSTANT,
		weight_func_constant=0,
		attraction_points=None, #this [0,0,10,10,20,20]
		points_per_attraction_polygon=None, 
		attraction_polygon_point_interval=0,
		attraction_values=None, #weigh by data [2.0,3.0,4.0] 
		attraction_distribution_func=AttractionDistributionFunc.DIVIDE,
		attraction_collection_func=AttractionCollectionFunc.AVARAGE,
		progress_callback = None, 
		out_scores=None)
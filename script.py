import urllib.request
import geopandas
import pandas as pd
import numpy as np
from geopandas.tools import sjoin
import momepy
import libpysal
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings



#C:\Users\vagva\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyogrio\gdal_data

def get_buildings_and_streets(country:str, city_boundary:str, crs:int, download=True):
    """Given the country and a shapefile as a city boundary
       this function downloads the .pbf data from geofabrik
       and extracts the buildings

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


    city_buildings = city_buildings[city_buildings['GS'] >= 15]


    #get streets
    df = geopandas.read_file("sweden-latest.osm.pbf",engine="pyogrio",layer = 'lines')
    df = df.to_crs(3006)
    filtered_df=df[df['highway'].notnull()]
    filtered_df = filtered_df.to_crs(3006)

    city_streets = sjoin(filtered_df, city, how='inner')

    return [city_buildings, city_streets]

def height_level_conv(height):
    try:
        #if ',' in height:
        #    float(height.split(",")[1].strip())
        #    return height.split(",")[1].strip()
        #elif 'm' in height:
        #    float(height.split(" ")[0])
        #    return height.split(" ")[0]
        #else:
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
    buildings.loc[:, 'E'] = momepy.Elongation(buildings).series

    # Convexity
    buildings.loc[:, 'C'] = momepy.Convexity(buildings).series

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



def floor_estimation(buildings, Training_ratio = 0.7):
    """https://github.com/perezjoan/Population-Potential-on-Catchment-Area---PPCA-Worldwide/blob/main/current%20release%20(v1.0.5)/STEP%203%20-%20FLOOR%20ESTIMATION.ipynb"""
    

    # Add a new column 'type' and apply the conditions
    buildings.loc[:, 'type'] = buildings['building'].apply(assign_type)
    buildings = buildings[buildings['type'] == 1]

    print(f"Running floor estimations")
    #we start by computing the morphometric indicators
    compute_morphometric_indicators(buildings)

    # Ensure height & level columns are numeric
    buildings['hght'] = pd.to_numeric(buildings['hght'], errors='coerce')
    buildings['lvl'] = pd.to_numeric(buildings['lvl'], errors='coerce')

    # checks if height is Null and if floor is non Null
    # If both conditions are met : multiplies the value of floor by 3 and assigns it to height
    buildings['hght'] = buildings.apply(lambda row: row['lvl'] * 3 if pd.isna(row['hght']) and not pd.isna(row['lvl']) else row['hght'], axis=1)

    # checks if the height is not Null and if the floor is Null
    # If both conditions are met : divides the value of height by 3 and assigns it to level
    buildings['lvl'] = buildings.apply(lambda row: round(row['hght'] / 3) if pd.isna(row['lvl']) and not pd.isna(row['hght']) else row['lvl'], axis=1)
    
    #calculate the building floorspace
    buildings = calculate_building_floorspace(buildings)

    ## 2. DECISION TREE CLASSIFIER TO EVALUATE THE MISSING NUMBER OF FLOORS
    print("Step 2: Decision tree classifier for missing floors")
    # 2.1 SUBSET DATA INTO TRAIN AND TEST DATA

    # List of columns to keep
    columns_to_keep = ['GS', 'P', 'E', 'C', 'FS', 'ECA', 'EA', 'SW', 'lvl']

    # Subset the DataFrame
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
    print("Step 2.2: Training decision tree classifier")
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
    print(f"Accuracy on test data: {accuracy:.2f}")

    # Apply the model to building_null
    print("Step 2.2: Applying model to missing floor data")
    # Ensure that we are using the same features as those used during training
    X_null = building_null.drop(columns=['lvl'])

    # Make sure there are no additional columns
    X_null = X_null[X_train.columns]

    # Predict the types for building_null
    building_null = building_null.copy()
    building_null['lvl'] = clf.predict(X_null)

    # 2.3 APPLY THE TREE TO THE NULL VALUES
    print("Step 2.3: Applying decision tree to the entire dataset")
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


    return building_final


def run_Reach_analysis():
    pass

def run_Attraction_Reach_analysis():
    pass

    
a=get_buildings_and_streets("sweden", "gbg", 3006, download=False)
b=floor_estimation(a[0])

b.to_file("bgbg_levels.shp")


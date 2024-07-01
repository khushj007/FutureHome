import numpy as np
import pandas as pd
import pathlib
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(path,path2):
    df = pd.read_csv(path/"gurgaon_properties_cleaned_v1.csv")
    df2 = pd.read_csv(path2/"appartments.csv")
    return df ,df2

def get_super_built_up_area(text):
    match = re.search(r'Super Built up area (\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

def get_area(text, area_type):
    match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

def convert_to_sqft(text, area_value):
    if area_value is None:
        return None
    match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
    if match:
        sq_m_value = float(match.group(1))
        return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
    return area_value

def extract_plot_area(area_with_type):
    match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
    return float(match.group(1)) if match else None

def convert_scale(row):
    if np.isnan(row['area']) or np.isnan(row['built_up_area']):
        return row['built_up_area']
    else:
        if round(row['area']/row['built_up_area']) == 9.0:
            return row['built_up_area'] * 9
        elif round(row['area']/row['built_up_area']) == 11.0:
            return row['built_up_area'] * 10.7
        else:
            return row['built_up_area']

def categorize_age_possession(value):
    if pd.isna(value):
        return "Undefined"
    if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
        return "New Property"
    if "1 to 5 Year Old" in value:
        return "Relatively New"
    if "5 to 10 Year Old" in value:
        return "Moderately Old"
    if "10+ Year Old" in value:
        return "Old Property"
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    try:
        # For entries like 'May 2024'
        int(value.split(" ")[-1])
        return "Under Construction"
    except:
        return "Undefined"

def get_furnishing_count(details, furnishing):
    if isinstance(details, str):
        if f"No {furnishing}" in details:
            return 0
        pattern = re.compile(rf"(\d+) {furnishing}")
        match = pattern.search(details)
        if match:
            return int(match.group(1))
        elif furnishing in details:
            return 1
    return 0

def feature_engineering(df,app_df):
    df['super_built_up_area'] = df['areaWithType'].apply(get_super_built_up_area)
    df['super_built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis=1)

    # Extract Built Up area and convert to sqft if needed
    df['built_up_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Built Up area'))
    df['built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

    # Extract Carpet area and convert to sqft if needed
    df['carpet_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
    df['carpet_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)

    all_nan_df = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]
    all_nan_index = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']].index
    all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(extract_plot_area)

    all_nan_df['built_up_area'] = all_nan_df.apply(convert_scale,axis=1)
    df.update(all_nan_df)

    new_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']

    # Populate the new columns based on the "additionalRoom" column
    for col in new_cols:
        df[col] = df['additionalRoom'].str.contains(col).astype(int)
    df.loc[:,'agePossession'] = df['agePossession'].apply(categorize_age_possession)

    all_furnishings = []
    for detail in df['furnishDetails'].dropna():
        furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
        all_furnishings.extend(furnishings)
    unique_furnishings = list(set(all_furnishings))

    # Simplify the furnishings list by removing "No" prefix and numbers
    columns_to_include = [re.sub(r'No |\d+', '', furnishing).strip() for furnishing in unique_furnishings]
    columns_to_include = list(set(columns_to_include))  # Get unique furnishings
    columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]  # Remove empty strings
    for furnishing in columns_to_include:
        df[furnishing] = df['furnishDetails'].apply(lambda x: get_furnishing_count(x, furnishing))
    
    # Create the new dataframe with the required columns
    furnishings_df = df[['furnishDetails'] + columns_to_include]

    furnishings_df=furnishings_df.drop(columns=['furnishDetails'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(furnishings_df)

    n_clusters = 3

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)

    # Predict the cluster assignments for each row
    cluster_assignments = kmeans.predict(scaled_data)

    df = df.iloc[:,:-18]
    df['furnishing_type'] = cluster_assignments
    app_df['PropertyName'] = app_df['PropertyName'].str.lower()
    temp_df = df[df['features'].isnull()]
    x = temp_df.merge(app_df,left_on='society',right_on='PropertyName',how='left')['TopFacilities']
    df.loc[temp_df.index,'features'] = x.values
    df.loc[:,'features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])
    mlb = MultiLabelBinarizer()
    features_binary_matrix = mlb.fit_transform(df['features_list'])

    # Convert the binary matrix into a DataFrame
    features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)
    weights = {
    '24/7 Power Backup': 8,
    '24/7 Water Supply': 4,
    '24x7 Security': 7,
    'ATM': 4,
    'Aerobics Centre': 6,
    'Airy Rooms': 8,
    'Amphitheatre': 7,
    'Badminton Court': 7,
    'Banquet Hall': 8,
    'Bar/Chill-Out Lounge': 9,
    'Barbecue': 7,
    'Basketball Court': 7,
    'Billiards': 7,
    'Bowling Alley': 8,
    'Business Lounge': 9,
    'CCTV Camera Security': 8,
    'Cafeteria': 6,
    'Car Parking': 6,
    'Card Room': 6,
    'Centrally Air Conditioned': 9,
    'Changing Area': 6,
    "Children's Play Area": 7,
    'Cigar Lounge': 9,
    'Clinic': 5,
    'Club House': 9,
    'Concierge Service': 9,
    'Conference room': 8,
    'Creche/Day care': 7,
    'Cricket Pitch': 7,
    'Doctor on Call': 6,
    'Earthquake Resistant': 5,
    'Entrance Lobby': 7,
    'False Ceiling Lighting': 6,
    'Feng Shui / Vaastu Compliant': 5,
    'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8,
    'Flower Garden': 7,
    'Food Court': 6,
    'Foosball': 5,
    'Football': 7,
    'Fountain': 7,
    'Gated Community': 7,
    'Golf Course': 10,
    'Grocery Shop': 6,
    'Gymnasium': 8,
    'High Ceiling Height': 8,
    'High Speed Elevators': 8,
    'Infinity Pool': 9,
    'Intercom Facility': 7,
    'Internal Street Lights': 6,
    'Internet/wi-fi connectivity': 7,
    'Jacuzzi': 9,
    'Jogging Track': 7,
    'Landscape Garden': 8,
    'Laundry': 6,
    'Lawn Tennis Court': 8,
    'Library': 8,
    'Lounge': 8,
    'Low Density Society': 7,
    'Maintenance Staff': 6,
    'Manicured Garden': 7,
    'Medical Centre': 5,
    'Milk Booth': 4,
    'Mini Theatre': 9,
    'Multipurpose Court': 7,
    'Multipurpose Hall': 7,
    'Natural Light': 8,
    'Natural Pond': 7,
    'Park': 8,
    'Party Lawn': 8,
    'Piped Gas': 7,
    'Pool Table': 7,
    'Power Back up Lift': 8,
    'Private Garden / Terrace': 9,
    'Property Staff': 7,
    'RO System': 7,
    'Rain Water Harvesting': 7,
    'Reading Lounge': 8,
    'Restaurant': 8,
    'Salon': 8,
    'Sauna': 9,
    'Security / Fire Alarm': 9,
    'Security Personnel': 9,
    'Separate entry for servant room': 8,
    'Sewage Treatment Plant': 6,
    'Shopping Centre': 7,
    'Skating Rink': 7,
    'Solar Lighting': 6,
    'Solar Water Heating': 7,
    'Spa': 9,
    'Spacious Interiors': 9,
    'Squash Court': 8,
    'Steam Room': 9,
    'Sun Deck': 8,
    'Swimming Pool': 8,
    'Temple': 5,
    'Theatre': 9,
    'Toddler Pool': 7,
    'Valet Parking': 9,
    'Video Door Security': 9,
    'Visitor Parking': 7,
    'Water Softener Plant': 7,
    'Water Storage': 7,
    'Water purifier': 7,
    'Yoga/Meditation Area': 7
    }
    # Calculate luxury score for each row
    luxury_score = features_binary_df[list(weights.keys())].multiply(list(weights.values())).sum(axis=1)
    df.loc[:,'luxury_score'] = luxury_score
    # cols to drop -> nearbyLocations,furnishDetails, features,features_list, additionalRoom
    df.drop(columns=['nearbyLocations','furnishDetails','features','features_list','additionalRoom'],inplace=True)
    return df 


def save_file(path,file):
    file.to_csv(path/'gurgaon_properties_cleaned_v2.csv',index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path1 = home_dir / "data" / "processed"
    input_path2 = home_dir / "data" / "raw"
    save_path = home_dir / "data" / "processed"
    
    df , df2 = load_data(input_path1,input_path2)
    df = feature_engineering(df,df2)
    
    save_file(save_path,df)


if __name__ == "__main__":
    main()
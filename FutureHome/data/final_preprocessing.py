import pandas as pd
import numpy as np
import pathlib

def load_data(path):
    df = pd.read_csv(path/"gurgaon_properties_missing_value_imputation.csv").drop_duplicates()
    return df 

def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  

def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None

def processing(df):
    df.drop(columns=["study room","pooja room","others","society","price_per_sqft"],inplace=True)
    df.loc[:,'luxury_category'] = df['luxury_score'].apply(categorize_luxury)
    df.loc[:,'floor_category'] = df['floorNum'].apply(categorize_floor)
    df.drop(columns=["luxury_score","floorNum"],inplace=True)
    df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
    df.drop(index=916,inplace=True)
    return df

def save_file(path,file):
    file.to_csv(path/'df_final.csv',index=False)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / "data" / "processed"
    save_path = home_dir / "data" / "processed"
    
    df = load_data(input_path)
    df = processing(df)
    
    save_file(save_path,df)


if __name__ == "__main__":
    main()
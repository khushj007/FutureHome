import numpy as np
import pandas as pd
import pathlib

def load_data(path):
    df = pd.read_csv(path/"gurgaon_properties_cleaned_v2.csv").drop_duplicates()
    return df 

def processing(df):
    Q1 = df['price_per_sqft'].quantile(0.25)
    Q3 = df['price_per_sqft'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]
    outliers_sqft.loc[:,'area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)
    outliers_sqft.loc[:,'price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])

    df.update(outliers_sqft)
    df = df[df['price_per_sqft'] <= 50000]
    df = df[df['area'] < 100000]

    df.drop(index=[3571, 2304, 1293, 79, 1292, 1415, 1289, 1294, 3042], inplace=True)
    df.loc[3222,'area'] = 1035
    df.loc[3294,'area'] = 7250
    df.loc[3471,'area'] = 5800
    df.loc[3049,'area'] = 2660
    df.loc[3533,'area'] = 2850
    df.loc[2399,'area'] = 1812
    df.loc[3056,'area'] = 2160
    df.loc[2801,'area'] = 1175
    df.loc[2399,'carpet_area'] = 1812
    df['price_per_sqft'] = round((df['price']*10000000)/df['area'])

    df["area_room_ratio"] = df["area"]/df["bedRoom"]
    df = df[df["area_room_ratio"] > 100]
    outliers_df = df[(df["area_room_ratio"] < 250) & (df["bedRoom"]>3)]
    outliers_df = round(outliers_df["bedRoom"]/outliers_df["floorNum"])
    df.update(outliers_df)
    df = df[~((df["area_room_ratio"] < 250) & (df["bedRoom"]>4))]
    return df

def save_file(path,file):
    file.to_csv(path/'gurgaon_properties_outlier_treated.csv',index=False)

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
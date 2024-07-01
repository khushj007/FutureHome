import pandas as pd
import numpy as np
import pathlib
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data(path):
    df = pd.read_csv(path/"df_final.csv").drop_duplicates()
    return df 

def model_training(df):
    
    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse_output=False),['agePossession']),
        ('target_enc', ce.TargetEncoder(), ['sector'])
    ], 
    remainder='passthrough')

    # model and hyperparemeters value received from model selection notebook
    pipeline = Pipeline([
        ("preprocessing",preprocessor),
        ("model",RandomForestRegressor(max_depth= 20,max_samples=1.0,n_estimators= 300))
    ])

    x = df.drop(columns="price")
    y=df["price"]

    pipeline.fit(x,y)

    return pipeline
    



def save_file(path,file):
    joblib.dump(file,path/"model_rf_1.pkl")

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / "data" / "processed"
    save_path = home_dir / "models"
    
    df = load_data(input_path)
    pipeline = model_training(df)
    
    save_file(save_path,pipeline)


if __name__ == "__main__":
    main()
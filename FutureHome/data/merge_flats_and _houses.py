import numpy as np
import pandas as pd
import pathlib



def load_data(path):
    flats = pd.read_csv(path/"flats_cleaned.csv")
    houses = pd.read_csv(path/"houses_cleaned.csv")
    return flats , houses

def merge(flats,houses):
    df = pd.concat([flats,houses],ignore_index=True)
    return df

def save_file(path,file):
    file.to_csv(path/'gurgaon_properties.csv',index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / "data" / "processed"
    save_path = home_dir / "data" / "processed"
    
    flats , houses = load_data(input_path)
    df = merge(flats,houses)
    
    save_file(save_path,df)

if __name__ == "__main__":
    main()
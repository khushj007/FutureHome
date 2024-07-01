import numpy as np
import pandas as pd
import pathlib
import re 


def load_data(path):
    df = pd.read_csv(path/"houses.csv")
    return df

def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)


def processing(df):
    df.drop(columns=['link','property_id'], inplace=True)
    df.rename(columns={'rate':'price_per_sqft'},inplace=True)
    df.loc[:,'society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    df.loc[:,'society'] = df['society'].str.replace('nan','independent')
    df = df[df['price'] != 'Price on Request']
    df.loc[:,'price'] = df['price'].str.split(' ').apply(treat_price)
    df.loc[:,'price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
    df = df[~df['bedRoom'].isnull()]
    df.loc[:,'bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df.loc[:,'bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df.loc[:,'balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
    df.loc[:,"additionalRoom"]=df['additionalRoom'].fillna('not available')
    df.loc[:,"additionalRoom"]=df['additionalRoom'] = df['additionalRoom'].str.lower()
    df.loc[:,'noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
    df.rename(columns={'noOfFloor':'floorNum'},inplace=True)
    df.loc[:,'facing']=df['facing'].fillna('NA')
    df['area'] = round((df['price']*10000000)/df['price_per_sqft'])
    df.insert(loc=1,column='property_type',value='house')

    return df


def save_file(path,file):
    file.to_csv(path / 'houses_cleaned.csv',index=False)




def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / "data" / "raw"
    save_path = home_dir / "data" / "processed"
    
    df = load_data(input_path)
    df = processing(df)
    
    save_file(save_path,df)

if __name__ == "__main__":
    main()
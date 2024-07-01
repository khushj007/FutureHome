import numpy as np
import pandas as pd
import pathlib

def load_data(path):
    df = pd.read_csv(path/"gurgaon_properties_outlier_treated.csv").drop_duplicates()
    return df 

def mode_based_imputation(row,df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation2(row,df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation3(row,df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def imputation(df):
    all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]
    super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()
    carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()

    # only buildup_area area is absent 
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    sbc_df.loc[:,'built_up_area']=sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2))
    df.update(sbc_df)

    # sb present c is null built up null
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
    sb_df.loc[:,'built_up_area']=sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105))
    df.update(sb_df)

    # sb null c is present built up null
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
    c_df.loc[:,'built_up_area']=c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9))
    df.update(c_df)


    anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]
    anamoly_df['built_up_area'] = anamoly_df['area']
    df.update(anamoly_df)

    df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)

    df.loc[:,'floorNum']=df['floorNum'].fillna(2.0)
    df.drop(columns=['facing'],inplace=True)

    
    val = df[df["society"].isnull()].index
    df.drop(index=val,inplace=True)

    df['agePossession'] = df.apply(lambda x : mode_based_imputation(x,df),axis=1)
    df['agePossession'] = df.apply(lambda x : mode_based_imputation2(x,df),axis=1)
    df['agePossession'] = df.apply(lambda x : mode_based_imputation3(x,df),axis=1)

    return df

def save_file(path,file):
    file.to_csv(path/'gurgaon_properties_missing_value_imputation.csv',index=False)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / "data" / "processed"
    save_path = home_dir / "data" / "processed"
    
    df = load_data(input_path)
    df = imputation(df)
    
    save_file(save_path,df)


if __name__ == "__main__":
    main()
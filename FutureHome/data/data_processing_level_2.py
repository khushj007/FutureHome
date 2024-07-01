import numpy as np
import pandas as pd
import pathlib


def load_data(path):
    df = pd.read_csv(path/"gurgaon_properties.csv")
    return df

def processing(df):
    df.insert(loc=3,column='sector',value=df['property_name'].str.split('in').str.get(1).str.replace('Gurgaon','').str.strip())
    df.loc[:,'sector'] = df['sector'].str.lower()
    df.loc[:,'sector'] = df['sector'].str.replace('dharam colony','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('krishna colony','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('suncity','sector 54')
    df.loc[:,'sector'] = df['sector'].str.replace('prem nagar','sector 13')
    df.loc[:,'sector'] = df['sector'].str.replace('mg road','sector 28')
    df.loc[:,'sector'] = df['sector'].str.replace('gandhi nagar','sector 28')
    df.loc[:,'sector'] = df['sector'].str.replace('laxmi garden','sector 11')
    df.loc[:,'sector'] = df['sector'].str.replace('shakti nagar','sector 11')
    df.loc[:,'sector'] = df['sector'].str.replace('baldev nagar','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('shivpuri','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('garhi harsaru','sector 17')
    df.loc[:,'sector'] = df['sector'].str.replace('imt manesar','manesar')
    df.loc[:,'sector'] = df['sector'].str.replace('adarsh nagar','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('shivaji nagar','sector 11')
    df.loc[:,'sector'] = df['sector'].str.replace('bhim nagar','sector 6')
    df.loc[:,'sector'] = df['sector'].str.replace('madanpuri','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('saraswati vihar','sector 28')
    df.loc[:,'sector'] = df['sector'].str.replace('arjun nagar','sector 8')
    df.loc[:,'sector'] = df['sector'].str.replace('ravi nagar','sector 9')
    df.loc[:,'sector'] = df['sector'].str.replace('vishnu garden','sector 105')
    df.loc[:,'sector'] = df['sector'].str.replace('bhondsi','sector 11')
    df.loc[:,'sector'] = df['sector'].str.replace('surya vihar','sector 21')
    df.loc[:,'sector'] = df['sector'].str.replace('devilal colony','sector 9')
    df.loc[:,'sector'] = df['sector'].str.replace('valley view estate','gwal pahari')
    df.loc[:,'sector'] = df['sector'].str.replace('mehrauli  road','sector 14')
    df.loc[:,'sector'] = df['sector'].str.replace('jyoti park','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('ansal plaza','sector 23')
    df.loc[:,'sector'] = df['sector'].str.replace('dayanand colony','sector 6')
    df.loc[:,'sector'] = df['sector'].str.replace('sushant lok phase 2','sector 55')
    df.loc[:,'sector'] = df['sector'].str.replace('chakkarpur','sector 28')
    df.loc[:,'sector'] = df['sector'].str.replace('greenwood city','sector 45')
    df.loc[:,'sector'] = df['sector'].str.replace('subhash nagar','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('sohna road road','sohna road')
    df.loc[:,'sector'] = df['sector'].str.replace('malibu town','sector 47')
    df.loc[:,'sector'] = df['sector'].str.replace('surat nagar 1','sector 104')
    df.loc[:,'sector'] = df['sector'].str.replace('new colony','sector 7')
    df.loc[:,'sector'] = df['sector'].str.replace('mianwali colony','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('jacobpura','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('rajiv nagar','sector 13')
    df.loc[:,'sector'] = df['sector'].str.replace('ashok vihar','sector 3')
    df.loc[:,'sector'] = df['sector'].str.replace('dlf phase 1','sector 26')
    df.loc[:,'sector'] = df['sector'].str.replace('nirvana country','sector 50')
    df.loc[:,'sector'] = df['sector'].str.replace('palam vihar','sector 2')
    df.loc[:,'sector'] = df['sector'].str.replace('dlf phase 2','sector 25')
    df.loc[:,'sector'] = df['sector'].str.replace('sushant lok phase 1','sector 43')
    df.loc[:,'sector'] = df['sector'].str.replace('laxman vihar','sector 4')
    df.loc[:,'sector'] = df['sector'].str.replace('dlf phase 4','sector 28')
    df.loc[:,'sector'] = df['sector'].str.replace('dlf phase 3','sector 24')
    df.loc[:,'sector'] = df['sector'].str.replace('sushant lok phase 3','sector 57')
    df.loc[:,'sector'] = df['sector'].str.replace('dlf phase 5','sector 43')
    df.loc[:,'sector'] = df['sector'].str.replace('rajendra park','sector 105')
    df.loc[:,'sector'] = df['sector'].str.replace('uppals southend','sector 49')
    df.loc[:,'sector'] = df['sector'].str.replace('sohna','sohna road')
    df.loc[:,'sector'] = df['sector'].str.replace('ashok vihar phase 3 extension','sector 5')
    df.loc[:,'sector'] = df['sector'].str.replace('south city 1','sector 41')
    df.loc[:,'sector'] = df['sector'].str.replace('ashok vihar phase 2','sector 5')
    a = df['sector'].value_counts()[df['sector'].value_counts() >= 3]
    df = df[df['sector'].isin(a.index)]
    df.loc[:,'sector'] = df['sector'].str.replace('sector 95a','sector 95')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 23a','sector 23')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 12a','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 3a','sector 3')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 110 a','sector 110')
    df.loc[:,'sector'] = df['sector'].str.replace('patel nagar','sector 15')
    df.loc[:,'sector'] = df['sector'].str.replace('a block sector 43','sector 43')
    df.loc[:,'sector'] = df['sector'].str.replace('maruti kunj','sector 12')
    df.loc[:,'sector'] = df['sector'].str.replace('b block sector 43','sector 43')
    df.loc[:,'sector'] = df['sector'].str.replace('sector-33 sohna road','sector 33')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 1 manesar','manesar')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 4 phase 2','sector 4')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 1a manesar','manesar')
    df.loc[:,'sector'] = df['sector'].str.replace('c block sector 43','sector 43')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 89 a','sector 89')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 2 extension','sector 2')
    df.loc[:,'sector'] = df['sector'].str.replace('sector 36 sohna road','sector 36')
    df[df['sector'] == 'new']
    df.loc[955,'sector'] = 'sector 37'
    df.loc[2800,'sector'] = 'sector 92'
    df.loc[2838,'sector'] = 'sector 90'
    df.loc[2857,'sector'] = 'sector 76'
    df[df['sector'] == 'new sector 2']
    df.loc[[311,1072,1486,3040,3875],'sector'] = 'sector 110'
    df.drop(columns=['property_name', 'address', 'description', 'rating'],inplace=True)
    
    return df



def save_file(path,file):
    file.to_csv(path / 'gurgaon_properties_cleaned_v1.csv',index=False)


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
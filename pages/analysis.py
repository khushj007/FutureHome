import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


st.set_page_config(page_title="Analysis")
st.title('Analytics')

current_dir = pathlib.Path(__file__)
home_dir = current_dir.parent.parent

feature = joblib.load(home_dir/"features.pkl")

new_df = pd.read_csv(home_dir / 'data/processed/viz_df.csv')
new_df["longitude"]=new_df["longitude"].astype(np.float64)
new_df["latitude"]=new_df["latitude"].astype(np.float64)

group_df = new_df[["sector",'price','price_per_sqft','built_up_area','latitude','longitude']].groupby('sector').mean()

st.header('Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(group_df, lat="longitude", lon="latitude", color="price_per_sqft", size='built_up_area',
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                  mapbox_style="open-street-map",width=1200,height=700,hover_name=group_df.index)

st.plotly_chart(fig,use_container_width=True)


# wordcloud

st.header('Features Wordcloud')
sector = st.selectbox('Select Sector', feature["sector"].tolist())

if sector :
    values = feature[feature["sector"] == sector]["features"].values[0]
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          stopwords = set(['s']),  # Any stopwords you'd like to exclude
                          min_font_size = 10).generate(", ".join(values))

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=None)
    
    # Display the word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    # Adjust layout
    plt.tight_layout(pad=0)
    
    # Display the figure using Streamlit
    st.pyplot(fig)


st.header('Area Vs Price')

property_type = st.selectbox('Select Property Type', ['flat','house'])

if property_type == 'house':
    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)
else:
    fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)


st.header('BHK Pie Chart')

sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0,'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':

    fig2 = px.pie(new_df, names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)
else:

    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

    st.plotly_chart(fig2, use_container_width=True)

st.header('Side by Side BHK price comparison')



selected_sector2 = st.selectbox('Select', sector_options)

if selected_sector2 == "overall" :
    fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')
    st.plotly_chart(fig3, use_container_width=True)
else :
    fig3 = px.box(new_df[(new_df['sector'] == selected_sector2)], x='bedRoom', y='price', title='BHK Price Range')
    st.plotly_chart(fig3, use_container_width=True)
    

st.header('Side by Side Distplot for property type')


selected_sector3 = st.selectbox('Select sector', sector_options)

if selected_sector3 == "overall" :
    fig3 = plt.figure(figsize=(10, 4))
    sns.distplot(new_df[new_df['property_type'] == 'house']['price'],label='house')
    sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
    plt.legend()
    st.pyplot(fig3)
else:
    fig3 = plt.figure(figsize=(10, 4))
    sns.distplot(new_df[(new_df['property_type'] == 'house') & (new_df["sector"] == selected_sector3)]['price'],label='house')
    sns.distplot(new_df[(new_df['property_type'] == 'flat')& (new_df["sector"] == selected_sector3)]['price'], label='flat')
    plt.legend()
    st.pyplot(fig3)

    

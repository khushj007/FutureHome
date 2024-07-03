import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pathlib

st.set_page_config(page_title="Recommend Appartments")

current_dir = pathlib.Path(__file__)
home_dir = current_dir.parent.parent

location_df = joblib.load(home_dir / "projectData"/"location_df.pkl")
cosine_sim1 = joblib.load(home_dir / "projectData"/"cosine_sim1.pkl")
cosine_sim2 = joblib.load(home_dir / "projectData"/"cosine_sim2.pkl")
cosine_sim3 = joblib.load(home_dir / "projectData"/"cosine_sim3.pkl")
appartment_df = pd.read_csv(home_dir/"data"/"raw"/"appartments.csv")



def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
    top_properties = location_df.index[top_indices].tolist()
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    return recommendations_df.merge(appartment_df,on="PropertyName")[["PropertyName","SimilarityScore","Link"]]


st.title('Select Location and Radius')

selected_location = st.selectbox('Location',sorted(location_df.columns.to_list()))

min_distance = int(location_df[selected_location].min()/1000)

max_distance = int(location_df[selected_location].max()/1000)

radius = st.slider('Select a range of values', min_distance, max_distance+1)

if st.button('Search'):
    result_ser = location_df[location_df[selected_location] <= radius*1000][selected_location].sort_values()
    st.title('Recommendation within radius')
    for key, value in result_ser.items():
        with st.expander(str(key) + " --> " + str(round(value/1000,2)) + ' kms'):
            st.write(f"Appartments similar to  ' {key} '")
            recommendation_df = recommend_properties_with_scores(key)
            st.dataframe(recommendation_df)





import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from src.models.collaborative_filtering import CollaborativeFiltering

st.set_page_config(page_title="Get Recommendations", page_icon="📊")

# models = ("dtc", "gnb", "mnb", "pn")
#
# option = st.selectbox(
#     "Which model would you like to use for computing recommendations?",
#     models,
#     index=None,
#     placeholder="Select a model..",
# )


artists_df = pd.read_csv(Path() / "data/processed/lastfm_2k/artists.csv")
artists = artists_df.name
selected_artists = st.multiselect(
    "Choose min. 10 artists that you like", options=artists
)
dataset = pd.read_csv(Path() / "data/processed/lastfm_2k/user_artists.csv")

if (
    st.button("Get artists recommendations", type="primary")
    and len(selected_artists) > 9
):
    num_selected_artists = len(selected_artists)
    st.write("Recommendations are being computed...")
    ids = list(artists_df[artists_df.name.isin(selected_artists)].id)
    new_data = pd.DataFrame(
        {
            "userID": [1] * num_selected_artists,
            "artistID": list(ids),
            "weight": [1] * num_selected_artists,
        }
    )
    dataset = pd.concat([dataset, new_data], ignore_index=True)
    model = CollaborativeFiltering(
        model_name="Collaborative Filtering", dataset=dataset
    )
    new_user_recommendations = model.get_recommendations(dataset).iloc[0]
    artist_names = artists_df[artists_df.id.isin(list(new_user_recommendations))].name
    st.dataframe(artist_names)

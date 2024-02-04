import time
from pathlib import Path

import pandas as pd
import streamlit as st
from src.models.streamlit_models.collaborative_filtering import CollaborativeFiltering
from pages.streamlit_utils.make_data_from_s3 import get_dataframes

st.set_page_config(page_title="Get Recommendations", page_icon="📊")

top_n_options = (10, 20, 30, 40)

top_n = st.selectbox(
    "How many recommended artists do you want to get?",
    top_n_options,
    index=None,
    placeholder="Select an amount..",
)

artists_df, _, DATASET, _, _ = get_dataframes()
artists = artists_df.name
selected_artists = st.multiselect(
    "Choose min. 10 artists that you like",
    options=artists,
    placeholder="Choose artists...",
)


def append_new_user_data(selected_items: list[str]):
    ids = list(artists_df[artists_df.name.isin(selected_artists)].id)
    new_data = pd.DataFrame(
        {
            "userID": [1] * len(selected_items),
            "artistID": list(ids),
            "weight": [1] * len(selected_items),
        }
    )
    return pd.concat([DATASET, new_data], ignore_index=True)


def display_artists(artist_dict, top_tracks):
    heart_emojis = ["🧡", "💛", "💚", "💙", "💜", "❤️"]
    for i, (name, link) in enumerate(artist_dict.items()):
        heart_emoji = heart_emojis[i % len(heart_emojis)]
        st.markdown(f"{heart_emoji} {name} - {link}")
        tracks = ", ".join(top_tracks.get(name, []))
        st.text(f"\v\t⭐️Top Songs: {tracks}")


def display_recommendations(selected_items: list[str], top_n_rec: int):
    updated_dataset = append_new_user_data(selected_items)
    model = CollaborativeFiltering(
        model_name="Collaborative Filtering", dataset=updated_dataset
    )
    new_user_recommendations = model.get_recommendations(
        updated_dataset, top_n_rec
    ).iloc[0]
    artist_names = artists_df[artists_df.id.isin(list(new_user_recommendations))].name
    artist_urls = artists_df[artists_df.id.isin(list(new_user_recommendations))].url
    top_tracks_dict = {}
    artist_dict = dict(
        zip(
            artist_names,
            artist_urls,
        )
    )
    display_artists(artist_dict, top_tracks_dict)


if st.button("Get artists recommendations", type="primary"):
    if not (len(selected_artists) > 9 or not top_n_options):
        st.write(
            "You have to choose number of recommendations and choose more than 10 artists."
        )
    else:
        computation_message = st.empty()

        computation_message.write("Recommendations will appear shortly.")
        time.sleep(3)
        computation_message.empty()
        display_recommendations(selected_artists, top_n)
